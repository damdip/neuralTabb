# modules/weaviate_manager.py
import pathlib
import weaviate
import json
import pandas as pd
from typing import List, Dict, Any
import uuid
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from pandasToWeaviate import  get_properties_from_map, deterministic_weaviate_types
from weaviateMain import checkExistingCollection, extractChunksAndInsertIntoWeaviateProgressBar, resetSchema, createSchema, createElementData, insertElement
import re
import requests
from datetime import datetime

# Per supporto Excel
try:
    import openpyxl
    import xlrd
except ImportError:
    print("Avviso: openpyxl e xlrd non installati. Supporto Excel limitato.")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None



class WeaviateManager:
    def __init__(self, client):
        self.client = client
    
    def setup_schema(self):
        """Crea gli schemi di base"""
        try:
            # Controlla se la collezione Documents esiste già
            try:
                collection = self.client.collections.get("Documents")
                print("Collezione Documents già esistente")
                return True
            except:
                # Crea la collezione Documents
                self.client.collections.create(
                    name="Documents",
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(),
                    properties=[
                        weaviate.classes.config.Property(name="title", data_type=weaviate.classes.config.DataType.TEXT),
                        weaviate.classes.config.Property(name="content", data_type=weaviate.classes.config.DataType.TEXT),
                        weaviate.classes.config.Property(name="source", data_type=weaviate.classes.config.DataType.TEXT),
                        weaviate.classes.config.Property(name="category", data_type=weaviate.classes.config.DataType.TEXT),
                        weaviate.classes.config.Property(name="timestamp", data_type=weaviate.classes.config.DataType.DATE),
                    ]
                )
                print("Collezione Documents creata")
                return True
            
        except Exception as e:
            print(f"Errore setup schema: {e}")
            return False
    
    def process_file(self, filepath: str) -> Dict[str, Any]:
        collection_name = pathlib.Path(filepath).stem
        """Processa un file e inserisce i documenti"""
        try:
            if ( checkExistingCollection(self.client ,collection_name) ):
                print(f"Collezione {collection_name} già esistente")
                return {"status": "success", "inserted": 0, "errors": 0}
            
            collection = self.client.collections.get("Documents")
            inserted = 0
            errors = 0

            

            if filepath.endswith(('.xlsx', '.xls')):
                # Approccio flessibile: ogni colonna viene inserita come dato separato
                try:
                    # Leggi il file Excel
                    df = pd.read_excel(filepath, engine='openpyxl' if filepath.endswith('.xlsx') else 'xlrd')
                    
                    columns = df.columns.tolist()
                    print(f"File Excel caricato. Righe: {len(df)}, Colonne: {columns}")
                    
                    #Crea mappa a partire dai dati del dataframe
                    mappings, type_map = deterministic_weaviate_types(df)
                    # Crea properties da passare a weaviate per la creazione della classe
                    properties = get_properties_from_map(type_map)

                    createSchema(self.client, collection_name, properties)
                    print(f"Schema creato per la collezione {collection_name}")


                    try:
                        extractChunksAndInsertIntoWeaviateProgressBar(self.client, collection_name, df)
                    except Exception as e:
                        errors += 1
                        print(f"Errore inserimento file Excel: {e}")
                                
                except Exception as e:
                    return {"inserted": 0, "errors": 1, "status": "error", "error": f"Errore lettura Excel: {str(e)}"}
                    
            else:
                print("Formato file non supportato. Solo Excel (.xlsx, .xls) al momento.")
                return {"status": "error", "inserted": 0, "errors": 0}

            status_msg = "success"
            if errors > 0:
                status_msg = f"partial_success - {errors} errori"
                
            return {
                "inserted": inserted, 
                "errors": errors,
                "status": status_msg
            }
            
        except Exception as e:
            return {"inserted": 0, "errors": 1, "status": "error", "error": str(e)}
    
    def _get_safe_value(self, row, column_name: str) -> str:
        """Estrae il valore dalla riga in modo sicuro"""
        try:
            value = row.get(column_name) if hasattr(row, 'get') else row[column_name]
            if pd.isna(value) or value is None or value == '':
                return ''
            return str(value).strip()
        except (KeyError, IndexError, Exception):
            return ''
    
    def _map_excel_columns(self, columns: List[str]) -> Dict[str, str]:
        """Mappa automaticamente le colonne Excel ai campi richiesti"""
        mapping = {"title": None, "content": None, "category": None}
        
        # Converti colonne in lowercase per confronto
        columns_lower = [col.lower() for col in columns]
        
        # Mappatura intelligente per TITLE
        title_keywords = ['title', 'titolo', 'nome', 'name', 'subject', 'oggetto', 'headline']
        for keyword in title_keywords:
            for i, col_lower in enumerate(columns_lower):
                if keyword in col_lower:
                    mapping['title'] = columns[i]
                    break
            if mapping['title']:
                break
        
        # Se non trova title, usa la prima colonna
        if not mapping['title'] and columns:
            mapping['title'] = columns[0]
        
        # Mappatura intelligente per CONTENT
        content_keywords = ['content', 'contenuto', 'text', 'testo', 'description', 'descrizione', 
                           'body', 'message', 'messaggio', 'detail', 'dettaglio', 'summary']
        for keyword in content_keywords:
            for i, col_lower in enumerate(columns_lower):
                if keyword in col_lower and columns[i] != mapping['title']:
                    mapping['content'] = columns[i]
                    break
            if mapping['content']:
                break
        
        # Se non trova content, usa la seconda colonna o la più lunga
        if not mapping['content'] and len(columns) > 1:
            remaining_cols = [col for col in columns if col != mapping['title']]
            mapping['content'] = remaining_cols[0] if remaining_cols else None
        
        # Mappatura intelligente per CATEGORY
        category_keywords = ['category', 'categoria', 'type', 'tipo', 'class', 'classe', 
                            'tag', 'label', 'etichetta', 'group', 'gruppo']
        for keyword in category_keywords:
            for i, col_lower in enumerate(columns_lower):
                if keyword in col_lower and columns[i] not in [mapping['title'], mapping['content']]:
                    mapping['category'] = columns[i]
                    break
            if mapping['category']:
                break
        
        return mapping
    
    def _get_mapped_value(self, row, column_name: str, default: str = '') -> str:
        """Estrae il valore dalla riga usando il nome della colonna mappata"""
        if not column_name:
            return default
            
        try:
            value = row.get(column_name, default)
            if pd.isna(value) or value is None:
                return default
            return str(value).strip()
        except Exception:
            return default
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """Lista tutte le collezioni"""
        try:
            collections = []
            
            # Ottieni tutte le collezioni esistenti
            all_collection_names = self.client.collections.list_all()
            
            for collection_name in all_collection_names:
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Conta documenti
                    response = collection.aggregate.over_all(total_count=True)
                    count = response.total_count
                    
                    # Ottieni schema per contare proprietà
                    # Prova a ottenere un documento di esempio per vedere le proprietà
                    try:
                        sample = collection.query.fetch_objects(limit=1)
                        properties_count = len(sample.objects[0].properties.keys()) if sample.objects else 0
                    except:
                        properties_count = 0
                    
                    # Ottieni configurazione vectorizer (se disponibile)
                    try:
                        # La configurazione del vectorizer potrebbe non essere facilmente accessibile
                        # Per ora usiamo un valore di default
                        vectorizer = "text2vec-transformers"
                    except:
                        vectorizer = "unknown"
                    
                    collections.append({
                        "name": collection_name,
                        "count": count,
                        "properties": properties_count,
                        "vectorizer": vectorizer
                    })
                    
                except Exception as e:
                    # Se c'è un errore con una collezione specifica, aggiungi info minime
                    collections.append({
                        "name": collection_name,
                        "count": 0,
                        "properties": 0,
                        "vectorizer": "error",
                        "error": str(e)
                    })
            
            return collections
            
        except Exception as e:
            print(f"Errore lista collezioni: {e}")
            return []

    def create_collection(self, name: str, properties: List[Dict[str, Any]]) -> bool:
        """Crea una nuova collezione"""
        try:
            # Conversione proprietà al formato v4
            props = []
            for prop in properties:
                props.append(
                    weaviate.classes.config.Property(
                        name=prop["name"], 
                        data_type=weaviate.classes.config.DataType.TEXT
                    )
                )
            
            self.client.collections.create(
                name=name,
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(),
                properties=props
            )
            return True
            
        except Exception as e:
            print(f"Errore creazione collezione: {e}")
            return False


class DataAnalyzer:
    def __init__(self, client):
        self.client = client
    
    def get_basic_stats(self, collection_name: str = "Documents") -> Dict[str, Any]:
        """Statistiche base della collezione"""
        try:
            # Ottieni collezione
            collection = self.client.collections.get(collection_name)
            
            # Conta documenti totali
            count_response = collection.aggregate.over_all(total_count=True)
            total_count = count_response.total_count
            
            # Ottieni campione per analisi
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["title", "content", "source", "category"]
            )
            
            documents = response.objects
            
            # Analisi categorie
            categories = [doc.properties.get('category') for doc in documents if doc.properties.get('category')]
            category_counts = Counter(categories)
            
            # Analisi fonti
            sources = [doc.properties.get('source') for doc in documents if doc.properties.get('source')]
            source_counts = Counter(sources)
            
            # Analisi lunghezza contenuti
            content_lengths = [len(doc.properties.get('content', '')) for doc in documents if doc.properties.get('content')]
            
            return {
                "total_documents": total_count,
                "analyzed_sample": len(documents),
                "categories": dict(category_counts.most_common(10)),
                "sources": dict(source_counts.most_common(10)),
                "content_stats": {
                    "avg_length": np.mean(content_lengths) if content_lengths else 0,
                    "min_length": min(content_lengths) if content_lengths else 0,
                    "max_length": max(content_lengths) if content_lengths else 0
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_clusters(self, collection_name: str = "Documents", n_clusters: int = 5) -> Dict[str, Any]:
        """Analizza cluster semantici"""
        try:
            # Ottieni documenti con vettori
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=500,
                return_properties=["title", "content"],
                include_vector=True
            )
            
            documents = response.objects
            
            if len(documents) < n_clusters:
                return {"error": "Troppi pochi documenti per il clustering"}
            
            # Estrai vettori
            vectors = np.array([doc.vector["default"] for doc in documents])
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            # Analizza cluster
            cluster_info = {}
            for i in range(n_clusters):
                cluster_docs = [documents[j] for j in range(len(documents)) if clusters[j] == i]
                
                # Titoli rappresentativi
                titles = [doc.properties.get('title', '') for doc in cluster_docs[:5]]
                
                # Parole chiave comuni
                contents = [doc.properties.get('content', '') for doc in cluster_docs]
                keywords = self._extract_keywords_from_texts(contents)
                
                cluster_info[f"cluster_{i}"] = {
                    "size": len(cluster_docs),
                    "sample_titles": titles,
                    "keywords": keywords[:10]
                }
            
            return cluster_info
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_topics(self, collection_name: str = "Documents", n_topics: int = 5) -> List[Dict[str, Any]]:
        """Estrae topic usando LDA"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["content"]
            )
            
            documents = response.objects
            texts = [doc.properties.get('content', '') for doc in documents if doc.properties.get('content')]
            
            if len(texts) < n_topics:
                return []
            
            # Vettorizzazione TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # LDA
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Estrai topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weight": topic.max()
                })
            
            return topics
            
        except Exception as e:
            return []
    
    def _extract_keywords_from_texts(self, texts: List[str]) -> List[str]:
        """Estrae parole chiave da una lista di testi"""
        try:
            all_text = " ".join(texts)
            
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([all_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Ordina per punteggio
            word_scores = list(zip(feature_names, tfidf_scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [word for word, score in word_scores]
            
        except Exception as e:
            return []


class DataCleaner:
    def __init__(self, client):
        self.client = client
    
    def find_duplicates(self, collection_name: str = "Documents", threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Trova documenti duplicati"""
        try:
            # Ottieni documenti con vettori
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["title", "content"],
                include_vector=True
            )
            
            documents = response.objects
            
            if len(documents) < 2:
                return []
            
            # Calcola similarità
            vectors = np.array([doc.vector["default"] for doc in documents])
            similarities = cosine_similarity(vectors)
            
            duplicates = []
            processed = set()
            
            for i in range(len(documents)):
                if i in processed:
                    continue
                    
                for j in range(i + 1, len(documents)):
                    if j in processed:
                        continue
                        
                    if similarities[i][j] > threshold:
                        duplicates.append({
                            "doc1": {
                                "id": str(documents[i].uuid),
                                "title": documents[i].properties.get('title', ''),
                                "content": documents[i].properties.get('content', '')[:100] + "..."
                            },
                            "doc2": {
                                "id": str(documents[j].uuid),
                                "title": documents[j].properties.get('title', ''),
                                "content": documents[j].properties.get('content', '')[:100] + "..."
                            },
                            "similarity": float(similarities[i][j])
                        })
                        processed.add(j)
            
            return duplicates
            
        except Exception as e:
            return []
    
    def remove_duplicates(self, duplicate_ids: List[str]) -> int:
        """Rimuove documenti duplicati"""
        try:
            collection = self.client.collections.get("Documents")
            removed = 0
            for doc_id in duplicate_ids:
                try:
                    collection.data.delete_by_id(doc_id)
                    removed += 1
                except Exception as e:
                    print(f"Errore rimozione documento {doc_id}: {e}")
            
            return removed
            
        except Exception as e:
            print(f"Errore rimozione duplicati: {e}")
            return 0
    
    def remove_low_quality_content(self, collection_name: str = "Documents") -> int:
        """Rimuove contenuti di bassa qualità"""
        try:
            # Ottieni tutti i documenti
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=10000,
                return_properties=["title", "content"]
            )
            
            documents = response.objects
            to_remove = []
            
            for doc in documents:
                content = doc.properties.get('content', '') or ""
                title = doc.properties.get('title', '') or ""
                
                # Criteri di qualità
                if (len(content) < 50 or  # Troppo corto
                    len(content.split()) < 10 or  # Poche parole
                    self._is_low_quality_text(content) or  # Testo di bassa qualità
                    len(title) < 3):  # Titolo troppo corto
                    to_remove.append(str(doc.uuid))
            
            # Rimuovi documenti di bassa qualità
            removed = 0
            for doc_id in to_remove:
                try:
                    collection.data.delete_by_id(doc_id)
                    removed += 1
                except Exception as e:
                    print(f"Errore rimozione documento {doc_id}: {e}")
            
            return removed
            
        except Exception as e:
            print(f"Errore pulizia contenuti: {e}")
            return 0
    
    def _is_low_quality_text(self, text: str) -> bool:
        """Determina se un testo è di bassa qualità"""
        if not text:
            return True
        
        # Troppi caratteri speciali
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        if special_chars / len(text) > 0.3:
            return True
        
        # Troppi line breaks
        if text.count('\n') / len(text) > 0.1:
            return True
        
        # Troppi numeri
        numbers = len(re.findall(r'\d', text))
        if numbers / len(text) > 0.5:
            return True
        
        # Parole troppo ripetitive
        words = text.lower().split()
        if len(set(words)) / len(words) < 0.3:
            return True
        
        return False


class DataIntegrator:
    def __init__(self, client):
        self.client = client
    
    def integrate_external_file(self, filepath: str, collection_name: str = "Documents") -> Dict[str, Any]:
        """Integra dati da file esterno"""
        try:
            collection = self.client.collections.get(collection_name)
            integrated = 0
            skipped = 0
            
            # Leggi il file
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                data = df.to_dict('records')
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath, engine='openpyxl' if filepath.endswith('.xlsx') else 'xlrd')
                data = df.to_dict('records')
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            else:
                return {"integrated": 0, "skipped": 0, "error": "Formato file non supportato (supportati: CSV, Excel, JSON)"}
            
            for item in data:
                content = item.get('content', '')
                title = item.get('title', '')
                
                if not content:
                    skipped += 1
                    continue
                
                # Controlla se esiste già un documento simile
                if self._document_exists(content, collection_name):
                    skipped += 1
                    continue
                
                # Crea nuovo documento
                document = {
                    "title": title,
                    "content": content,
                    "source": f"integrated_from_{filepath}",
                    "category": item.get('category', 'integrated'),
                    "timestamp": datetime.now().isoformat(),
                }
                
                try:
                    collection.data.insert(document)
                    integrated += 1
                except Exception as e:
                    print(f"Errore inserimento documento: {e}")
                    skipped += 1
            
            return {"integrated": integrated, "skipped": skipped}
            
        except Exception as e:
            return {"integrated": 0, "skipped": 0, "error": str(e)}
    
    def integrate_from_api(self, api_url: str, collection_name: str = "Documents") -> Dict[str, Any]:
        """Integra dati da API"""
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not isinstance(data, list):
                data = [data]
            
            collection = self.client.collections.get(collection_name)
            integrated = 0
            skipped = 0
            
            for item in data:
                content = item.get('content', item.get('text', ''))
                title = item.get('title', item.get('name', ''))
                
                if not content:
                    skipped += 1
                    continue
                
                # Controlla se esiste già
                if self._document_exists(content, collection_name):
                    skipped += 1
                    continue
                
                # Crea documento
                document = {
                    "title": title,
                    "content": content,
                    "source": f"api_{api_url}",
                    "category": "api_integrated",
                    "timestamp": datetime.now().isoformat(),
                }
                
                try:
                    collection.data.insert(document)
                    integrated += 1
                except Exception as e:
                    print(f"Errore inserimento da API: {e}")
                    skipped += 1
            
            return {"integrated": integrated, "skipped": skipped}
            
        except Exception as e:
            return {"integrated": 0, "skipped": 0, "error": str(e)}
    
    def _document_exists(self, content: str, collection_name: str, threshold: float = 0.9) -> bool:
        """Controlla se un documento simile esiste già"""
        try:
            # Ricerca per contenuto simile
            collection = self.client.collections.get(collection_name)
            response = collection.query.near_text(
                query=content[:500],
                limit=1,
                distance=1-threshold
            )
            
            return len(response.objects) > 0
            
        except Exception as e:
            print(f"Errore controllo esistenza: {e}")
            return False


class KnowledgeExtractor:
    def __init__(self, client):
        self.client = client
        self.nlp = nlp
    
    def extract_entities(self, collection_name: str = "Documents") -> Dict[str, Any]:
        """Estrae entità nominate"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=500,
                return_properties=["title", "content"]
            )
            
            documents = response.objects
            
            if not self.nlp:
                return {"entities": [], "error": "spaCy non disponibile"}
            
            all_entities = []
            entity_counts = defaultdict(int)
            
            for doc in documents:
                content = doc.properties.get('content', '') or ""
                title = doc.properties.get('title', '') or ""
                
                # Processa contenuto
                doc_nlp = self.nlp(content[:1000])  # Limita per performance
                
                for ent in doc_nlp.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "document_id": str(doc.uuid),
                        "document_title": title
                    }
                    
                    all_entities.append(entity_info)
                    entity_counts[f"{ent.text}_{ent.label_}"] += 1
            
            # Statistiche
            entity_stats = {
                "total_entities": len(all_entities),
                "unique_entities": len(entity_counts),
                "top_entities": dict(Counter(entity_counts).most_common(20)),
                "entity_types": dict(Counter([e["label"] for e in all_entities]).most_common())
            }
            
            return {"entities": all_entities[:100], "stats": entity_stats}
            
        except Exception as e:
            return {"entities": [], "error": str(e)}
    
    def extract_keywords(self, collection_name: str = "Documents") -> List[Dict[str, Any]]:
        """Estrae parole chiave usando TF-IDF"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["content"]
            )
            
            documents = response.objects
            texts = [doc.properties.get('content', '') for doc in documents if doc.properties.get('content')]
            
            if not texts:
                return []
            
            # TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calcola punteggi medi
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Crea lista keywords
            keywords = []
            for i, score in enumerate(mean_scores):
                keywords.append({
                    "keyword": feature_names[i],
                    "score": float(score),
                    "type": "tfidf"
                })
            
            # Ordina per punteggio
            keywords.sort(key=lambda x: x['score'], reverse=True)
            
            return keywords[:50]
            
        except Exception as e:
            return []
    
    def extract_topics(self, collection_name: str = "Documents", n_topics: int = 5) -> List[Dict[str, Any]]:
        """Estrae topic usando LDA"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["content"]
            )
            
            documents = response.objects
            texts = [doc.properties.get('content', '') for doc in documents if doc.properties.get('content')]
            
            if len(texts) < n_topics:
                return []
            
            # Preprocessing e vectorizzazione
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(doc_term_matrix)
            
            # Estrai topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": top_weights,
                    "coherence": float(topic.max())
                })
            
            return topics
            
        except Exception as e:
            return []


class QASystem:
    def __init__(self, client):
        self.client = client
    
    def ask_question(self, question: str, collection_name: str = "Documents") -> Dict[str, Any]:
        """Risponde a una domanda usando RAG"""
        try:
            # Ricerca semantica
            collection = self.client.collections.get(collection_name)
            response = collection.query.near_text(
                query=question,
                limit=5,
                distance=0.3
            )
            
            if not response.objects:
                return {
                    "answer": "Non ho trovato informazioni pertinenti per rispondere alla tua domanda.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Costruisci il contesto
            context_parts = []
            sources = []
            
            for doc in response.objects:
                content = doc.properties.get('content', '')
                title = doc.properties.get('title', '')
                source = doc.properties.get('source', '')
                
                context_parts.append(f"Titolo: {title}\nContenuto: {content[:500]}...")
                sources.append({
                    "title": title,
                    "source": source,
                    "distance": getattr(doc.metadata, 'distance', 0),
                    "id": str(doc.uuid)
                })
            
            context = "\n\n".join(context_parts)
            
            # Genera risposta usando il contesto (semplificata)
            answer = self._generate_answer(question, context)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": 1.0 - min([s['distance'] for s in sources if 'distance' in s])
            }
            
        except Exception as e:
            return {
                "answer": f"Errore durante la ricerca: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Genera una risposta basata sul contesto (versione semplificata)"""
        words = question.lower().split()
        context_lower = context.lower()
        
        # Trova le frasi più rilevanti
        sentences = context.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return " ".join(relevant_sentences[:3])
        else:
            return "Basandomi sui documenti trovati, non riesco a fornire una risposta specifica alla tua domanda."
    
    def search_documents(self, query: str, collection_name: str = "Documents", limit: int = 10) -> List[Dict[str, Any]]:
        """Ricerca documenti per query"""
        try:
            collection = self.client.collections.get(collection_name)
            response = collection.query.near_text(
                query=query,
                limit=limit,
                return_properties=["title", "content", "source", "category"],
                distance=0.5
            )
            
            documents = []
            for doc in response.objects:
                content = doc.properties.get('content', '')
                documents.append({
                    "id": str(doc.uuid),
                    "title": doc.properties.get('title', ''),
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "source": doc.properties.get('source', ''),
                    "category": doc.properties.get('category', ''),
                    "relevance": 1.0 - getattr(doc.metadata, 'distance', 0)
                })
            
            return documents
            
        except Exception as e:
            return []
