# modules/weWaviate_manager.py
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
        """Statistiche base della collezione - versione flessibile"""
        try:
            print(f"DEBUG: Tentando di ottenere statistiche per collezione: {collection_name}")
            
            # Ottieni collezione
            collection = self.client.collections.get(collection_name)
            print(f"DEBUG: Collezione ottenuta: {collection}")
            
            # Conta documenti totali
            count_response = collection.aggregate.over_all(total_count=True)
            total_count = count_response.total_count
            print(f"DEBUG: Conteggio documenti: {total_count}")
            
            if total_count == 0:
                print("DEBUG: Nessun documento nella collezione")
                return {
                    "total_documents": 0,
                    "analyzed_sample": 0,
                    "categories": [],
                    "sources": [],
                    "content_stats": {"avg_length": 0},
                    "available_properties": [],
                    "error": "Nessun documento nella collezione"
                }
            
            # Ottieni un campione per scoprire le proprietà disponibili
            sample_response = collection.query.fetch_objects(limit=1)
            if not sample_response.objects:
                return {
                    "total_documents": total_count,
                    "analyzed_sample": 0,
                    "categories": [],
                    "sources": [],
                    "content_stats": {"avg_length": 0},
                    "available_properties": [],
                    "error": "Impossibile recuperare documenti di esempio"
                }
            
            # Scopri tutte le proprietà disponibili
            available_properties = list(sample_response.objects[0].properties.keys())
            print(f"Proprietà disponibili nella collezione {collection_name}: {available_properties}")
            
            # Ottieni campione per analisi (senza specificare proprietà)
            response = collection.query.fetch_objects(limit=1000)
            documents = response.objects
            
            # Analisi flessibile delle proprietà
            property_stats = {}
            categories = set()
            sources = set()
            content_lengths = []
            
            for prop_name in available_properties:
                prop_values = []
                for doc in documents:
                    prop_value = doc.properties.get(prop_name)
                    if prop_value is not None:
                        prop_values.append(prop_value)
                        
                        # Estrai categorie da campi che potrebbero contenere categorie
                        if 'category' in prop_name.lower() or 'genre' in prop_name.lower() or 'type' in prop_name.lower():
                            categories.add(str(prop_value))
                        
                        # Estrai sorgenti da campi che potrebbero contenere sorgenti  
                        if 'source' in prop_name.lower() or 'author' in prop_name.lower() or 'publisher' in prop_name.lower():
                            sources.add(str(prop_value))
                        
                        # Calcola lunghezza del contenuto da campi testuali
                        if isinstance(prop_value, str) and ('content' in prop_name.lower() or 'text' in prop_name.lower() or 'description' in prop_name.lower()):
                            content_lengths.append(len(prop_value))
                
                if prop_values:
                    # Statistiche per questa proprietà
                    if isinstance(prop_values[0], str):
                        # Proprietà testuale
                        lengths = [len(str(val)) for val in prop_values]
                        unique_values = list(set(prop_values))
                        
                        property_stats[prop_name] = {
                            "type": "text",
                            "total_values": len(prop_values),
                            "unique_values": len(unique_values),
                            "avg_length": np.mean(lengths) if lengths else 0,
                            "max_length": max(lengths) if lengths else 0,
                            "sample_values": unique_values[:5]  # Prime 5 per esempio
                        }
                    else:
                        # Proprietà numerica o altro
                        unique_values = list(set(prop_values))
                        property_stats[prop_name] = {
                            "type": "other",
                            "total_values": len(prop_values),
                            "unique_values": len(unique_values),
                            "sample_values": unique_values[:5]
                        }
            
            # Se non abbiamo trovato contenuti specifici, usa tutti i campi testuali
            if not content_lengths:
                for doc in documents:
                    for prop_name, prop_value in doc.properties.items():
                        if isinstance(prop_value, str):
                            content_lengths.append(len(prop_value))
            
            result = {
                "total_documents": total_count,
                "analyzed_sample": len(documents),
                "categories": list(categories),
                "sources": list(sources),
                "content_stats": {
                    "avg_length": np.mean(content_lengths) if content_lengths else 0,
                    "max_length": max(content_lengths) if content_lengths else 0,
                    "total_content": len(content_lengths)
                },
                "available_properties": available_properties,
                "property_stats": property_stats
            }
            
            print(f"DEBUG: Risultato finale: {result}")
            return result
            
        except Exception as e:
            print(f"DEBUG: Errore in get_basic_stats: {str(e)}")
            return {
                "total_documents": 0,
                "analyzed_sample": 0,
                "categories": [],
                "sources": [],
                "content_stats": {"avg_length": 0},
                "error": str(e)
            }
    
    def analyze_clusters(self, collection_name: str = "Documents", n_clusters: int = 5) -> Dict[str, Any]:
        """Analizza cluster semantici - versione flessibile"""
        try:
            # Ottieni documenti con vettori (senza specificare proprietà specifiche)
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                limit=500,
                include_vector=True
            )
            
            documents = response.objects
            
            if len(documents) == 0:
                return {"error": "Nessun documento trovato nella collezione"}
            
            if len(documents) < n_clusters:
                # Riduci automaticamente il numero di cluster
                n_clusters = min(len(documents), 2)
                if n_clusters < 2:
                    return {"error": "Troppi pochi documenti per il clustering (minimo 2)"}
            
            # Verifica che i documenti abbiano vettori
            try:
                vectors = np.array([doc.vector["default"] for doc in documents])
            except (KeyError, AttributeError):
                return {"error": "I documenti non hanno vettori associati"}
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            # Scopri le proprietà testuali disponibili
            if documents:
                available_props = list(documents[0].properties.keys())
                text_props = []
                for prop in available_props:
                    sample_value = documents[0].properties.get(prop)
                    if isinstance(sample_value, str) and len(sample_value) > 10:
                        text_props.append(prop)
                
                if not text_props:
                    text_props = available_props[:2]  # Usa le prime 2 proprietà
            
            # Analizza cluster
            cluster_info = {}
            for i in range(n_clusters):
                cluster_docs = [documents[j] for j in range(len(documents)) if clusters[j] == i]
                
                # Usa proprietà flessibili
                sample_data = []
                for doc in cluster_docs[:5]:  # Prime 5 per esempio
                    doc_sample = {}
                    for prop in text_props:
                        value = doc.properties.get(prop, '')
                        if isinstance(value, str) and len(value) > 100:
                            doc_sample[prop] = value[:100] + "..."
                        else:
                            doc_sample[prop] = str(value)
                    sample_data.append(doc_sample)
                
                cluster_info[f"cluster_{i}"] = {
                    "size": len(cluster_docs),
                    "sample_documents": sample_data,
                    "available_properties": text_props
                }
            
            return {
                "n_clusters": n_clusters,
                "total_documents": len(documents),
                "clusters": cluster_info,
                "properties_used": text_props
            }
            
        except Exception as e:
            return {"error": str(e)}
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_topics(self, collection_name: str = "Documents", n_topics: int = 5) -> List[Dict[str, Any]]:
        """Estrae topic usando LDA"""
    def extract_topics(self, collection_name: str = "Documents", n_topics: int = 5) -> List[Dict[str, Any]]:
        """Estrae topic usando LDA - versione flessibile"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            response = collection.query.fetch_objects(limit=1000)
            
            documents = response.objects
            
            if not documents:
                return []
            
            # Trova proprietà testuali
            sample_doc = documents[0]
            text_properties = []
            
            for prop_name, prop_value in sample_doc.properties.items():
                if isinstance(prop_value, str) and len(prop_value) > 50:
                    text_properties.append(prop_name)
            
            if not text_properties:
                return []
            
            # Usa la proprietà testuale più lunga in media
            best_prop = None
            max_avg_length = 0
            
            for prop in text_properties:
                lengths = []
                for doc in documents[:100]:  # Campione per velocità
                    value = doc.properties.get(prop, '')
                    if isinstance(value, str):
                        lengths.append(len(value))
                
                if lengths:
                    avg_length = np.mean(lengths)
                    if avg_length > max_avg_length:
                        max_avg_length = avg_length
                        best_prop = prop
            
            if not best_prop:
                return []
            
            # Estrai testi dalla proprietà migliore
            texts = []
            for doc in documents:
                text = doc.properties.get(best_prop, '')
                if isinstance(text, str) and len(text.strip()) > 20:
                    texts.append(text)
            
            if len(texts) < n_topics:
                n_topics = min(len(texts), 2)
                if n_topics < 2:
                    return []
            
            try:
                # Vettorizzazione TF-IDF
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                doc_term_matrix = vectorizer.fit_transform(texts)
                
                # LDA
                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
                lda.fit(doc_term_matrix)
                
                # Estrai topics
                feature_names = vectorizer.get_feature_names_out()
                topics = []
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                    topics.append({
                        "topic_id": topic_idx,
                        "words": top_words,
                        "weight": float(topic.max()),
                        "property_used": best_prop
                    })
                
                return topics
                
            except Exception as ve:
                return [{"error": f"Errore nella vettorizzazione: {str(ve)}"}]
            
        except Exception as e:
            return [{"error": str(e)}]
    
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
        """Estrae entità nominate - versione flessibile"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            
            # Prima ottieni un campione per scoprire le proprietà disponibili
            sample_response = collection.query.fetch_objects(limit=1)
            if not sample_response.objects:
                return {"entities": [], "error": "Nessun documento nella collezione"}
            
            # Scopri proprietà testuali disponibili
            available_properties = list(sample_response.objects[0].properties.keys())
            text_properties = []
            
            for prop_name in available_properties:
                sample_value = sample_response.objects[0].properties.get(prop_name)
                if isinstance(sample_value, str) and len(sample_value) > 10:
                    text_properties.append(prop_name)
            
            if not text_properties:
                return {"entities": [], "error": "Nessuna proprietà testuale trovata nella collezione"}
            
            # Ottieni documenti usando solo le proprietà disponibili
            response = collection.query.fetch_objects(limit=500)
            documents = response.objects
            
            if not self.nlp:
                return {"entities": [], "error": "spaCy non disponibile"}
            
            all_entities = []
            entity_counts = defaultdict(int)
            
            for doc in documents:
                # Trova la proprietà con più testo per questo documento
                best_text = ""
                doc_title = ""
                
                for prop_name in text_properties:
                    prop_value = doc.properties.get(prop_name, '')
                    if isinstance(prop_value, str):
                        if len(prop_value) > len(best_text):
                            best_text = prop_value
                        # Usa la prima proprietà corta come "titolo"
                        if not doc_title and len(prop_value) < 200:
                            doc_title = prop_value
                
                if not best_text:
                    continue
                
                # Processa contenuto
                try:
                    doc_nlp = self.nlp(best_text[:1000])  # Limita per performance
                    
                    for ent in doc_nlp.ents:
                        entity_info = {
                            "text": str(ent.text),
                            "label": str(ent.label_),
                            "start": int(ent.start_char),
                            "end": int(ent.end_char),
                            "document_id": str(doc.uuid),
                            "document_title": str(doc_title)[:100] if doc_title else "N/A"
                        }
                        
                        all_entities.append(entity_info)
                        entity_counts[f"{ent.text}_{ent.label_}"] += 1
                        
                except Exception as e:
                    print(f"Errore processing doc {doc.uuid}: {e}")
                    continue
            
            # Statistiche
            entity_stats = {
                "total_entities": len(all_entities),
                "unique_entities": len(entity_counts),
                "top_entities": dict(Counter(entity_counts).most_common(20)),
                "entity_types": dict(Counter([e["label"] for e in all_entities]).most_common()),
                "properties_used": text_properties
            }
            
            return {"entities": all_entities[:100], "stats": entity_stats}
            
        except Exception as e:
            return {"entities": [], "error": str(e)}
    
    def extract_keywords(self, collection_name: str = "Documents") -> List[Dict[str, Any]]:
        """Estrae parole chiave usando TF-IDF - versione flessibile"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            
            # Prima ottieni un campione per scoprire le proprietà disponibili
            sample_response = collection.query.fetch_objects(limit=1)
            if not sample_response.objects:
                return []
            
            # Scopri proprietà testuali disponibili
            available_properties = list(sample_response.objects[0].properties.keys())
            text_properties = []
            
            for prop_name in available_properties:
                sample_value = sample_response.objects[0].properties.get(prop_name)
                if isinstance(sample_value, str) and len(sample_value) > 20:
                    text_properties.append(prop_name)
            
            if not text_properties:
                return []
            
            # Scegli la proprietà migliore (quella con testo più lungo in media)
            response = collection.query.fetch_objects(limit=100)  # Campione per valutare
            sample_docs = response.objects
            
            best_property = None
            max_avg_length = 0
            
            for prop_name in text_properties:
                lengths = []
                for doc in sample_docs:
                    prop_value = doc.properties.get(prop_name, '')
                    if isinstance(prop_value, str):
                        lengths.append(len(prop_value))
                
                if lengths:
                    avg_length = sum(lengths) / len(lengths)
                    if avg_length > max_avg_length:
                        max_avg_length = avg_length
                        best_property = prop_name
            
            if not best_property:
                return []
            
            # Ottieni tutti i documenti e estrai testi dalla proprietà migliore
            response = collection.query.fetch_objects(limit=1000)
            documents = response.objects
            
            texts = []
            for doc in documents:
                text = doc.properties.get(best_property, '')
                if isinstance(text, str) and len(text.strip()) > 20:
                    texts.append(text)
            
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
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Calcola punteggi medi
                mean_scores = tfidf_matrix.mean(axis=0).A1
                
                # Crea lista keywords
                keywords = []
                for i, score in enumerate(mean_scores):
                    keywords.append({
                        "keyword": str(feature_names[i]),
                        "score": float(score),
                        "type": "tfidf",
                        "property_used": best_property
                    })
                
                # Ordina per punteggio
                keywords.sort(key=lambda x: x['score'], reverse=True)
                
                return keywords[:50]
                
            except Exception as ve:
                return [{"error": f"Errore nella vettorizzazione: {str(ve)}"}]
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def extract_topics(self, collection_name: str = "Documents", n_topics: int = 5) -> List[Dict[str, Any]]:
        """Estrae topic usando LDA - versione flessibile"""
        try:
            # Ottieni documenti
            collection = self.client.collections.get(collection_name)
            
            # Prima ottieni un campione per scoprire le proprietà disponibili
            sample_response = collection.query.fetch_objects(limit=1)
            if not sample_response.objects:
                return []
            
            # Scopri proprietà testuali disponibili
            available_properties = list(sample_response.objects[0].properties.keys())
            text_properties = []
            
            for prop_name in available_properties:
                sample_value = sample_response.objects[0].properties.get(prop_name)
                if isinstance(sample_value, str) and len(sample_value) > 50:
                    text_properties.append(prop_name)
            
            if not text_properties:
                return []
            
            # Scegli la proprietà migliore (quella con testo più lungo in media)
            response = collection.query.fetch_objects(limit=100)  # Campione per valutare
            sample_docs = response.objects
            
            best_property = None
            max_avg_length = 0
            
            for prop_name in text_properties:
                lengths = []
                for doc in sample_docs:
                    prop_value = doc.properties.get(prop_name, '')
                    if isinstance(prop_value, str):
                        lengths.append(len(prop_value))
                
                if lengths:
                    avg_length = sum(lengths) / len(lengths)
                    if avg_length > max_avg_length:
                        max_avg_length = avg_length
                        best_property = prop_name
            
            if not best_property:
                return []
            
            # Ottieni tutti i documenti e estrai testi dalla proprietà migliore
            response = collection.query.fetch_objects(limit=1000)
            documents = response.objects
            
            texts = []
            for doc in documents:
                text = doc.properties.get(best_property, '')
                if isinstance(text, str) and len(text.strip()) > 20:
                    texts.append(text)
            
            if len(texts) < n_topics:
                n_topics = min(len(texts), 2)
                if n_topics < 2:
                    return []
            
            try:
                # Preprocessing e vectorizzazione
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
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
                    top_words = [str(feature_names[i]) for i in top_words_idx]
                    top_weights = [float(topic[i]) for i in top_words_idx]
                    
                    topics.append({
                        "topic_id": int(topic_idx),
                        "words": top_words,
                        "weights": top_weights,
                        "coherence": float(topic.max()),
                        "property_used": best_property
                    })
                
                return topics
                
            except Exception as ve:
                return [{"error": f"Errore nella vettorizzazione: {str(ve)}"}]
            
        except Exception as e:
            return [{"error": str(e)}]


class QASystem:
    def __init__(self, client):
        self.client = client
    
    def ask_question(self, question: str, collection_name: str = "Documents", limit: int = 5) -> Dict[str, Any]:
        """Cerca documenti in base a una domanda"""
        try:
            # Ricerca semantica sui documenti della collezione scelta
            collection = self.client.collections.get(collection_name)
            response = collection.query.near_text(
                query=question,
                limit=limit,
                distance=0.5
            )
            
            # Ottieni tutte le proprietà disponibili dal primo documento per essere generalizzabile
            available_properties = []
            if response.objects:
                available_properties = list(response.objects[0].properties.keys())
            
            sources = []
            
            if response.objects:
                for doc in response.objects:
                    distance = getattr(doc.metadata, 'distance', None)
                    if distance is None:
                        distance = 0.0
                    source_info = {"id": str(doc.uuid), "distance": distance}
                    
                    # Debug: stampa tutte le proprietà del documento
                    print(f"Documento {doc.uuid}:")
                    print(f"Proprietà disponibili: {list(doc.properties.keys())}")
                    
                    for prop_name, prop_value in doc.properties.items():
                        # Debug: stampa ogni proprietà
                        print(f"  {prop_name}: {type(prop_value)} - {str(prop_value)[:100]}...")
                        
                        source_info[prop_name] = prop_value if prop_value is not None else ""
                    
                    sources.append(source_info)
            
            return {
                "sources": sources,
                "total_found": len(sources),
                "collection_name": collection_name,
                "available_properties": available_properties,
                "query": question,
                "limit_used": limit
            }
            
        except Exception as e:
            return {
                "sources": [],
                "total_found": 0,
                "collection_name": collection_name,
                "available_properties": [],
                "query": question,
                "limit_used": limit,
                "error": str(e)
            }
    
    def search_documents(self, query: str, collection_name: str = "Documents", limit: int = 10) -> List[Dict[str, Any]]:
        """Ricerca documenti per query"""
        try:
            collection = self.client.collections.get(collection_name)
            response = collection.query.near_text(
                query=query,
                limit=limit,
                distance=0.5
            )
            
            documents = []
            for doc in response.objects:
                # Costruisci il documento in modo generalizzabile
                doc_data = {"id": str(doc.uuid)}
                
                # Aggiungi tutte le proprietà disponibili
                for prop_name, prop_value in doc.properties.items():
                    if prop_value:
                        # Per contenuti lunghi, accorcia per la visualizzazione
                        if isinstance(prop_value, str) and len(prop_value) > 200:
                            doc_data[prop_name] = prop_value[:200] + "..."
                        else:
                            doc_data[prop_name] = prop_value
                
                # Aggiungi la rilevanza
                doc_data["relevance"] = 1.0 - getattr(doc.metadata, 'distance', 0)
                documents.append(doc_data)
            
            return documents
            
        except Exception as e:
            return []
