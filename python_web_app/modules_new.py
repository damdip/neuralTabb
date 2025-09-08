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
import google.generativeai as genai

# Import per le query Weaviate native
try:
    from weaviate.classes.query import Filter, Metrics, MetadataQuery
    from weaviate.classes.aggregate import GroupByAggregate
except ImportError:
    print("Avviso: Impossibile importare le classi Weaviate. Assicurati di avere weaviate-client v4+ installato.")

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
        """Lista tutte le collezioni con dettagli completi delle proprietà"""
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
                    
                    # Ottieni dettagli completi delle proprietà
                    property_details = []
                    properties_count = 0
                    
                    try:
                        # Prova a ottenere un documento di esempio per vedere le proprietà
                        sample = collection.query.fetch_objects(limit=1)
                        if sample.objects:
                            sample_properties = sample.objects[0].properties
                            properties_count = len(sample_properties.keys())
                            
                            # Crea dettagli per ogni proprietà
                            for prop_name, prop_value in sample_properties.items():
                                prop_type = self._get_property_type(prop_value)
                                property_details.append({
                                    "name": prop_name,
                                    "type": prop_type,
                                    "sample_value": str(prop_value)[:50] + "..." if len(str(prop_value)) > 50 else str(prop_value)
                                })
                    except:
                        properties_count = 0
                        property_details = []
                    
                    # Ottieni configurazione vectorizer (se disponibile)
                    try:
                        # Per ora usiamo un valore di default
                        vectorizer = "text2vec-transformers"
                    except:
                        vectorizer = "unknown"
                    
                    collections.append({
                        "name": collection_name,
                        "count": count,
                        "properties": properties_count,
                        "property_details": property_details,
                        "vectorizer": vectorizer
                    })
                    
                except Exception as e:
                    # Se c'è un errore con una collezione specifica, aggiungi info minime
                    collections.append({
                        "name": collection_name,
                        "count": 0,
                        "properties": 0,
                        "property_details": [],
                        "vectorizer": "error"
                    })
            
            return collections
            
        except Exception as e:
            print(f"Errore nel listare le collezioni: {e}")
            return []
    
    def _get_property_type(self, value) -> str:
        """Determina il tipo di una proprietà dal suo valore"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            if len(value) > 100:
                return "text"
            else:
                return "string"
        elif isinstance(value, list):
            return f"array[{len(value)}]"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
    
    def delete_collection(self, collection_name: str) -> bool:
        """Elimina una collezione e tutti i suoi dati"""
        try:
            # Verifica se la collezione esiste
            if collection_name not in self.client.collections.list_all():
                raise ValueError(f"Collezione '{collection_name}' non trovata")
            
            # Elimina la collezione
            self.client.collections.delete(collection_name)
            print(f"Collezione '{collection_name}' eliminata con successo")
            return True
            
        except Exception as e:
            print(f"Errore nell'eliminazione della collezione '{collection_name}': {e}")
            return False
    
    def get_collection_sample_data(self, collection_name: str, limit: int = 20) -> Dict[str, Any]:
        """Ottiene dati campione da una collezione per l'esplorazione"""
        try:
            collection = self.client.collections.get(collection_name)
            
            # Ottieni statistiche generali
            response = collection.aggregate.over_all(total_count=True)
            total_count = response.total_count
            
            # Ottieni dati campione
            sample_response = collection.query.fetch_objects(limit=limit)
            sample_data = []
            property_info = []
            
            for obj in sample_response.objects:
                try:
                    # Estrai le proprietà direttamente dall'oggetto
                    obj_properties = {}
                    if hasattr(obj, 'properties') and obj.properties:
                        obj_properties = obj.properties
                    
                    # Ottieni l'UUID se disponibile
                    obj_id = str(obj.uuid) if hasattr(obj, 'uuid') else "N/A"
                    
                    sample_data.append({
                        "id": obj_id,
                        "properties": obj_properties
                    })
                    
                    # Raccogli informazioni sulle proprietà dalla prima riga
                    if not property_info and obj_properties:
                        for prop_name, prop_value in obj_properties.items():
                            prop_type = self._get_property_type(prop_value)
                            property_info.append({
                                "name": prop_name,
                                "type": prop_type
                            })
                            
                except Exception as obj_error:
                    print(f"Errore nel processare oggetto: {obj_error}")
                    # Aggiungi comunque un oggetto vuoto per mantenere la consistenza
                    sample_data.append({
                        "id": "Error",
                        "properties": {"error": f"Errore nel processare oggetto: {str(obj_error)}"}
                    })
                    continue
            
            return {
                "collection_name": collection_name,
                "total_count": total_count,
                "sample_data": sample_data,
                "property_info": property_info,
                "sample_size": len(sample_data)
            }
            
        except Exception as e:
            print(f"Errore nel recuperare dati campione per '{collection_name}': {e}")
            return {
                "collection_name": collection_name,
                "error": str(e),
                "total_count": 0,
                "sample_data": [],
                "property_info": [],
                "sample_size": 0
            }

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




class QASystemWithGemini:
    def __init__(self, client, api_key_path="/config/configLLM.txt"):
        self.client = client
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            genai.configure(api_key=api_key)
            
            # Prova diversi modelli in ordine di preferenza (con Gemini 2.0 Flash come primo)
            models_to_try = [
                'gemini-2.0-flash-exp',      # Gemini 2.0 Flash (experimental) - il più recente
                'gemini-2.0-flash',          # Gemini 2.0 Flash (se disponibile in versione stabile)
                'gemini-1.5-flash',          # Fallback ai modelli 1.5
                'gemini-1.5-pro', 
                'gemini-pro'
            ]
            
            for model_name in models_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test del modello con una richiesta semplice
                    test_response = self.model.generate_content("Test")
                    self.current_model_name = model_name  # Salva il nome del modello utilizzato
                    print(f"Modello Gemini '{model_name}' inizializzato con successo")
                    break
                except Exception as model_error:
                    print(f"Modello '{model_name}' non disponibile: {model_error}")
                    continue
            else:
                # Se nessun modello funziona, prova a elencare i modelli disponibili
                try:
                    available_models = [m.name for m in genai.list_models()]
                    raise Exception(f"Nessun modello Gemini disponibile. Modelli supportati: {available_models}")
                except:
                    raise Exception("Impossibile inizializzare qualsiasi modello Gemini e recuperare la lista dei modelli disponibili")
                    
        except FileNotFoundError:
            raise Exception(f"File della chiave API non trovato in '{api_key_path}'. Assicurati che il file esista.")
        except Exception as e:
            raise Exception(f"Errore durante la configurazione di Gemini: {e}")

    def classify_question(self, question: str) -> str:
        """Usa Gemini per classificare la domanda in una delle quattro categorie disponibili."""
        prompt = f"""
        Classifica la seguente domanda in una di queste quattro categorie:
        
        1. 'conversazionale': Saluti, ringraziamenti, domande sul funzionamento del sistema, cortesie
           Esempi: "Ciao", "Come stai?", "Grazie", "Come funzioni?", "Che cosa sai fare?", "Chi sei?", "Help", "Aiuto"
        
        2. 'analitica': Richiede dati specifici, conteggi, aggregazioni o elenchi filtrati dai dati
           Esempi: "Quanti libri ha scritto X?", "Elenca i libri dopo il 2020", "Conta gli autori", "Mostra i titoli che contengono..."
        
        3. 'generale': Domande aperte sui contenuti che richiedono analisi testuale e contesto
           Esempi: "Parlami dei temi principali", "Riassumi il contenuto", "Qual è l'argomento principale?"
        
        4. 'pulizia': Richieste di pulizia, correzione, normalizzazione, validazione dei dati e identificazione di problemi
           Esempi: "Pulisci i dati", "Rimuovi i duplicati", "Correggi gli errori", "Normalizza i valori", "Valida i campi",
           "Trova record incompleti", "Identifica valori anomali", "Standardizza i formati", "Elimina spazi extra",
           "Controlla la consistenza", "Verifica l'integrità", "Rimuovi caratteri speciali", "Unifica le categorie"

        Domanda: "{question}"
        
        Rispondi SOLO con una delle quattro parole: conversazionale, analitica, generale, pulizia
        """
        try:
            response = self.model.generate_content(prompt)
            classification = response.text.strip().lower()
            
            if "conversazionale" in classification:
                return "conversazionale"
            elif "analitica" in classification:
                return "analitica"
            elif "generale" in classification:
                return "generale"
            elif "pulizia" in classification:
                return "pulizia"
            elif "integrazione" in classification:
                return "integrazione"
            else:
                # Se non riconosce, prova a fare una classificazione basata su parole chiave
                question_lower = question.lower()
                
                # Parole chiave conversazionali
                conversational_keywords = [
                    "ciao", "salve", "buongiorno", "buonasera", "hello", "hi",
                    "grazie", "thank", "prego", "scusa", "scusami",
                    "come stai", "come va", "tutto bene",
                    "chi sei", "cosa sei", "come funzioni", "cosa fai", "come fai",
                    "aiuto", "help", "guida", "istruzioni",
                    "arrivederci", "addio", "bye", "ciao ciao"
                ]
                
                if any(keyword in question_lower for keyword in conversational_keywords):
                    return "conversazionale"
                
                # Parole chiave analitiche
                analytical_keywords = [
                    "quanti", "quanto", "conta", "elenca", "lista", "mostra",
                    "trova", "cerca", "filtra", "dove", "quando",
                    "maggiore", "minore", "primo", "ultimo", "media", "somma"
                ]
                
                if any(keyword in question_lower for keyword in analytical_keywords):
                    return "analitica"
                
                # Parole chiave pulizia
                cleaning_keywords = [
                    "pulisci", "pulire", "pulizia", "clean", "rimuovi duplicati", "duplicati",
                    "correggi", "correggere", "correzione", "fix", "normalizza", "normalizzare",
                    "valida", "validare", "validazione", "validate", "formatta", "formato"
                ]
                
                if any(keyword in question_lower for keyword in cleaning_keywords):
                    return "pulizia"
                
                # Parole chiave integrazione
                integration_keywords = [
                    "integra", "integrare", "integrazione", "integrate", "unisci", "unire",
                    "merge", "join", "collega", "collegare", "combina", "combinare",
                    "fusion", "fusione", "concatena", "concatenare"
                ]
                
                if any(keyword in question_lower for keyword in integration_keywords):
                    return "integrazione"
                
                return "generale"  # Default
        except Exception as e:
            print(f"Errore nella classificazione con Gemini: {e}")
            print(f"Tipo di errore: {type(e).__name__}")
            if "404" in str(e) or "not found" in str(e).lower():
                print("Il modello potrebbe non essere supportato. Prova a riavviare l'applicazione.")
            return "generale" # Default a generale

    def test_connection(self) -> bool:
        """Testa la connessione con Gemini"""
        try:
            response = self.model.generate_content("Ciao, questo è un test di connessione.")
            return True
        except Exception as e:
            print(f"Test connessione Gemini fallito: {e}")
            return False

    def get_current_model_info(self) -> dict:
        """Restituisce informazioni sul modello correntemente in uso"""
        return {
            "model_name": getattr(self, 'current_model_name', 'Sconosciuto'),
            "is_gemini_2_0": getattr(self, 'current_model_name', '').startswith('gemini-2.0'),
            "model_object": str(self.model) if hasattr(self, 'model') else 'Non disponibile'
        }

    def switch_model(self, model_name: str) -> bool:
        """Cambia il modello Gemini utilizzato"""
        try:
            new_model = genai.GenerativeModel(model_name)
            # Test del nuovo modello
            test_response = new_model.generate_content("Test")
            
            # Se il test va a buon fine, cambia il modello
            self.model = new_model
            self.current_model_name = model_name
            print(f"Modello cambiato con successo a: {model_name}")
            return True
        except Exception as e:
            print(f"Impossibile cambiare al modello '{model_name}': {e}")
            return False

    def list_available_models(self):
        """Elenca i modelli Gemini disponibili"""
        try:
            models = genai.list_models()
            available_models = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
            print(f"Modelli disponibili per generateContent: {available_models}")
            return available_models
        except Exception as e:
            print(f"Errore nel recuperare i modelli disponibili: {e}")
            return []

    def list_available_models(self):
        """Elenca i modelli Gemini disponibili"""
        try:
            models = genai.list_models()
            available_models = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
            print(f"Modelli disponibili per generateContent: {available_models}")
            return available_models
        except Exception as e:
            print(f"Errore nel recuperare i modelli disponibili: {e}")
            return []

    def handle_analytical_question(self, question: str, class_name: str) -> str:
        """Genera ed esegue query native Weaviate per domande analitiche usando Gemini."""
        try:
            collection = self.client.collections.get(class_name)
            sample = collection.query.fetch_objects(limit=1)
            if not sample.objects:
                return f"La collezione '{class_name}' è vuota o non esiste."
            
            properties = list(sample.objects[0].properties.keys())
            
            # Prompt per generare codice Python nativo Weaviate
            model_info = f" (usando {getattr(self, 'current_model_name', 'Gemini')})" if hasattr(self, 'current_model_name') else ""
            prompt = f"""
Sei un esperto di Weaviate database{model_info}. Data una domanda in linguaggio naturale, devi creare codice Python per eseguire query native di Weaviate (NON GraphQL).

COLLEZIONE: {class_name}
PROPRIETÀ DISPONIBILI: {', '.join(properties)}
DOMANDA: {question}

IMPORTANTE - GESTIONE PERFORMANCE:
- Per query di AGGREGAZIONE/RAGGRUPPAMENTO: usa filtri per limitare il dataset se necessario
- Per query di CONTEGGIO: usare total_count=True (veloce)
- Per query di RICERCA: limit massimo 50 per prestazioni ottimali

ESEMPI DI SINTASSI WEAVIATE PYTHON CLIENT v4:

1. Per CONTARE oggetti (aggregazione):
```python
# Conta totale
response = collection.aggregate.over_all(total_count=True)
count = response.total_count

# Conta con filtro
response = collection.aggregate.over_all(
    filters=Filter.by_property("author").equal("Stephen King"),
    total_count=True
)
count = response.total_count
```

2. Per CERCARE/RECUPERARE dati:
```python
# Cerca con filtro
response = collection.query.fetch_objects(
    filters=Filter.by_property("author").equal("Stephen King"),
    limit=10
)

# Cerca per testo semantico
response = collection.query.near_text(
    query="fantasy adventure",
    limit=10
)
```

3. FILTRI disponibili:
- Filter.by_property("campo").equal("valore")
- Filter.by_property("campo").like("*pattern*")  
- Filter.by_property("campo").greater_than(numero)
- Filter.by_property("campo").less_than(numero)
- Combinazioni: Filter.by_property("a").equal("x") & Filter.by_property("b").greater_than(10)

4. Per AGGREGAZIONI NUMERICHE:
```python
# Calcola statistiche su numeri
response = collection.aggregate.over_all(
    return_metrics=Metrics("rating").number(
        sum_=True,
        maximum=True, 
        minimum=True,
        mean=True
    )
)
```

5. Per RAGGRUPPAMENTI:
```python
# Raggruppamento semplice - le aggregazioni processano tutti i dati
response = collection.aggregate.over_all(
    group_by=GroupByAggregate(prop="genre")
)

# Raggruppamento con filtro per limitare il dataset
response = collection.aggregate.over_all(
    group_by=GroupByAggregate(prop="category"),
    filters=Filter.by_property("year").greater_than(2000)
)
```

6. REGOLE PER I LIMITI:
- Aggregazioni/Raggruppamenti: NON supportano 'limit' - usa filtri per ridurre il dataset
- Conteggi semplici: total_count=True (veloce, no limite necessario)  
- Ricerche/Recupero dati: limit max 50
- Per dataset grandi: usa filtri specifici nelle aggregazioni

Genera SOLO il codice Python necessario per rispondere alla domanda. 
Il risultato deve essere salvato in una variabile chiamata 'response'.
Non includere import o spiegazioni extra.

ESEMPI SPECIFICI PER RAGGRUPPAMENTI:
Per "raggruppa per genere": 
```python
response = collection.aggregate.over_all(
    group_by=GroupByAggregate(prop="genre")
)
```

Per "conta libri per autore":
```python  
response = collection.aggregate.over_all(
    group_by=GroupByAggregate(prop="author")
)
```
```python  
response = collection.aggregate.over_all(
    group_by=GroupByAggregate(prop="proprietà"),
    limit=300
)
```

CODICE PYTHON:"""

            # Genera il codice con Gemini
            response = self.model.generate_content(prompt)
            weaviate_code = response.text.strip()
            
            # Pulisci il codice rimuovendo markdown
            weaviate_code = weaviate_code.replace('```python', '').replace('```', '').strip()
            
            print(f"Codice Weaviate generato da Gemini: {weaviate_code}")
            
            # Verifica e correggi il codice se necessario
            weaviate_code = self._validate_and_fix_weaviate_code(weaviate_code)
            
            # Prepara l'ambiente per l'esecuzione
            exec_namespace = {
                'collection': collection,
                'Filter': Filter,
                'Metrics': Metrics,
                'GroupByAggregate': GroupByAggregate,
                'MetadataQuery': MetadataQuery
            }
            
            try:
                # Esegui il codice generato con timeout compatibile Windows
                import threading
                import time
                
                # Variabili per gestire timeout e risultati
                execution_result = {'completed': False, 'error': None}
                
                def execute_code():
                    try:
                        exec(weaviate_code, exec_namespace)
                        execution_result['completed'] = True
                    except Exception as e:
                        execution_result['error'] = e
                
                # Avvia l'esecuzione in un thread separato
                thread = threading.Thread(target=execute_code)
                thread.daemon = True
                thread.start()
                
                # Aspetta al massimo 30 secondi
                thread.join(timeout=30)
                
                if thread.is_alive():
                    # Il thread è ancora in esecuzione = timeout
                    return "⏱️ Query troppo complessa - hai provato a raggruppare troppi dati. Prova con un filtro più specifico o un dataset più piccolo."
                
                if execution_result['error']:
                    raise execution_result['error']
                    
                if not execution_result['completed']:
                    return "⚠️ Esecuzione interrotta - prova a semplificare la query."
                
            except Exception as e:
                return f"Errore nell'esecuzione della query: {str(e)}. Prova a riformulare la domanda con termini più specifici."
            
            # Il risultato dovrebbe essere in 'response' 
            response_data = exec_namespace.get('response', None)
            
            if response_data is None:
                return "Errore: il codice generato non ha prodotto risultati."
            
            # Formatta la risposta in base al tipo di risultato
            return self._format_weaviate_response(response_data, question)
            
        except Exception as e:
            print(f"Errore nell'esecuzione query analitica: {e}")
            
            # Fallback con suggerimenti specifici per query di raggruppamento
            if any(word in question.lower() for word in ['raggruppa', 'group', 'conta per', 'quanti per']):
                return f"""❌ **Errore nella query di raggruppamento**

La query "{question}" ha causato un errore di timeout o complessità.

💡 **Suggerimenti per migliorare la query:**
- Prova con un filtro più specifico (es: "raggruppa i libri pubblicati dopo il 2000 per genere")  
- Usa termini più precisi per le proprietà
- Prova a dividere la query in parti più piccole

🔍 **Query alternative che potresti provare:**
- "quanti libri ci sono in totale?"
- "mostra i primi 10 libri per genere"
- "conta i libri per un genere specifico"

Errore tecnico: {str(e)}"""
            
            # Fallback alla ricerca semantica per altri tipi di errore
            return self.handle_general_question(question, class_name)

    def handle_conversational_question(self, question: str, class_name: str = None) -> str:
        """Gestisce domande conversazionali come saluti, ringraziamenti e domande sul sistema."""
        try:
            question_lower = question.lower().strip()
            
            # Risposte predefinite per risposte rapide
            quick_responses = {
                # Saluti
                "ciao": "Ciao! 👋 Sono qui per aiutarti ad analizzare i tuoi dati. Che cosa vorresti sapere?",
                "salve": "Salve! Come posso aiutarti oggi con l'analisi dei tuoi dati?",
                "buongiorno": "Buongiorno! 🌅 Pronto ad aiutarti con le tue domande sui dati.",
                "buonasera": "Buonasera! 🌅 Come posso assisterti?",
                "hello": "Hello! How can I help you analyze your data today?",
                "hi": "Hi there! 👋 Ready to explore your data together?",
                
                # Ringraziamenti
                "grazie": "Prego! 😊 È stato un piacere aiutarti. Hai altre domande?",
                "thanks": "You're welcome! Any other questions about your data?",
                "thank you": "You're very welcome! Happy to help with your data analysis.",
                
                # Saluti di commiato
                "arrivederci": "Arrivederci! 👋 Torna quando vuoi per analizzare altri dati!",
                "addio": "Addio! Spero di averti aiutato. A presto!",
                "bye": "Bye! 👋 Come back anytime for more data insights!",
                "ciao ciao": "Ciao ciao! 👋👋 È stato un piacere aiutarti!",
                
                # Cortesie
                "come stai": "Sto bene, grazie! 😊 Sono pronto ad aiutarti con l'analisi dei dati. Tu come stai?",
                "come va": "Tutto bene! Sto elaborando dati e rispondendo a domande. Come posso aiutarti?",
                "tutto bene": "Sì, tutto perfetto! Sono qui per te. Che cosa vorresti analizzare?",
            }
            
            # Controlla risposte rapide
            for key, response in quick_responses.items():
                if key in question_lower:
                    return response
            
            # Per domande più complesse, usa Gemini con un prompt specializzato
            model_name = getattr(self, 'current_model_name', 'Gemini')
            
            prompt = f"""
            Sei un assistente AI specializzato nell'analisi di dati chiamato NeuralTabb, che utilizza {model_name} e Weaviate.
            
            Rispondi a questa domanda conversazionale in modo amichevole e utile:
            "{question}"
            
            INFORMAZIONI SU DI TE:
            - Sono NeuralTabb, un sistema di Q&A per dati tabulari
            - Uso {model_name} per comprendere le domande in linguaggio naturale
            - Uso Weaviate come database vettoriale per archiviare e cercare nei dati
            - Posso rispondere a domande analitiche (conteggi, filtri), generali (riassunti, temi)
            - Posso gestire richieste di pulizia dati (rimozione duplicati, normalizzazione)
            - Posso gestire richieste di integrazione dati (merge, join, fusione dataset)
            - Posso analizzare file Excel, pulire dati, fare integrazioni e estrarre conoscenza
            
            STILE DI RISPOSTA:
            - Sii amichevole e professionale
            - Usa emoji quando appropriato
            - Se ti chiedono cosa sai fare, spiega le tue capacità principali
            - Se ti chiedono come funzioni, spiega brevemente la tecnologia
            - Mantieni un tono conversazionale ma informativo
            
            Risposta:
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Errore nella gestione domanda conversazionale: {e}")
            # Fallback a una risposta generica
            return "Ciao! 😊 Sono NeuralTabb, il tuo assistente per l'analisi dei dati. Come posso aiutarti oggi?"

    def handle_general_question(self, question: str, class_name: str) -> str:
        """Usa l'approccio RAG con Gemini per domande generali."""
        try:
            collection = self.client.collections.get(class_name)
            
            # Scopri le proprietà testuali
            sample = collection.query.fetch_objects(limit=1)
            if not sample.objects:
                 return "La collezione è vuota."
            
            text_properties = [k for k, v in sample.objects[0].properties.items() if isinstance(v, str)]
            if not text_properties:
                return "Nessuna proprietà testuale trovata per la ricerca."

            result = collection.query.near_text(
                query=question,
                limit=3,
                return_properties=text_properties
            )
            
            context_documents = result.objects
            if not context_documents:
                return "Non ho trovato informazioni pertinenti per rispondere alla tua domanda."

            context = "\n".join([f"- Documento: {doc.properties}" for doc in context_documents])
            
            prompt = f"""
            Basandoti esclusivamente sul seguente contesto, rispondi alla domanda.

            Contesto:
            {context}

            Domanda: {question}

            Risposta:
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Impossibile generare la risposta con Gemini: {e}"

    def handle_cleaning_question(self, question: str, class_name: str = None) -> str:
        """Gestisce domande relative alla pulizia e normalizzazione dei dati."""
        try:
            # Per ora restituiamo una risposta che indica che la funzionalità sarà implementata
            model_name = getattr(self, 'current_model_name', 'Gemini')
            
            prompt = f"""
            Sei NeuralTabb, un assistente AI specializzato nella pulizia dei dati usando {model_name}.
            
            L'utente ha fatto questa richiesta di pulizia dati:
            "{question}"
            
            ATTUALMENTE DISPONIBILE:
            - Riconoscimento delle richieste di pulizia dati
            - Classificazione dei tipi di pulizia richiesti
            
            FUNZIONALITÀ IN SVILUPPO (prossime implementazioni):
            - Rimozione duplicati
            - Normalizzazione valori
            - Correzione errori di formato
            - Validazione campi
            - Pulizia dati mancanti
            - Standardizzazione testo
            
            Rispondi in modo professionale spiegando che hai compreso la richiesta di pulizia
            e che questa funzionalità è in fase di sviluppo. Suggerisci quali operazioni 
            di pulizia potrebbero essere utili per il tipo di richiesta ricevuta.
            
            Mantieni un tono incoraggiante e professionale.
            """
            
            response = self.model.generate_content(prompt)
            return f"🧹 **Richiesta di Pulizia Dati Riconosciuta**\n\n{response.text.strip()}"
            
        except Exception as e:
            print(f"Errore nella gestione domanda di pulizia: {e}")
            return """🧹 **Richiesta di Pulizia Dati**

Ho riconosciuto che stai chiedendo operazioni di pulizia sui dati. 

**Funzionalità di pulizia in fase di sviluppo:**
- Rimozione duplicati
- Normalizzazione valori
- Correzione errori di formato  
- Validazione campi
- Gestione dati mancanti

La tua richiesta è stata registrata e sarà disponibile nel prossimo aggiornamento! 🚀"""

    def handle_integration_question(self, question: str, class_name: str = None) -> str:
        """Gestisce domande relative all'integrazione e fusione di dataset."""
        try:
            # Per ora restituiamo una risposta che indica che la funzionalità sarà implementata
            model_name = getattr(self, 'current_model_name', 'Gemini')
            
            prompt = f"""
            Sei NeuralTabb, un assistente AI specializzato nell'integrazione di dati usando {model_name}.
            
            L'utente ha fatto questa richiesta di integrazione dati:
            "{question}"
            
            ATTUALMENTE DISPONIBILE:
            - Riconoscimento delle richieste di integrazione dati
            - Classificazione dei tipi di integrazione richiesti
            
            FUNZIONALITÀ IN SVILUPPO (prossime implementazioni):
            - Merge di dataset multipli
            - Join basato su chiavi comuni
            - Concatenazione di file
            - Fusione intelligente di schemi
            - Risoluzione conflitti dati
            - Mappatura automatica campi
            
            Rispondi in modo professionale spiegando che hai compreso la richiesta di integrazione
            e che questa funzionalità è in fase di sviluppo. Suggerisci quali operazioni 
            di integrazione potrebbero essere utili per il tipo di richiesta ricevuta.
            
            Mantieni un tono incoraggiante e professionale.
            """
            
            response = self.model.generate_content(prompt)
            return f"🔗 **Richiesta di Integrazione Dati Riconosciuta**\n\n{response.text.strip()}"
            
        except Exception as e:
            print(f"Errore nella gestione domanda di integrazione: {e}")
            return """🔗 **Richiesta di Integrazione Dati**

Ho riconosciuto che stai chiedendo operazioni di integrazione tra dataset.

**Funzionalità di integrazione in fase di sviluppo:**
- Merge di dataset multipli
- Join basato su chiavi comuni  
- Concatenazione di file
- Fusione intelligente di schemi
- Risoluzione conflitti dati
- Mappatura automatica campi

La tua richiesta è stata registrata e sarà disponibile nel prossimo aggiornamento! 🚀"""

    def smart_answer(self, question: str, class_name: str = None) -> str:
        """Classifica la domanda e la indirizza alla funzione corretta."""
        question_type = self.classify_question(question)
        
        print(f"Tipo di domanda rilevato da Gemini: {question_type}")

        if question_type == "conversazionale":
            return self.handle_conversational_question(question, class_name)
        elif question_type == "analitica":
            if not class_name:
                return "Per domande analitiche, devi selezionare una collezione di dati da analizzare. 📊"
            return self.handle_analytical_question(question, class_name)
        elif question_type == "pulizia":
            return self.handle_cleaning_question(question, class_name)
        elif question_type == "integrazione":
            return self.handle_integration_question(question, class_name)
        else: # 'generale'
            if not class_name:
                return "Per domande sui contenuti, devi selezionare una collezione di dati da esplorare. 🔍"
            return self.handle_general_question(question, class_name)

    def _validate_and_fix_weaviate_code(self, code: str) -> str:
        """Valida e corregge il codice Weaviate generato da Gemini."""
        try:
            # Correzioni comuni
            fixes = [
                # Assicurati che ci sia una variabile response
                ("result =", "response ="),
                ("data =", "response ="),
                ("query_result =", "response ="),
                # Fix import statements che potrebbero essere inserite
                ("from weaviate.classes", "# from weaviate.classes"),
                ("import Filter", "# import Filter"),
            ]
            
            for old, new in fixes:
                code = code.replace(old, new)
            
            # Assicurati che il codice finisca con l'assegnazione a response
            if "response =" not in code:
                if code.strip().split('\n')[-1].startswith(('collection.', 'result', 'data')):
                    last_line = code.strip().split('\n')[-1]
                    code = code.replace(last_line, f"response = {last_line}")
            
            return code
            
        except Exception as e:
            print(f"Errore nella correzione del codice: {e}")
            return code

    def _format_weaviate_response(self, response_data, question: str) -> str:
        """Formatta la risposta di Weaviate in un formato leggibile."""
        try:
            # Verifica il tipo di risposta
            if hasattr(response_data, 'total_count'):
                # Risposta di aggregazione con conteggio
                return f"Ho trovato {response_data.total_count} risultati per la tua domanda: '{question}'"
            
            elif hasattr(response_data, 'objects') and response_data.objects:
                # Risposta con oggetti (query normale)
                count = len(response_data.objects)
                result = f"Ho trovato {count} risultati per: '{question}'\n\n"
                
                for i, obj in enumerate(response_data.objects[:10], 1):  # Mostra max 10 risultati
                    result += f"{i}. "
                    props = obj.properties
                    
                    # Mostra le prime 3 proprietà disponibili (o tutte se sono meno di 3)
                    property_names = list(props.keys())
                    max_props_to_show = min(3, len(property_names))
                    
                    displayed_props = []
                    for j in range(max_props_to_show):
                        prop_name = property_names[j]
                        prop_value = props[prop_name]
                        
                        # Gestisci diversi tipi di dati
                        if prop_value is None:
                            prop_value_str = "N/A"
                        elif isinstance(prop_value, str):
                            # Se il testo è troppo lungo, troncalo
                            prop_value_str = prop_value[:100] + "..." if len(str(prop_value)) > 100 else str(prop_value)
                        else:
                            prop_value_str = str(prop_value)
                        
                        displayed_props.append(f"{prop_name}: {prop_value_str}")
                    
                    result += " | ".join(displayed_props)
                    
                    # Se ci sono più di 3 proprietà, indica che ce ne sono altre
                    if len(property_names) > 3:
                        result += f" | ... (+{len(property_names) - 3} altre proprietà)"
                    
                    result += "\n"
                
                if count > 10:
                    result += f"\n... e altri {count-10} risultati."
                
                return result
            
            elif hasattr(response_data, 'properties'):
                # Risposta di aggregazione con proprietà numeriche
                props = response_data.properties
                result = f"Statistiche per: '{question}'\n\n"
                
                for prop_name, values in props.items():
                    if hasattr(values, 'sum_'):
                        result += f"• {prop_name} - Somma: {values.sum_}\n"
                    if hasattr(values, 'maximum'):
                        result += f"• {prop_name} - Massimo: {values.maximum}\n"
                    if hasattr(values, 'minimum'):
                        result += f"• {prop_name} - Minimo: {values.minimum}\n"
                    if hasattr(values, 'mean'):
                        result += f"• {prop_name} - Media: {values.mean}\n"
                
                return result
            
            elif hasattr(response_data, 'groups'):
                # Risposta di raggruppamento (nuova gestione per Weaviate v4)
                result = f"📊 **Raggruppamento per: '{question}'**\n\n"
                
                if hasattr(response_data.groups, 'items'):
                    # Nuova struttura Weaviate v4
                    for group in response_data.groups:
                        group_name = getattr(group, 'grouped_by', {}).get('value', 'Non specificato')
                        count = getattr(group, 'total_count', 0)
                        result += f"• **{group_name}**: {count} elementi\n"
                else:
                    # Struttura legacy
                    for group_name, group_data in response_data.groups.items():
                        count = getattr(group_data, 'total_count', 0)
                        result += f"• **{group_name}**: {count} elementi\n"
                
                return result
                
            # Gestione specifica per aggregazioni Weaviate v4
            elif hasattr(response_data, '__dict__') and any(hasattr(response_data, attr) for attr in ['groups', 'total_count']):
                result = f"📊 **Risultati per: '{question}'**\n\n"
                
                # Prova a gestire diverse strutture di aggregazione
                if hasattr(response_data, 'groups') and response_data.groups:
                    # Gestione gruppi
                    groups = response_data.groups if hasattr(response_data.groups, '__iter__') else [response_data.groups]
                    for i, group in enumerate(groups):
                        if hasattr(group, 'grouped_by') and hasattr(group, 'total_count'):
                            group_value = group.grouped_by.get('value', f'Gruppo {i+1}')
                            result += f"• **{group_value}**: {group.total_count} elementi\n"
                        elif hasattr(group, 'name') and hasattr(group, 'count'):
                            result += f"• **{group.name}**: {group.count} elementi\n"
                        else:
                            # Fallback - prova a estrarre qualsiasi informazione disponibile
                            result += f"• **Gruppo {i+1}**: {getattr(group, 'total_count', 'N/D')} elementi\n"
                
                elif hasattr(response_data, 'total_count'):
                    result += f"Totale elementi: {response_data.total_count}\n"
                
                return result if result != f"📊 **Risultati per: '{question}'**\n\n" else "Risultati di aggregazione ricevuti ma formato non riconosciuto."
            
            else:
                # Fallback: prova a convertire in JSON
                import json
                return f"Risultato per '{question}':\n{json.dumps(response_data, indent=2, default=str)}"
                
        except Exception as e:
            print(f"Errore nella formattazione della risposta: {e}")
            return f"Ho ottenuto una risposta per '{question}', ma non riesco a formattarla correttamente. Errore: {str(e)}"

# Esempio di utilizzo della classe QASystemWithGemini
if __name__ == '__main__':
    try:
        # Connessione a Weaviate
        weaviate_client = weaviate.Client("http://localhost:8080")

        # Inizializza il sistema QA con Gemini
        # Assicurati che 'chiave.txt' sia nel percorso corretto
        qa_system = QASystemWithGemini(weaviate_client)
        
        collection_to_query = "Book" # Sostituisci con il nome della tua collezione

        # Esempio domanda generale
        general_q = "Parlami dei libri che trattano di magia"
        print(f"--- Domanda Generale ---\n{general_q}")
        general_a = qa_system.smart_answer(general_q, collection_to_query)
        print(f"Risposta: {general_a}\n")

        # Esempio domanda analitica
        analytical_q = "Quanti libri ha scritto Isaac Asimov?"
        print(f"--- Domanda Analitica ---\n{analytical_q}")
        analytical_a = qa_system.smart_answer(analytical_q, collection_to_query)
        print(f"Risposta: {analytical_a}\n")
        
        # Esempio domanda di pulizia
        cleaning_q = "Pulisci i dati rimuovendo i duplicati"
        print(f"--- Domanda di Pulizia ---\n{cleaning_q}")
        cleaning_a = qa_system.smart_answer(cleaning_q, collection_to_query)
        print(f"Risposta: {cleaning_a}\n")
        
        # Esempio domanda di integrazione
        integration_q = "Integra questo dataset con altri file"
        print(f"--- Domanda di Integrazione ---\n{integration_q}")
        integration_a = qa_system.smart_answer(integration_q, collection_to_query)
        print(f"Risposta: {integration_a}\n")

    except Exception as e:
        print(f"Si è verificato un errore principale: {e}")