# modules/weaviate_manager.py
import weaviate
import json
import pandas as pd
from typing import List, Dict, Any
import uuid
import os

class WeaviateManager:
    def __init__(self, client):
        self.client = client
    
    def setup_schema(self):
        """Crea gli schemi di base"""
        try:
            # Schema principale per documenti
            documents_schema = {
                "class": "Documents",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False,
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                },
                "properties": [
                    {"name": "title", "dataType": ["string"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "source", "dataType": ["string"]},
                    {"name": "category", "dataType": ["string"]},
                    {"name": "timestamp", "dataType": ["date"]},
                    {"name": "metadata", "dataType": ["object"]}
                ]
            }
            
            # Schema per entità estratte
            entities_schema = {
                "class": "Entities",
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "text", "dataType": ["string"]},
                    {"name": "label", "dataType": ["string"]},
                    {"name": "confidence", "dataType": ["number"]},
                    {"name": "document_id", "dataType": ["string"]}
                ]
            }
            
            # Schema per relazioni
            relations_schema = {
                "class": "Relations",
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "subject", "dataType": ["string"]},
                    {"name": "predicate", "dataType": ["string"]},
                    {"name": "object", "dataType": ["string"]},
                    {"name": "confidence", "dataType": ["number"]},
                    {"name": "document_id", "dataType": ["string"]}
                ]
            }
            
            # Crea le classi se non esistono
            existing_classes = [cls['class'] for cls in self.client.schema.get()['classes']]
            
            for schema in [documents_schema, entities_schema, relations_schema]:
                if schema['class'] not in existing_classes:
                    self.client.schema.create_class(schema)
                    print(f"Creata classe: {schema['class']}")
            
            return True
            
        except Exception as e:
            print(f"Errore setup schema: {e}")
            return False
    
    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Processa un file e inserisce i documenti"""
        try:
            inserted = 0
            
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    document = {
                        "title": str(row.get('title', '')),
                        "content": str(row.get('content', '')),
                        "source": filepath,
                        "category": str(row.get('category', 'general')),
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "metadata": row.to_dict()
                    }
                    
                    self.client.data_object.create(document, "Documents")
                    inserted += 1
                    
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for item in data:
                        document = {
                            "title": item.get('title', ''),
                            "content": item.get('content', ''),
                            "source": filepath,
                            "category": item.get('category', 'general'),
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "metadata": item
                        }
                        
                        self.client.data_object.create(document, "Documents")
                        inserted += 1
                        
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Dividi in chunks se il file è troppo grande
                chunk_size = 5000
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    document = {
                        "title": f"{os.path.basename(filepath)} - Parte {i+1}",
                        "content": chunk,
                        "source": filepath,
                        "category": "text",
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "metadata": {"chunk_number": i+1, "total_chunks": len(chunks)}
                    }
                    
                    self.client.data_object.create(document, "Documents")
                    inserted += 1
            
            return {"inserted": inserted, "status": "success"}
            
        except Exception as e:
            return {"inserted": 0, "status": "error", "error": str(e)}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """Lista tutte le collezioni"""
        try:
            schema = self.client.schema.get()
            collections = []
            
            for cls in schema['classes']:
                # Conta i documenti in ogni collezione
                count_result = self.client.query.aggregate(cls['class']).with_fields("meta { count }").do()
                count = count_result['data']['Aggregate'][cls['class']][0]['meta']['count']
                
                collections.append({
                    "name": cls['class'],
                    "count": count,
                    "properties": len(cls['properties']),
                    "vectorizer": cls.get('vectorizer', 'none')
                })
            
            return collections
            
        except Exception as e:
            print(f"Errore lista collezioni: {e}")
            return []
    
    def create_collection(self, name: str, properties: List[Dict[str, Any]]) -> bool:
        """Crea una nuova collezione"""
        try:
            schema = {
                "class": name,
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False,
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                },
                "properties": properties
            }
            
            self.client.schema.create_class(schema)
            return True
            
        except Exception as e:
            print(f"Errore creazione collezione: {e}")
            return False


# modules/analyzer.py
import weaviate
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
from typing import Dict, List, Any

class DataAnalyzer:
    def __init__(self, client):
        self.client = client
    
    def get_basic_stats(self, collection_name: str = "Documents") -> Dict[str, Any]:
        """Statistiche base della collezione"""
        try:
            # Conta totale documenti usando la nuova API
            collection = self.client.collections.get(collection_name)
            
            # Ottieni campione per analisi
            response = collection.query.fetch_objects(
                limit=1000,
                return_properties=["title", "content", "source", "category"]
            )
            
            documents = response.objects
            total_count = len(documents)
            
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
            # Ottieni documenti con vettori usando la nuova API
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
            vectors = np.array([doc.vector for doc in documents])
            
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
            result = self.client.query.get(collection_name, ["content"]) \
                .with_limit(1000) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            texts = [doc['content'] for doc in documents if doc['content']]
            
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

# modules/cleaner.py
import weaviate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import re

class DataCleaner:
    def __init__(self, client):
        self.client = client
    
    def find_duplicates(self, collection_name: str = "Documents", threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Trova documenti duplicati"""
        try:
            # Ottieni documenti con vettori
            result = self.client.query.get(collection_name, ["title", "content"]) \
                .with_additional(["id", "vector"]) \
                .with_limit(1000) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            
            if len(documents) < 2:
                return []
            
            # Calcola similarità
            vectors = np.array([doc['_additional']['vector'] for doc in documents])
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
                                "id": documents[i]['_additional']['id'],
                                "title": documents[i]['title'],
                                "content": documents[i]['content'][:100] + "..."
                            },
                            "doc2": {
                                "id": documents[j]['_additional']['id'],
                                "title": documents[j]['title'],
                                "content": documents[j]['content'][:100] + "..."
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
            removed = 0
            for doc_id in duplicate_ids:
                try:
                    self.client.data_object.delete(doc_id)
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
            result = self.client.query.get(collection_name, ["title", "content"]) \
                .with_additional(["id"]) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            to_remove = []
            
            for doc in documents:
                content = doc['content'] or ""
                title = doc['title'] or ""
                
                # Criteri di qualità
                if (len(content) < 50 or  # Troppo corto
                    len(content.split()) < 10 or  # Poche parole
                    self._is_low_quality_text(content) or  # Testo di bassa qualità
                    len(title) < 3):  # Titolo troppo corto
                    to_remove.append(doc['_additional']['id'])
            
            # Rimuovi documenti di bassa qualità
            removed = 0
            for doc_id in to_remove:
                try:
                    self.client.data_object.delete(doc_id)
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
    
    def clean_text_content(self, collection_name: str = "Documents") -> int:
        """Pulisce il contenuto testuale dei documenti"""
        try:
            # Ottieni tutti i documenti
            result = self.client.query.get(collection_name, ["title", "content"]) \
                .with_additional(["id"]) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            cleaned = 0
            
            for doc in documents:
                original_content = doc['content'] or ""
                original_title = doc['title'] or ""
                
                # Pulisci contenuto
                cleaned_content = self._clean_text(original_content)
                cleaned_title = self._clean_text(original_title)
                
                # Aggiorna solo se c'è stata una modifica
                if cleaned_content != original_content or cleaned_title != original_title:
                    try:
                        self.client.data_object.update(
                            doc['_additional']['id'],
                            {
                                "content": cleaned_content,
                                "title": cleaned_title
                            }
                        )
                        cleaned += 1
                    except Exception as e:
                        print(f"Errore aggiornamento documento {doc['_additional']['id']}: {e}")
            
            return cleaned
            
        except Exception as e:
            print(f"Errore pulizia testo: {e}")
            return 0
    
    def _clean_text(self, text: str) -> str:
        """Pulisce un testo"""
        if not text:
            return ""
        
        # Rimuovi caratteri di controllo
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalizza spazi
        text = re.sub(r'\s+', ' ', text)
        
        # Rimuovi spazi all'inizio e alla fine
        text = text.strip()
        
        # Rimuovi pattern ripetitivi
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        return text

# modules/integrator.py
import weaviate
import pandas as pd
import json
import requests
from typing import Dict, Any, List
import uuid
from datetime import datetime

class DataIntegrator:
    def __init__(self, client):
        self.client = client
    
    def integrate_external_file(self, filepath: str, collection_name: str = "Documents") -> Dict[str, Any]:
        """Integra dati da file esterno"""
        try:
            integrated = 0
            skipped = 0
            
            # Leggi il file
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                data = df.to_dict('records')
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            else:
                return {"integrated": 0, "skipped": 0, "error": "Formato file non supportato"}
            
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
                    "metadata": item
                }
                
                try:
                    self.client.data_object.create(document, collection_name)
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
                    "metadata": item
                }
                
                try:
                    self.client.data_object.create(document, collection_name)
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
            result = self.client.query.get(collection_name, ["content"]) \
                .with_near_text({"concepts": [content[:500]]}) \
                .with_limit(1) \
                .with_additional(["distance"]) \
                .do()
            
            if (result['data']['Get'][collection_name] and 
                result['data']['Get'][collection_name][0]['_additional']['distance'] < (1 - threshold)):
                return True
            
            return False
            
        except Exception as e:
            print(f"Errore controllo esistenza: {e}")
            return False
    
    def merge_collections(self, source_collection: str, target_collection: str) -> Dict[str, Any]:
        """Merge due collezioni"""
        try:
            # Ottieni tutti i documenti dalla collezione sorgente
            result = self.client.query.get(source_collection, ["title", "content", "source", "category", "metadata"]) \
                .with_additional(["id"]) \
                .do()
            
            documents = result['data']['Get'][source_collection]
            
            merged = 0
            skipped = 0
            
            for doc in documents:
                # Controlla se esiste già nella collezione target
                if self._document_exists(doc['content'], target_collection):
                    skipped += 1
                    continue
                
                # Crea nella collezione target
                new_doc = {
                    "title": doc['title'],
                    "content": doc['content'],
                    "source": doc['source'],
                    "category": doc['category'],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": doc['metadata']
                }
                
                try:
                    self.client.data_object.create(new_doc, target_collection)
                    merged += 1
                except Exception as e:
                    print(f"Errore merge documento: {e}")
                    skipped += 1
            
            return {"merged": merged, "skipped": skipped}
            
        except Exception as e:
            return {"merged": 0, "skipped": 0, "error": str(e)}

# modules/extractor.py
import weaviate
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter, defaultdict
import re
from typing import List, Dict, Any
import json

class KnowledgeExtractor:
    def __init__(self, client):
        self.client = client
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Modello spaCy non trovato. Installa con: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities(self, collection_name: str = "Documents") -> Dict[str, Any]:
        """Estrae entità nominate"""
        try:
            # Ottieni documenti
            result = self.client.query.get(collection_name, ["title", "content"]) \
                .with_additional(["id"]) \
                .with_limit(500) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            
            if not self.nlp:
                return {"entities": [], "error": "spaCy non disponibile"}
            
            all_entities = []
            entity_counts = defaultdict(int)
            
            for doc in documents:
                content = doc['content'] or ""
                title = doc['title'] or ""
                
                # Processa contenuto
                doc_nlp = self.nlp(content[:1000])  # Limita per performance
                
                for ent in doc_nlp.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "document_id": doc['_additional']['id'],
                        "document_title": title
                    }
                    
                    all_entities.append(entity_info)
                    entity_counts[f"{ent.text}_{ent.label_}"] += 1
            
            # Salva entità in collezione separata
            self._save_entities(all_entities)
            
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
    
    def extract_relations(self, collection_name: str = "Documents") -> Dict[str, Any]:
        """Estrae relazioni tra entità"""
        try:
            # Ottieni documenti
            result = self.client.query.get(collection_name, ["title", "content"]) \
                .with_additional(["id"]) \
                .with_limit(200) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            
            if not self.nlp:
                return {"relations": [], "error": "spaCy non disponibile"}
            
            all_relations = []
            
            for doc in documents:
                content = doc['content'] or ""
                
                # Processa contenuto
                doc_nlp = self.nlp(content[:1000])
                
                # Estrai relazioni semplici basate su dependency parsing
                for token in doc_nlp:
                    if token.dep_ in ["nsubj", "dobj", "pobj"] and token.head.pos_ == "VERB":
                        # Cerca entità correlate
                        subject = self._find_entity_for_token(token, doc_nlp.ents)
                        predicate = token.head.lemma_
                        
                        # Cerca oggetto
                        obj_token = None
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj"]:
                                obj_token = child
                                break
                        
                        if obj_token:
                            obj = self._find_entity_for_token(obj_token, doc_nlp.ents)
                            
                            if subject and obj and subject != obj:
                                relation = {
                                    "subject": subject,
                                    "predicate": predicate,
                                    "object": obj,
                                    "document_id": doc['_additional']['id'],
                                    "confidence": 0.7
                                }
                                all_relations.append(relation)
            
            # Salva relazioni
            self._save_relations(all_relations)
            
            return {"relations": all_relations[:50], "total": len(all_relations)}
            
        except Exception as e:
            return {"relations": [], "error": str(e)}
    
    def extract_keywords(self, collection_name: str = "Documents") -> List[Dict[str, Any]]:
        """Estrae parole chiave usando TF-IDF"""
        try:
            # Ottieni documenti
            result = self.client.query.get(collection_name, ["content"]) \
                .with_limit(1000) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            texts = [doc['content'] for doc in documents if doc['content']]
            
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
            result = self.client.query.get(collection_name, ["content"]) \
                .with_limit(1000) \
                .do()
            
            documents = result['data']['Get'][collection_name]
            texts = [doc['content'] for doc in documents if doc['content']]
            
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
    
    def _find_entity_for_token(self, token, entities):
        """Trova l'entità che contiene un token"""
        for ent in entities:
            if ent.start <= token.i < ent.end:
                return ent.text
        return token.text
    
    def _save_entities(self, entities: List[Dict[str, Any]]):
        """Salva entità in collezione Weaviate"""
        try:
            for entity in entities:
                self.client.data_object.create(entity, "Entities")
        except Exception as e:
            print(f"Errore salvataggio entità: {e}")
    
    def _save_relations(self, relations: List[Dict[str, Any]]):
        """Salva relazioni in collezione Weaviate"""
        try:
            for relation in relations:
                self.client.data_object.create(relation, "Relations")
        except Exception as e:
            print(f"Errore salvataggio relazioni: {e}")


            # modules/qa_system.py
import weaviate
from typing import List, Dict, Any

class QASystem:
    def __init__(self, client):
        self.client = client
    
    def ask_question(self, question: str, collection_name: str = "Documents") -> Dict[str, Any]:
        """Risponde a una domanda usando RAG"""
        try:
            # Ricerca semantica
            search_result = self.client.query.get(collection_name, ["content", "title", "source"]) \
                .with_near_text({"concepts": [question]}) \
                .with_limit(5) \
                .with_additional(["distance", "id"]) \
                .do()
            
            if not search_result['data']['Get'][collection_name]:
                return {
                    "answer": "Non ho trovato informazioni pertinenti per rispondere alla tua domanda.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Costruisci il contesto
            context_parts = []
            sources = []
            
            for doc in search_result['data']['Get'][collection_name]:
                context_parts.append(f"Titolo: {doc['title']}\nContenuto: {doc['content'][:500]}...")
                sources.append({
                    "title": doc['title'],
                    "source": doc['source'],
                    "distance": doc['_additional']['distance'],
                    "id": doc['_additional']['id']
                })
            
            context = "\n\n".join(context_parts)
            
            # Genera risposta usando il contesto (semplificata)
            # In una versione reale, useresti un LLM come OpenAI GPT
            answer = self._generate_answer(question, context)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": 1.0 - min([s['distance'] for s in sources])
            }
            
        except Exception as e:
            return {
                "answer": f"Errore durante la ricerca: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Genera una risposta basata sul contesto (versione semplificata)"""
        # Questa è una versione semplificata
        # In produzione, integreresti con OpenAI o altri LLM
        
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
            result = self.client.query.get(collection_name, ["title", "content", "source", "category"]) \
                .with_near_text({"concepts": [query]}) \
                .with_limit(limit) \
                .with_additional(["distance", "id"]) \
                .do()
            
            documents = []
            for doc in result['data']['Get'][collection_name]:
                documents.append({
                    "id": doc['_additional']['id'],
                    "title": doc['title'],
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "source": doc['source'],
                    "category": doc['category'],
                    "relevance": 1.0 - doc['_additional']['distance']
                })
            
            return documents
            
        except Exception as e:
            return []
