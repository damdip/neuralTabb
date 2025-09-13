# modules/weWaviate_manager.py
import pathlib
import weaviate
import json
import pandas as pd
import os
from typing import List, Dict, Any
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
import threading
import time

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
        """Non crea più schemi hardcoded - usa solo collezioni esistenti"""
        try:
            # Verifica solo la connessione a Weaviate
            collections = self.client.collections.list_all()
            print(f"Connessione verificata: {len(collections)} collezioni disponibili")
            return True
        except Exception as e:
            print(f"Errore connessione Weaviate: {e}")
            return False
    
    def process_file(self, filepath: str) -> Dict[str, Any]:
        collection_name = pathlib.Path(filepath).stem
        """Processa un file e inserisce i documenti"""
        try:
            if ( checkExistingCollection(self.client ,collection_name) ):
                print(f"Collezione {collection_name} già esistente")
                return {"status": "success", "inserted": 0, "errors": 0}
            
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

    def count_objects(self, collection_name: str) -> int:
        """Conta rapidamente il numero di oggetti in una collezione usando aggregazione."""
        try:
            if not self.client.collections.exists(collection_name):
                return 0
                
            collection = self.client.collections.get(collection_name)
            response = collection.aggregate.over_all(total_count=True)
            
            return response.total_count
            
        except Exception as e:
            print(f"Errore nel conteggio veloce per collezione '{collection_name}': {e}")
            return 0

class QASystem:
    def __init__(self, client):
        self.client = client
    
    def ask_question(self, question: str, collection_name: str, limit: int = 5) -> Dict[str, Any]:
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
    
    def search_documents(self, query: str, collection_name: str, limit: int = 10) -> List[Dict[str, Any]]:
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
        self.gemini_call_count = 0  # Contatore per monitorare le chiamate
        self.tokens_saved = 0       # Stima token risparmiati
        self.weaviateManager = WeaviateManager(client)
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
    
    def _track_gemini_call(self, prompt_length: int):
        """Traccia le chiamate a Gemini per monitorare i costi."""
        self.gemini_call_count += 1
        print(f"[GEMINI CALL #{self.gemini_call_count}] Token stimati: ~{prompt_length}")
    
    def get_usage_stats(self) -> dict:
        """Restituisce statistiche sull'uso di Gemini."""
        return {
            "total_calls": self.gemini_call_count,
            "estimated_tokens_saved": self.tokens_saved,
            "model_used": getattr(self, 'current_model_name', 'Unknown')
        }

    def askGeminiHowManyElementsInvolved(self, question: str, collection_items) -> int:
        """Chiede a Gemini quanti elementi/documenti dovrebbero essere coinvolti nella risposta."""
        try:
            # Usa l'istanza del modello dalla classe
            prompt = f"""
            Analizza questa domanda e determina quanti documenti/elementi dovrebbero essere recuperati dal database per rispondere adeguatamente:
            
            Domanda: "{question}"

            Rispondi SOLO con un numero intero tra 1 e {collection_items}.
            """
            
            
            response = self.model.generate_content(prompt)
            
            # Estrai il numero dalla risposta
            import re
            numbers = re.findall(r'\b(\d{1,3})\b', response.text.strip())
            
            return int(numbers[0]) if numbers else 15  # Fallback sicuro
                
        except Exception as e:
            print(f"Errore askGeminiHowManyElementsInvolved: {e}")
            return 15  # Fallback sicuro
    
    def askGeminiAboutPropertiesInvolved(self, question: str, properties_info: dict) -> list:
        """Chiede a Gemini quali proprietà sono rilevanti per rispondere alla domanda."""
        try:
            if not hasattr(self, 'model'):
                return properties_info.get('all', [])[:3]  # Fallback sicuro
                
            all_properties = properties_info.get('all', [])
            text_properties = properties_info.get('text', [])
            number_properties = properties_info.get('number', [])
            
            if not all_properties:
                return []
                
            prompt = f"""
            Analizza questa domanda e seleziona le proprietà più rilevanti per rispondere:
            
            Domanda: "{question}"
            
            PROPRIETÀ DISPONIBILI: {all_properties}
            
            Seleziona le proprietà più rilevanti che potrebbero contenere informazioni utili per rispondere alla domanda.
            
            Rispondi con una lista di nomi di proprietà separati da virgole, SOLO nomi esistenti dalla lista fornita.
            Esempio: title, content, category
            """
            
            response = self.model.generate_content(prompt)
            
            # Estrai le proprietà dalla risposta
            response_text = response.text.strip().lower()
            selected_properties = []
            
            # Cerca le proprietà menzionate nella risposta
            for prop in all_properties:
                if prop.lower() in response_text:
                    selected_properties.append(prop)
            
            
            # Limita a massimo 4 proprietà per non sovraccaricare
            return selected_properties
            
        except Exception as e:
            print(f"Errore askGeminiAboutPropertiesInvolved: {e}")
            # Fallback intelligente
            all_props = properties_info.get('all', [])
            return all_props[:3] if all_props else []

    def performWeaviateQueries(self, question: str, elements_involved: int, properties_involved: dict) -> str:
        """Esegue query Weaviate ottimizzate e formatta la risposta."""
        print(f"Ecco la domanda {question}, gli elementi coinvolti {elements_involved}, le proprietà coinvolte {properties_involved}")
        try:
            if not hasattr(self, 'client'):
                return "Errore: client Weaviate non disponibile"
                
            # Determina il tipo di query dalla domanda
            question_lower = question.lower()
            
            # Query di conteggio
            if any(word in question_lower for word in ['quanti', 'quante', 'count', 'how many', 'numero di']):
                #return self._perform_count_query(question, elements_involved, properties_involved)
                return "Funzionalità di conteggio non ancora implementata."
            # Query di raggruppamento
            elif any(word in question_lower for word in ['raggruppa', 'group by', 'per categoria', 'by category']):
                #return self._perform_group_query(question, elements_involved, properties_involved)
                return "Funzionalità di raggruppamento non ancora implementata."
            # Query di ricerca con filtri
            elif any(word in question_lower for word in ['trova', 'search', 'where', 'con', 'che hanno', 'filter']):
                #return self._perform_filtered_search(question, elements_involved, properties_involved)
                return "Funzionalità di ricerca con filtri non ancora implementata."
        except Exception as e:
            return f"Errore nell'esecuzione della query: {str(e)}"

    def classify_question(self, question: str) -> str:
        
        # Solo per casi molto ambigui usiamo Gemini (riduce del 90% le chiamate)
        #if len(question.split()) > 15:  # Solo per domande molto lunghe
        return self._classify_with_gemini(question)
        
        # Default: generale (per domande sui contenuti)
        #return "generale"
    
    def _classify_with_gemini(self, question: str) -> str:
        """Usa Gemini solo per classificazioni complesse (fallback)."""
        prompt = f"Classifica '{question}' in: conversazionale, analitica, generale, pulizia, integrazione, estrazione_conoscenza. Rispondi solo con una parola."
        
        try:
            response = self.model.generate_content(prompt)
            classification = response.text.strip().lower()
            
            categories = ["conversazionale", "analitica", "generale", "pulizia", "integrazione", "estrazione_conoscenza"]
            for cat in categories:
                if cat in classification:
                    return cat
                    
        except Exception as e:
            print(f"Errore classificazione Gemini: {e}")
        
        # Fallback sicuro
        return "generale"

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


    #---------------------------------------#
    #      Analytical Questions             #
    #                                       #
    #---------------------------------------#
    def handle_analytical_question(self, question: str, class_name: str) -> str:
        """Gestisce domande analitiche in modo deterministico (senza code-gen/exec).

        Supporta:
        - conteggi: "quanti ...?" (+ filtro opzionale)
        - elenchi: "mostra/elenca ..." (+ filtro e limit)
        - raggruppamenti: "conta per <campo>" / "raggruppa per <campo>"
        - filtri semplici: equality su campi testuali, >/< su campi numerici/anno

        Fallback: messaggi chiari o ricerca semantica con conteggio.
        """
        try:
            collection = self.client.collections.get(class_name)
            properties_info = self._get_collection_properties(collection)
            """
            intent = self._parse_analytical_intent(question, properties_info)
            if intent is None:
                # Prova un conteggio semantico come fallback
                try:
                    result = collection.query.near_text(query=question, limit=50, return_metadata=MetadataQuery(total_count=True))
                    if hasattr(result, 'total_count'):
                        return f"Stima (ricerca semantica): ho trovato circa {result.total_count} risultati pertinenti."
                except Exception:
                    pass
                return "Non riesco a interpretare la richiesta analitica. Prova con: 'quanti ...', 'elenca ...', 'conta per <campo>'."

            op = intent['op']
            filter_obj = self._build_filter(intent.get('filter')) if intent.get('filter') else None

            if op == 'count':
                resp = collection.aggregate.over_all(total_count=True, filters=filter_obj)
                total = getattr(resp, 'total_count', None)
                if total is not None:
                    return f"Totale elementi{self._fmt_filter_hint(intent)}: {total}"
                return "Conteggio non disponibile."

            if op == 'group_count':
                group_prop = intent.get('group_by')
                if not group_prop:
                    return "Specifica il campo per il raggruppamento (es: 'conta per categoria')."
                resp = collection.aggregate.over_all(group_by=GroupByAggregate(prop=group_prop), filters=filter_obj)
                # Riusa il formatter esistente
                return self._format_weaviate_response(resp, question)

            if op == 'list':
                limit = intent.get('limit', 10)
                limit = max(1, min(50, int(limit)))
                return_props = intent.get('return_properties') or props_info['text'][:3] or props_info['all'][:3]
                resp = collection.query.fetch_objects(filters=filter_obj, limit=limit, return_properties=return_props)
                return self._format_weaviate_response(resp, question)

            return "Operazione analitica non supportata al momento."
            """

            elements_involved = self.askGeminiHowManyElementsInvolved(question, self.weaviateManager.count_objects(class_name))
            
            properties_involved = self.askGeminiAboutPropertiesInvolved(question, properties_info)

            answer = self.performWeaviateQueries(question, elements_involved, properties_involved)
            return answer


        except Exception as e:
            print(f"Errore handle_analytical_question: {e}")
            return f"Errore durante l'elaborazione della domanda analitica: {str(e)}"

    def _get_collection_properties(self, collection) -> dict:
        """Rileva proprietà disponibili e prova a classificarle per tipo."""
        try:
            sample = collection.query.fetch_objects(limit=3)
            if not getattr(sample, 'objects', None):
                return {'all': [], 'text': [], 'number': [], 'datetime': []}
            props_set = set()
            text, number, dt = set(), set(), set()
            for obj in sample.objects:
                for k, v in (obj.properties or {}).items():
                    props_set.add(k)
                    if isinstance(v, str):
                        text.add(k)
                    elif isinstance(v, (int, float)):
                        number.add(k)
                    elif k.lower() in ('timestamp', 'date', 'datetime', 'data'):
                        dt.add(k)
            all_props = list(props_set)
            return {
                'all': all_props,
                'text': list(text),
                'number': list(number),
                'datetime': list(dt)
            }
        except Exception:
            return {'all': [], 'text': [], 'number': [], 'datetime': []}

    # ---- Similarity helpers (no static field assumptions) ----
    def _normalize(self, s: str) -> str:
        try:
            import unicodedata
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(c for c in s if not unicodedata.combining(c))
        except Exception:
            pass
        return s.lower().replace('_', ' ').strip()

    def _tokenize(self, s: str) -> list[str]:
        import re
        s = self._normalize(s)
        return re.findall(r"[a-z0-9]+", s)

    def _score_prop_similarity(self, question: str, prop_name: str) -> float:
        q_norm = self._normalize(question)
        p_norm = self._normalize(prop_name)
        q_tokens = set(self._tokenize(q_norm))
        p_tokens = self._tokenize(p_norm)
        if not p_tokens:
            return 0.0
        token_overlap = len([t for t in p_tokens if t in q_tokens]) / max(1, len(p_tokens))
        substring = 1.0 if p_norm in q_norm else 0.0
        return token_overlap + 0.5 * substring

    def _choose_property_by_similarity(self, question: str, candidates: list[str]) -> str | None:
        best = None
        best_score = 0.0
        for prop in candidates or []:
            s = self._score_prop_similarity(question, prop)
            if s > best_score:
                best, best_score = prop, s
        return best

    def _parse_analytical_intent(self, question: str, props_info: dict) -> dict | None:
        """Estrae operazione, filtro, group_by e limit dalla domanda (IT/EN basico)."""
        q_raw = question.strip()
        q = q_raw.lower()

        # Operazione
        is_count = any(w in q for w in ["quanti", "quante", "quanto", "count", "how many", "numero di"])
        is_list = any(w in q for w in ["elenca", "lista", "mostra", "list", "show"])
        group_by = None

        # Group by rilevato da 'per <campo>' / 'by <field>' / 'raggruppa'
        import re as _re
        by_match = _re.search(r"\b(per|by|for)\s+([a-zA-Z_]+)\b", q)
        if by_match:
            candidate = by_match.group(2)
            group_by = self._infer_property_name(candidate, props_info)
        if any(w in q for w in ["raggruppa", "raggruppare", "group by"]):
            # prova anche a cercare dopo queste parole
            gb_match = _re.search(r"(raggruppa(?:re)?|group by)\s+(per\s+)?([a-zA-Z_]+)", q)
            if gb_match and not group_by:
                group_by = self._infer_property_name(gb_match.group(3), props_info)

        # Limite
        limit = 10
        lim_match = _re.search(r"\b(primi|prime|top|first)\s+(\d{1,3})\b", q)
        if lim_match:
            limit = int(lim_match.group(2))

        # Filtro semplice (autore/categoria/anno)
        filtr = self._infer_simple_filter(q_raw, q, props_info)

        if is_count and group_by:
            return {'op': 'group_count', 'group_by': group_by, 'filter': filtr}
        if is_count:
            return {'op': 'count', 'filter': filtr}
        if is_list or group_by:
            # se c'è un group_by senza 'count', preferisci group_count
            if group_by:
                return {'op': 'group_count', 'group_by': group_by, 'filter': filtr}
            return {'op': 'list', 'filter': filtr, 'limit': limit}
        # Heuristic: domande come "titoli dopo il 2020"
        if any(w in q for w in ["dopo", "prima", "after", "before"]) and filtr:
            return {'op': 'list', 'filter': filtr, 'limit': limit}
        return None

    def _infer_property_name(self, token: str, props_info: dict, prefer_types: list[str] | None = None) -> str | None:
        """Sceglie una proprietà simile al token usando sola similarità, senza sinonimi hard-coded."""
        candidates = props_info.get('all', [])
        # Applica bias di tipo se richiesto
        if prefer_types:
            typed_list = []
            for t in prefer_types:
                typed_list.extend(props_info.get(t, []))
            if typed_list:
                candidates = list(dict.fromkeys(typed_list + candidates))
        return self._choose_property_by_similarity(token, candidates)

    def _infer_simple_filter(self, q_raw: str, q_lower: str, props_info: dict) -> dict | None:
        """Estrae filtri semplici senza campi statici: sceglie la proprietà testuale via similarità e validazione con Weaviate."""
        import re as _re
        # Estrai entity dopo 'di/by/autore' o 'scritto [da]' come valore
        value = None
        # Cattura espressioni tra virgolette
        m = _re.search(r'"([^"]{2,60})"', q_raw)
        if m:
            value = m.group(1).strip()
        else:
            # pattern autore: ... (di|by|autore) <parole>
            m2 = _re.search(r"\b(di|by|autore|author)\s+([a-zàèéìòùA-Z'\-\s]{2,60})", q_raw)
            if m2:
                value = m2.group(2).strip()
            else:
                # pattern: 'scritto [da] <nome>' / 'scritti ...'
                m3 = _re.search(r"\b(scritto|scritti|pubblicati?|written|published)\s+(da\s+)?([a-zàèéìòù'\-\s]{2,60})", q_lower)
                if m3:
                    value = m3.group(3).strip()

        # Seleziona proprietà testuale candidata in modo dinamico
        prop_for_value = None
        if value:
            prop_for_value = self._best_text_property_for_value(q_raw, value, props_info)

        conditions = []
        if value and prop_for_value:
            # usa like per tollerare maiuscole/minuscole e varianti
            like_val = value
            # aggiungi wildcard semplici se non ci sono virgolette
            if not (like_val.startswith('*') or like_val.endswith('*')):
                like_val = f"*{like_val}*"
            conditions.append({'property': prop_for_value, 'op': 'like', 'value': like_val})

        # Filtri numerici su anno (dopo/prima N)
        year_match = _re.search(r"(dopo|after|prima|before)\s+(il\s+)?(\d{4})", q_lower)
        if year_match:
            year = int(year_match.group(3))
            cmp_op = 'greater_than' if year_match.group(1) in ('dopo', 'after') else 'less_than'
            # scegli proprietà numerica/datetime più simile alla domanda
            num_dt_candidates = (props_info.get('number', []) or []) + (props_info.get('datetime', []) or [])
            year_prop = self._choose_property_by_similarity(q_raw, num_dt_candidates)
            if year_prop:
                conditions.append({'property': year_prop, 'op': cmp_op, 'value': year})

        if conditions:
            return {'conditions': conditions}
        return None

    def _best_text_property_for_value(self, question: str, value: str, props_info: dict) -> str | None:
        """Sceglie la proprietà testuale migliore per un certo valore:
        1) ordina per similarità del nome proprietà con la domanda
        2) valida con aggregate count per i primi candidati e sceglie quella con conteggio massimo
        """
        candidates = props_info.get('text', []) or props_info.get('all', [])
        if not candidates:
            return None
        # ordina per similarità
        scored = sorted(((p, self._score_prop_similarity(question, p)) for p in candidates), key=lambda x: x[1], reverse=True)
        top = [p for p, s in scored[:5] if s > 0] or [p for p, _ in scored[:3]]
        # validazione leggera via aggregate
        try:
            collection = None
            # recupera una collection dal client usando un trucco: props_info non ha class_name, quindi questa funzione deve essere chiamata dal flusso che ha la collection corrente.
            # Qui non possiamo accedere a collection, quindi demandiamo la validazione al chiamante se necessario.
            # Workaround: eseguiamo la validazione più avanti nel flusso della query quando costruiamo i filtri.
        except Exception:
            pass
        # Non avendo accesso diretto alla collection qui, restituiamo la migliore candidata per similarità.
        return top[0] if top else candidates[0]

    def _build_filter(self, spec: dict):
        """Costruisce un Filter a partire da una specifica semplice.

        spec = {'conditions': [{'property': 'author', 'op': 'equal', 'value': 'Stephen King'}, ...]}
        """
        if not spec or 'conditions' not in spec or not spec['conditions']:
            return None
        built = None
        for cond in spec['conditions']:
            prop = cond['property']
            op = cond.get('op', 'equal')
            val = cond.get('value')
            try:
                node = None
                if op == 'equal':
                    node = Filter.by_property(prop).equal(val)
                elif op == 'like':
                    node = Filter.by_property(prop).like(val)
                elif op == 'greater_than':
                    node = Filter.by_property(prop).greater_than(val)
                elif op == 'less_than':
                    node = Filter.by_property(prop).less_than(val)
                else:
                    node = Filter.by_property(prop).equal(val)
                built = node if built is None else (built & node)
            except Exception:
                continue
        return built

    def _fmt_filter_hint(self, intent: dict) -> str:
        """Restituisce una stringa descrittiva del filtro per la risposta utente."""
        try:
            f = intent.get('filter')
            if not f:
                return ''
            parts = []
            for c in f.get('conditions', []):
                op = c.get('op')
                if op == 'equal':
                    parts.append(f"{c.get('property')} = '{c.get('value')}'")
                elif op in ('greater_than', 'less_than'):
                    sym = '>' if op == 'greater_than' else '<'
                    parts.append(f"{c.get('property')} {sym} {c.get('value')}")
            return f" (filtro: {', '.join(parts)})" if parts else ''
        except Exception:
            return ''

    def _format_weaviate_response(self, response, question: str) -> str:
        """Formatta le risposte di Weaviate in formato leggibile."""
        try:
            # Gestisci diversi tipi di risposta di Weaviate
            if hasattr(response, 'objects') and response.objects:
                # Query con oggetti (fetch_objects, near_text, etc.)
                results = []
                for i, obj in enumerate(response.objects[:10], 1):  # Limita a 10 risultati
                    obj_data = []
                    if hasattr(obj, 'properties') and obj.properties:
                        for key, value in obj.properties.items():
                            if value is not None and str(value).strip():
                                # Tronca valori molto lunghi
                                value_str = str(value)
                                if len(value_str) > 100:
                                    value_str = value_str[:100] + "..."
                                obj_data.append(f"**{key}**: {value_str}")
                    
                    if obj_data:
                        results.append(f"**{i}.** " + " | ".join(obj_data))
                
                if results:
                    total = len(response.objects)
                    header = f"🔍 **Risultati per:** {question}\n\n"
                    if total > 10:
                        header += f"**Mostro i primi 10 di {total} risultati:**\n\n"
                    else:
                        header += f"**{total} risultat{'o' if total == 1 else 'i'} trovat{'o' if total == 1 else 'i'}:**\n\n"
                    
                    return header + "\n\n".join(results)
                else:
                    return f"❌ **Nessun risultato trovato** per: {question}"
            
            elif hasattr(response, 'groups') and response.groups:
                # Aggregazioni con raggruppamento
                results = []
                for group in response.groups[:20]:  # Limita a 20 gruppi
                    group_by = group.grouped_by.value if hasattr(group.grouped_by, 'value') else str(group.grouped_by)
                    count = group.total_count if hasattr(group, 'total_count') else 'N/A'
                    results.append(f"**{group_by}**: {count}")
                
                if results:
                    header = f"📊 **Raggruppamento per:** {question}\n\n"
                    return header + "\n".join(results)
                else:
                    return f"❌ **Nessun gruppo trovato** per: {question}"
            
            elif hasattr(response, 'total_count'):
                # Semplice conteggio
                count = response.total_count
                return f"🔢 **Risultato conteggio**: {count} element{'o' if count == 1 else 'i'}"
            
            else:
                # Fallback per tipi di risposta non gestiti
                return f"📋 **Risposta**: {str(response)}"
                
        except Exception as e:
            print(f"Errore nel formattare la risposta Weaviate: {e}")
            return f"❌ **Errore nella formattazione**: {str(e)}"
    #---------------------------------------#
    #      Conversational Questions         #
    #                                       #
    #---------------------------------------#
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

    
    #---------------------------------------#
    #      General semantic Questions       #
    #                                       #
    #---------------------------------------#

    def handle_general_question(self, question: str, class_name: str) -> str:
        """RAG ottimizzato con proprietà dinamiche basate sullo schema effettivo."""
        try:
            collection = self.client.collections.get(class_name)
            
            # Ottieni le proprietà della collezione
            props_info = self._get_collection_properties(collection)
            available_props = list(props_info.keys())
            
            if not available_props:
                return f"La collezione '{class_name}' non ha proprietà disponibili."
            
            # Scegli le proprietà migliori per la ricerca (max 3 per ridurre token)
            search_props = self._select_best_search_properties(available_props)
            #aggiungi questo:  ----> search_props = self.select_best_search_properties_with_gemini(available_props)
            
            elements_involved = self.askGeminiHowManyElementsInvolved(question, self.weaviateManager.count_objects(class_name))
            properties_involed = self.askGeminiAboutPropertiesInvolved(question, props_info)

            
            # Cerca documenti pertinenti
            result = collection.query.near_text(
                query=question,
                limit= elements_involved , 
                return_properties=properties_involed
            )
            
            if not result.objects:
                return f"Non ho trovato informazioni pertinenti per la tua domanda nella collezione '{class_name}'."

            # Estrai il contenuto dalle proprietà disponibili
            contexts = []
            for doc in result.objects:
                doc_content = []
                for prop in search_props:
                    value = doc.properties.get(prop, '')
                    if value and str(value).strip():
                        # Tronca contenuti molto lunghi
                        value_str = str(value)
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "..."
                        doc_content.append(f"{prop}: {value_str}")
                
                if doc_content:
                    contexts.append(" | ".join(doc_content))
            
            if not contexts:
                return f"I documenti trovati nella collezione '{class_name}' non contengono informazioni utili."
            
            context = "\n".join(contexts)
            
            # Prompt compatto con informazioni sulla collezione
            prompt = f"""Basato sui dati della collezione "{class_name}", rispondi: "{question}"

Dati disponibili:
{context}

Risposta breve e diretta:"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Errore nell'analisi della collezione '{class_name}': {e}"


    ##da eliminare
    def _select_best_search_properties(self, available_props: list) -> list:
        """Seleziona le migliori proprietà per la ricerca semantica."""
        # Ordina per preferenza le proprietà più utili per la ricerca
        preferred_order = [
            # Proprietà di contenuto testuale
            'content', 'contenuto', 'text', 'testo', 'description', 'descrizione',
            'summary', 'abstract', 'body', 'details', 'info', 'information',
            # Proprietà di titolo/nome
            'title', 'titolo', 'name', 'nome', 'heading', 'subject', 'oggetto',
            # Altre proprietà testuali
            'category', 'categoria', 'type', 'tipo', 'genre', 'genere'
        ]
        
        selected_props = []
        available_lower = [prop.lower() for prop in available_props]
        
        # Prima aggiungi le proprietà preferite che esistono
        for pref in preferred_order:
            for i, prop_lower in enumerate(available_lower):
                if pref in prop_lower and available_props[i] not in selected_props:
                    selected_props.append(available_props[i])
                    if len(selected_props) >= 3:  # Limite a 3 proprietà
                        return selected_props
        
        # Se non ne abbiamo abbastanza, aggiungi altre proprietà testuali
        for prop in available_props:
            if prop not in selected_props and len(selected_props) < 3:
                selected_props.append(prop)
        
        return selected_props[:3] if selected_props else available_props[:3]



    #---------------------------------------#
    #      Cleaning Questions               #
    #                                       #
    #---------------------------------------#


    def handle_cleaning_question(self, question: str, class_name: str = None) -> str:
        """Gestisce domande di pulizia dati con operazioni reali sui dati."""
        if not class_name:
            return "🧹 **Pulizia Dati**\n\nPer eseguire operazioni di pulizia, devi prima selezionare una collezione di dati da pulire."
        
        try:
            collection = self.client.collections.get(class_name)
            
            # Usa Gemini per identificare intelligentemente il tipo di pulizia richiesta
            cleaning_type = self._identify_cleaning_type_with_gemini(question, class_name)
            
            # Esegui l'operazione di pulizia appropriata basata sulla classificazione di Gemini
            if cleaning_type == "duplicates":
                return self._handle_duplicate_removal(collection, class_name, question)
            elif cleaning_type == "empty_values":
                return self._handle_empty_values(collection, class_name, question)
            elif cleaning_type == "text_normalization":
                return self._handle_text_normalization(collection, class_name, question)
            elif cleaning_type == "whitespace":
                return self._handle_whitespace_cleaning(collection, class_name, question)
            elif cleaning_type == "encoding":
                return self._handle_encoding_issues(collection, class_name, question)
            elif cleaning_type == "validation":
                return self._handle_data_validation(collection, class_name, question)
            elif cleaning_type == "outliers":
                return self._handle_outlier_removal(collection, class_name, question)
            elif cleaning_type == "general":
                return self._handle_general_cleaning(collection, class_name, question)
            else:
                # Per operazioni personalizzate o non standard
                return self._handle_custom_cleaning_with_gemini(question, collection, class_name)
                
        except Exception as e:
            return f"🧹 **Errore durante la pulizia**\n\nSi è verificato un errore: {str(e)}"

    def _identify_cleaning_type_with_gemini(self, question: str, class_name: str) -> str:
        """Usa Gemini per identificare intelligentemente il tipo di pulizia richiesta."""
        try:
            # Ottieni informazioni sulla collezione per dare contesto a Gemini
            collection = self.client.collections.get(class_name)
            
            # Prendi un campione per capire la struttura dei dati
            sample_response = collection.query.fetch_objects(limit=3)
            sample_data = ""
            
            if sample_response.objects:
                sample_properties = []
                for obj in sample_response.objects[:2]:  # Solo 2 esempi per non sovraccaricare
                    obj_props = []
                    for key, value in obj.properties.items():
                        # Tronca valori lunghi
                        value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        obj_props.append(f"{key}: {value_str}")
                    sample_properties.append(" | ".join(obj_props))
                
                sample_data = "\n".join([f"Esempio {i+1}: {props}" for i, props in enumerate(sample_properties)])
            
            prompt = f"""
            Analizza questa richiesta di pulizia dati e identifica il tipo di operazione più appropriato.
            
            RICHIESTA: "{question}"
            
            COLLEZIONE: {class_name}
            DATI DI ESEMPIO:
            {sample_data}
            
            TIPI DI PULIZIA DISPONIBILI:
            1. "duplicates" - Rimuovere record duplicati o molto simili
            2. "empty_values" - Gestire valori vuoti, null, NaN o mancanti
            3. "text_normalization" - Normalizzare testo (maiuscole/minuscole, formattazione)
            4. "whitespace" - Rimuovere spazi extra, trim, pulizia whitespace
            5. "encoding" - Correggere problemi di encoding, caratteri speciali, accenti
            6. "validation" - Validare formato dati, controllare integrità
            7. "outliers" - Rimuovere valori anomali o outliers
            8. "general" - Pulizia generale o combinazione di operazioni
            9. "custom" - Operazione personalizzata non coperta dai tipi standard
            
            ISTRUZIONI:
            - Analizza la richiesta nel contesto dei dati disponibili
            - Considera sia il linguaggio naturale che la struttura dei dati
            - Se la richiesta è ambigua, scegli il tipo più generale appropriato
            - Rispondi con UNA SOLA parola chiave tra quelle elencate sopra
            
            RISPOSTA:"""
            
            response = self.model.generate_content(prompt)
            classification = response.text.strip().lower()
            
            # Mappa le possibili risposte ai tipi validi
            valid_types = [
                "duplicates", "empty_values", "text_normalization", 
                "whitespace", "encoding", "validation", "outliers", 
                "general", "custom"
            ]
            
            # Cerca la classificazione nella risposta
            for valid_type in valid_types:
                if valid_type in classification:
                    print(f"🤖 Gemini ha classificato la pulizia come: {valid_type}")
                    return valid_type
            
            # Fallback: prova a fare un match parziale più intelligente
            if any(word in classification for word in ["duplicat", "duplicate", "doppi"]):
                return "duplicates"
            elif any(word in classification for word in ["vuot", "null", "empty", "mancant"]):
                return "empty_values"
            elif any(word in classification for word in ["normal", "format", "maiuscol", "minuscol"]):
                return "text_normalization"
            elif any(word in classification for word in ["spazi", "space", "trim"]):
                return "whitespace"
            elif any(word in classification for word in ["encoding", "caratteri", "accenti"]):
                return "encoding"
            elif any(word in classification for word in ["valid", "controlla", "verifica"]):
                return "validation"
            elif any(word in classification for word in ["outlier", "anomal"]):
                return "outliers"
            else:
                return "general"  # Fallback sicuro
                
        except Exception as e:
            print(f"Errore nella classificazione con Gemini: {e}")
            # Fallback a classificazione locale semplice
            question_lower = question.lower()
            if any(word in question_lower for word in ["duplicat", "duplicate", "doppi"]):
                return "duplicates"
            elif any(word in question_lower for word in ["vuot", "null", "empty"]):
                return "empty_values"
            else:
                return "general"
                
        except Exception as e:
            return f"🧹 **Errore durante la pulizia**\n\nSi è verificato un errore: {str(e)}"

    def _handle_duplicate_removal(self, collection, class_name: str, question: str) -> str:
        """Gestisce la rimozione dei duplicati."""
        try:
            # Ottieni un campione per analizzare i duplicati
            response = collection.query.fetch_objects(limit=1000, include_vector=True)
            documents = response.objects
            
            if len(documents) < 2:
                return "🧹 **Rimozione Duplicati**\n\nNon ci sono abbastanza documenti per rilevare duplicati (minimo 2 richiesti)."
            
            # Calcola similarità tra documenti
            vectors = []
            doc_info = []
            
            for doc in documents:
                if hasattr(doc, 'vector') and doc.vector:
                    vectors.append(doc.vector.get("default", []))
                    doc_info.append({
                        "id": str(doc.uuid),
                        "properties": doc.properties
                    })
            
            if len(vectors) < 2:
                return "🧹 **Rimozione Duplicati**\n\nNon sono disponibili vettori per il calcolo della similarità."
            
            # Calcola matrice di similarità
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectors_array = np.array(vectors)
            similarity_matrix = cosine_similarity(vectors_array)
            
            # Trova duplicati (soglia di similarità alta)
            threshold = 0.95
            duplicates_found = []
            processed_indices = set()
            
            for i in range(len(similarity_matrix)):
                if i in processed_indices:
                    continue
                    
                similar_docs = []
                for j in range(i + 1, len(similarity_matrix)):
                    if j in processed_indices:
                        continue
                        
                    if similarity_matrix[i][j] > threshold:
                        if not similar_docs:  # Prima volta che troviamo un duplicato per doc i
                            similar_docs.append(doc_info[i])
                        similar_docs.append(doc_info[j])
                        processed_indices.add(j)
                
                if similar_docs:
                    duplicates_found.append(similar_docs)
                    processed_indices.add(i)
            
            if not duplicates_found:
                return f"🧹 **Rimozione Duplicati Completata**\n\nNessun duplicato trovato nella collezione '{class_name}' con soglia di similarità {threshold:.0%}.\n\n✅ La collezione è già pulita!"
            
            # Conta i duplicati
            total_duplicates = sum(len(group) - 1 for group in duplicates_found)  # -1 perché teniamo l'originale
            
            result = f"🧹 **Duplicati Rilevati**\n\n"
            result += f"📊 **Statistiche:**\n"
            result += f"• Documenti analizzati: {len(documents)}\n"
            result += f"• Gruppi di duplicati: {len(duplicates_found)}\n"
            result += f"• Duplicati da rimuovere: {total_duplicates}\n\n"
            
            result += f"🔍 **Dettagli gruppi duplicati:**\n"
            
            for i, group in enumerate(duplicates_found[:5]):  # Mostra max 5 gruppi
                result += f"\n**Gruppo {i+1}** ({len(group)} documenti):\n"
                
                for j, doc in enumerate(group[:3]):  # Mostra max 3 doc per gruppo
                    # Trova la prima proprietà testuale per il preview
                    preview = "N/A"
                    for prop_name, prop_value in doc["properties"].items():
                        if isinstance(prop_value, str) and len(prop_value) > 10:
                            preview = prop_value[:100] + "..." if len(prop_value) > 100 else prop_value
                            break
                    
                    result += f"  • Doc {j+1}: {preview}\n"
                
                if len(group) > 3:
                    result += f"  ... e altri {len(group)-3} documenti simili\n"
            
            if len(duplicates_found) > 5:
                result += f"\n... e altri {len(duplicates_found)-5} gruppi di duplicati.\n"
            
            # Rimuovi automaticamente i duplicati se richiesto esplicitamente
            if any(word in question.lower() for word in ["rimuovi", "elimina", "cancella", "remove", "delete"]):
                removed_count = 0
                for group in duplicates_found:
                    # Tieni il primo documento di ogni gruppo, rimuovi gli altri
                    for doc_to_remove in group[1:]:
                        try:
                            collection.data.delete_by_id(doc_to_remove["id"])
                            removed_count += 1
                        except Exception as e:
                            print(f"Errore rimozione documento {doc_to_remove['id']}: {e}")
                
                result += f"\n✅ **Rimozione completata!**\n"
                result += f"• Documenti rimossi: {removed_count}\n"
                result += f"• Documenti rimanenti: {len(documents) - removed_count}"
            else:
                result += f"\n💡 **Suggerimento:** Per rimuovere automaticamente i duplicati, chiedi: 'Rimuovi i duplicati dalla collezione {class_name}'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Rimozione Duplicati**\n\nErrore durante l'analisi duplicati: {str(e)}"

    def _handle_empty_values(self, collection, class_name: str, question: str) -> str:
        """Gestisce la pulizia di valori vuoti/null."""
        try:
            # Ottieni campione per analizzare valori vuoti
            response = collection.query.fetch_objects(limit=500)
            documents = response.objects
            
            if not documents:
                return "🧹 **Pulizia Valori Vuoti**\n\nNessun documento trovato nella collezione."
            
            # Analizza valori vuoti
            property_stats = {}
            total_docs = len(documents)
            
            # Ottieni tutte le proprietà
            all_properties = set()
            for doc in documents:
                all_properties.update(doc.properties.keys())
            
            for prop_name in all_properties:
                empty_count = 0
                non_empty_values = []
                
                for doc in documents:
                    value = doc.properties.get(prop_name)
                    
                    # Controlla se è vuoto
                    if (value is None or 
                        (isinstance(value, str) and value.strip() == "") or
                        (isinstance(value, (list, dict)) and len(value) == 0)):
                        empty_count += 1
                    else:
                        non_empty_values.append(value)
                
                if empty_count > 0:
                    property_stats[prop_name] = {
                        "empty_count": empty_count,
                        "non_empty_count": total_docs - empty_count,
                        "empty_percentage": (empty_count / total_docs) * 100,
                        "sample_values": non_empty_values[:3]  # Primi 3 valori non vuoti
                    }
            
            if not property_stats:
                return f"🧹 **Pulizia Valori Vuoti Completata**\n\n✅ Nessun valore vuoto trovato nella collezione '{class_name}'!"
            
            result = f"🧹 **Analisi Valori Vuoti**\n\n"
            result += f"📊 **Statistiche generali:**\n"
            result += f"• Documenti analizzati: {total_docs}\n"
            result += f"• Proprietà con valori vuoti: {len(property_stats)}\n\n"
            
            result += f"🔍 **Dettaglio per proprietà:**\n"
            
            for prop_name, stats in sorted(property_stats.items(), key=lambda x: x[1]["empty_percentage"], reverse=True):
                result += f"\n**{prop_name}:**\n"
                result += f"  • Valori vuoti: {stats['empty_count']} ({stats['empty_percentage']:.1f}%)\n"
                result += f"  • Valori presenti: {stats['non_empty_count']}\n"
                
                if stats['sample_values']:
                    sample_str = ", ".join([str(v)[:50] for v in stats['sample_values']])
                    result += f"  • Esempi valori: {sample_str}\n"
            
            # Se richiesto, rimuovi documenti con troppi valori vuoti
            if any(word in question.lower() for word in ["rimuovi", "elimina", "remove", "delete", "pulisci"]):
                # Rimuovi documenti che hanno >70% di campi vuoti
                threshold = 0.7
                removed_count = 0
                
                for doc in documents:
                    empty_fields = 0
                    total_fields = len(all_properties)
                    
                    for prop_name in all_properties:
                        value = doc.properties.get(prop_name)
                        if (value is None or 
                            (isinstance(value, str) and value.strip() == "") or
                            (isinstance(value, (list, dict)) and len(value) == 0)):
                            empty_fields += 1
                    
                    empty_ratio = empty_fields / total_fields if total_fields > 0 else 0
                    
                    if empty_ratio > threshold:
                        try:
                            collection.data.delete_by_id(str(doc.uuid))
                            removed_count += 1
                        except Exception as e:
                            print(f"Errore rimozione documento {doc.uuid}: {e}")
                
                result += f"\n✅ **Pulizia completata!**\n"
                result += f"• Documenti rimossi (>{threshold:.0%} campi vuoti): {removed_count}\n"
                result += f"• Documenti rimanenti: {total_docs - removed_count}"
            else:
                result += f"\n💡 **Suggerimenti:**\n"
                result += f"• Per rimuovere documenti con troppi campi vuoti: 'Rimuovi i documenti con valori vuoti'\n"
                result += f"• Per sostituire valori vuoti: 'Sostituisci i valori vuoti con [valore]'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Valori Vuoti**\n\nErrore durante l'analisi: {str(e)}"

    def _handle_text_normalization(self, collection, class_name: str, question: str) -> str:
        """Gestisce la normalizzazione del testo."""
        try:
            response = collection.query.fetch_objects(limit=100)
            documents = response.objects
            
            if not documents:
                return "🧹 **Normalizzazione Testo**\n\nNessun documento trovato nella collezione."
            
            # Trova proprietà testuali
            text_properties = []
            for doc in documents[:5]:  # Campione per identificare proprietà testuali
                for prop_name, prop_value in doc.properties.items():
                    if isinstance(prop_value, str) and len(prop_value) > 5:
                        if prop_name not in text_properties:
                            text_properties.append(prop_name)
            
            if not text_properties:
                return f"🧹 **Normalizzazione Testo**\n\nNessuna proprietà testuale trovata nella collezione '{class_name}'."
            
            # Analizza problemi di normalizzazione
            normalization_issues = {
                'mixed_case': 0,
                'extra_spaces': 0,
                'special_chars': 0,
                'encoding_issues': 0
            }
            
            examples = {
                'mixed_case': [],
                'extra_spaces': [],
                'special_chars': [],
                'encoding_issues': []
            }
            
            import re
            
            for doc in documents:
                for prop_name in text_properties:
                    value = doc.properties.get(prop_name, '')
                    if not isinstance(value, str) or len(value.strip()) == 0:
                        continue
                    
                    # Controlla case misto inconsistente
                    if re.search(r'[A-Z][a-z]+[A-Z]', value) or re.search(r'[a-z][A-Z]', value):
                        normalization_issues['mixed_case'] += 1
                        if len(examples['mixed_case']) < 3:
                            examples['mixed_case'].append(f"{prop_name}: '{value[:50]}...'")
                    
                    # Controlla spazi extra
                    if '  ' in value or value != value.strip():
                        normalization_issues['extra_spaces'] += 1
                        if len(examples['extra_spaces']) < 3:
                            examples['extra_spaces'].append(f"{prop_name}: '{value[:50]}...'")
                    
                    # Controlla caratteri speciali eccessivi
                    special_count = len(re.findall(r'[^\w\s\.\,\!\?\-\'\"]', value))
                    if special_count / len(value) > 0.1:  # >10% caratteri speciali
                        normalization_issues['special_chars'] += 1
                        if len(examples['special_chars']) < 3:
                            examples['special_chars'].append(f"{prop_name}: '{value[:50]}...'")
                    
                    # Controlla possibili problemi di encoding
                    if any(char in value for char in ['Ã', 'â', 'Â', 'Ã©', 'Ã¨']):
                        normalization_issues['encoding_issues'] += 1
                        if len(examples['encoding_issues']) < 3:
                            examples['encoding_issues'].append(f"{prop_name}: '{value[:50]}...'")
            
            total_issues = sum(normalization_issues.values())
            
            if total_issues == 0:
                return f"🧹 **Normalizzazione Testo Completata**\n\n✅ Nessun problema di normalizzazione trovato nella collezione '{class_name}'!"
            
            result = f"🧹 **Analisi Normalizzazione Testo**\n\n"
            result += f"📊 **Statistiche:**\n"
            result += f"• Documenti analizzati: {len(documents)}\n"
            result += f"• Proprietà testuali: {len(text_properties)}\n"
            result += f"• Problemi totali rilevati: {total_issues}\n\n"
            
            result += f"🔍 **Dettaglio problemi:**\n"
            
            issue_labels = {
                'mixed_case': 'Case inconsistente',
                'extra_spaces': 'Spazi extra',
                'special_chars': 'Caratteri speciali eccessivi',
                'encoding_issues': 'Problemi encoding'
            }
            
            for issue_type, count in normalization_issues.items():
                if count > 0:
                    result += f"\n**{issue_labels[issue_type]}:** {count} occorrenze\n"
                    if examples[issue_type]:
                        result += "  Esempi:\n"
                        for example in examples[issue_type]:
                            result += f"  • {example}\n"
            
            # Suggerimenti per la normalizzazione
            result += f"\n💡 **Suggerimenti di normalizzazione:**\n"
            if normalization_issues['mixed_case'] > 0:
                result += f"• Per uniformare il case: 'Converti tutto in minuscolo' o 'Converti in Title Case'\n"
            if normalization_issues['extra_spaces'] > 0:
                result += f"• Per rimuovere spazi extra: 'Rimuovi spazi doppi e trim'\n"
            if normalization_issues['special_chars'] > 0:
                result += f"• Per pulire caratteri speciali: 'Rimuovi caratteri speciali'\n"
            if normalization_issues['encoding_issues'] > 0:
                result += f"• Per correggere encoding: 'Correggi problemi di encoding UTF-8'\n"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Normalizzazione Testo**\n\nErrore durante l'analisi: {str(e)}"

    def _handle_whitespace_cleaning(self, collection, class_name: str, question: str) -> str:
        """Gestisce la pulizia degli spazi bianchi."""
        try:
            response = collection.query.fetch_objects(limit=200)
            documents = response.objects
            
            if not documents:
                return "🧹 **Pulizia Spazi**\n\nNessun documento trovato nella collezione."
            
            # Trova proprietà testuali e analizza problemi di spazi
            whitespace_issues = 0
            cleaned_count = 0
            examples = []
            
            for doc in documents:
                doc_updated = False
                updated_properties = {}
                
                for prop_name, prop_value in doc.properties.items():
                    if isinstance(prop_value, str):
                        original_value = prop_value
                        
                        # Pulisci spazi
                        cleaned_value = prop_value.strip()  # Rimuovi spazi iniziali/finali
                        cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Sostituisci spazi multipli con singoli
                        
                        if original_value != cleaned_value:
                            whitespace_issues += 1
                            updated_properties[prop_name] = cleaned_value
                            doc_updated = True
                            
                            if len(examples) < 5:
                                examples.append({
                                    "property": prop_name,
                                    "before": original_value[:50] + "..." if len(original_value) > 50 else original_value,
                                    "after": cleaned_value[:50] + "..." if len(cleaned_value) > 50 else cleaned_value
                                })
                
                # Se richiesto, aggiorna il documento
                if doc_updated and any(word in question.lower() for word in ["pulisci", "rimuovi", "clean", "trim", "remove"]):
                    try:
                        # Aggiorna tutte le proprietà del documento
                        all_properties = doc.properties.copy()
                        all_properties.update(updated_properties)
                        
                        collection.data.replace(
                            uuid=str(doc.uuid),
                            properties=all_properties
                        )
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Errore aggiornamento documento {doc.uuid}: {e}")
            
            if whitespace_issues == 0:
                return f"🧹 **Pulizia Spazi Completata**\n\n✅ Nessun problema di spazi trovato nella collezione '{class_name}'!"
            
            result = f"🧹 **Analisi Spazi Bianchi**\n\n"
            result += f"📊 **Statistiche:**\n"
            result += f"• Documenti analizzati: {len(documents)}\n"
            result += f"• Problemi di spazi rilevati: {whitespace_issues}\n"
            
            if examples:
                result += f"\n🔍 **Esempi di pulizia:**\n"
                for example in examples:
                    result += f"\n**{example['property']}:**\n"
                    result += f"  Prima:  '{example['before']}'\n"
                    result += f"  Dopo:   '{example['after']}'\n"
            
            if cleaned_count > 0:
                result += f"\n✅ **Pulizia completata!**\n"
                result += f"• Documenti aggiornati: {cleaned_count}\n"
                result += f"• Proprietà corrette: {whitespace_issues}"
            else:
                result += f"\n💡 **Suggerimento:** Per applicare automaticamente la pulizia degli spazi, chiedi: 'Pulisci gli spazi dalla collezione {class_name}'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Spazi**\n\nErrore durante la pulizia: {str(e)}"

    def _handle_encoding_issues(self, collection, class_name: str, question: str) -> str:
        """Gestisce i problemi di encoding dei caratteri."""
        return f"""🧹 **Correzione Encoding**

Ho rilevato una richiesta per correggere problemi di encoding nella collezione '{class_name}'.

**Problemi comuni di encoding:**
- Caratteri accentati malformati (es: Ã, â, Â)
- Simboli strani al posto di caratteri normali
- Testo illeggibile dopo import da fonti diverse

**Funzionalità di correzione encoding in sviluppo:**
- ✨ Auto-rilevamento encoding problematici
- ✨ Correzione UTF-8 automatica
- ✨ Normalizzazione caratteri accentati
- ✨ Conversione tra diversi encoding

Questa funzionalità sarà presto disponibile! 🚀

Nel frattempo, puoi:
- Verificare l'encoding originale dei file prima dell'import
- Usare UTF-8 come encoding standard
- Controllare i caratteri speciali nei tuoi dati"""

    def _handle_data_validation(self, collection, class_name: str, question: str) -> str:
        """Gestisce la validazione dell'integrità dei dati."""
        try:
            response = collection.query.fetch_objects(limit=300)
            documents = response.objects
            
            if not documents:
                return "🧹 **Validazione Dati**\n\nNessun documento trovato nella collezione."
            
            # Ottieni schema delle proprietà
            all_properties = set()
            property_types = {}
            
            for doc in documents[:50]:  # Campione per determinare tipi
                for prop_name, prop_value in doc.properties.items():
                    all_properties.add(prop_name)
                    
                    if prop_name not in property_types:
                        if isinstance(prop_value, str):
                            property_types[prop_name] = 'string'
                        elif isinstance(prop_value, (int, float)):
                            property_types[prop_name] = 'number'
                        elif isinstance(prop_value, bool):
                            property_types[prop_name] = 'boolean'
                        elif isinstance(prop_value, list):
                            property_types[prop_name] = 'array'
                        else:
                            property_types[prop_name] = 'unknown'
            
            # Validazione
            validation_results = {
                'missing_properties': 0,
                'type_mismatches': 0,
                'invalid_values': 0,
                'inconsistent_formats': 0
            }
            
            issues_details = []
            
            for doc in documents:
                doc_id = str(doc.uuid)
                
                # Controlla proprietà mancanti
                missing_props = all_properties - set(doc.properties.keys())
                if missing_props:
                    validation_results['missing_properties'] += len(missing_props)
                    issues_details.append(f"Doc {doc_id[:8]}: proprietà mancanti {list(missing_props)[:3]}")
                
                # Controlla tipi inconsistenti
                for prop_name, prop_value in doc.properties.items():
                    expected_type = property_types.get(prop_name)
                    actual_type = type(prop_value).__name__
                    
                    if expected_type == 'string' and not isinstance(prop_value, str):
                        validation_results['type_mismatches'] += 1
                        issues_details.append(f"Doc {doc_id[:8]}: {prop_name} dovrebbe essere stringa, trovato {actual_type}")
                    
                    elif expected_type == 'number' and not isinstance(prop_value, (int, float)):
                        validation_results['type_mismatches'] += 1
                        issues_details.append(f"Doc {doc_id[:8]}: {prop_name} dovrebbe essere numero, trovato {actual_type}")
                    
                    # Validazione valori specifici
                    if isinstance(prop_value, str):
                        # Email malformate
                        if 'email' in prop_name.lower() and '@' in prop_value and '.' not in prop_value.split('@')[-1]:
                            validation_results['invalid_values'] += 1
                            issues_details.append(f"Doc {doc_id[:8]}: email malformata in {prop_name}")
                        
                        # URL malformati
                        if 'url' in prop_name.lower() and not (prop_value.startswith('http://') or prop_value.startswith('https://')):
                            validation_results['invalid_values'] += 1
                            issues_details.append(f"Doc {doc_id[:8]}: URL malformato in {prop_name}")
            
            total_issues = sum(validation_results.values())
            
            result = f"🧹 **Validazione Integrità Dati**\n\n"
            result += f"📊 **Statistiche:**\n"
            result += f"• Documenti analizzati: {len(documents)}\n"
            result += f"• Proprietà uniche rilevate: {len(all_properties)}\n"
            result += f"• Problemi totali: {total_issues}\n\n"
            
            if total_issues == 0:
                result += "✅ **Validazione completata con successo!**\nNessun problema di integrità rilevato.\n"
            else:
                result += f"🔍 **Dettaglio problemi:**\n"
                for issue_type, count in validation_results.items():
                    if count > 0:
                        issue_name = issue_type.replace('_', ' ').title()
                        result += f"• {issue_name}: {count}\n"
                
                result += f"\n📝 **Primi esempi di problemi:**\n"
                for detail in issues_details[:10]:
                    result += f"• {detail}\n"
                
                if len(issues_details) > 10:
                    result += f"... e altri {len(issues_details) - 10} problemi.\n"
            
            result += f"\n📋 **Schema proprietà rilevato:**\n"
            for prop_name, prop_type in sorted(property_types.items()):
                result += f"• {prop_name}: {prop_type}\n"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Validazione Dati**\n\nErrore durante la validazione: {str(e)}"

    def _handle_outlier_removal(self, collection, class_name: str, question: str) -> str:
        """Gestisce la rimozione di outliers e valori anomali."""
        return f"""🧹 **Rimozione Outliers**

Ho rilevato una richiesta per identificare e rimuovere outliers nella collezione '{class_name}'.

**Tipi di outliers che posso rilevare:**
- 📊 Outliers numerici (valori statisticamente anomali)
- 📝 Testi anomali (lunghezza, pattern strani)
- 📅 Date fuori range ragionevole
- 🔢 Valori fuori dai limiti logici

**Funzionalità di rimozione outliers in sviluppo:**
- ✨ Rilevamento automatico outliers statistici (Z-score, IQR)
- ✨ Analisi anomalie testuali
- ✨ Validazione range logici
- ✨ Opzioni di rimozione/correzione selettive

Questa funzionalità avanzata sarà presto disponibile! 🚀

**Nel frattempo puoi:**
- Identificare valori sospetti con query analitiche
- Usare "mostra valori estremi per [campo]"
- Controllare manualmente range dei dati numerici"""

    def _handle_general_cleaning(self, collection, class_name: str, question: str) -> str:
        """Gestisce richieste di pulizia generale (combina più operazioni)."""
        try:
            result = f"🧹 **Pulizia Generale Avviata**\n\nEseguo una pulizia completa della collezione '{class_name}'...\n\n"
            
            # 1. Controlla duplicati
            result += "🔍 **1. Analisi Duplicati...**\n"
            duplicate_result = self._handle_duplicate_removal(collection, class_name, "analizza duplicati")
            if "Nessun duplicato" in duplicate_result:
                result += "✅ Nessun duplicato trovato\n\n"
            else:
                # Estrai solo il numero di duplicati dal risultato
                import re
                match = re.search(r'Duplicati da rimuovere: (\d+)', duplicate_result)
                if match:
                    result += f"⚠️ Trovati {match.group(1)} duplicati\n\n"
            
            # 2. Controlla valori vuoti
            result += "🔍 **2. Analisi Valori Vuoti...**\n"
            empty_result = self._handle_empty_values(collection, class_name, "analizza valori vuoti")
            if "Nessun valore vuoto" in empty_result:
                result += "✅ Nessun valore vuoto problematico\n\n"
            else:
                # Estrai info sui valori vuoti
                import re
                match = re.search(r'Proprietà con valori vuoti: (\d+)', empty_result)
                if match:
                    result += f"⚠️ {match.group(1)} proprietà hanno valori vuoti\n\n"
            
            # 3. Controlla normalizzazione testo
            result += "🔍 **3. Analisi Normalizzazione...**\n"
            norm_result = self._handle_text_normalization(collection, class_name, "analizza normalizzazione")
            if "Nessun problema" in norm_result:
                result += "✅ Testo già normalizzato correttamente\n\n"
            else:
                import re
                match = re.search(r'Problemi totali rilevati: (\d+)', norm_result)
                if match:
                    result += f"⚠️ {match.group(1)} problemi di normalizzazione\n\n"
            
            # 4. Controlla spazi
            result += "🔍 **4. Analisi Spazi Bianchi...**\n"
            space_result = self._handle_whitespace_cleaning(collection, class_name, "analizza spazi")
            if "Nessun problema" in space_result:
                result += "✅ Spazi già puliti\n\n"
            else:
                import re
                match = re.search(r'Problemi di spazi rilevati: (\d+)', space_result)
                if match:
                    result += f"⚠️ {match.group(1)} problemi di spazi\n\n"
            
            # 5. Validazione generale
            result += "🔍 **5. Validazione Integrità...**\n"
            validation_result = self._handle_data_validation(collection, class_name, "valida dati")
            if "Nessun problema" in validation_result:
                result += "✅ Integrità dati confermata\n\n"
            else:
                import re
                match = re.search(r'Problemi totali: (\d+)', validation_result)
                if match:
                    result += f"⚠️ {match.group(1)} problemi di integrità\n\n"
            
            result += "🎯 **Riepilogo Pulizia Generale:**\n"
            result += "La scansione è completata. Per applicare automaticamente le correzioni, usa comandi specifici come:\n"
            result += "• 'Rimuovi i duplicati'\n"
            result += "• 'Pulisci gli spazi bianchi'\n"
            result += "• 'Normalizza il testo'\n"
            result += "• 'Rimuovi valori vuoti'\n\n"
            result += "💡 **Suggerimento:** Per una pulizia automatica completa, chiedi: 'Applica tutte le correzioni di pulizia'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Generale**\n\nErrore durante la pulizia generale: {str(e)}"

    #---------------------------------------#
    #      Knowledge Extraction Questions   #
    #                                       #
    #---------------------------------------#
  
    def handle_knowledge_extraction_question(self, question: str, class_name: str) -> str:
        """Gestisce domande di estrazione di conoscenza con operazioni avanzate sui dati."""
        try:
            collection = self.client.collections.get(class_name)
            
            # Usa Gemini per identificare il tipo specifico di estrazione di conoscenza richiesta
            extraction_type = self._identify_knowledge_extraction_type_with_gemini(question, class_name)
            
            # Esegui l'operazione di estrazione appropriata
            if extraction_type == "pattern_discovery":
                return self._extract_patterns(collection, class_name, question)
            elif extraction_type == "topic_modeling":
                return self._extract_topics(collection, class_name, question)
            elif extraction_type == "entity_extraction":
                return self._extract_entities(collection, class_name, question)
            elif extraction_type == "sentiment_analysis":
                return self._analyze_sentiment(collection, class_name, question)
            else:
                # Estrazione generica con approccio ibrido
                return self._general_knowledge_extraction(collection, class_name, question)
                
        except Exception as e:
            return f"🧠 **Errore nell'estrazione di conoscenza**\n\nSi è verificato un errore: {str(e)}"

    def _identify_knowledge_extraction_type_with_gemini(self, question: str, class_name: str) -> str:
        """Usa Gemini per identificare il tipo specifico di estrazione di conoscenza richiesta."""
        try:
            # Ottieni informazioni sulla collezione
            collection = self.client.collections.get(class_name)
            props_info = self._get_collection_properties(collection)
            
            # Prendi un campione per capire il tipo di dati
            sample_response = collection.query.fetch_objects(limit=3)
            sample_data = ""
            
            if sample_response.objects:
                sample_properties = []
                for obj in sample_response.objects[:2]:
                    obj_props = []
                    for key, value in obj.properties.items():
                        value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        obj_props.append(f"{key}: {value_str}")
                    sample_properties.append(" | ".join(obj_props))
                
                sample_data = "\n".join([f"Esempio {i+1}: {props}" for i, props in enumerate(sample_properties)])
            
            prompt = f"""
            Analizza questa richiesta di estrazione di conoscenza e identifica il tipo specifico di operazione.
            
            RICHIESTA: "{question}"
            
            COLLEZIONE: {class_name}
            PROPRIETÀ DISPONIBILI: {props_info.get('all', [])}
            PROPRIETÀ TESTUALI: {props_info.get('text', [])}
            PROPRIETÀ NUMERICHE: {props_info.get('number', [])}
            
            DATI DI ESEMPIO:
            {sample_data}
            
            TIPI DI ESTRAZIONE DISPONIBILI:
            1. "pattern_discovery" - Scoprire pattern nascosti, regularità, comportamenti ricorrenti
            2. "topic_modeling" - Identificare temi, argomenti, categorie concettuali
            3. "entity_extraction" - Estrarre entità (persone, luoghi, organizzazioni, brand)
            4. "sentiment_analysis" - Analizzare sentiment, emozioni, opinioni
            5. "correlation_analysis" - Trovare correlazioni tra variabili, dipendenze
            6. "trend_analysis" - Analizzare tendenze temporali, evoluzione
            7. "clustering" - Raggruppare elementi simili, segmentazione
            8. "summarization" - Riassumere, sintetizzare, estrarre punti chiave
            9. "keyword_extraction" - Estrarre parole chiave, termini importanti
            10. "relationship_mapping" - Mappare relazioni, connessioni, network
            11. "general" - Estrazione generica o combinazione di tecniche
            
            ISTRUZIONI:
            - Considera sia la richiesta che la natura dei dati disponibili
            - Per dati testuali: preferisci topic_modeling, sentiment, entity_extraction
            - Per dati numerici: preferisci correlation_analysis, trend_analysis, clustering
            - Per richieste di sintesi: usa summarization
            - Per richieste di connessioni: usa relationship_mapping
            
            Rispondi con UNA SOLA parola chiave tra quelle elencate sopra.
            """
            
            response = self.model.generate_content(prompt)
            classification = response.text.strip().lower()
            
            # Mappa le risposte ai tipi validi
            valid_types = [
                "pattern_discovery", "topic_modeling", "entity_extraction", 
                "sentiment_analysis", "correlation_analysis", "trend_analysis",
                "clustering", "summarization", "keyword_extraction", 
                "relationship_mapping", "general"
            ]
            
            # Cerca la classificazione nella risposta
            for valid_type in valid_types:
                if valid_type in classification:
                    print(f"🤖 Gemini ha classificato l'estrazione come: {valid_type}")
                    return valid_type
            
            # Fallback con match parziale
            if any(word in classification for word in ["pattern", "regularità", "comportament"]):
                return "pattern_discovery"
            elif any(word in classification for word in ["topic", "temi", "argomenti"]):
                return "topic_modeling"
            elif any(word in classification for word in ["entità", "entity", "persone", "luoghi"]):
                return "entity_extraction"
            elif any(word in classification for word in ["sentiment", "emotion", "opinioni"]):
                return "sentiment_analysis"
            elif any(word in classification for word in ["correlazione", "dipendenze"]):
                return "correlation_analysis"
            elif any(word in classification for word in ["trend", "tendenze", "evoluzione"]):
                return "trend_analysis"
            elif any(word in classification for word in ["cluster", "raggrup", "segment"]):
                return "clustering"
            elif any(word in classification for word in ["riassun", "sintesi", "summary"]):
                return "summarization"
            elif any(word in classification for word in ["keyword", "parole chiave", "termini"]):
                return "keyword_extraction"
            elif any(word in classification for word in ["relazioni", "connessioni", "network"]):
                return "relationship_mapping"
            else:
                return "general"
                
        except Exception as e:
            print(f"Errore nella classificazione dell'estrazione: {e}")
            return "general"

    def _extract_patterns(self, collection, class_name: str, question: str) -> str:
        """Estrae pattern ricorrenti dai dati."""
        try:
            # Ottieni un campione significativo di dati
            response = collection.query.fetch_objects(limit=100)
            
            if not response.objects:
                return "🧠 **Pattern Discovery**\n\nNon ci sono dati sufficienti per l'analisi dei pattern."
            
            # Analizza le proprietà per identificare pattern
            props_info = self._get_collection_properties(collection)
            text_props = props_info.get('text', [])
            number_props = props_info.get('number', [])
            
            patterns_found = []
            
            # Pattern sui dati testuali
            if text_props:
                text_patterns = self._analyze_text_patterns(response.objects, text_props[:2])
                patterns_found.extend(text_patterns)
            
            # Pattern sui dati numerici
            if number_props:
                numeric_patterns = self._analyze_numeric_patterns(response.objects, number_props[:2])
                patterns_found.extend(numeric_patterns)
            
            # Usa Gemini per interpretare i pattern trovati
            interpretation = self._interpret_patterns_with_gemini(patterns_found, question, class_name)
            
            return f"🧠 **Pattern Discovery**\n\n{interpretation}"
            
        except Exception as e:
            return f"🧠 **Errore Pattern Discovery**\n\nErrore: {str(e)}"

    def _extract_topics(self, collection, class_name: str, question: str) -> str:
        """Estrae topic e temi principali dai contenuti testuali."""
        try:
            props_info = self._get_collection_properties(collection)
            text_props = props_info.get('text', [])
            
            if not text_props:
                return "🧠 **Topic Modeling**\n\nNon ci sono proprietà testuali sufficienti per l'analisi dei topic."
            
            # Prendi un campione di testi
            response = collection.query.fetch_objects(
                limit=50,
                return_properties=text_props[:2]  # Prendi le prime 2 proprietà testuali
            )
            
            if not response.objects:
                return "🧠 **Topic Modeling**\n\nNon ci sono dati sufficienti per l'analisi dei topic."
            
            # Estrai i contenuti testuali
            texts = []
            for obj in response.objects:
                obj_text = []
                for prop in text_props[:2]:
                    if prop in obj.properties and obj.properties[prop]:
                        text_content = str(obj.properties[prop])
                        if len(text_content) > 50:  # Solo testi significativi
                            obj_text.append(text_content)
                
                if obj_text:
                    texts.append(" ".join(obj_text))
            
            if len(texts) < 5:
                return "🧠 **Topic Modeling**\n\nNon ci sono abbastanza testi per un'analisi significativa dei topic."
            
            # Usa Gemini per analizzare i topic
            topics = self._analyze_topics_with_gemini(texts, question, class_name)
            
            return f"🧠 **Topic Modeling**\n\n{topics}"
            
        except Exception as e:
            return f"🧠 **Errore Topic Modeling**\n\nErrore: {str(e)}"

    def _extract_entities(self, collection, class_name: str, question: str) -> str:
        """Estrae entità nominate dai testi."""
        try:
            props_info = self._get_collection_properties(collection)
            text_props = props_info.get('text', [])
            
            if not text_props:
                return "🧠 **Entity Extraction**\n\nNon ci sono proprietà testuali per l'estrazione di entità."
            
            # Prendi un campione di testi
            response = collection.query.fetch_objects(
                limit=30,
                return_properties=text_props[:2]
            )
            
            if not response.objects:
                return "🧠 **Entity Extraction**\n\nNon ci sono dati sufficienti per l'estrazione di entità."
            
            # Combina i testi per l'analisi
            combined_text = []
            for obj in response.objects:
                for prop in text_props[:2]:
                    if prop in obj.properties and obj.properties[prop]:
                        text_content = str(obj.properties[prop])
                        if len(text_content) > 20:
                            combined_text.append(text_content)
            
            if not combined_text:
                return "🧠 **Entity Extraction**\n\nNon ci sono testi sufficienti per l'estrazione di entità."
            
            # Usa Gemini per estrarre entità
            entities = self._extract_entities_with_gemini(combined_text[:10], question, class_name)
            
            return f"🧠 **Entity Extraction**\n\n{entities}"
            
        except Exception as e:
            return f"🧠 **Errore Entity Extraction**\n\nErrore: {str(e)}"

    def _analyze_sentiment(self, collection, class_name: str, question: str) -> str:
        """Analizza il sentiment dei contenuti testuali."""
        try:
            props_info = self._get_collection_properties(collection)
            text_props = props_info.get('text', [])
            
            if not text_props:
                return "🧠 **Sentiment Analysis**\n\nNon ci sono proprietà testuali per l'analisi del sentiment."
            
            # Prendi un campione di testi
            response = collection.query.fetch_objects(
                limit=40,
                return_properties=text_props[:2]
            )
            
            if not response.objects:
                return "🧠 **Sentiment Analysis**\n\nNon ci sono dati sufficienti per l'analisi del sentiment."
            
            # Prepara i testi per l'analisi
            texts_for_analysis = []
            for obj in response.objects:
                obj_texts = []
                for prop in text_props[:2]:
                    if prop in obj.properties and obj.properties[prop]:
                        text_content = str(obj.properties[prop])
                        if len(text_content) > 30:
                            obj_texts.append(text_content[:300])  # Limita la lunghezza
                
                if obj_texts:
                    texts_for_analysis.append(" | ".join(obj_texts))
            
            if len(texts_for_analysis) < 3:
                return "🧠 **Sentiment Analysis**\n\nNon ci sono abbastanza testi per un'analisi significativa del sentiment."
            
            # Usa Gemini per analizzare il sentiment
            sentiment_analysis = self._analyze_sentiment_with_gemini(texts_for_analysis[:15], question, class_name)
            
            return f"🧠 **Sentiment Analysis**\n\n{sentiment_analysis}"
            
        except Exception as e:
            return f"🧠 **Errore Sentiment Analysis**\n\nErrore: {str(e)}"

    def _general_knowledge_extraction(self, collection, class_name: str, question: str) -> str:
        """Estrazione di conoscenza generica con approccio ibrido."""
        try:
            # Ottieni informazioni complete sulla collezione
            props_info = self._get_collection_properties(collection)
            
            # Prendi un campione rappresentativo
            response = collection.query.fetch_objects(limit=30)
            
            if not response.objects:
                return "🧠 **Knowledge Extraction**\n\nNon ci sono dati sufficienti per l'estrazione di conoscenza."
            
            # Prepara i dati per l'analisi
            sample_data = []
            for obj in response.objects[:10]:  # Limita per non sovraccaricare
                obj_summary = []
                for key, value in obj.properties.items():
                    if value is not None:
                        value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        obj_summary.append(f"{key}: {value_str}")
                
                if obj_summary:
                    sample_data.append(" | ".join(obj_summary))
            
            # Usa Gemini per un'analisi completa e personalizzata
            analysis = self._perform_comprehensive_analysis_with_gemini(
                sample_data, props_info, question, class_name
            )
            
            return f"🧠 **Knowledge Extraction**\n\n{analysis}"
            
        except Exception as e:
            return f"🧠 **Errore Knowledge Extraction**\n\nErrore: {str(e)}"

    # Metodi di supporto per l'analisi con Gemini

    def _analyze_topics_with_gemini(self, texts: list, question: str, class_name: str) -> str:
        """Usa Gemini per analizzare i topic nei testi."""
        try:
            # Prepara un campione dei testi per Gemini
            sample_texts = texts[:10]  # Limita per non sovraccaricare
            combined_sample = "\n---\n".join([f"Testo {i+1}: {text[:200]}..." 
                                            for i, text in enumerate(sample_texts)])
            
            prompt = f"""
            Analizza questi testi dalla collezione "{class_name}" e identifica i topic/temi principali.
            
            RICHIESTA ORIGINALE: "{question}"
            
            TESTI DA ANALIZZARE:
            {combined_sample}
            
            COMPITO:
            1. Identifica i 5-8 topic/temi principali presenti nei testi
            2. Per ogni topic, fornisci:
               - Nome del topic
               - Breve descrizione
               - Frequenza stimata (alta/media/bassa)
               - Parole chiave associate
            3. Identifica eventuali topic emergenti o minoritari interessanti
            4. Suggerisci insights o pattern tematici significativi
            
            FORMATO RISPOSTA:
            **Topic Principali:**
            • **[Nome Topic]** (Frequenza: X) - Descrizione
              Keywords: parola1, parola2, parola3
            
            **Insights Tematici:**
            - Insight 1
            - Insight 2
            
            Mantieni la risposta chiara, strutturata e focalizzata sui risultati più significativi.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Errore nell'analisi dei topic: {str(e)}"

    def _extract_entities_with_gemini(self, texts: list, question: str, class_name: str) -> str:
        """Usa Gemini per estrarre entità nominate."""
        try:
            sample_texts = texts[:8]
            combined_sample = "\n---\n".join([f"Testo {i+1}: {text[:250]}..." 
                                            for i, text in enumerate(sample_texts)])
            
            prompt = f"""
            Estrai entità nominate da questi testi della collezione "{class_name}".
            
            RICHIESTA ORIGINALE: "{question}"
            
            TESTI DA ANALIZZARE:
            {combined_sample}
            
            COMPITO:
            Identifica e categorizza le seguenti entità:
            1. **PERSONE** - Nomi di persone, autori, personaggi
            2. **LUOGHI** - Città, paesi, regioni, location specifiche
            3. **ORGANIZZAZIONI** - Aziende, istituzioni, gruppi
            4. **BRAND/PRODOTTI** - Marchi, prodotti, servizi
            5. **DATE/EVENTI** - Date, eventi storici, occasioni
            6. **ALTRI** - Altre entità rilevanti per il dominio
            
            FORMATO RISPOSTA:
            **👥 PERSONE:**
            • Nome1, Nome2, Nome3...
            
            **📍 LUOGHI:**
            • Luogo1, Luogo2, Luogo3...
            
            **🏢 ORGANIZZAZIONI:**
            • Org1, Org2, Org3...
            
            **Insights:**
            - Osservazione principale sulle entità trovate
            - Pattern geografici/temporali notevoli
            
            Se una categoria è vuota, non includerla. Concentrati sulle entità più frequenti e significative.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Errore nell'estrazione di entità: {str(e)}"

    def _analyze_sentiment_with_gemini(self, texts: list, question: str, class_name: str) -> str:
        """Usa Gemini per analizzare il sentiment."""
        try:
            sample_texts = texts[:12]
            combined_sample = "\n---\n".join([f"Testo {i+1}: {text[:200]}..." 
                                            for i, text in enumerate(sample_texts)])
            
            prompt = f"""
            Analizza il sentiment e il tono emotivo di questi testi dalla collezione "{class_name}".
            
            RICHIESTA ORIGINALE: "{question}"
            
            TESTI DA ANALIZZARE:
            {combined_sample}
            
            COMPITO:
            1. **Sentiment Generale**: Determina il sentiment predominante (Positivo/Neutro/Negativo) con percentuali
            2. **Emozioni Specifiche**: Identifica emozioni presenti (gioia, tristezza, rabbia, paura, sorpresa, etc.)
            3. **Tono**: Descrivi il tono complessivo (formale/informale, tecnico/colloquiale, etc.)
            4. **Variazioni**: Evidenzia eventuali variazioni di sentiment tra i testi
            5. **Insights**: Osservazioni significative sul sentiment
            
            FORMATO RISPOSTA:
            **📊 Sentiment Generale:**
            • Positivo: X% | Neutro: Y% | Negativo: Z%
            
            **😊 Emozioni Rilevate:**
            • Emozione principale: descrizione
            • Emozione secondaria: descrizione
            
            **🎭 Tono e Stile:**
            • Caratteristiche del tono identificate
            
            **💡 Insights:**
            • Osservazione chiave 1
            • Osservazione chiave 2
            
            Sii specifico e fornisci esempi quando possibile.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Errore nell'analisi del sentiment: {str(e)}"

    def _perform_comprehensive_analysis_with_gemini(self, sample_data: list, props_info: dict, question: str, class_name: str) -> str:
        """Esegue un'analisi completa e personalizzata con Gemini."""
        try:
            # Prepara il campione dati
            data_sample = "\n".join([f"Record {i+1}: {data}" for i, data in enumerate(sample_data[:8])])
            
            prompt = f"""
            Esegui un'analisi completa di estrazione di conoscenza per questa richiesta specifica.
            
            RICHIESTA: "{question}"
            COLLEZIONE: "{class_name}"
            
            STRUTTURA DATI:
            - Proprietà totali: {props_info.get('all', [])}
            - Proprietà testuali: {props_info.get('text', [])}
            - Proprietà numeriche: {props_info.get('number', [])}
            
            CAMPIONE DATI:
            {data_sample}
            
            COMPITO:
            Basandoti sulla richiesta specifica e sui dati disponibili, esegui l'analisi più appropriata tra:
            
            1. **Pattern Analysis** - Se cerchi regolarità, comportamenti ricorrenti
            2. **Content Analysis** - Se vuoi analizzare contenuti, temi, significati
            3. **Statistical Insights** - Se cerchi correlazioni, trend numerici
            4. **Structural Analysis** - Se vuoi capire relazioni, connessioni
            5. **Comparative Analysis** - Se vuoi confronti, classificazioni
            
            FORMATO RISPOSTA:
            **🎯 Tipo di Analisi Eseguita:** [Nome Analisi]
            
            **📋 Risultati Principali:**
            • Risultato 1 con dettagli
            • Risultato 2 con dettagli
            • Risultato 3 con dettagli
            
            **💡 Insights Chiave:**
            • Insight significativo 1
            • Insight significativo 2
            
            **🔍 Raccomandazioni:**
            • Cosa approfondire ulteriormente
            • Domande aggiuntive da esplorare
            
            Concentrati sui risultati più rilevanti per la richiesta specifica e fornisci insights azionabili.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Errore nell'analisi completa: {str(e)}"


    def smart_answer(self, question: str, collection_name: str = None) -> Dict[str, Any]:
        """
        Metodo principale per rispondere alle domande con classificazione intelligente
        e gestione ottimizzata delle performance con GeminiValidator.
        """
        start_time = time.time()
        
        try:
            # Classifica la domanda
            question_type = self.classify_question(question)
            self._track_gemini_call(len(question))
            
            # Per domande che richiedono dati, verifica che ci sia una collezione
            if question_type in ["analitica", "pulizia", "generale", "integrazione", "estrazione_conoscenza"] and not collection_name:
                return {
                    'answer': "❌ **Collezione richiesta**\n\nPer questo tipo di domanda devi selezionare una collezione che contenga i dati da analizzare.",
                    'type': "error",
                    'question': question,
                    'collection': None,
                    'response_time': round(time.time() - start_time, 2),
                    'error': "Collezione non specificata"
                }
                
            # Verifica che la collezione esista (se specificata)
            if collection_name:
                try:
                    if  not self.client.collections.exists(collection_name):
                        return {
                            'answer': f"❌ **Collezione non trovata**\n\nLa collezione '{collection_name}' non esiste. Collezioni disponibili: {', '.join(collection_names)}",
                            'type': "error", 
                            'question': question,
                            'collection': collection_name,
                            'response_time': round(time.time() - start_time, 2),
                            'error': f"Collezione '{collection_name}' non trovata"
                        }
                except Exception as e:
                    print(f"Errore controllo collezione: {e}")
            
            # Gestisci in base al tipo
            if question_type == "analitica":
                answer = self.handle_analytical_question(question, collection_name)
                response_type = "analytical"
            elif question_type == "conversazionale":
                answer = self.handle_conversational_question(question, collection_name)
                response_type = "conversational"
            elif question_type == "pulizia":
                answer = self.handle_cleaning_question(question, collection_name)
                response_type = "cleaning"
            elif question_type == "integrazione":
                # Per ora usiamo la gestione generale per le integrazioni
                answer = self.handle_general_question(question, collection_name)
                response_type = "integration"
            elif question_type == "estrazione_conoscenza":
                answer = self.handle_knowledge_extraction_question(question, collection_name)
                response_type = "knowledge_extraction"
            else:  # general/semantic
                answer = self.handle_general_question(question, collection_name)
                response_type = "semantic"
            
            # Calcola tempo di risposta
            response_time = time.time() - start_time
            
            # Formato di risposta standard
            return {
                'answer': answer,
                'type': response_type,
                'question': question,
                'collection': collection_name,
                'response_time': round(response_time, 2),
                'model_info': self.get_current_model_info(),
                'usage_stats': self.get_usage_stats()
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            return {
                'answer': f"❌ **Errore nel processamento**\n\nSi è verificato un errore: {str(e)}",
                'type': "error",
                'question': question,
                'collection': collection_name,
                'response_time': round(error_time, 2),
                'error': str(e)
            }

    def prepare_response_for_download(self, response_data: dict, format_type: str = 'txt') -> dict:
        """Prepara la risposta per il download in vari formati."""
        try:
            import os
            import json
            from datetime import datetime
            
            # Crea la cartella exports se non esiste
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Genera nome file unico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            question_short = response_data.get('question', 'query')[:30].replace(' ', '_').replace('?', '').replace(':', '')
            collection_name = response_data.get('collection', 'unknown')
            response_type = response_data.get('type', 'general')
            
            base_filename = f"{response_type}_{collection_name}_{question_short}_{timestamp}"
            
            if format_type == 'txt':
                return self._export_as_text(response_data, export_dir, base_filename)
            elif format_type == 'json':
                return self._export_as_json(response_data, export_dir, base_filename)
            elif format_type == 'csv':
                return self._export_as_csv(response_data, export_dir, base_filename)
            elif format_type == 'md':
                return self._export_as_markdown(response_data, export_dir, base_filename)
            else:
                return {"error": f"Formato {format_type} non supportato"}
                
        except Exception as e:
            return {"error": f"Errore nella preparazione del download: {str(e)}"}

    def _export_as_text(self, response_data: dict, export_dir: str, base_filename: str) -> dict:
        """Esporta la risposta come file di testo formattato."""
        try:
            from datetime import datetime
            
            filepath = os.path.join(export_dir, f"{base_filename}.txt")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("NEURALABB - REPORT ANALISI\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"📊 TIPO ANALISI: {response_data.get('type', 'N/A').upper()}\n")
                f.write(f"📅 DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"🗂️  COLLEZIONE: {response_data.get('collection', 'N/A')}\n")
                f.write(f"⏱️  TEMPO ELABORAZIONE: {response_data.get('response_time', 'N/A')} secondi\n\n")
                
                f.write("❓ DOMANDA:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{response_data.get('question', 'N/A')}\n\n")
                
                f.write("💡 RISPOSTA:\n")
                f.write("-" * 40 + "\n")
                answer = response_data.get('answer', 'N/A')
                # Rimuovi markdown per il file di testo
                answer = self._clean_markdown_for_text(answer)
                f.write(f"{answer}\n\n")
                
                # Aggiungi informazioni tecniche se disponibili
                if 'model_info' in response_data:
                    f.write("🤖 INFORMAZIONI MODELLO:\n")
                    f.write("-" * 40 + "\n")
                    model_info = response_data['model_info']
                    f.write(f"Modello: {model_info.get('model_name', 'N/A')}\n")
                    f.write(f"Versione 2.0: {model_info.get('is_gemini_2_0', False)}\n\n")
                
                if 'usage_stats' in response_data:
                    f.write("📈 STATISTICHE UTILIZZO:\n")
                    f.write("-" * 40 + "\n")
                    stats = response_data['usage_stats']
                    f.write(f"Chiamate totali: {stats.get('total_calls', 'N/A')}\n")
                    f.write(f"Token risparmiati: {stats.get('estimated_tokens_saved', 'N/A')}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("Fine del report\n")
                f.write("=" * 80 + "\n")
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": f"{base_filename}.txt",
                "format": "text"
            }
            
        except Exception as e:
            return {"error": f"Errore nell'export testo: {str(e)}"}

    def _export_as_json(self, response_data: dict, export_dir: str, base_filename: str) -> dict:
        """Esporta la risposta come file JSON."""
        try:
            import json
            from datetime import datetime
            
            filepath = os.path.join(export_dir, f"{base_filename}.json")
            
            # Prepara i dati JSON con metadata aggiuntivi
            json_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "tool": "NeuralTabb",
                    "version": "1.0"
                },
                "analysis": response_data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": f"{base_filename}.json",
                "format": "json"
            }
            
        except Exception as e:
            return {"error": f"Errore nell'export JSON: {str(e)}"}

    def _export_as_markdown(self, response_data: dict, export_dir: str, base_filename: str) -> dict:
        """Esporta la risposta come file Markdown."""
        try:
            from datetime import datetime
            
            filepath = os.path.join(export_dir, f"{base_filename}.md")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# 🧠 NeuralTabb - Report Analisi\n\n")
                
                f.write("## 📋 Informazioni Generali\n\n")
                f.write(f"- **Tipo Analisi**: {response_data.get('type', 'N/A')}\n")
                f.write(f"- **Data**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Collezione**: `{response_data.get('collection', 'N/A')}`\n")
                f.write(f"- **Tempo Elaborazione**: {response_data.get('response_time', 'N/A')} secondi\n\n")
                
                f.write("## ❓ Domanda\n\n")
                f.write(f"```\n{response_data.get('question', 'N/A')}\n```\n\n")
                
                f.write("## 💡 Risposta\n\n")
                answer = response_data.get('answer', 'N/A')
                f.write(f"{answer}\n\n")
                
                # Sezione tecnica
                f.write("## 🔧 Dettagli Tecnici\n\n")
                
                if 'model_info' in response_data:
                    f.write("### 🤖 Modello AI\n\n")
                    model_info = response_data['model_info']
                    f.write(f"- **Modello**: {model_info.get('model_name', 'N/A')}\n")
                    f.write(f"- **Gemini 2.0**: {model_info.get('is_gemini_2_0', False)}\n\n")
                
                if 'usage_stats' in response_data:
                    f.write("### 📈 Statistiche\n\n")
                    stats = response_data['usage_stats']
                    f.write(f"- **Chiamate Totali**: {stats.get('total_calls', 'N/A')}\n")
                    f.write(f"- **Token Risparmiati**: {stats.get('estimated_tokens_saved', 'N/A')}\n\n")
                
                f.write("---\n")
                f.write("*Report generato da NeuralTabb*\n")
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": f"{base_filename}.md",
                "format": "markdown"
            }
            
        except Exception as e:
            return {"error": f"Errore nell'export Markdown: {str(e)}"}

    def _export_as_csv(self, response_data: dict, export_dir: str, base_filename: str) -> dict:
        """Esporta la risposta come file CSV (per dati strutturati)."""
        try:
            import csv
            from datetime import datetime
            
            filepath = os.path.join(export_dir, f"{base_filename}.csv")
            
            # Per CSV, esportiamo i metadati principali
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['Campo', 'Valore'])
                
                # Dati principali
                writer.writerow(['Data Export', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['Tipo Analisi', response_data.get('type', 'N/A')])
                writer.writerow(['Collezione', response_data.get('collection', 'N/A')])
                writer.writerow(['Tempo Elaborazione (sec)', response_data.get('response_time', 'N/A')])
                writer.writerow(['Domanda', response_data.get('question', 'N/A')])
                
                # Pulisci la risposta per CSV
                answer = response_data.get('answer', 'N/A')
                answer_clean = self._clean_markdown_for_text(answer).replace('\n', ' | ')
                writer.writerow(['Risposta', answer_clean])
                
                # Informazioni modello
                if 'model_info' in response_data:
                    model_info = response_data['model_info']
                    writer.writerow(['Modello AI', model_info.get('model_name', 'N/A')])
                    writer.writerow(['Gemini 2.0', model_info.get('is_gemini_2_0', False)])
                
                # Statistiche
                if 'usage_stats' in response_data:
                    stats = response_data['usage_stats']
                    writer.writerow(['Chiamate Totali', stats.get('total_calls', 'N/A')])
                    writer.writerow(['Token Risparmiati', stats.get('estimated_tokens_saved', 'N/A')])
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": f"{base_filename}.csv",
                "format": "csv"
            }
            
        except Exception as e:
            return {"error": f"Errore nell'export CSV: {str(e)}"}

    def _clean_markdown_for_text(self, text: str) -> str:
        """Rimuove la formattazione Markdown per export di testo pulito."""
        import re
        
        # Rimuovi headers markdown
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # Rimuovi bold/italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Rimuovi emoji markdown-style
        text = re.sub(r':\w+:', '', text)
        
        # Rimuovi blocchi di codice
        text = re.sub(r'```.*?```', '[CODICE]', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Rimuovi link
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        return text.strip()
