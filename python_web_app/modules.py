
# modules/weWaviate_manager.py
import pathlib
import weaviate
import json
import pandas as pd
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



class ChatHistory:
    """Gestisce la cronologia di una conversazione per mantenere il contesto."""
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Aggiunge un messaggio (utente o modello) alla cronologia."""
        if len(self.history) >= self.max_history:
            self.history.pop(0)  # Rimuovi il messaggio più vecchio
        self.history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Restituisce la cronologia formattata per l'API di Gemini."""
        return self.history

class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.chat_history = ChatHistory()
        self._configure_gemini()

    def _configure_gemini(self):
        """Configura l'API di Gemini."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Modello Gemini '{self.model_name}' configurato con successo.")
        except Exception as e:
            print(f"Errore durante la configurazione di Gemini: {e}")
            raise

    def _generate_content(self, prompt: str) -> str:
        """Genera contenuti utilizzando il modello Gemini, gestendo la cronologia."""
        try:
            # Aggiungi il prompt dell'utente alla cronologia
            self.chat_history.add_message("user", prompt)
            
            # Crea una sessione di chat con la cronologia corrente
            chat_session = self.model.start_chat(history=self.chat_history.get_history())
            
            # Invia il messaggio (il prompt è già l'ultimo messaggio nella cronologia)
            response = chat_session.send_message(prompt)
            
            # Aggiungi la risposta del modello alla cronologia
            self.chat_history.add_message("model", response.text)
            
            return response.text.strip()
        except Exception as e:
            print(f"Errore durante la generazione di contenuti: {e}")
            return f"Si è verificato un errore: {e}"

    def classify_question(self, question: str) -> str:
        """Classifica la domanda dell'utente in una delle categorie predefinite."""
        prompt = f"""
        Classifica la seguente domanda in una delle seguenti categorie:
        - 'conversational': Saluti, domande generali su chi sei, come stai, ecc.
        - 'analytical': Domande che richiedono calcoli, conteggi, medie, filtri, raggruppamenti.
        - 'general_knowledge': Domande che richiedono di riassumere o estrarre informazioni dai dati.
        - 'data_cleaning': Domande relative alla pulizia dei dati (duplicati, valori mancanti, ecc.).
        - 'data_integration': Domande su come unire, fondere o integrare set di dati.

        Domanda: "{question}"
        Categoria:"
        """
        # Per la classificazione, non usiamo la cronologia per evitare bias
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().lower()
        except Exception as e:
            print(f"Errore durante la classificazione: {e}")
            return "conversational" # Fallback sicuro

    def handle_conversational_question(self, question: str) -> str:
        """Gestisce domande conversazionali in modo amichevole."""
        prompt = f"""
        Sei NeuralTabb, un assistente AI amichevole e professionale per l'analisi dei dati.
        Rispondi alla seguente domanda in modo conversazionale.

        Domanda: "{question}"
        Risposta:"
        """
        return self._generate_content(prompt)

    def handle_analytical_question(self, question: str, collection_name: str, client) -> str:
        """Gestisce domande analitiche generando ed eseguendo query Weaviate."""
        try:
            collection = client.collections.get(collection_name)
            
            # Usa Gemini per generare la query Weaviate
            prompt = f"""
            Genera una query Weaviate Python per rispondere alla seguente domanda analitica.
            La collezione si chiama \'{collection_name}\'.
            Le proprietà disponibili sono quelle della collezione.
            
            Domanda: \"{question}\"\n\nQuery Weaviate (solo codice Python, senza spiegazioni):\n"""
            
            query_code = self._generate_content(prompt)
            
            # Esegui la query generata
            # ATTENZIONE: Eseguire codice generato da LLM può essere rischioso. 
            # In un ambiente di produzione, questa parte dovrebbe essere gestita con molta cautela
            # e idealmente con una validazione o un parsing più robusto.
            
            # Per questo esempio, assumiamo che la query generata sia sicura e valida.
            # Si potrebbe usare eval() o exec(), ma è estremamente sconsigliato per sicurezza.
            # Invece, si dovrebbe parsare la query e costruire l'oggetto query di Weaviate.
            
            # Placeholder per l'esecuzione della query
            return f"Ho generato una query per la tua domanda analitica: \n```python\n{query_code}\n```\nL'esecuzione della query è in fase di sviluppo. "
            
        except Exception as e:
            return f"Errore durante l'analisi analitica: {e}"

    def handle_general_knowledge_question(self, question: str, collection_name: str, client) -> str:
        """Gestisce domande di conoscenza generale tramite RAG."""
        try:
            collection = client.collections.get(collection_name)
            
            # Cerca documenti pertinenti
            result = collection.query.near_text(
                query=question,
                limit=3,
                return_properties=["title", "content"]
            )
            
            if not result.objects:
                return "Non ho trovato informazioni pertinenti per la tua domanda nei dati."

            contexts = []
            for doc in result.objects:
                content = doc.properties.get('content', '')
                contexts.append(content)
            
            context = "\n".join(contexts)
            
            prompt = f"""Basato su questi dati, rispondi: "{question}"\n\nDati:\n{context}\n\nRisposta:"""
            
            return self._generate_content(prompt)
            
        except Exception as e:
            return f"Errore nell'analisi: {e}"

    def handle_data_cleaning_question(self, question: str, collection_name: str, client) -> str:
        """Gestisce domande sulla pulizia dei dati."""
        if not collection_name:
            return "Per eseguire operazioni di pulizia, devi prima selezionare una collezione di dati da pulire."
        
        try:
            collection = client.collections.get(collection_name)
            question_lower = question.lower()
            
            if any(word in question_lower for word in ["duplicat", "duplicate", "doppi", "ripetut"]):
                return self._handle_duplicate_removal(collection, collection_name, question)
            
            elif any(word in question_lower for word in ["vuot", "null", "empty", "mancant", "missing", "nan"]):
                return self._handle_empty_values(collection, collection_name, question)
            
            elif any(word in question_lower for word in ["normaliz", "standard", "maiuscol", "minuscol", "uppercase", "lowercase", "format"]):
                return self._handle_text_normalization(collection, collection_name, question)
            
            elif any(word in question_lower for word in ["spazi", "spaces", "trim", "whitespace", "pulisci spazi"]):
                return self._handle_whitespace_cleaning(collection, collection_name, question)
            
            else:
                return self._handle_custom_cleaning_with_gemini(question, collection, collection_name)
                
        except Exception as e:
            return f"Errore durante la pulizia: {str(e)}"

    def handle_data_integration_question(self, question: str, collection_names: List[str], client) -> str:
        """Gestisce domande sull'integrazione dei dati."""
        try:
            if not collection_names or len(collection_names) < 2:
                return "Per l\'integrazione dei dati, sono necessarie almeno due collezioni."

            # Usa Gemini per analizzare la richiesta di integrazione
            prompt = f"""
            Sei un esperto nell\'integrazione di dati in Weaviate.
            
            RICHIESTA UTENTE: "{question}"
            
            COLLEZIONI DISPONIBILI: {collection_names}
            
            Analizza la richiesta di integrazione e fornisci:
            1. Tipo di operazione di integrazione (es. merge, join, append)
            2. Collezioni coinvolte
            3. Campi chiave per l\'integrazione (se applicabile)
            4. Potenziali problemi o considerazioni
            5. Passi specifici per l\'implementazione in Weaviate (anche se concettuali)
            
            IMPORTANTE: Sii specifico e tecnico nella tua risposta.
            """
            
            response = self._generate_content(prompt)
            
            return f"📊 **Analisi Integrazione Dati**\n\n{response.text.strip()}\n\n💡 **Nota:** L\'implementazione automatica di operazioni di integrazione complesse è in fase di sviluppo."
            
        except Exception as e:
            return f"Errore durante l\'integrazione dei dati: {e}"



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
                return f"🧹 **Rimozione Duplicati Completata**\n\nNessun duplicato trovato nella collezione \'{class_name}\' con soglia di similarità {threshold:.0%}.\n\n✅ La collezione è già pulita!"
            
            # Conta i duplicati
            total_duplicates = sum(len(group) - 1 for group in duplicates_found)  # -1 perché teniamo l\'originale
            
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
                result += f"\n💡 **Suggerimento:** Per rimuovere automaticamente i duplicati, chiedi: \'Rimuovi i duplicati dalla collezione {class_name}\'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Rimozione Duplicati**\n\nErrore durante l\'analisi duplicati: {str(e)}"

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
                return f"🧹 **Pulizia Valori Vuoti Completata**\n\n✅ Nessun valore vuoto trovato nella collezione \'{class_name}\'!"
            
            result = f"🧹 **Analisi Valori Vuoti**\n\n"
            result += f"📊 **Statistiche:**\n"
            result += f"• Documenti analizzati: {total_docs}\n"
            result += f"• Proprietà con valori vuoti: {len(property_stats)}\n\n"
            
            result += f"🔍 **Dettaglio per proprietà:**\n"
            
            for prop_name, stats in sorted(property_stats.items(), key=lambda x: x[1]["empty_percentage"], reverse=True):
                result += f"\n**{prop_name}:**\n"
                result += f"  • Valori vuoti: {stats['empty_count']} ({stats['empty_percentage']:.1f}%)\n"
                result += f"  • Valori presenti: {stats['non_empty_count']}\n"
                if stats["sample_values"]:
                    sample_str = ", ".join([str(v)[:50] for v in stats["sample_values"]])
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
                result += f"• Documenti rimossi: {removed_count}\n"
                result += f"• Documenti rimanenti: {total_docs - removed_count}"
            else:
                result += f"\n💡 **Suggerimento:** Per rimuovere automaticamente i documenti con molti valori vuoti, chiedi: \'Rimuovi i documenti con valori vuoti dalla collezione {class_name}\'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Valori Vuoti**\n\nErrore durante la pulizia: {str(e)}"

    def _handle_text_normalization(self, collection, class_name: str, question: str) -> str:
        """Gestisce la normalizzazione del testo (minuscolo, maiuscolo, punteggiatura)."""
        try:
            response = collection.query.fetch_objects(limit=500)
            documents = response.objects
            
            if not documents:
                return "🧹 **Normalizzazione Testo**\n\nNessun documento trovato nella collezione."
            
            # Analisi e applicazione della normalizzazione
            normalized_count = 0
            issues_found = 0
            
            for doc in documents:
                updated_properties = {}
                original_properties = doc.properties
                
                for prop_name, prop_value in original_properties.items():
                    if isinstance(prop_value, str):
                        original_text = prop_value
                        cleaned_text = original_text.strip() # Rimuovi spazi extra
                        
                        # Normalizzazione a minuscolo (esempio)
                        if "minuscol" in question.lower() or "lowercase" in question.lower():
                            cleaned_text = cleaned_text.lower()
                        # Normalizzazione a maiuscolo (esempio)
                        elif "maiuscol" in question.lower() or "uppercase" in question.lower():
                            cleaned_text = cleaned_text.upper()
                        
                        # Rimuovi punteggiatura extra o caratteri speciali (esempio)
                        if "punteggiatura" in question.lower() or "speciali" in question.lower():
                            cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text) # Rimuove tutto tranne lettere, numeri, spazi
                        
                        if original_text != cleaned_text:
                            updated_properties[prop_name] = cleaned_text
                            issues_found += 1
                
                if updated_properties:
                    try:
                        # Aggiorna il documento in Weaviate
                        all_properties = original_properties.copy()
                        all_properties.update(updated_properties)
                        
                        collection.data.replace(
                            uuid=str(doc.uuid),
                            properties=all_properties
                        )
                        normalized_count += 1
                    except Exception as e:
                        print(f"Errore aggiornamento documento {doc.uuid}: {e}")
            
            if issues_found == 0:
                return f"🧹 **Normalizzazione Testo Completata**\n\n✅ Nessun problema di normalizzazione trovato nella collezione \'{class_name}\'!"
            
            result = f"🧹 **Analisi e Normalizzazione Testo**\n\n"
            result += f"📊 **Statistiche:**\n"
            result += f"• Documenti analizzati: {len(documents)}\n"
            result += f"• Problemi totali rilevati: {issues_found}\n"
            result += f"• Documenti aggiornati: {normalized_count}\n\n"
            
            if normalized_count > 0:
                result += f"✅ **Normalizzazione completata!**\n"
            else:
                result += f"💡 **Suggerimento:** Per applicare la normalizzazione, chiedi in modo specifico (es: \'Normalizza il testo in minuscolo nella collezione {class_name}\' o \'Rimuovi la punteggiatura dalla collezione {class_name}\')"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Normalizzazione Testo**\n\nErrore durante la normalizzazione: {str(e)}"

    def _handle_whitespace_cleaning(self, collection, class_name: str, question: str) -> str:
        """Gestisce la rimozione di spazi bianchi extra."""
        try:
            response = collection.query.fetch_objects(limit=500)
            documents = response.objects
            
            if not documents:
                return "🧹 **Pulizia Spazi Bianchi**\n\nNessun documento trovato nella collezione."
            
            cleaned_count = 0
            whitespace_issues = 0
            examples = []
            
            for doc in documents:
                updated_properties = {}
                for prop_name, prop_value in doc.properties.items():
                    if isinstance(prop_value, str):
                        original_value = prop_value
                        cleaned_value = original_value.strip() # Rimuove spazi all\'inizio e alla fine
                        cleaned_value = re.sub(r'\s+', ' ', cleaned_value) # Rimuove spazi multipli
                        
                        if original_value != cleaned_value:
                            updated_properties[prop_name] = cleaned_value
                            whitespace_issues += 1
                            if len(examples) < 3: # Raccogli 3 esempi
                                examples.append({
                                    "property": prop_name,
                                    "before": original_value,
                                    "after": cleaned_value
                                })
                
                if updated_properties:
                    try:
                        # Aggiorna il documento in Weaviate
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
                return f"🧹 **Pulizia Spazi Completata**\n\n✅ Nessun problema di spazi trovato nella collezione \'{class_name}\'!"
            
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
                result += f"\n💡 **Suggerimento:** Per applicare automaticamente la pulizia degli spazi, chiedi: \'Pulisci gli spazi dalla collezione {class_name}\'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Spazi**\n\nErrore durante la pulizia: {str(e)}"

    def _handle_encoding_issues(self, collection, class_name: str, question: str) -> str:
        """Gestisce i problemi di encoding dei caratteri."""
        return f"""🧹 **Correzione Encoding**\n\nHo rilevato una richiesta per correggere problemi di encoding nella collezione \'{class_name}\'.\n\n**Problemi comuni di encoding:**\n- Caratteri accentati malformati (es: Ã, â, Â)\n- Simboli strani al posto di caratteri normali\n- Testo illeggibile dopo import da fonti diverse\n\n**Funzionalità di correzione encoding in sviluppo:**\n- ✨ Auto-rilevamento encoding problematici\n- ✨ Correzione UTF-8 automatica\n- ✨ Normalizzazione caratteri accentati\n- ✨ Conversione tra diversi encoding\n\nQuesta funzionalità sarà presto disponibile! 🚀\n\nNel frattempo, puoi:\n- Verificare l\'encoding originale dei file prima dell\'import\n- Usare UTF-8 come encoding standard\n- Controllare i caratteri speciali nei tuoi dati"""

    def _handle_data_validation(self, collection, class_name: str, question: str) -> str:
        """Gestisce la validazione dell\'integrità dei dati."""
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
        return f"""🧹 **Rimozione Outliers**\n\nHo rilevato una richiesta per identificare e rimuovere outliers nella collezione \'{class_name}\'.\n\n**Tipi di outliers che posso rilevare:**\n- 📊 Outliers numerici (valori statisticamente anomali)\n- 📝 Testi anomali (lunghezza, pattern strani)\n- 📅 Date fuori range ragionevole\n- 🔢 Valori fuori dai limiti logici\n\n**Funzionalità di rimozione outliers in sviluppo:**\n- ✨ Rilevamento automatico outliers statistici (Z-score, IQR)\n- ✨ Analisi anomalie testuali\n- ✨ Validazione range logici\n- ✨ Opzioni di rimozione/correzione selettive\n\nQuesta funzionalità avanzata sarà presto disponibile! 🚀\n\n**Nel frattempo puoi:**\n- Identificare valori sospetti con query analitiche\n- Usare \"mostra valori estremi per [campo]\"\n- Controllare manualmente range dei dati numerici"""

    def _handle_general_cleaning(self, collection, class_name: str, question: str) -> str:
        """Gestisce richieste di pulizia generale (combina più operazioni)."""
        try:
            result = f"🧹 **Pulizia Generale Avviata**\n\nEseguo una pulizia completa della collezione \'{class_name}\'...\n\n"
            
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
            result += "💡 **Suggerimento:** Per una pulizia automatica completa, chiedi: \'Applica tutte le correzioni di pulizia\'"
            
            return result
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Generale**\n\nErrore durante la pulizia generale: {str(e)}"

    def _handle_custom_cleaning_with_gemini(self, question: str, collection, class_name: str) -> str:
        """Usa Gemini per operazioni di pulizia personalizzate non coperte dai metodi standard."""
        try:
            # Ottieni informazioni sulla collezione
            sample_response = collection.query.fetch_objects(limit=3)
            if not sample_response.objects:
                return "🧹 **Pulizia Personalizzata**\n\nNessun documento trovato nella collezione per l\'analisi."
            
            # Prepara contesto per Gemini
            properties = list(sample_response.objects[0].properties.keys())
            sample_data = []
            
            for doc in sample_response.objects:
                doc_sample = {}
                for prop_name, prop_value in doc.properties.items():
                    if isinstance(prop_value, str) and len(prop_value) > 100:
                        doc_sample[prop_name] = prop_value[:100] + "..."
                    else:
                        doc_sample[prop_name] = str(prop_value) if prop_value is not None else "NULL"
                sample_data.append(doc_sample)
            
            prompt = f"""
            Sei un esperto in pulizia e normalizzazione dati per la collezione \'{class_name}\'.\n            \n            RICHIESTA UTENTE: \"{question}\"\n            \n            PROPRIETÀ DISPONIBILI: {properties}\n            \n            CAMPIONE DATI:\n            {sample_data}\n            \n            Analizza la richiesta di pulizia e fornisci:\n            1. Tipo di operazione richiesta\n            2. Campi da processare  \n            3. Metodo di pulizia suggerito\n            4. Potenziali problemi da considerare\n            5. Passi specifici per l\'implementazione\n            \n            IMPORTANTE: Sii specifico sui campi e metodi da usare.\n            """
            
            response = self._generate_content(prompt)
            
            return f"🧹 **Analisi Pulizia Personalizzata**\n\n{response.text.strip()}\n\n💡 **Nota:** Questa è un\'analisi della richiesta. L\'implementazione automatica di operazioni personalizzate sarà disponibile nelle prossime versioni."
            
        except Exception as e:
            return f"🧹 **Errore Pulizia Personalizzata**\n\nErrore nell\'analisi: {str(e)}"

