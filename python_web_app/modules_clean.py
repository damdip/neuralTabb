"""
Moduli puliti per NeuralTabb - Solo funzionalità Weaviate essenziali, con integrazione Gemini
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import weaviate
import json
from datetime import datetime
import google.generativeai as genai


class GeminiValidator:
    """Validatore Gemini ULTRA-VELOCE con cache e prompt ottimizzati"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self._configure_gemini()
        
        # Cache aggressiva per evitare chiamate duplicate
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _configure_gemini(self):
        """Configura Gemini per massima velocità"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    'temperature': 0,  # Deterministica
                    'max_output_tokens': 10,  # MOLTO breve
                    'top_p': 1.0,
                    'top_k': 1
                }
            )
            print("🚀 Gemini configurato per velocità massima")
        except Exception as e:
            print(f"❌ Errore configurazione Gemini: {e}")
            self.model = None
    
    def quick_validate(self, question: str, local_prediction: str) -> Dict[str, Any]:
        """
        Validazione ULTRA-VELOCE - solo check di accordo, non riclassifica
        """
        start_time = datetime.now()
        
        # Cache check PRIMO
        cache_key = f"{question.lower().strip()}:{local_prediction}"
        if cache_key in self.cache:
            self.cache_hits += 1
            result = self.cache[cache_key].copy()
            result['time_ms'] = 1
            result['source'] = 'cache'
            return result
        
        self.cache_misses += 1
        
        # Se Gemini non disponibile, accetta sempre la predizione locale
        if not self.model:
            return {
                'validated_type': local_prediction,
                'confidence': 'medium',
                'source': 'local_only',
                'time_ms': 1,
                'gemini_available': False
            }
        
        try:
            # Prompt ULTRA-CONCISO per velocità massima
            prompt = f"""Q: "{question}"
Local: {local_prediction}

Valid types: conversational, analytical, semantic

Agree? Answer ONLY: yes/no"""

            response = self.model.generate_content(prompt)
            gemini_response = response.text.strip().lower()
            
            # Parse veloce
            agrees = 'yes' in gemini_response or 'sì' in gemini_response or gemini_response == 'si'
            
            end_time = datetime.now()
            time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = {
                'validated_type': local_prediction,  # Manteniamo sempre la predizione locale
                'confidence': 'high' if agrees else 'medium',
                'source': 'gemini_quick',
                'time_ms': time_ms,
                'gemini_available': True,
                'agreement': agrees,
                'gemini_response': gemini_response
            }
            
            # Cache il risultato
            self.cache[cache_key] = result.copy()
            
            # Pulisci cache se troppo grande (mantieni solo le ultime 50)
            if len(self.cache) > 50:
                oldest_keys = list(self.cache.keys())[:-25]  # Rimuovi le più vecchie
                for key in oldest_keys:
                    del self.cache[key]
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return {
                'validated_type': local_prediction,
                'confidence': 'medium',
                'source': 'error_fallback',
                'time_ms': time_ms,
                'gemini_available': False,
                'error': str(e)
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiche cache per debugging"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 1)
        }
                


class WeaviateManager:
    """Manager semplificato per Weaviate con operazioni base"""
    
    def __init__(self, client):
        self.client = client
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """Lista tutte le collezioni"""
        try:
            collections = []
            all_collection_names = self.client.collections.list_all()
            
            for collection_name in all_collection_names:
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Conta documenti
                    response = collection.aggregate.over_all(total_count=True)
                    count = response.total_count
                    
                    collections.append({
                        "name": collection_name,
                        "count": count
                    })
                    
                except Exception as e:
                    collections.append({
                        "name": collection_name,
                        "count": 0,
                        "error": str(e)
                    })
            
            return collections
            
        except Exception as e:
            print(f"Errore nel listare le collezioni: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Ottieni informazioni su una specifica collezione"""
        try:
            collection = self.client.collections.get(collection_name)
            
            # Conta documenti
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count
            
            # Ottieni un campione per vedere le proprietà
            sample = collection.query.fetch_objects(limit=1)
            properties = []
            
            if sample.objects:
                for prop_name in sample.objects[0].properties.keys():
                    properties.append(prop_name)
            
            return {
                "name": collection_name,
                "count": count,
                "properties": properties,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "name": collection_name,
                "count": 0,
                "properties": [],
                "status": "error",
                "error": str(e)
            }
    
    def delete_collection(self, collection_name: str) -> bool:
        """Elimina una collezione"""
        try:
            self.client.collections.delete(collection_name)
            print(f"Collezione '{collection_name}' eliminata con successo")
            return True
        except Exception as e:
            print(f"Errore nell'eliminazione della collezione '{collection_name}': {e}")
            return False
    
    def get_collection_sample_data(self, collection_name: str, limit: int = 20) -> Dict[str, Any]:
        """Ottieni dati campione da una collezione"""
        try:
            collection = self.client.collections.get(collection_name)
            
            # Ottieni i dati campione
            response = collection.query.fetch_objects(limit=limit)
            
            documents = []
            properties = []
            
            if response.objects:
                # Ottieni le proprietà dal primo documento
                properties = list(response.objects[0].properties.keys())
                
                for obj in response.objects:
                    doc_data = {"id": str(obj.uuid)}
                    
                    # Aggiungi tutte le proprietà
                    for prop_name, prop_value in obj.properties.items():
                        if prop_value is not None:
                            # Per testi lunghi, accorcia
                            if isinstance(prop_value, str) and len(prop_value) > 100:
                                doc_data[prop_name] = prop_value[:100] + "..."
                            else:
                                doc_data[prop_name] = prop_value
                        else:
                            doc_data[prop_name] = ""
                    
                    documents.append(doc_data)
            
            return {
                "success": True,
                "documents": documents,
                "properties": properties,
                "total_documents": len(documents),
                "collection_name": collection_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "properties": [],
                "total_documents": 0,
                "collection_name": collection_name
            }


class SimpleSearchManager:
    """Manager per ricerche semplici in Weaviate senza AI"""
    
    def __init__(self, client):
        self.client = client
    
    def search_by_keyword(self, collection_name: str, keyword: str, limit: int = 5) -> Dict[str, Any]:
        """Ricerca semplice per parola chiave - adattabile a qualsiasi schema"""
        try:
            collection = self.client.collections.get(collection_name)
            
            # Prima ottieni le proprietà disponibili
            sample = collection.query.fetch_objects(limit=1)
            available_properties = []
            
            if sample.objects:
                available_properties = list(sample.objects[0].properties.keys())
                print(f"Proprietà disponibili in {collection_name}: {available_properties}")
            
            # Ricerca usando near_text senza specificare return_properties
            response = collection.query.near_text(
                query=keyword,
                limit=limit
            )
            
            results = []
            for obj in response.objects:
                # Gestisci lo score in modo sicuro
                score = getattr(obj.metadata, 'score', None) if hasattr(obj, 'metadata') else None
                if score is None:
                    score = 0.0
                
                # Assicurati che score sia un numero valido
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                
                result_data = {
                    "id": str(obj.uuid),
                    "score": score
                }
                
                # Aggiungi tutte le proprietà disponibili dinamicamente
                for prop_name, prop_value in obj.properties.items():
                    if prop_value is not None:
                        # Per testi lunghi, accorcia per la visualizzazione
                        if isinstance(prop_value, str) and len(prop_value) > 200:
                            result_data[prop_name] = prop_value[:200] + "..."
                        else:
                            result_data[prop_name] = prop_value
                    else:
                        result_data[prop_name] = ""
                
                results.append(result_data)
            
            return {
                "success": True,
                "results": results,
                "total": len(results),
                "query": keyword,
                "available_properties": available_properties
            }
            
        except Exception as e:
            print(f"Errore in search_by_keyword: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total": 0,
                "available_properties": []
            }
    
    def get_sample_documents(self, collection_name: str, limit: int = 10) -> Dict[str, Any]:
        """Ottieni documenti campione dalla collezione - adattabile a qualsiasi schema"""
        try:
            collection = self.client.collections.get(collection_name)
            
            response = collection.query.fetch_objects(limit=limit)
            
            results = []
            available_properties = []
            
            if response.objects:
                # Ottieni le proprietà dal primo documento
                available_properties = list(response.objects[0].properties.keys())
                print(f"Proprietà disponibili in {collection_name}: {available_properties}")
                
                for obj in response.objects:
                    result_data = {"id": str(obj.uuid)}
                    
                    # Aggiungi tutte le proprietà disponibili dinamicamente
                    for prop_name, prop_value in obj.properties.items():
                        if prop_value is not None:
                            # Per testi lunghi, accorcia per la visualizzazione
                            if isinstance(prop_value, str) and len(prop_value) > 150:
                                result_data[prop_name] = prop_value[:150] + "..."
                            else:
                                result_data[prop_name] = prop_value
                        else:
                            result_data[prop_name] = ""
                    
                    results.append(result_data)
            
            return {
                "success": True,
                "results": results,
                "total": len(results),
                "available_properties": available_properties
            }
            
        except Exception as e:
            print(f"Errore in get_sample_documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total": 0,
                "available_properties": []
            }


class ConversationalHandler:
    """Gestisce le domande conversazionali senza AI"""
    
    def __init__(self, weaviate_manager=None):
        self.weaviate_manager = weaviate_manager
        self.responses = {
            # Saluti
            'saluti': [
                "Ciao! Sono NeuralTabb, il tuo assistente per l'analisi di dati con Weaviate. Come posso aiutarti oggi?",
                "Salve! Sono qui per aiutarti a esplorare e analizzare i tuoi dati. Cosa vorresti sapere?",
                "Buongiorno! Sono NeuralTabb, pronto ad aiutarti con le tue ricerche sui dati."
            ],
            
            # Chi sono
            'identita': [
                "Sono NeuralTabb, un sistema di ricerca e analisi dati basato su Weaviate. Posso aiutarti a trovare informazioni nei tuoi dataset, fare ricerche semantiche e analizzare i dati.",
                "Mi chiamo NeuralTabb e sono specializzato nell'analisi di dati utilizzando Weaviate. Posso cercare documenti, analizzare contenuti e rispondere a domande sui tuoi dati.",
            ],
            
            # Aiuto
            'aiuto': [
                "Posso aiutarti in diversi modi:\n• Ricerca semantica nei tuoi dati\n• Analisi di collezioni Weaviate\n• Rispondere a domande sui contenuti\n• Esplorare e gestire le tue collezioni\n\nCosa ti interessa di più?",
                "Ecco cosa posso fare per te:\n• Cercare informazioni specifiche nei tuoi dataset\n• Analizzare e riassumere contenuti\n• Gestire collezioni Weaviate\n• Fornire statistiche sui dati\n\nSeleziona una collezione e fammi una domanda!"
            ],
            
            # Ringraziamenti
            'ringraziamenti': [
                "Prego! Sono sempre qui per aiutarti con i tuoi dati. Se hai altre domande, non esitare a chiedere!",
                "Di niente! È un piacere aiutarti. C'è altro che vorresti sapere sui tuoi dati?",
                "Felice di essere utile! Se hai bisogno di altre analisi o ricerche, sono a disposizione."
            ],
            
            # Come stai
            'stato': [
                "Sto bene, grazie! Tutti i sistemi sono operativi e sono pronto ad analizzare i tuoi dati. Come posso aiutarti?",
                "Perfettamente funzionante! I miei sistemi di ricerca sono attivi e pronti. Cosa vorresti cercare oggi?"
            ],
            
            # Default
            'default': [
                "Mi dispiace, non sono sicuro di aver capito. Sono specializzato nell'analisi di dati con Weaviate. Puoi farmi domande sui tuoi dataset o cercare informazioni specifiche.",
                "Non ho capito bene la tua richiesta. Sono progettato per aiutarti con ricerche e analisi di dati. Prova a selezionare una collezione e farmi una domanda specifica!"
            ]
        }
    
    def classify_conversational_type(self, question: str) -> str:
        """Classifica il tipo di domanda conversazionale"""
        question_lower = question.lower().strip()
        
        # Saluti
        if any(word in question_lower for word in ['ciao', 'salve', 'buongiorno', 'buonasera', 'hello', 'hi']):
            return 'saluti'
        
        # Identità
        if any(phrase in question_lower for phrase in ['chi sei', 'cosa sei', 'come ti chiami', 'what are you', 'who are you']):
            return 'identita'
        
        # Aiuto
        if any(word in question_lower for word in ['aiuto', 'help', 'cosa puoi fare', 'come funzioni', 'cosa fai']):
            return 'aiuto'
        
        # Ringraziamenti
        if any(word in question_lower for word in ['grazie', 'thanks', 'thank you', 'perfetto', 'ottimo', 'bene']):
            return 'ringraziamenti'
        
        # Come stai
        if any(phrase in question_lower for phrase in ['come stai', 'come va', 'tutto bene', 'how are you']):
            return 'stato'
        
        return 'default'
    
    def get_response(self, question: str) -> str:
        """Restituisce una risposta appropriata per la domanda conversazionale"""
        response_type = self.classify_conversational_type(question)
        responses = self.responses.get(response_type, self.responses['default'])
        
        # Scegli una risposta casuale per varietà
        import random
        base_response = random.choice(responses)
        
        # Aggiungi informazioni contestuali se disponibile
        if self.weaviate_manager and response_type in ['aiuto', 'saluti']:
            try:
                collections = self.weaviate_manager.list_collections()
                if collections:
                    collection_info = f"\n\nCollezioni disponibili: {', '.join([col['name'] for col in collections[:5]])}"
                    if len(collections) > 5:
                        collection_info += f" e altre {len(collections) - 5}..."
                    base_response += collection_info
            except:
                pass
        
        return base_response


class SimpleQASystem:
    """Sistema Q&A con integrazione Gemini per validazione classificazione"""
    
    def __init__(self, client, gemini_api_key: str = None):
        self.client = client
        self.weaviate_manager = WeaviateManager(client)
        self.search_manager = SimpleSearchManager(client)
        self.conversational_handler = ConversationalHandler(self.weaviate_manager)
        
        # Inizializza validatore Gemini se disponibile
        self.gemini_validator = None
        if gemini_api_key:
            try:
                self.gemini_validator = GeminiValidator(gemini_api_key)
            except Exception as e:
                print(f"Impossibile inizializzare Gemini: {e}")
    
    def ask_question(self, question: str, collection_name: str, limit: int = 5) -> Dict[str, Any]:
        """Risponde alle domande con validazione Gemini della classificazione"""
        try:
            # Fase 1: Classificazione locale
            local_prediction = self.classify_question_local(question)
            print(f"Classificazione locale: {local_prediction}")
            
            # Fase 2: Validazione con Gemini (se disponibile)
            final_classification = local_prediction
            validation_info = {}
            
            if self.gemini_validator:
                validation = self.gemini_validator.quick_validate(question, local_prediction)
                validation_info = validation
                
                # Nel nuovo formato, usiamo sempre validated_type
                final_classification = validation['validated_type']
                
                print(f"🚀 Validazione veloce completata in {validation.get('time_ms', 0)}ms")
                print(f"📊 Classificazione: {final_classification} (fiducia: {validation.get('confidence', 'medium')})")
                
            else:
                validation_info = {
                    'validated_type': local_prediction,
                    'confidence': 'local_only',
                    'source': 'no_gemini',
                    'time_ms': 0,
                    'gemini_available': False
                }
                final_classification = local_prediction
            
            # Fase 3: Elabora risposta basata sulla classificazione finale
            if final_classification == 'conversational':
                response = self.conversational_handler.get_response(question)
                return {
                    "answer": response,
                    "type": "conversational",
                    "success": True,
                    "total_found": 0,
                    "documents": [],
                    "collection_info": {"name": "Sistema conversazionale"},
                    "is_conversational": True,
                    "classification_info": {
                        "local_prediction": local_prediction,
                        "final_classification": final_classification,
                        "validation": validation_info
                    }
                }
            
            elif final_classification == 'analytical':
                return {
                    "answer": f"Ho riconosciuto che stai facendo una domanda analitica: '{question}'. "
                             f"Attualmente il sistema di analisi avanzata non è disponibile, ma posso fare "
                             f"una ricerca semantica nella collezione '{collection_name}' per trovare dati pertinenti.",
                    "type": "analytical_fallback",
                    "success": True,
                    "total_found": 0,
                    "documents": [],
                    "collection_info": {"name": collection_name},
                    "suggestion": "Prova a fare una ricerca semantica per parole chiave specifiche.",
                    "classification_info": {
                        "local_prediction": local_prediction,
                        "final_classification": final_classification,
                        "validation": validation_info
                    }
                }
            
            else:  # semantic
                keywords = self._extract_keywords(question)
                search_query = " ".join(keywords)
                
                search_result = self.search_manager.search_by_keyword(collection_name, search_query, limit)
                
                base_response = {
                    "collection_info": {"name": collection_name},
                    "classification_info": {
                        "local_prediction": local_prediction,
                        "final_classification": final_classification,
                        "validation": validation_info
                    }
                }
                
                if search_result["success"]:
                    return {
                        **base_response,
                        "answer": f"Ho trovato {search_result['total']} risultati per la tua ricerca su '{search_query}'.",
                        "type": "search",
                        "success": True,
                        "total_found": search_result["total"],
                        "documents": search_result["results"]
                    }
                else:
                    return {
                        **base_response,
                        "answer": "Non ho trovato risultati per la tua ricerca.",
                        "type": "search",
                        "success": False,
                        "total_found": 0,
                        "documents": [],
                        "error": search_result.get("error", "")
                    }
                
        except Exception as e:
            return {
                "answer": f"Errore durante la ricerca: {str(e)}",
                "type": "error",
                "success": False,
                "total_found": 0,
                "documents": [],
                "error": str(e)
            }
    
    def classify_question_local(self, question: str) -> str:
        """Classificazione locale (rinominata per chiarezza)"""
        question_lower = question.lower().strip()
        
        # Parole chiave conversazionali
        conversational_patterns = [
            'ciao', 'salve', 'buongiorno', 'buonasera', 'hello', 'hi',
            'come stai', 'come va', 'tutto bene', 'how are you',
            'chi sei', 'cosa sei', 'come ti chiami', 'what are you', 'who are you',
            'aiuto', 'help', 'cosa puoi fare', 'come funzioni', 'cosa fai',
            'grazie', 'thanks', 'thank you', 'perfetto', 'ottimo', 'bene'
        ]
        
        if any(pattern in question_lower for pattern in conversational_patterns):
            return 'conversational'
        
        # Parole chiave analitiche
        analytical_patterns = [
            'quanti', 'conta', 'numero', 'media', 'somma', 'totale', 'percentuale', 'statistica',
            'calcola', 'misura', 'analizza', 'confronta', 'massimo', 'minimo', 'count', 'sum',
            'average', 'statistics', 'analyze', 'compare', 'maximum', 'minimum'
        ]
        
        if any(pattern in question_lower for pattern in analytical_patterns):
            return 'analytical'
        
        # Default: semantica
        return 'semantic'
    
    def classify_question(self, question: str) -> str:
        """Classificazione con validazione Gemini (per compatibilità)"""
        # Per mantenere compatibilità, restituisce solo la classificazione finale
        result = self.ask_question(question, "", 0)  # Chiamata dummy per ottenere classificazione
        return result.get('classification_info', {}).get('final_classification', 'semantic')
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Estrae parole chiave semplici dalla domanda"""
        import re
        # Rimuovi parole comuni
        stop_words = {'il', 'la', 'di', 'che', 'e', 'un', 'una', 'da', 'in', 'con', 'su', 'per', 'come', 'cosa', 'dove', 'quando', 'chi', 'perché'}
        
        # Estrai parole (solo lettere, no numeri o simboli)
        words = re.findall(r'\b[a-zA-ZàáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ]+\b', question.lower())
        
        # Filtra parole troppo corte o stop words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:5]  # Massimo 5 parole chiave


# Classe di compatibilità per il main.py esistente  
class QASystemWithGemini:
    """Wrapper di compatibilità che usa solo Weaviate (niente Gemini)"""
    
    def __init__(self, client, api_key_path: str = None):
        self.client = client
        self.qa_system = SimpleQASystem(client)
        self.weaviate_manager = WeaviateManager(client)
    
    def ask_question(self, question: str, collection_name: str, limit: int = 5) -> Dict[str, Any]:
        """Compatibilità con l'interfaccia esistente"""
        return self.qa_system.ask_question(question, collection_name, limit)
    
    def classify_question(self, question: str) -> str:
        """Classificazione migliorata delle domande - delega al SimpleQASystem"""
        return self.qa_system.classify_question(question)
    
    def get_current_model_info(self) -> Dict[str, str]:
        return {
            "model_name": "Ricerca Weaviate (senza AI)",
            "status": "active"
        }



# Compatibilità per SemanticSearch
class SemanticSearch:
    def __init__(self, client):
        self.client = client
    
    def ask_question(self, question: str, collection_name: str = "Documents", limit: int = 5) -> Dict[str, Any]:
        """Cerca documenti in base a una domanda - adattabile a qualsiasi schema"""
        try:
            # Ricerca semantica sui documenti della collezione scelta
            collection = self.client.collections.get(collection_name)
            
            # Prima ottieni la configurazione della collezione per vedere le proprietà disponibili
            response = collection.query.fetch_objects(limit=1)
            available_properties = []
            
            if response.objects:
                available_properties = list(response.objects[0].properties.keys())
                print(f"Proprietà disponibili in {collection_name}: {available_properties}")
            
            # Esegui la ricerca senza specificare return_properties (prende tutte)
            response = collection.query.near_text(
                query=question,
                limit=limit,
                distance=0.5
            )
            
            sources = []
            
            if response.objects:
                for doc in response.objects:
                    distance = getattr(doc.metadata, 'distance', None)
                    if distance is None:
                        distance = 0.0
                    
                    # Assicurati che distance sia un numero valido
                    try:
                        distance = float(distance)
                    except (TypeError, ValueError):
                        distance = 0.0
                    
                    source_info = {
                        "id": str(doc.uuid), 
                        "distance": distance,
                        "relevance": max(0.0, 1.0 - distance)  # Assicura che sia >= 0
                    }
                    
                    # Aggiungi tutte le proprietà disponibili dinamicamente
                    for prop_name, prop_value in doc.properties.items():
                        if prop_value is not None:
                            # Per testi lunghi, accorcia per la visualizzazione
                            if isinstance(prop_value, str) and len(prop_value) > 300:
                                source_info[prop_name] = prop_value[:300] + "..."
                            else:
                                source_info[prop_name] = prop_value
                        else:
                            source_info[prop_name] = ""
                    
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
            print(f"Errore in ask_question: {e}")
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
        """Ricerca documenti per query - adattabile a qualsiasi schema"""
        try:
            collection = self.client.collections.get(collection_name)
            
            # Esegui ricerca senza specificare proprietà specifiche
            response = collection.query.near_text(
                query=query,
                limit=limit,
                distance=0.5
            )
            
            documents = []
            for doc in response.objects:
                # Gestisci la distance in modo sicuro
                distance = getattr(doc.metadata, 'distance', None)
                if distance is None:
                    distance = 0.0
                
                # Assicurati che distance sia un numero valido
                try:
                    distance = float(distance)
                except (TypeError, ValueError):
                    distance = 0.0
                
                # Costruisci il documento in modo generalizzabile
                doc_data = {
                    "id": str(doc.uuid),
                    "relevance": max(0.0, 1.0 - distance)  # Assicura che sia >= 0
                }
                
                # Aggiungi tutte le proprietà disponibili dinamicamente
                for prop_name, prop_value in doc.properties.items():
                    if prop_value is not None:
                        # Per contenuti lunghi, accorcia per la visualizzazione
                        if isinstance(prop_value, str) and len(prop_value) > 200:
                            doc_data[prop_name] = prop_value[:200] + "..."
                        else:
                            doc_data[prop_name] = prop_value
                
                documents.append(doc_data)
            
            return documents
            
        except Exception as e:
            print(f"Errore in search_documents: {e}")
            return []

