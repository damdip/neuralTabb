from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import weaviate
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import traceback

# Import dei moduli personalizzati
from modules import WeaviateManager, QASystemWithGemini

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000MB max file size (provvisorio)

# Assicurati che la cartella uploads esista
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inizializza Weaviate client
try:
    client = weaviate.connect_to_local()
    weaviate_manager = WeaviateManager(client)
    
    # Inizializza il sistema QA con Gemini
    try:
        qa_gemini = QASystemWithGemini(client, api_key_path="config/configLLM.txt")
        print("Sistema QA con Gemini inizializzato con successo")
    except Exception as e:
        print(f"Errore nell'inizializzazione del sistema QA con Gemini: {e}")
        qa_gemini = None
    
    # Crea lo schema se non esistente
    weaviate_manager.setup_schema()
    
except Exception as e:
    print(f"Errore connessione Weaviate: {e}")
    client = None
    qa_gemini = None

@app.route('/')
def index():
    """Homepage con dashboard intelligente"""
    if not client:
        return render_template('error.html', error="Connessione Weaviate non disponibile")
    
    try:
        # Prima controlla se esistono collezioni
        collections = weaviate_manager.list_collections()
        print(f"Collezioni disponibili: {[col.get('name') for col in collections]}")
        
        # Se non ci sono collezioni, mostra schermata di benvenuto
        if not collections:
            return render_template('index.html', 
                                 show_welcome=True, 
                                 collections=[])
        
        # Controlla se ci sono documenti nelle collezioni
        total_documents = sum(col.get('count', 0) for col in collections)
        print(f"Documenti totali nelle collezioni: {total_documents}")
        
        # Se non ci sono documenti, mostra schermata di avvio
        if total_documents == 0:
            return render_template('index.html', 
                                 show_empty_collections=True, 
                                 collections=collections)
        
        # Se ci sono dati, mostra la dashboard
        return render_template('index.html', 
                             show_dashboard=True,
                             collections=collections)
        
    except Exception as e:
        print(f"Errore nella homepage: {e}")
        return render_template('error.html', error=str(e))

@app.route('/upload', methods=['GET', 'POST'])
def upload_documents():
    """Upload e inserimento documenti"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('Nessun file selezionato')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('Nessun file selezionato')
                return redirect(request.url)
            
            if file:
                # Controlla la dimensione del file prima di salvarlo
                file.seek(0, 2)  # Vai alla fine del file
                file_size = file.tell()
                file.seek(0)  # Torna all'inizio
                
                max_size = app.config['MAX_CONTENT_LENGTH']
                if file_size > max_size:
                    flash(f'File troppo grande! Dimensione massima: {max_size // (1024*1024)}MB, File: {file_size // (1024*1024)}MB')
                    return redirect(request.url)
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Elabora il file
                result = weaviate_manager.process_file(filepath)
                
                if result.get("status") == "success":
                    flash(f'Caricati {result["inserted"]} documenti con successo')
                else:
                    flash(f'Errore durante l\'elaborazione: {result.get("error", "Errore sconosciuto")}')
                
                # Pulisci il file temporaneo
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Errore durante il caricamento: {str(e)}')
    
    return render_template('upload.html')

@app.route('/qa', methods=['GET', 'POST'])
def question_answering():
    # Controllo se il client Weaviate è disponibile
    if not client:
        return render_template('error.html', error="Connessione Weaviate non disponibile")
    
    # Ottieni le collezioni disponibili
    try:
        collections = weaviate_manager.list_collections()
    except Exception as e:
        print(f"Errore nel recuperare collezioni: {e}")
        collections = []
        flash('Errore nel recuperare le collezioni disponibili')
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        collection_name = request.form.get('collection', '').strip()
        limit = request.form.get('limit', 5)
        
        # Converti limit a intero
        try:
            limit = int(limit)
            if limit < 1 or limit > 50:  # Limita tra 1 e 50
                limit = 5
        except ValueError:
            limit = 5
        
        # Validazione input
        if not question:
            flash('Inserisci una domanda')
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections,
                                 limit=limit)
        
        if not collection_name:
            flash('Seleziona una collezione')
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections,
                                 limit=limit)
        
        try:
            # Verifica che la collezione esista
            collection_names = [col.get('name') for col in collections]
            if collection_name not in collection_names:
                flash(f'Collezione "{collection_name}" non trovata')
                return render_template('qa.html', 
                                     question=question, 
                                     collection=collection_name,
                                     collections=collections,
                                     limit=limit)
            
            # Usa il sistema QA con Gemini per ottenere una risposta intelligente
            answer = qa_gemini.smart_answer(question, collection_name) if qa_gemini else {"error": "Sistema QA non disponibile"}
            
            # Log della risposta per debug
            print(f"Documenti trovati: {answer.get('total_found', 0)}")
            
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections,
                                 answer=answer,
                                 limit=limit)
                                 
        except Exception as e:
            print(f"Errore durante la ricerca: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            flash(f'Errore nella ricerca: {str(e)}')
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections)
    
    return render_template('qa.html', collections=collections)



@app.route('/chat')
def chat_interface():
    """Interfaccia di chat per il sistema Q&A con Gemini"""
    if not client:
        return render_template('error.html', error="Connessione Weaviate non disponibile")
    
    if not qa_gemini:
        return render_template('error.html', error="Sistema QA con Gemini non disponibile. Verifica che il file 'chiave.txt' esista.")
    
    # Ottieni informazioni sul modello corrente
    model_info = qa_gemini.get_current_model_info() if qa_gemini else {}
    
    return render_template('chat.html', model_info=model_info)

@app.route('/chat/ask', methods=['POST'])
def chat_ask():
    """Endpoint API per le domande della chat"""
    if not client or not qa_gemini:
        return jsonify({
            'success': False,
            'error': 'Sistema non disponibile'
        })
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        collection_name = data.get('collection', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Domanda non fornita'
            })
        
        # Classifica prima la domanda per determinare se serve una collezione
        question_type = qa_gemini.classify_question(question)
        
        # Per domande conversazionali non serve una collezione
        if question_type != "conversazionale":
            if not collection_name:
                return jsonify({
                    'success': False,
                    'error': 'Per questo tipo di domanda devi selezionare una collezione',
                    'question_type': question_type
                })
            
            # Verifica che la collezione esista
            try:
                collections = weaviate_manager.list_collections()
                collection_names = [col.get('name') for col in collections]
                if collection_name not in collection_names:
                    return jsonify({
                        'success': False,
                        'error': f'Collezione "{collection_name}" non trovata'
                    })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Errore nel verificare la collezione: {str(e)}'
                })
        
        # Misura il tempo di elaborazione
        start_time = datetime.now()
        
        # Ottieni la risposta (passa None come collection_name se conversazionale)
        result = qa_gemini.smart_answer(question, collection_name if question_type != "conversazionale" else None)
        
        # Estrai la risposta testuale dal risultato
        answer = result.get('answer', 'Nessuna risposta disponibile') if isinstance(result, dict) else str(result)
        
        end_time = datetime.now()
        processing_time = int((end_time - start_time).total_seconds() * 1000)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'question_type': question_type,
            'processing_time': processing_time,
            'collection': collection_name
        })
        
    except Exception as e:
        print(f"Errore nella chat: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Errore durante l\'elaborazione: {str(e)}'
        })



@app.route('/collections')
def manage_collections():
    """Gestione collezioni"""
    try:
        collections = weaviate_manager.list_collections()
        return render_template('collections.html', collections=collections)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/collections')
def api_collections():
    """API endpoint per ottenere la lista delle collezioni"""
    if not client:
        return jsonify({
            'success': False,
            'error': 'Connessione Weaviate non disponibile'
        })
    
    try:
        collections = weaviate_manager.list_collections()
        return jsonify({
            'success': True,
            'collections': collections
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/collections/<collection_name>', methods=['DELETE'])
def delete_collection(collection_name):
    """API endpoint per eliminare una collezione"""
    if not client:
        return jsonify({
            'success': False,
            'message': 'Connessione Weaviate non disponibile'
        })
    
    try:
        success = weaviate_manager.delete_collection(collection_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'Collezione {collection_name} eliminata con successo'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Errore nell\'eliminazione della collezione {collection_name}'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Errore: {str(e)}'
        })


@app.route('/explore/<collection_name>')
def explore_collection(collection_name):
    """Pagina per esplorare i dati di una collezione"""
    if not client:
        return render_template('error.html', error="Connessione Weaviate non disponibile")
    
    try:
        # Ottieni dati campione della collezione
        collection_data = weaviate_manager.get_collection_sample_data(collection_name, limit=50)
        
        if 'error' in collection_data:
            return render_template('error.html', error=f"Errore nel caricamento della collezione: {collection_data['error']}")
        
        return render_template('explore_collection.html', 
                             collection_data=collection_data,
                             collection_name=collection_name)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/collections/<collection_name>/data')
def api_collection_data(collection_name):
    """API endpoint per ottenere dati paginati di una collezione"""
    if not client:
        return jsonify({'success': False, 'error': 'Connessione Weaviate non disponibile'})
    
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        offset = (page - 1) * limit
        
        # Per Weaviate, non c'è un offset diretto, ma possiamo simularlo
        # Per ora prendiamo più dati e filtriamo lato client
        collection_data = weaviate_manager.get_collection_sample_data(collection_name, limit=limit*page)
        
        # Simula la paginazione
        sample_data = collection_data['sample_data']
        paginated_data = sample_data[offset:offset+limit] if len(sample_data) > offset else []
        
        return jsonify({
            'success': True,
            'data': paginated_data,
            'total': collection_data['total_count'],
            'page': page,
            'limit': limit,
            'has_more': len(sample_data) == limit*page
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    flash(f'File troppo grande! Dimensione massima consentita: {max_size_mb}MB')
    return redirect(url_for('upload_documents'))

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Errore interno del server"), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Pagina non trovata"), 404

if __name__ == '__main__':
    # Controlla se Weaviate è disponibile
    if client is None:
        print("ATTENZIONE: Weaviate non è disponibile. Avvia il container Docker.")
        print("docker-compose up -d")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Weaviate connesso con successo!")
        print("Applicazione disponibile su: http://localhost:5000")
        
        try:
            app.run(debug=True, host='0.0.0.0', port=5000)
        finally:
            # Chiudi la connessione quando l'app viene fermata
            if client:
                client.close()
