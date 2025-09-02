from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import weaviate
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import traceback

# Import dei moduli personalizzati
from modules_new import QASystem, DataAnalyzer, DataCleaner, DataIntegrator, KnowledgeExtractor, WeaviateManager

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000MB max file size (aumentato da 50MB)

# Assicurati che la cartella uploads esista
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inizializza Weaviate client
try:
    client = weaviate.connect_to_local()
    weaviate_manager = WeaviateManager(client)
    
    # Inizializza i sistemi
    qa_system = QASystem(client)
    analyzer = DataAnalyzer(client)
    cleaner = DataCleaner(client)
    integrator = DataIntegrator(client)
    extractor = KnowledgeExtractor(client)
    
    # Crea lo schema se non esistente
    weaviate_manager.setup_schema()
    
except Exception as e:
    print(f"Errore connessione Weaviate: {e}")
    client = None

@app.route('/')
def index():
    """Homepage con dashboard intelligente"""
    if not client:
        return render_template('error.html', error="Connessione Weaviate non disponibile")
    
    try:
        # Prima controlla se esistono collezioni
        collections = weaviate_manager.list_collections()
        
        # Se non ci sono collezioni, mostra schermata di benvenuto
        if not collections:
            return render_template('index.html', 
                                 show_welcome=True, 
                                 collections=[],
                                 stats=None)
        
        # Controlla se ci sono documenti nelle collezioni
        total_documents = sum(col.get('count', 0) for col in collections)
        
        # Se non ci sono documenti, mostra schermata di avvio
        if total_documents == 0:
            return render_template('index.html', 
                                 show_empty_collections=True, 
                                 collections=collections,
                                 stats=None)
        
        # Se ci sono dati, mostra le statistiche complete
        stats = analyzer.get_basic_stats()
        return render_template('index.html', 
                             show_dashboard=True,
                             collections=collections,
                             stats=stats)
        
    except Exception as e:
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

# Nel tuo file app.py o routes.py

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
        
        # Validazione input
        if not question:
            flash('Inserisci una domanda')
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections)
        
        if not collection_name:
            flash('Seleziona una collezione')
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections)
        
        try:
            # Verifica che la collezione esista
            collection_names = [col.get('name') for col in collections]
            if collection_name not in collection_names:
                flash(f'Collezione "{collection_name}" non trovata')
                return render_template('qa.html', 
                                     question=question, 
                                     collection=collection_name,
                                     collections=collections)
            
            # Usa il sistema QA per ottenere la risposta
            answer = qa_system.ask_question(question, collection_name)
            
            if not answer or not answer.get('answer'):
                flash('Nessuna risposta trovata per la tua domanda')
            
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections,
                                 answer=answer)
                                 
        except Exception as e:
            print(f"Errore durante la ricerca: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            flash(f'Errore nella ricerca: {str(e)}')
            return render_template('qa.html', 
                                 question=question, 
                                 collection=collection_name,
                                 collections=collections)
    
    return render_template('qa.html', collections=collections)

@app.route('/analyze')
def analyze_data():
    """Analisi dei dati"""
    try:
        collection = request.args.get('collection', 'Documents')
        
        # Analisi completa
        stats = analyzer.get_basic_stats(collection)
        clusters = analyzer.analyze_clusters(collection)
        topics = analyzer.extract_topics(collection)
        
        analysis_result = {
            'stats': stats,
            'clusters': clusters,
            'topics': topics,
            'timestamp': datetime.now().isoformat()
        }
        
        return render_template('analyze.html', analysis=analysis_result)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/clean', methods=['GET', 'POST'])
def clean_data():
    """Pulizia dei dati"""
    if request.method == 'POST':
        try:
            action = request.form.get('action')
            collection = request.form.get('collection', 'Documents')
            
            if action == 'find_duplicates':
                threshold = float(request.form.get('threshold', 0.95))
                duplicates = cleaner.find_duplicates(collection, threshold)
                return render_template('clean.html', duplicates=duplicates)
                
            elif action == 'remove_low_quality':
                removed_count = cleaner.remove_low_quality_content(collection)
                flash(f'Rimossi {removed_count} documenti di bassa qualità')
                
            elif action == 'remove_duplicates':
                duplicate_ids = request.form.getlist('duplicate_ids')
                removed_count = cleaner.remove_duplicates(duplicate_ids)
                flash(f'Rimossi {removed_count} duplicati')
                
            return redirect(url_for('clean_data'))
            
        except Exception as e:
            flash(f'Errore: {str(e)}')
    
    return render_template('clean.html')

@app.route('/integrate', methods=['GET', 'POST'])
def integrate_data():
    """Integrazione dati esterni"""
    if request.method == 'POST':
        try:
            if 'file' in request.files:
                file = request.files['file']
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    result = integrator.integrate_external_file(filepath)
                    flash(f'Integrati {result["integrated"]} nuovi documenti')
                    
                    os.remove(filepath)
            
            elif 'api_url' in request.form:
                api_url = request.form.get('api_url')
                result = integrator.integrate_from_api(api_url)
                flash(f'Integrati {result["integrated"]} documenti da API')
                
            return redirect(url_for('integrate_data'))
            
        except Exception as e:
            flash(f'Errore: {str(e)}')
    
    return render_template('integrate.html')

@app.route('/extract')
def extract_knowledge():
    """Estrazione conoscenza"""
    try:
        collection = request.args.get('collection', 'Documents')
        
        # Estrazione entità
        entities = extractor.extract_entities(collection)
        
        # Estrazione relazioni
        relations = extractor.extract_relations(collection)
        
        # Topic modeling
        topics = extractor.extract_topics(collection)
        
        # Keyword extraction
        keywords = extractor.extract_keywords(collection)
        
        extraction_result = {
            'entities': entities,
            'relations': relations,
            'topics': topics,
            'keywords': keywords,
            'timestamp': datetime.now().isoformat()
        }
        
        return render_template('extract.html', extraction=extraction_result)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/search')
def api_search():
    """API endpoint per ricerca"""
    try:
        query = request.args.get('q')
        collection = request.args.get('collection', 'Documents')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query richiesta'}), 400
        
        results = qa_system.search_documents(query, collection, limit)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint per statistiche"""
    try:
        collection = request.args.get('collection', 'Documents')
        stats = analyzer.get_basic_stats(collection)
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/collections')
def manage_collections():
    """Gestione collezioni"""
    try:
        collections = weaviate_manager.list_collections()
        return render_template('collections.html', collections=collections)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/collections/create', methods=['POST'])
def create_collection():
    """Crea nuova collezione"""
    try:
        name = request.form.get('name')
        properties = request.form.get('properties', '[]')
        
        if not name:
            flash('Nome collezione richiesto')
            return redirect(url_for('manage_collections'))
        
        properties = json.loads(properties)
        result = weaviate_manager.create_collection(name, properties)
        
        if result:
            flash(f'Collezione {name} creata con successo')
        else:
            flash('Errore nella creazione della collezione')
            
        return redirect(url_for('manage_collections'))
        
    except Exception as e:
        flash(f'Errore: {str(e)}')
        return redirect(url_for('manage_collections'))

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
                print("Connessione Weaviate chiusa.")