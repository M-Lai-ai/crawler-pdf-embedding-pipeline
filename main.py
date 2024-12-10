# main.py
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import logging
from pipeline.crawler import WebCrawler
from pipeline.pdf_doc_extractor import PDFExtractor
from pipeline.embedding_processor import EmbeddingProcessor
from pipeline.content_rewriter import ContentRewriter
from config import (
    PIPELINE_STEPS, 
    CRAWLER_OUTPUT_DIR, 
    PDF_DOC_OUTPUT_DIR, 
    EMBEDDING_OUTPUT_DIR, 
    CONTENT_REWRITER_OUTPUT_DIR,
    OPENAI_API_KEYS,
    LLM_PROVIDER,
    EMBEDDING_PROVIDER,
    VERBOSE,
    CONTENT_REWRITER_ENABLED,
    CONTENT_REWRITER_API_KEY,
    CONTENT_REWRITER_MODEL
)
from utils.event_manager import event_manager

# Initialiser l'application Flask et SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Changez ceci pour une clé secrète sécurisée
socketio = SocketIO(app, async_mode='eventlet')

# Initialiser le logger principal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction d'écoute des événements
def event_listener():
    while True:
        event = event_manager.get_event()
        if event:
            event_type = event.get('type')
            data = event.get('data')
            if event_type == 'log':
                level = data.get('level', 'info')
                message = data.get('message', '')
                socketio.emit('log', {'level': level, 'message': message})
            elif event_type == 'download':
                file_type = data.get('file_type')
                filename = data.get('filename')
                socketio.emit('download', {'file_type': file_type, 'filename': filename})
            elif event_type == 'progress':
                socketio.emit('progress', data)
            elif event_type == 'embedding_processed':
                socketio.emit('embedding_processed', data)
            elif event_type == 'content_extracted':
                socketio.emit('content_extracted', data)
            elif event_type == 'content_rewritten':
                socketio.emit('content_rewritten', data)
            elif event_type == 'crawl_completed':
                socketio.emit('crawl_completed', data)
        else:
            time.sleep(0.1)

# Route pour le tableau de bord
@app.route('/')
def index():
    return render_template('index.html')

# Démarrer le thread d'écoute des événements
listener_thread = threading.Thread(target=event_listener, daemon=True)
listener_thread.start()

# Fonction pour exécuter le pipeline
def run_pipeline():
    try:
        if "crawler" in PIPELINE_STEPS:
            crawler = WebCrawler(base_dir=CRAWLER_OUTPUT_DIR, resume=True)
            crawler.crawl()
        
        if "pdf_doc_extractor" in PIPELINE_STEPS:
            extractor = PDFExtractor(
                input_dir=CRAWLER_OUTPUT_DIR, 
                output_dir=PDF_DOC_OUTPUT_DIR, 
                openai_api_keys=OPENAI_API_KEYS, 
                llm_provider=LLM_PROVIDER, 
                verbose=VERBOSE
            )
            extractor.process_all_pdfs()
            extractor.process_all_docs()
        
        if "embedding" in PIPELINE_STEPS:
            embedding_processor = EmbeddingProcessor(
                input_dir=PDF_DOC_OUTPUT_DIR, 
                output_dir=EMBEDDING_OUTPUT_DIR, 
                openai_api_keys=OPENAI_API_KEYS, 
                llm_provider=LLM_PROVIDER, 
                embedding_provider=EMBEDDING_PROVIDER, 
                verbose=VERBOSE
            )
            embedding_processor.process_all_files()
        
        if "content_rewriter" in PIPELINE_STEPS and CONTENT_REWRITER_ENABLED:
            content_rewriter = ContentRewriter(
                input_dir=EMBEDDING_OUTPUT_DIR, 
                output_dir=CONTENT_REWRITER_OUTPUT_DIR, 
                api_key=CONTENT_REWRITER_API_KEY, 
                model=CONTENT_REWRITER_MODEL, 
                verbose=VERBOSE
            )
            content_rewriter.rewrite_all_contents()
        
        logger.info("Pipeline terminé avec succès.")
        event_manager.emit('log', {'level': 'info', 'message': "Pipeline terminé avec succès."})
        event_manager.emit('crawl_completed', {'duration_seconds': 0, 'status': 'success'})  # Vous pouvez ajuster la durée
    except Exception as e:
        logger.error(f"Erreur critique dans le pipeline : {str(e)}")
        event_manager.emit('log', {'level': 'error', 'message': f"Erreur critique dans le pipeline : {str(e)}"})
        event_manager.emit('crawl_completed', {'duration_seconds': 0, 'status': 'error', 'error': str(e)})

# Démarrer le pipeline après la connexion d'un client
@socketio.on('connect')
def handle_connect():
    logger.info("Client connecté")
    event_manager.emit('log', {'level': 'info', 'message': "Client connecté"})
    threading.Thread(target=run_pipeline, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
