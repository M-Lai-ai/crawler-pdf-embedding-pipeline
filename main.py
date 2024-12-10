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
from config import PIPELINE_STEPS, CRAWLER_OUTPUT_DIR, PDF_DOC_OUTPUT_DIR, EMBEDDING_OUTPUT_DIR, CONTENT_REWRITER_OUTPUT_DIR, CONTENT_REWRITER_ENABLED
from utils.event_manager import event_manager

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Initialize logger for main
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Event listener function
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
            elif event_type == 'crawl_completed':
                socketio.emit('crawl_completed', data)
        else:
            time.sleep(0.1)

# Define the dashboard route
@app.route('/')
def index():
    return render_template('index.html')

# Start the event listener thread
listener_thread = threading.Thread(target=event_listener, daemon=True)
listener_thread.start()

# Function to run the pipeline
def run_pipeline():
    crawler = WebCrawler(base_dir=CRAWLER_OUTPUT_DIR, resume=True)
    extractor = PDFExtractor(input_dir=CRAWLER_OUTPUT_DIR, output_dir=PDF_DOC_OUTPUT_DIR, openai_api_keys=OPENAI_API_KEYS, llm_provider=LLM_PROVIDER, verbose=VERBOSE)
    embedding_processor = EmbeddingProcessor(input_dir=PDF_DOC_OUTPUT_DIR, output_dir=EMBEDDING_OUTPUT_DIR, openai_api_keys=OPENAI_API_KEYS, llm_provider=LLM_PROVIDER, embedding_provider=EMBEDDING_PROVIDER, verbose=VERBOSE)
    content_rewriter = ContentRewriter(input_dir=EMBEDDING_OUTPUT_DIR, output_dir=CONTENT_REWRITER_OUTPUT_DIR, api_key=CONTENT_REWRITER_API_KEY, model=CONTENT_REWRITER_MODEL, verbose=VERBOSE)

    for step in PIPELINE_STEPS:
        if step == "crawler":
            crawler.crawl()
        elif step == "pdf_doc_extractor":
            extractor.process_all_pdfs()
            extractor.process_all_docs()
        elif step == "embedding":
            embedding_processor.process_all_files()
        elif step == "content_rewriter" and CONTENT_REWRITER_ENABLED:
            content_rewriter.rewrite_all_contents()
        else:
            logger.warning(f"Unknown or disabled step: {step}")

    logger.info("Pipeline completed successfully.")
    event_manager.emit('log', {'level': 'info', 'message': "Pipeline completed successfully."})

# Start the pipeline in a separate thread after the server starts
@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    event_manager.emit('log', {'level': 'info', 'message': "Client connected"})
    threading.Thread(target=run_pipeline, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
