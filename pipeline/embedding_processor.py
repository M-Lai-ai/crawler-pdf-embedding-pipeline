# pipeline/embedding_processor.py
import os
import json
import numpy as np
import requests
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
import threading
from config import (
    EMBEDDING_OUTPUT_DIR,
    OPENAI_API_KEYS,
    EMBEDDING_LLM_PROVIDER,
    EMBEDDING_PROVIDER,
    VERBOSE,
    MAX_TOKENS,
    CHECKPOINT_FILE
)
from utils.event_manager import event_manager

class EmbeddingProcessor:
    def __init__(self, input_dir, output_dir, openai_api_keys, llm_provider="openai", embedding_provider="openai", verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_embeddings = []
        self.all_metadata = []
        
        self.llm_provider = llm_provider.lower()
        self.embedding_provider = embedding_provider.lower()

        self.openai_api_keys = openai_api_keys
        self.headers_cycle = cycle([
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            } for key in self.openai_api_keys
        ])
        self.lock = threading.Lock()

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY", "")

        logging.basicConfig(
            level=logging.INFO if not verbose else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('embedding_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.verbose = verbose

    def chunk_text(self, text, chunk_size=400, overlap_size=100):
        tokens = text.split(' ')
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap_size):
            chunk = ' '.join(tokens[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def get_contextualized_chunk(self, chunk, full_text, headers, document_name, page_num, chunk_id):
        system_content = (
            "Vous êtes un analyste expert chargé de fournir un contexte détaillé pour un extrait de texte."
        )
        user_content = f"Document : {full_text}\n\nExtrait : {chunk}\n\nVeuillez fournir un contexte pour cet extrait."

        if self.llm_provider == "openai":
            endpoint = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0,
                "max_tokens": 200,
                "top_p": 1
            }
        elif self.llm_provider == "anthropic":
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-3.5",
                "max_tokens": 1024,
                "stop_sequences": [],
                "temperature": 0,
                "top_p": 0,
                "system": system_content,
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                "stream": False
            }
        elif self.llm_provider == "mistral":
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "model": "mistral-large-latest",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": 200,
                "stream": False
            }
        else:
            self.logger.error(f"Fournisseur LLM inconnu : {self.llm_provider}")
            event_manager.emit('log', {'level': 'error', 'message': f"Fournisseur LLM inconnu : {self.llm_provider}"})
            return None

        if self.verbose:
            self.logger.info(f"Appel LLM context {document_name} p{page_num} c{chunk_id} : {payload}")
            event_manager.emit('log', {'level': 'info', 'message': f"Appel LLM context {document_name} p{page_num} c{chunk_id}"})

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"Erreur API LLM context {self.llm_provider} : {str(e)}, {response.text}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur API LLM context {self.llm_provider} : {str(e)}, {response.text}"})
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de l'appel LLM context : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de l'appel LLM context : {str(e)}"})
            return None

        response_json = response.json()
        if self.llm_provider == "openai":
            context = response_json['choices'][0]['message']['content']
        elif self.llm_provider == "anthropic":
            content_parts = response_json.get("content", [])
            context = "".join([part["text"] for part in content_parts if part["type"] == "text"])
        elif self.llm_provider == "mistral":
            context = response_json['choices'][0]['message']['content']
        else:
            context = ""

        return context

    def get_embedding(self, text, headers, document_name, page_num, chunk_id):
        if self.embedding_provider == "openai":
            endpoint = 'https://api.openai.com/v1/embeddings'
            payload = {
                "input": text,
                "model": "text-embedding-ada-002",
                "encoding_format": "float"
            }
        elif self.embedding_provider == "mistral":
            endpoint = "https://api.mistral.ai/v1/embeddings"
            payload = {
                "input": [text],
                "model": "mistral-embed",
                "encoding_format": "float"
            }
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
        elif self.embedding_provider == "voyage":
            endpoint = "https://api.voyageai.com/v1/embeddings"
            payload = {
                "input": [text],
                "model": "voyage-large-2"
            }
            headers = {
                "Authorization": f"Bearer {self.voyage_api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.logger.error(f"Fournisseur embedding inconnu : {self.embedding_provider}")
            event_manager.emit('log', {'level': 'error', 'message': f"Fournisseur embedding inconnu : {self.embedding_provider}"})
            return None

        if self.verbose:
            self.logger.info(f"Appel embedding {self.embedding_provider} p{page_num} c{chunk_id} : {payload}")
            event_manager.emit('log', {'level': 'info', 'message': f"Appel embedding {self.embedding_provider} p{page_num} c{chunk_id}"})

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"Erreur API Embedding {self.embedding_provider} : {str(e)}, {response.text}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur API Embedding {self.embedding_provider} : {str(e)}, {response.text}"})
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de l'appel Embedding : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de l'appel Embedding : {str(e)}"})
            return None

        response_json = response.json()
        if self.embedding_provider in ["openai", "mistral", "voyage"]:
            embedding = response_json['data'][0]['embedding']
        else:
            embedding = None

        return embedding

    def process_chunk(self, chunk_info):
        txt_file_path, chunk_id, chunk, full_text, document_name, page_num = chunk_info

        with self.lock:
            chosen_headers = next(self.headers_cycle) if (self.llm_provider == "openai" and self.embedding_provider == "openai") else {}

        headers_for_llm = chosen_headers if self.llm_provider == "openai" else {}
        context = self.get_contextualized_chunk(chunk, full_text, headers_for_llm, document_name, page_num, chunk_id)
        if context:
            combined_text = f"{context}\n\nContext:\n{chunk}"
            headers_for_embedding = chosen_headers if self.embedding_provider == "openai" else {}
            embedding = self.get_embedding(combined_text, headers_for_embedding, document_name, page_num, chunk_id)
            if embedding:
                metadata = {
                    "filename": txt_file_path.name,
                    "chunk_id": chunk_id,
                    "text_raw": chunk,
                    "context": context,
                    "text": combined_text
                }
                return (embedding, metadata)
        return (None, None)

    def process_file(self, txt_file_path):
        self.logger.info(f"Traitement embedding : {txt_file_path}")
        event_manager.emit('log', {'level': 'info', 'message': f"Traitement embedding : {txt_file_path}"})
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()

        chunks = self.chunk_text(full_text)
        chunk_infos = [
            (txt_file_path, i_page, chunk, full_text, txt_file_path.stem, i_page)
            for i_page, chunk in enumerate(chunks, 1)
        ]
        return chunk_infos

    def process_all_files(self):
        txt_files = list(self.input_dir.glob('*.txt'))
        total_files = len(txt_files)
        self.logger.info(f"Démarrage du traitement embedding de {total_files} fichiers")
        event_manager.emit('log', {'level': 'info', 'message': f"Démarrage du traitement embedding de {total_files} fichiers"})
        self.all_embeddings = []
        self.all_metadata = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i, txt_file_path in enumerate(txt_files, 1):
                self.logger.info(f"Fichier {i}/{total_files} : {txt_file_path.name}")
                event_manager.emit('log', {'level': 'info', 'message': f"Fichier {i}/{total_files} : {txt_file_path.name}"})
                chunk_infos = self.process_file(txt_file_path)
                for chunk_info in chunk_infos:
                    futures.append(executor.submit(self.process_chunk, chunk_info))

            for future in as_completed(futures):
                embedding, metadata = future.result()
                if embedding and metadata:
                    self.all_embeddings.append(embedding)
                    self.all_metadata.append(metadata)
                    event_manager.emit('embedding_processed', {'filename': metadata['filename'], 'chunk_id': metadata['chunk_id']})

        if self.all_embeddings:
            chunks_json_path = self.output_dir / "chunks.json"
            try:
                with open(chunks_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump({
                        "metadata": self.all_metadata
                    }, json_file, ensure_ascii=False, indent=4)
                self.logger.info(f"Fichier JSON : {chunks_json_path}")
                event_manager.emit('log', {'level': 'info', 'message': f"Fichier JSON : {chunks_json_path}"})
            except Exception as e:
                self.logger.error(f"Erreur lors de la sauvegarde du fichier JSON : {str(e)}")
                event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la sauvegarde du fichier JSON : {str(e)}"})

            embeddings_npy_path = self.output_dir / "embeddings.npy"
            try:
                np.save(embeddings_npy_path, np.array(self.all_embeddings))
                self.logger.info(f"Fichier NPY : {embeddings_npy_path}")
                event_manager.emit('log', {'level': 'info', 'message': f"Fichier NPY : {embeddings_npy_path}"})
            except Exception as e:
                self.logger.error(f"Erreur lors de la sauvegarde du fichier NPY : {str(e)}")
                event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la sauvegarde du fichier NPY : {str(e)}"})

        self.logger.info("Traitement embedding terminé.")
        event_manager.emit('log', {'level': 'info', 'message': "Traitement embedding terminé."})
