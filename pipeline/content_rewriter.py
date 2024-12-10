# pipeline/content_rewriter.py
import os
import json
import requests
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from config import (
    CONTENT_REWRITER_OUTPUT_DIR,
    CONTENT_REWRITER_API_KEY,
    CONTENT_REWRITER_MODEL,
    VERBOSE
)
from utils.event_manager import event_manager

class ContentRewriter:
    def __init__(self, input_dir, output_dir, api_key, model="openai", verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key
        self.model = model.lower()
        self.verbose = verbose

        logging.basicConfig(
            level=logging.INFO if not verbose else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('content_rewriter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def rewrite_content(self, text, document_name, file_name):
        system_prompt = (
            "Vous êtes un expert en réécriture de contenu. Reformulez le texte ci-dessous pour améliorer sa clarté et sa lisibilité tout en conservant le sens original."
        )
        user_content = text

        if self.model == "openai":
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0,
                "max_tokens": 3000,
                "top_p": 1
            }
        elif self.model == "anthropic":
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-3.5",
                "max_tokens": 1024,
                "stop_sequences": [],
                "temperature": 0,
                "top_p": 0,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                "stream": False
            }
        elif self.model == "mistral":
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "model": "mistral-large-latest",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": 3000,
                "stream": False
            }
        else:
            self.logger.error(f"Fournisseur LLM inconnu : {self.model}")
            event_manager.emit('log', {'level': 'error', 'message': f"Fournisseur LLM inconnu : {self.model}"})
            return None

        if self.verbose:
            self.logger.info(f"Appel LLM {self.model} pour {document_name} : {payload}")
            event_manager.emit('log', {'level': 'info', 'message': f"Appel LLM {self.model} pour {document_name}"})

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"Erreur API LLM {self.model} : {str(e)}, réponse : {response.text}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur API LLM {self.model} : {str(e)}, réponse : {response.text}"})
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de l'appel LLM {self.model} : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de l'appel LLM {self.model} : {str(e)}"})
            return None

        response_json = response.json()
        if self.model == "openai":
            rewritten_text = response_json['choices'][0]['message']['content']
        elif self.model == "anthropic":
            content_parts = response_json.get("content", [])
            rewritten_text = "".join([part["text"] for part in content_parts if part["type"] == "text"])
        elif self.model == "mistral":
            rewritten_text = response_json['choices'][0]['message']['content']
        else:
            rewritten_text = ""

        return rewritten_text

    def rewrite_file(self, file_path):
        document_name = file_path.stem
        self.logger.info(f"Réécriture du contenu de {file_path.name}")
        event_manager.emit('log', {'level': 'info', 'message': f"Réécriture du contenu de {file_path.name}"})

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            self.logger.warning(f"Aucun texte à réécrire dans {file_path.name}")
            event_manager.emit('log', {'level': 'warning', 'message': f"Aucun texte à réécrire dans {file_path.name}"})
            return False

        rewritten_text = self.rewrite_content(text, document_name, file_path.name)
        if not rewritten_text:
            self.logger.warning(f"Réécriture échouée pour {file_path.name}")
            event_manager.emit('log', {'level': 'warning', 'message': f"Réécriture échouée pour {file_path.name}"})
            return False

        output_file_name = self.output_dir / f"{document_name}_rewritten.txt"
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write(rewritten_text)
            self.logger.info(f"Fichier réécrit créé : {output_file_name}")
            event_manager.emit('log', {'level': 'info', 'message': f"Fichier réécrit créé : {output_file_name}"})
            event_manager.emit('content_rewritten', {'filename': output_file_name.name})
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de {file_path} : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la sauvegarde de {file_path} : {str(e)}"})
            return False
        return True

    def rewrite_all_contents(self):
        txt_files = list(self.input_dir.glob('*.txt'))
        total_files = len(txt_files)
        self.logger.info(f"Démarrage de la réécriture de {total_files} fichier(s)")
        event_manager.emit('log', {'level': 'info', 'message': f"Démarrage de la réécriture de {total_files} fichier(s)"})

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.rewrite_file, txt_file_path): txt_file_path for txt_file_path in txt_files}

            for future in as_completed(futures):
                txt_file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.logger.info(f"Réécriture réussie pour {txt_file_path.name}")
                        event_manager.emit('log', {'level': 'info', 'message': f"Réécriture réussie pour {txt_file_path.name}"})
                except Exception as e:
                    self.logger.error(f"Erreur lors de la réécriture de {txt_file_path.name} : {str(e)}")
                    event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la réécriture de {txt_file_path.name} : {str(e)}"})
        self.logger.info("Réécriture de contenu terminée.")
        event_manager.emit('log', {'level': 'info', 'message': "Réécriture de contenu terminée."})
