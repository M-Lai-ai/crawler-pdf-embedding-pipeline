# pipeline/pdf_doc_extractor.py
import os
import logging
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import cv2
from PIL import Image
import pypdf
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import subprocess
from docx import Document
from pathlib import Path
from itertools import cycle
from config import (
    PDF_DOC_LLM_PROVIDER,
    OPENAI_API_KEYS,
    PDF_DOC_OUTPUT_DIR,
    VERBOSE,
    MAX_TOKENS,
    CHECKPOINT_FILE
)
import yaml
from utils.event_manager import event_manager

def split_text_into_chunks(text, max_tokens=10000):
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    except KeyError:
        raise ValueError("Encodage 'cl100k_base' non trouvé.")
    
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

class PDFExtractor:
    def __init__(self, input_dir, output_dir, openai_api_keys, llm_provider="openai", verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_provider = llm_provider.lower()
        self.openai_api_keys = openai_api_keys
        self.api_key_cycle = cycle(self.openai_api_keys)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY", "")
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO if not verbose else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pdf_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.verbose = verbose

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)

        if img is None:
            raise ValueError("Image est None.")
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def extract_text_with_ocr(self, pdf_path):
        try:
            images = convert_from_path(pdf_path)
            ocr_texts = []
            for i, image in enumerate(images, 1):
                self.logger.info(f"OCR page {i}/{len(images)}")
                event_manager.emit('log', {'level': 'info', 'message': f"OCR page {i}/{len(images)}"})

                temp_path = self.temp_dir / f"temp_{i}.png"
                try:
                    image.save(temp_path)
                except Exception as e:
                    self.logger.error(f"Erreur enregistrement image {temp_path} : {str(e)}")
                    event_manager.emit('log', {'level': 'error', 'message': f"Erreur enregistrement image {temp_path} : {str(e)}"})
                    ocr_texts.append("")
                    continue

                img = cv2.imread(str(temp_path))
                if img is None:
                    self.logger.error(f"cv2.imread a échoué pour {temp_path}")
                    event_manager.emit('log', {'level': 'error', 'message': f"cv2.imread a échoué pour {temp_path}"})
                    ocr_texts.append("")
                    temp_path.unlink(missing_ok=True)
                    continue

                try:
                    processed_img = self.preprocess_image(img)
                except ValueError as ve:
                    self.logger.error(f"Erreur prétraitement image {temp_path} : {str(ve)}")
                    event_manager.emit('log', {'level': 'error', 'message': f"Erreur prétraitement image {temp_path} : {str(ve)}"})
                    ocr_texts.append("")
                    temp_path.unlink(missing_ok=True)
                    continue

                try:
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang='fra+eng',
                        config='--psm 1'
                    )
                    if len(text.strip()) < 100:
                        text = pytesseract.image_to_string(
                            processed_img,
                            lang='fra+eng',
                            config='--psm 3 --oem 1'
                        )
                    ocr_texts.append(text)
                except Exception as e:
                    self.logger.error(f"Erreur OCR {temp_path} : {str(e)}")
                    event_manager.emit('log', {'level': 'error', 'message': f"Erreur OCR {temp_path} : {str(e)}"})
                    ocr_texts.append("")
                
                temp_path.unlink(missing_ok=True)

            return ocr_texts
        except Exception as e:
            self.logger.error(f"Erreur OCR : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur OCR : {str(e)}"})
            return []

    def extract_text_with_pypdf(self, pdf_path):
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    self.logger.info(f"Extraction PyPDF page {page_num}")
                    event_manager.emit('log', {'level': 'info', 'message': f"Extraction PyPDF page {page_num}"})
                    text = page.extract_text() or ''
                    text_content.append(text)
            return text_content
        except Exception as e:
            self.logger.error(f"Erreur PyPDF : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur PyPDF : {str(e)}"})
            return []

    def process_with_llm(self, content):
        system_prompt = (
            "Vous êtes un analyste expert chargé de reformuler le contenu extrait pour améliorer sa clarté et sa lisibilité."
        )

        chunks = split_text_into_chunks(content, max_tokens=3000)
        processed_contents = []

        for idx, chunk in enumerate(chunks, 1):
            if self.llm_provider == "openai":
                headers = {
                    "Authorization": f"Bearer {next(self.api_key_cycle)}",
                    "Content-Type": "application/json"
                }
                payload = {
                  "model": "gpt-4",
                  "messages": [
                    {
                      "role": "system",
                      "content": system_prompt
                    },
                    {
                      "role": "user",
                      "content": chunk
                    }
                  ],
                  "temperature": 0,
                  "max_tokens": 13000,
                  "top_p": 1,
                  "frequency_penalty": 0,
                  "presence_penalty": 0,
                  "stream": False
                }
                endpoint = "https://api.openai.com/v1/chat/completions"
            elif self.llm_provider == "anthropic":
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
                  "system": system_prompt,
                  "messages": [
                    {
                      "role": "user",
                      "content": chunk
                    }
                  ],
                  "stream": False
                }
                endpoint = "https://api.anthropic.com/v1/messages"
            elif self.llm_provider == "mistral":
                headers = {
                    "Authorization": f"Bearer {self.mistral_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                payload = {
                    "model": "mistral-large-latest",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    "temperature": 0.7,
                    "top_p": 1,
                    "max_tokens": 100,
                    "stream": False
                }
                endpoint = "https://api.mistral.ai/v1/chat/completions"
            else:
                self.logger.error(f"Fournisseur LLM inconnu : {self.llm_provider}")
                event_manager.emit('log', {'level': 'error', 'message': f"Fournisseur LLM inconnu : {self.llm_provider}"})
                return None

            if self.verbose:
                self.logger.info(f"Appel LLM segment {idx}/{len(chunks)} : {payload}")
                event_manager.emit('log', {'level': 'info', 'message': f"Appel LLM segment {idx}/{len(chunks)}"})

            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
            except requests.HTTPError as e:
                self.logger.error(f"Erreur API LLM : {str(e)}, réponse : {response.text}")
                event_manager.emit('log', {'level': 'error', 'message': f"Erreur API LLM : {str(e)}, réponse : {response.text}"})
                return None
            except Exception as e:
                self.logger.error(f"Erreur lors de l'appel LLM : {str(e)}")
                event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de l'appel LLM : {str(e)}"})
                return None

            response_json = response.json()
            if self.llm_provider == "openai":
                processed_content = response_json['choices'][0]['message']['content']
            elif self.llm_provider == "anthropic":
                content_parts = response_json.get("content", [])
                processed_content = "".join([part["text"] for part in content_parts if part["type"] == "text"])
            elif self.llm_provider == "mistral":
                processed_content = response_json['choices'][0]['message']['content']
            else:
                processed_content = ""

            processed_contents.append(processed_content)
            time.sleep(1)  # Pour éviter les limitations d'API

        return "\n\n".join(processed_contents)

    def process_pdf(self, pdf_path):
        document_name = pdf_path.stem
        self.logger.info(f"Traitement de {pdf_path}")
        event_manager.emit('log', {'level': 'info', 'message': f"Traitement de {pdf_path}"})

        ocr_texts = self.extract_text_with_ocr(pdf_path)
        pypdf_texts = self.extract_text_with_pypdf(pdf_path)
        num_pages = max(len(ocr_texts), len(pypdf_texts))
        
        processed_contents = []
        for page_num in range(num_pages):
            self.logger.info(f"Traitement page {page_num + 1}")
            event_manager.emit('log', {'level': 'info', 'message': f"Traitement page {page_num + 1}"})
            page_text = ""
            if page_num < len(ocr_texts):
                page_text += ocr_texts[page_num] + "\n\n"
            if page_num < len(pypdf_texts):
                page_text += pypdf_texts[page_num]

            if not page_text.strip():
                self.logger.warning(f"Aucun texte extrait pour la page {page_num + 1}")
                event_manager.emit('log', {'level': 'warning', 'message': f"Aucun texte extrait pour la page {page_num + 1}"})
                continue
            
            processed_content = self.process_with_llm(page_text)
            if processed_content:
                processed_contents.append(processed_content)
                event_manager.emit('progress', {'type': 'content_processed', 'page_num': page_num + 1, 'document': document_name})

        if not processed_contents:
            self.logger.warning(f"Aucun contenu traité pour {pdf_path}")
            event_manager.emit('log', {'level': 'warning', 'message': f"Aucun contenu traité pour {pdf_path}"})
            return False

        final_content = "\n\n".join(processed_contents)
        output_file_name = self.output_dir / f"{document_name}.txt"
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write(f"Document ID : {document_name}\n\n{final_content}")
            self.logger.info(f"Fichier créé : {output_file_name}")
            event_manager.emit('log', {'level': 'info', 'message': f"Fichier créé : {output_file_name}"})
            event_manager.emit('content_extracted', {'filename': output_file_name.name})
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de {pdf_path} : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la sauvegarde de {pdf_path} : {str(e)}"})
            return False
        return True

    def convert_doc_to_txt(self, input_path, output_path):
        try:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_file = os.path.join(output_path, f"{base_name}.txt")
            result = subprocess.run(['antiword', input_path], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"antiword a échoué : {result.stderr}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            self.logger.info(f"Converti (DOC) : {input_path}")
            event_manager.emit('log', {'level': 'info', 'message': f"Converti (DOC) : {input_path}"})
            return output_file
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion de {input_path} : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la conversion de {input_path} : {str(e)}"})
            return None

    def convert_docx_to_txt(self, input_path, output_path):
        try:
            doc = Document(input_path)
            text = '\n'.join([p.text for p in doc.paragraphs])
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_file = os.path.join(output_path, f"{base_name}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"Converti (DOCX) : {input_path}")
            event_manager.emit('log', {'level': 'info', 'message': f"Converti (DOCX) : {input_path}"})
            return output_file
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion de {input_path} : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la conversion de {input_path} : {str(e)}"})
            return None

    def process_doc_file(self, doc_path):
        document_name = doc_path.stem
        self.logger.info(f"Traitement de {doc_path}")
        event_manager.emit('log', {'level': 'info', 'message': f"Traitement de {doc_path}"})
        
        converted_file = None
        if doc_path.suffix.lower() == '.doc':
            try:
                subprocess.run(['antiword', '-h'], capture_output=True)
            except FileNotFoundError:
                self.logger.error("antiword non installé.")
                event_manager.emit('log', {'level': 'error', 'message': "antiword non installé."})
                return False
            converted_file = self.convert_doc_to_txt(str(doc_path), str(self.output_dir))
        elif doc_path.suffix.lower() == '.docx':
            converted_file = self.convert_docx_to_txt(str(doc_path), str(self.output_dir))
        else:
            self.logger.error(f"Format non supporté : {doc_path.suffix}")
            event_manager.emit('log', {'level': 'error', 'message': f"Format non supporté : {doc_path.suffix}"})
            return False
        
        if not converted_file or not os.path.exists(converted_file):
            self.logger.error(f"Fichier non converti : {doc_path}")
            event_manager.emit('log', {'level': 'error', 'message': f"Fichier non converti : {doc_path}"})
            return False
        
        with open(converted_file, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            self.logger.warning(f"Aucun texte extrait de {doc_path}")
            event_manager.emit('log', {'level': 'warning', 'message': f"Aucun texte extrait de {doc_path}"})
            return False
        
        processed_content = self.process_with_llm(text)
        if not processed_content:
            self.logger.warning(f"Aucun contenu LLM pour {doc_path}")
            event_manager.emit('log', {'level': 'warning', 'message': f"Aucun contenu LLM pour {doc_path}"})
            return False
        
        output_file_name = self.output_dir / f"{document_name}_rewritten.txt"
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write(f"Document ID : {document_name}\n\n{processed_content}")
            self.logger.info(f"Fichier créé : {output_file_name}")
            event_manager.emit('log', {'level': 'info', 'message': f"Fichier créé : {output_file_name}"})
            event_manager.emit('content_rewritten', {'filename': output_file_name.name})
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de {doc_path} : {str(e)}")
            event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors de la sauvegarde de {doc_path} : {str(e)}"})
            return False
        return True

    def process_all_pdfs(self):
        pdf_files = list(self.input_dir.glob('*.pdf'))
        total_files = len(pdf_files)
        self.logger.info(f"Démarrage du traitement de {total_files} PDF(s)")
        event_manager.emit('log', {'level': 'info', 'message': f"Démarrage du traitement de {total_files} PDF(s)"})
        successful = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_pdf = {executor.submit(self.process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                except Exception as e:
                    self.logger.error(f"Erreur lors du traitement de {pdf_path} : {str(e)}")
                    event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors du traitement de {pdf_path} : {str(e)}"})
        self.logger.info(f"Terminé. {successful}/{total_files} PDF traités avec succès.")
        event_manager.emit('log', {'level': 'info', 'message': f"Terminé. {successful}/{total_files} PDF traités avec succès."})

    def process_all_docs(self):
        doc_files = list(self.input_dir.glob('*.doc')) + list(self.input_dir.glob('*.docx'))
        total_files = len(doc_files)
        
        if total_files == 0:
            self.logger.info("Aucun fichier DOC/DOCX à traiter.")
            event_manager.emit('log', {'level': 'info', 'message': "Aucun fichier DOC/DOCX à traiter."})
            return
        
        self.logger.info(f"Démarrage du traitement de {total_files} fichier(s) DOC/DOCX")
        event_manager.emit('log', {'level': 'info', 'message': f"Démarrage du traitement de {total_files} fichier(s) DOC/DOCX"})
        successful = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_doc = {executor.submit(self.process_doc_file, doc_path): doc_path for doc_path in doc_files}
            for future in as_completed(future_to_doc):
                doc_path = future_to_doc[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                except Exception as e:
                    self.logger.error(f"Erreur lors du traitement de {doc_path} : {str(e)}")
                    event_manager.emit('log', {'level': 'error', 'message': f"Erreur lors du traitement de {doc_path} : {str(e)}"})
        self.logger.info(f"Terminé. {successful}/{total_files} DOC/DOCX traités avec succès.")
        event_manager.emit('log', {'level': 'info', 'message': f"Terminé. {successful}/{total_files} DOC/DOCX traités avec succès."})
