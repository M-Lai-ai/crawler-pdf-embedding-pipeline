# config.py
import os
import yaml
from pathlib import Path

# Load the configuration file
CONFIG_FILE = Path(__file__).parent / "config.yaml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found.")

with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Pipeline Steps
PIPELINE_STEPS = config.get('pipeline_steps', ["crawler", "pdf_doc_extractor", "embedding"])

# Providers
LLM_PROVIDER = config.get('llm_provider', "openai")
EMBEDDING_PROVIDER = config.get('embedding_provider', "openai")

# API Keys
API_KEYS = config.get('api_keys', {})
OPENAI_API_KEYS = API_KEYS.get('openai_api_keys', [])
ANTHROPIC_API_KEY = API_KEYS.get('anthropic_api_key', "")
MISTRAL_API_KEY = API_KEYS.get('mistral_api_key', "")
VOYAGE_API_KEY = API_KEYS.get('voyage_api_key', "")

# Model and Tokens
MAX_TOKENS = config.get('model', {}).get('max_tokens', 10000)

# Directories
INPUT_DIR = config.get('directories', {}).get('input_dir', "input")
OUTPUT_DIR = config.get('directories', {}).get('output_dir', "output")
CRAWLER_OUTPUT_DIR = config.get('directories', {}).get('crawler_output_dir', os.path.join(OUTPUT_DIR, "crawler_output"))
PDF_DOC_OUTPUT_DIR = config.get('directories', {}).get('pdf_doc_output_dir', os.path.join(OUTPUT_DIR, "pdf_doc_extracted"))
EMBEDDING_OUTPUT_DIR = config.get('directories', {}).get('embedding_output_dir', os.path.join(OUTPUT_DIR, "embedding_output"))
CONTENT_REWRITER_OUTPUT_DIR = config.get('directories', {}).get('content_rewriter_output_dir', os.path.join(OUTPUT_DIR, "content_rewritten"))

# Crawler Parameters
CRAWLER_PARAMS = config.get('crawler_params', {})
START_URL = CRAWLER_PARAMS.get('start_url', "https://your-example-site.com/fr-ca/")
MAX_DEPTH = CRAWLER_PARAMS.get('max_depth', 1)
USE_PLAYWRIGHT = CRAWLER_PARAMS.get('use_playwright', False)
DOWNLOAD_PDF = CRAWLER_PARAMS.get('download_pdf', True)
DOWNLOAD_DOC = CRAWLER_PARAMS.get('download_doc', True)
DOWNLOAD_IMAGE = CRAWLER_PARAMS.get('download_image', False)
DOWNLOAD_OTHER = CRAWLER_PARAMS.get('download_other', False)
MAX_URLS = CRAWLER_PARAMS.get('max_urls', None)
EXCLUDED_PATHS = CRAWLER_PARAMS.get('excluded_paths', ['product-selector'])

# Checkpoint
CHECKPOINT_FILE = config.get('checkpoint_file', os.path.join(OUTPUT_DIR, "checkpoint.json"))

# Logging
VERBOSE = config.get('logging', {}).get('verbose', False)
LOG_LEVEL = config.get('logging', {}).get('log_level', "INFO")

# Content Rewriter
CONTENT_REWRITER_ENABLED = config.get('content_rewriter', {}).get('enabled', False)
CONTENT_REWRITER_MODEL = config.get('content_rewriter', {}).get('model', "openai")
CONTENT_REWRITER_API_KEY = config.get('content_rewriter', {}).get('api_key', "")
