import datetime 
from pathlib import Path
import sys 

# Directory Navigation
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logging"
SCHEMA_DIR = BASE_DIR / "schema"
PROMPT_DIR = BASE_DIR / "prompts"

# Models
# Embedder
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
AI_AGENT_MODEL = "llama2"

# RAG DATA
TEXT_DATASET = "freddie_mac.txt"