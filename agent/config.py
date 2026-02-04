import datetime 
from pathlib import Path
import sys 
from loguru import logger

# Directory Navigation
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logging"
SCHEMA_DIR = BASE_DIR / "schema"
PROMPT_DIR = BASE_DIR / "prompts"
EVALUATION_DIR = BASE_DIR / "evaluation"

# Models
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
AI_AGENT_MODEL = "llama2"

# RAG DATA
TEXT_DATASET = "freddie_mac.txt"

# LOGGING 
LOGGING_FILE_NAME = "llm_logs.log"

# initialize logger
log_format = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}"
)

logger.remove(0)
logger.level("COSTS", no=15, color = "<yellow>", icon="$")
logger.add(
    sys.stderr,
    level="INFO",
    format = log_format
)
logger.add(
    LOG_DIR / LOGGING_FILE_NAME,
    format = log_format,
    level="TRACE"
)