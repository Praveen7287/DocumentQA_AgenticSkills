"""
Configuration for Document Q&A Agent
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
SKILLS_DIR = BASE_DIR / "skills"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"

# Create directories
for directory in [SKILLS_DIR, DATA_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# Ollama settings
OLLAMA_MODEL = "qwen3:1.7b"  # Options: llama3.1, mistral, codellama, etc.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120

# Skill settings
SKILL_CACHE_ENABLED = True
SKILL_CACHE_EXPIRY = 3600  # seconds

# Document processing
MAX_DOCUMENT_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt', '.md', '.doc']

# Output settings
RESPONSE_MAX_TOKENS = 2000
ENABLE_RICH_OUTPUT = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "agent.log"