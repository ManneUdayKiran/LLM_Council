"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Council members - list of Groq model identifiers
COUNCIL_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "groq/compound",
    
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
