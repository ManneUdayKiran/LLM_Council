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

# Model tier configuration (cost per 1M tokens - estimates)
MODEL_TIERS = {
    # Cheap models (try first)
    "cheap": {
        "models": ["qwen/qwen3-32b", "llama-3.3-70b-versatile"],
        "cost_per_1m_tokens": 0.05,
    },
    # Medium models
    "medium": {
        "models": ["groq/compound", "openai/gpt-oss-120b"],
        "cost_per_1m_tokens": 0.15,
    },
    # Premium models (escalate on disagreement)
    "premium": {
        "models": ["google/gemini-2.5-flash", "anthropic/claude-sonnet-4.5"],
        "cost_per_1m_tokens": 3.00,
    },
}

# Disagreement threshold for escalation
DISAGREEMENT_THRESHOLD = 0.6  # If confidence < 60%, escalate

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
