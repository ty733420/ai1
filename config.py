# config.py
#
# GitHub Copilot: Create a Python configuration file.
# It should define settings for a LangChain LLM application.
# Include an 'environment' variable that defaults to 'development'.
# Define LLM specific settings for 'development' (Ollama) and 'production' (Gemini).
# For Ollama, include its base URL (e.g., http://192.168.1.100:11434) and model name.
# For Gemini, include its model name (e.g., gemini-pro) and potentially a default API key placeholder.
# Use os.getenv for environment variables.

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

class Config:
    """Base configuration class."""
    APP_NAME = "MyAIAgent"
    # Shared settings can go here

class DevelopmentConfig(Config):
    """Development configuration using local Ollama."""
    LLM_PROVIDER = "ollama"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.100:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
    # Local memory persistence
    MEMORY_DB_PATH = "chat_histories/dev_memory.db"

class ProductionConfig(Config):
    """Production configuration using Gemini API."""
    LLM_PROVIDER = "gemini"
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Google Cloud / Gemini API key
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    # Cloud memory persistence (e.g., PostgreSQL connection string)
    MEMORY_DB_URL = os.getenv("DATABASE_URL")

def get_config():
    """Returns the appropriate configuration based on ENVIRONMENT."""
    if ENVIRONMENT == "production":
        return ProductionConfig()
    return DevelopmentConfig()

config = get_config()

print(f"Loaded environment: {ENVIRONMENT}")
print(f"LLM Provider: {config.LLM_PROVIDER}")