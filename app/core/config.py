import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Core settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "3200"))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Local LLM settings
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL")
    LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "dummy-key")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")
    LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4000"))
    LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "1.0"))

    # Application settings
    SUMMARIES_DIR = "app/summaries"
    PROMPT_FILE = "app/paper_prompt.md"
    MAX_PROMPT_LENGTH = 16000

settings = Settings()
