from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    LLM_MODEL_PATH: str = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    LLM_CONTEXT_LENGTH: int = 4096
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.7

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    USE_OLLAMA: bool = True

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "video_comparison_collection"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    WHISPER_MODEL: str = "base"

    FRONTEND_URL: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()