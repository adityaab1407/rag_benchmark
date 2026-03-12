from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # LLM - Groq (free tier)
    groq_api_key: Optional[str] = None
    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_model_fast: str = "groq/llama-3.1-8b-instant"

    # Embeddings - runs locally on CPU, no API key needed
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Qdrant - local folder, no Docker needed
    qdrant_path: str = "./qdrant_storage"
    collection_name: str = "rag_benchmark"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()