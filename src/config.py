"""
VinaCompare Configuration Module
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "VinaCompare"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API
    API_V1_PREFIX: str = "/api/v1"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "vinarag"
    POSTGRES_USER: str = "vinarag"
    POSTGRES_PASSWORD: str = "vinarag123"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "vietnamese_tech_docs"

    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_MODE: str = "hybrid"  # "dense", "sparse", "hybrid"
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RERANK_ENABLED: bool = False

    # LLM Models
    AVAILABLE_MODELS: List[str] = [
        "Vistral-7B-Chat",
        "Arcee-VyLinh-3B",
        "GemSUra-7B",
        "VinaLLaMA-2.7B",
        "PhoGPT-7B5"
    ]
    DEFAULT_MODEL: str = "Vistral-7B-Chat"

    # Model paths (will be downloaded)
    MODEL_CACHE_DIR: str = "./models"

    # Generation
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.1

    # Evaluation
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    JUDGE_MODEL: str = "gpt-4-turbo-preview"

    # Hallucination Detection
    HALLUCINATION_THRESHOLD: float = 0.3
    SELF_CONSISTENCY_SAMPLES: int = 3

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    METRICS_PORT: int = 8001

    # Logging
    LOG_FILE: str = "logs/vinacompare.log"
    LOG_ROTATION: str = "100 MB"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create settings instance
settings = get_settings()
