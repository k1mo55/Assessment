from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "pdf_chunks_bge_small_384"
    embedding_dimension: int = Field(default=384, ge=1)
    embedding_service_url: str = "http://embedding-service:8002"
    embedding_batch_size: int = Field(default=32, ge=1)
    semantic_similarity_threshold: float = Field(default=0.58, ge=-1, le=1)
    chunk_minimum_characters: int = Field(default=250, ge=1)
    chunk_maximum_characters: int = Field(default=2_000, ge=1)
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "pdf-documents"
    minio_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
