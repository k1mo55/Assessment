from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "pdf_chunks_bge_small_384"
    embedding_dimension: int = Field(default=384, ge=1)
    embedding_service_url: str = "http://embedding-service:8002"
    ingestion_service_url: str = "http://ingestion-service:8001"
    ingestion_request_timeout_seconds: float = Field(default=300.0, gt=0)
    search_top_k: int = Field(default=5, ge=1)
    search_score_threshold: float = Field(default=0.0, ge=0, le=1)
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "pdf-documents"
    minio_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
