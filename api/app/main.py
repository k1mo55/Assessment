from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .clients.embedding_client import EmbeddingClient
from .clients.ingestion_client import IngestionClient
from .clients.object_storage_client import ObjectStorageClient
from .clients.qdrant_client import QdrantClient
from .config import get_settings
from .routes.ingest import router as ingest_router
from .routes.search import router as search_router
from .services.ingestion_service import IngestionService
from .services.search_service import SearchService

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None ]:
    settings = get_settings()
    storage = ObjectStorageClient(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
    )
    qdrant = QdrantClient(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_dimension,
    )
    embedding_client = EmbeddingClient(settings.embedding_service_url)
    ingestion_client = IngestionClient(
        settings.ingestion_service_url,
        settings.ingestion_request_timeout_seconds,
    )

    await storage.ensure_bucket()
    await qdrant.ensure_collection()
    app.state.ingestion_service = IngestionService(storage, ingestion_client, qdrant)
    app.state.search_service = SearchService(
        embedding_client,
        qdrant,
        settings.search_top_k,
        settings.search_score_threshold or None,
    )
    yield
    await qdrant.close()


app = FastAPI(
    title="PDF Ingestor & Semantic Search API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(ingest_router)
app.include_router(search_router)
