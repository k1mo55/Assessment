from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .clients.embedding_client import EmbeddingClient
from .clients.object_storage_client import ObjectStorageClient
from .config import get_settings
from .schemas import ProcessRequest, ProcessResponse
from .services.chunker import SemanticChunker
from .services.document_processor import DocumentProcessor
from .services.qdrant_service import QdrantService

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    storage = ObjectStorageClient(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
    )
    embeddings = EmbeddingClient(
        settings.embedding_service_url,
        settings.embedding_batch_size,
    )
    qdrant = QdrantService(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_dimension,
    )
    await qdrant.ensure_collection()
    app.state.document_processor = DocumentProcessor(
        storage,
        embeddings,
        SemanticChunker(
            embeddings,
            similarity_threshold=settings.semantic_similarity_threshold,
            minimum_characters=settings.chunk_minimum_characters,
            maximum_characters=settings.chunk_maximum_characters,
        ),
        qdrant,
    )
    yield
    await qdrant.close()


app = FastAPI(title="PDF Ingestion Service", lifespan=lifespan)


@app.post("/process", response_model=ProcessResponse)
async def process(body: ProcessRequest, request: Request) -> ProcessResponse:
    try:
        await request.app.state.document_processor.process(body)
    except Exception as exc:
        return ProcessResponse(
            batch_id=body.batch_id,
            document_id=body.document_id,
            status="failed",
            error=str(exc) or "Document processing failed.",
        )
    return ProcessResponse(
        batch_id=body.batch_id,
        document_id=body.document_id,
        status="success",
    )
