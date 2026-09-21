import asyncio
import tempfile
from pathlib import Path

from ..clients.embedding_client import EmbeddingClient
from ..clients.object_storage_client import ObjectStorageClient
from ..schemas import ProcessRequest
from .chunker import SemanticChunker
from .pdf_parser import parse_pdf
from .qdrant_service import QdrantService

class DocumentProcessor:
    def __init__(
        self,
        storage: ObjectStorageClient,
        embedding_client: EmbeddingClient,
        chunker: SemanticChunker,
        qdrant: QdrantService,
    ) -> None:
        self._storage = storage
        self._embedding_client = embedding_client
        self._chunker = chunker
        self._qdrant = qdrant

    async def process(self, request: ProcessRequest) -> None:
        temporary_path = self._temporary_pdf_path()
        try:
            await self._storage.download_pdf(request.object_key, temporary_path)
            pages = await asyncio.to_thread(parse_pdf, temporary_path)
            chunks = await self._chunker.chunk(pages)
            embeddings = await self._embedding_client.embed(
                [chunk.text for chunk in chunks]
            )
            await self._qdrant.store_chunks(
                batch_id=request.batch_id,
                document_id=request.document_id,
                filename=request.filename,
                object_key=request.object_key,
                chunks=chunks,
                embeddings=embeddings,
            )
        except Exception:
            await self._clean_partial_document(request.document_id)
            raise
        finally:
            await asyncio.to_thread(temporary_path.unlink, missing_ok=True)

    async def _clean_partial_document(self, document_id: str) -> None:
        try:
            await self._qdrant.delete_document(document_id)
        except Exception:
            pass

    @staticmethod
    def _temporary_pdf_path() -> Path:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        path = Path(temporary_file.name)
        temporary_file.close()
        return path
