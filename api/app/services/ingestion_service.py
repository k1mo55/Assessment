import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ..clients.ingestion_client import IngestionClient, IngestionDocument
from ..clients.object_storage_client import ObjectStorageClient, build_object_key
from ..clients.qdrant_client import QdrantClient

class InvalidInputError(ValueError):
    pass


class BatchIngestionError(RuntimeError):
    pass


@dataclass(frozen=True)
class NormalizedPdf:
    filename: str
    content: bytes


class UploadedFileLike(Protocol):
    filename: str | None

    async def read(self) -> bytes: ...


async def normalize_input(values: list[object]) -> list[NormalizedPdf]:
    if not values:
        raise InvalidInputError("Invalid input.")

    if all(isinstance(value, str) for value in values):
        if len(values) != 1:
            raise InvalidInputError("Provide one directory path.")
        return await _read_directory(Path(str(values[0])))

    if any(isinstance(value, str) for value in values):
        raise InvalidInputError("Do not mix file uploads and directory paths.")

    pdfs: list[NormalizedPdf] = []
    for value in values:
        filename = getattr(value, "filename", None)
        read = getattr(value, "read", None)
        if not filename or read is None or Path(filename).suffix.lower() != ".pdf":
            raise InvalidInputError("Only PDF files are accepted.")
        content = await read()
        if not content:
            raise InvalidInputError("PDF files cannot be empty.")
        pdfs.append(NormalizedPdf(filename=Path(filename).name, content=content))
    return pdfs


async def _read_directory(directory: Path) -> list[NormalizedPdf]:
    if not await asyncio.to_thread(directory.is_dir):
        raise InvalidInputError("Directory does not exist.")
    paths = await asyncio.to_thread(
        lambda: sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
    )
    if not paths:
        raise InvalidInputError("Directory contains no PDF files.")

    pdfs: list[NormalizedPdf] = []
    for path in paths:
        content = await asyncio.to_thread(path.read_bytes)
        if content:
            pdfs.append(NormalizedPdf(filename=path.name, content=content))
    if not pdfs:
        raise InvalidInputError("Directory contains no usable PDF files.")
    return pdfs


class IngestionService:
    def __init__(
        self,
        storage: ObjectStorageClient,
        ingestion_client: IngestionClient,
        qdrant: QdrantClient,
    ) -> None:
        self._storage = storage
        self._ingestion_client = ingestion_client
        self._qdrant = qdrant

    async def ingest(self, pdfs: list[NormalizedPdf]) -> list[str]:
        batch_id = str(uuid4())
        documents = [self._make_document(batch_id, pdf.filename) for pdf in pdfs]

        try:
            await self._upload_documents(pdfs, documents)
            results = await asyncio.gather(
                *(self._ingestion_client.process(document) for document in documents),
                return_exceptions=True,
            )
        except Exception as exc:
            await self._rollback(batch_id)
            raise BatchIngestionError from exc

        failed = [
            result
            for result in results
            if isinstance(result, BaseException) or result.get("status") != "success"
        ]
        if failed:
            await self._rollback(batch_id)
            raise BatchIngestionError

        return [pdf.filename for pdf in pdfs]

    @staticmethod
    def _make_document(batch_id: str, filename: str) -> IngestionDocument:
        document_id = str(uuid4())
        return IngestionDocument(
            batch_id=batch_id,
            document_id=document_id,
            object_key=build_object_key(batch_id, document_id),
            filename=filename,
        )

    async def _upload_documents(
        self,
        pdfs: list[NormalizedPdf],
        documents: list[IngestionDocument],
    ) -> None:
        for pdf, document in zip(pdfs, documents, strict=True):
            await self._storage.upload_pdf(document.object_key, pdf.content)

    async def _rollback(self, batch_id: str) -> None:
        try:
            await self._qdrant.delete_batch(batch_id)
        except Exception:
            pass

        try:
            await self._storage.delete_batch(batch_id)
        except Exception:
            pass
