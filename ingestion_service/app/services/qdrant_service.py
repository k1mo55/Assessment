from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .chunker import SemanticChunk


class QdrantService:
    def __init__(self, url: str, collection: str, vector_size: int = 384) -> None:
        self.collection = collection
        self.vector_size = vector_size
        self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self.collection):
            try:
                await self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            except Exception:
                if not await self._client.collection_exists(self.collection):
                    raise
        await self._validate_collection()

    async def _validate_collection(self) -> None:
        collection = await self._client.get_collection(self.collection)
        vectors = collection.config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise RuntimeError("Qdrant collection must use one unnamed dense vector")
        if vectors.size != self.vector_size or vectors.distance != Distance.COSINE:
            raise RuntimeError(
                f"Qdrant collection {self.collection!r} is incompatible: "
                f"expected size={self.vector_size} and distance=Cosine. "
                "Reset the development collection and re-ingest the PDFs."
            )

    async def store_chunks(
        self,
        *,
        batch_id: str,
        document_id: str,
        filename: str,
        object_key: str,
        chunks: list[SemanticChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match")

        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = str(uuid4())
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "batch_id": batch_id,
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "filename": filename,
                        "object_key": object_key,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "text": chunk.text,
                    },
                )
            )
        await self._client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )

    async def delete_document(self, document_id: str) -> None:
        document_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )
        await self._client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(filter=document_filter),
            wait=True,
        )

    async def close(self) -> None:
        await self._client.close()
