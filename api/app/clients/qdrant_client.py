from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    VectorParams,
)


class QdrantClient:
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

    async def search(
        self,
        vector: list[float],
        limit: int,
        score_threshold: float | None = None,
    ) -> list[dict]:
        response = await self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return [
            {"score": point.score, "payload": point.payload or {}}
            for point in response.points
        ]

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

    async def delete_batch(self, batch_id: str) -> None:
        batch_filter = Filter(
            must=[
                FieldCondition(key="batch_id", match=MatchValue(value=batch_id))
            ]
        )
        await self._client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(filter=batch_filter),
            wait=True,
        )

    async def close(self) -> None:
        await self._client.close()
