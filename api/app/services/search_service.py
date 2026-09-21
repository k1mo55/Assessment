from ..clients.embedding_client import EmbeddingClient
from ..clients.qdrant_client import QdrantClient

class SearchService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        qdrant: QdrantClient,
        top_k: int,
        score_threshold: float | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._qdrant = qdrant
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def search(self, query: str) -> list[dict]:
        vector = (await self._embedding_client.embed([query]))[0]
        matches = await self._qdrant.search(
            vector,
            self._top_k,
            self._score_threshold,
        )
        return [
            {
                "document": match["payload"].get("filename", ""),
                "score": match["score"],
                "content": match["payload"].get("text", ""),
            }
            for match in matches
        ]
