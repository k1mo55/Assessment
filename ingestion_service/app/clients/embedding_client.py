import httpx


class EmbeddingClient:
    def __init__(self, base_url: str, batch_size: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient() as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                response = await client.post(
                    f"{self._base_url}/embed",
                    json={"texts": batch},
                )
                response.raise_for_status()
                batch_embeddings = response.json()["embeddings"]
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError("Embedding service returned an unexpected result count")
                embeddings.extend(batch_embeddings)
        return embeddings
