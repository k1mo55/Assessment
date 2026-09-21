import httpx


class EmbeddingClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/embed",
                json={"texts": texts},
            )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding service returned an unexpected result count")
        return embeddings
