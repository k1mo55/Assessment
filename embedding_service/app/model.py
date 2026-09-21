import asyncio
from typing import Any


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        expected_dimension: int,
        requested_device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._requested_device = requested_device
        self._device: str | None = None
        self._model: Any = None

    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self._device = self._resolve_device(torch)
        self._model = SentenceTransformer(self._model_name, device=self._device)

    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError("Embedding model has not been loaded")
        return self._device

    def _resolve_device(self, torch: Any) -> str:
        requested = self._requested_device.strip().lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" or requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"EMBEDDING_DEVICE={self._requested_device!r} was requested, "
                    "but CUDA is not available inside the container"
                )
            return requested
        raise ValueError(
            "EMBEDDING_DEVICE must be 'auto', 'cpu', 'cuda', or a CUDA device "
            "such as 'cuda:0'"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Embedding model has not been loaded")
        vectors = await asyncio.to_thread(self._encode, texts)
        if any(len(vector) != self._expected_dimension for vector in vectors):
            raise RuntimeError("Embedding model returned an unexpected vector dimension")
        return vectors

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
