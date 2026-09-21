from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .config import get_settings
from .model import EmbeddingModel
from .schemas import EmbedRequest, EmbedResponse

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    model = EmbeddingModel(
        settings.embedding_model,
        settings.embedding_dimension,
        settings.embedding_device,
    )
    model.load()
    # Force lazy PyTorch/CUDA initialization to finish before this service is
    # considered ready. Without this, the first real request can pay the
    # one-time initialization cost and time out in a caller.
    await model.embed(["embedding service readiness check"])
    app.state.embedding_model = model
    yield


app = FastAPI(title="Text Embedding Service", lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    vectors = await request.app.state.embedding_model.embed(body.texts)
    return EmbedResponse(embeddings=vectors)
