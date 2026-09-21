from pydantic import BaseModel


class IngestResponse(BaseModel):
    message: str
    files: list[str]


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    document: str
    score: float
    content: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ErrorResponse(BaseModel):
    error: str
