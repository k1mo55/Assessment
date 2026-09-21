from typing import Literal

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    batch_id: str
    document_id: str
    object_key: str
    filename: str


class ProcessResponse(BaseModel):
    batch_id: str
    document_id: str
    status: Literal["success", "failed"]
    error: str | None = None

