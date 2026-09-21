from dataclasses import asdict, dataclass

import httpx


@dataclass(frozen=True)
class IngestionDocument:
    batch_id: str
    document_id: str
    object_key: str
    filename: str


class IngestionClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def process(self, document: IngestionDocument) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{self._base_url}/process",
                json=asdict(document),
            )
        response.raise_for_status()
        result = response.json()
        if result.get("document_id") != document.document_id:
            raise RuntimeError("Ingestion service returned a mismatched document ID")
        return result
