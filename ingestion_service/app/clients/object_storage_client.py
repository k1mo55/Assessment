import asyncio
from pathlib import Path

from minio import Minio


class ObjectStorageClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
    ) -> None:
        self.bucket = bucket
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    async def download_pdf(self, object_key: str, destination: Path) -> None:
        await asyncio.to_thread(
            self._client.fget_object,
            self.bucket,
            object_key,
            str(destination),
        )

