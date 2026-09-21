import asyncio
import io

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

    async def ensure_bucket(self) -> None:
        await asyncio.to_thread(self._ensure_bucket_sync)

    def _ensure_bucket_sync(self) -> None:
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)

    async def upload_pdf(self, object_key: str, content: bytes) -> None:
        await asyncio.to_thread(self._upload_pdf_sync, object_key, content)

    def _upload_pdf_sync(self, object_key: str, content: bytes) -> None:
        stream = io.BytesIO(content)
        self._client.put_object(
            self.bucket,
            object_key,
            stream,
            length=len(content),
            content_type="application/pdf",
        )

    async def delete_object(self, object_key: str) -> None:
        await asyncio.to_thread(self._client.remove_object, self.bucket, object_key)

    async def delete_batch(self, batch_id: str) -> None:
        await asyncio.to_thread(self._delete_batch_sync, batch_id)

    def _delete_batch_sync(self, batch_id: str) -> None:
        prefix = f"batches/{batch_id}/"
        objects = self._client.list_objects(self.bucket, prefix=prefix, recursive=True)
        for stored_object in objects:
            self._client.remove_object(self.bucket, stored_object.object_name)


def build_object_key(batch_id: str, document_id: str) -> str:
    return f"batches/{batch_id}/{document_id}.pdf"
