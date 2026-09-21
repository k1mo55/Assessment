from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..schemas import ErrorResponse, IngestResponse
from ..services.ingestion_service import (
    BatchIngestionError,
    IngestionService,
    InvalidInputError,
    normalize_input,
)

router = APIRouter()

def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


@router.post(
    "/ingest/",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ingest(
    request: Request,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse | JSONResponse:
    try:
        form = await request.form()
        pdfs = await normalize_input(list(form.getlist("input")))
    except InvalidInputError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid input."})

    try:
        filenames = await service.ingest(pdfs)
    except BatchIngestionError:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process uploaded file."},
        )

    count = len(filenames)
    noun = "document" if count == 1 else "documents"
    return IngestResponse(
        message=f"Successfully ingested {count} PDF {noun}.",
        files=filenames,
    )
