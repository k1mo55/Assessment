from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..schemas import ErrorResponse, SearchRequest, SearchResponse
from ..services.search_service import SearchService

router = APIRouter()


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


@router.post(
    "/search/",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def search(
    body: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse | JSONResponse:
    query = body.query.strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})

    try:
        results = await service.search(query)
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Search processing failed."})
    return SearchResponse(results=results)
