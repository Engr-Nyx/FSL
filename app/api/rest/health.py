"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from app.model.predictor import FSLPredictor
    predictor = FSLPredictor.get()
    return HealthResponse(status="ok", model_loaded=predictor._weights_loaded)
