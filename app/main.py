"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.rest.health import router as health_router
from app.api.rest.upload import router as upload_router
from app.api.rest.training import router as training_router
from app.api.ws.endpoint import router as ws_router
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.use_ai_interpreter and settings.anthropic_api_key:
        logger.info("FSL API starting — interpreter=Claude Vision (model=%s)", settings.ai_model)
        try:
            from app.ai.vision_interpreter import VisionInterpreter
            VisionInterpreter.get()
        except Exception as exc:
            logger.warning("Could not pre-load VisionInterpreter: %s", exc)
    else:
        logger.info("FSL API starting — interpreter=FSLTransformer (device=%s)", settings.model_device)
        try:
            from app.model.predictor import FSLPredictor
            FSLPredictor.get()
        except Exception as exc:
            logger.warning("Could not pre-load predictor: %s", exc)
    yield
    logger.info("FSL API shutting down")


app = FastAPI(
    title="FSL Interpreter API",
    description=(
        "Filipino Sign Language interpretation API. "
        "Accepts MP4 video uploads and real-time webcam WebSocket streams, "
        "extracts skeletal keypoints via MediaPipe Holistic, runs them through "
        "a multi-branch Spatial-Temporal Transformer, and returns Tagalog + English translations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router, tags=["health"])
app.include_router(upload_router, tags=["interpret"])
app.include_router(training_router)
app.include_router(ws_router, tags=["stream"])

# ── Serve frontend static files if they exist ─────────────────────────────────
import os
if os.path.exists("index.html"):
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_frontend():
        return FileResponse("index.html")
