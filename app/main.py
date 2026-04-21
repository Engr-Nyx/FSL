"""FastAPI application entry point."""

import json
import logging
import os
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


def _init_database() -> None:
    """Create all SQLite tables and migrate legacy user_signs.json if present."""
    from app.database.engine import Base, engine, SessionLocal
    from app.database import models  # noqa: F401 — registers ORM models

    Base.metadata.create_all(bind=engine)
    logger.info("DB: tables ready (SQLite)")

    # One-time migration: import existing user_signs.json into SQLite
    legacy_path = os.path.join("models", "user_signs.json")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, encoding="utf-8") as f:
                legacy = json.load(f)
            if legacy:
                from app.database.crud import get_sign, save_sign
                session = SessionLocal()
                try:
                    migrated = 0
                    for gloss_key, entry in legacy.items():
                        if not get_sign(session, gloss_key):
                            save_sign(
                                session,
                                gloss=entry.get("gloss", gloss_key),
                                fil=entry.get("fil", ""),
                                en=entry.get("en", ""),
                                samples=entry.get("samples", []),
                            )
                            migrated += 1
                    logger.info("DB: migrated %d signs from user_signs.json", migrated)
                finally:
                    session.close()
                # Rename legacy file so migration won't run again
                os.rename(legacy_path, legacy_path + ".migrated")
        except Exception as exc:
            logger.warning("DB: could not migrate user_signs.json: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_database()
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
if os.path.exists("index.html"):
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_frontend():
        return FileResponse("index.html")

if os.path.exists("fsl_embedded_data.js"):
    from fastapi.responses import FileResponse as _FR

    @app.get("/fsl_embedded_data.js")
    async def serve_fsl_embedded():
        return _FR("fsl_embedded_data.js", media_type="application/javascript")

if os.path.exists("dataset.json"):
    from fastapi.responses import FileResponse as _FR2

    @app.get("/dataset.json")
    async def serve_dataset():
        return _FR2("dataset.json", media_type="application/json")
