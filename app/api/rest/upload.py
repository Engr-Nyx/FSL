"""REST endpoint for MP4 video upload → FSL interpretation.

Two interpretation paths:
  AI path   (use_ai_interpreter=True + ANTHROPIC_API_KEY set):
            Samples frames from the video and sends them to Claude Vision.
            No trained model weights required.

  Model path (fallback):
            Extracts MediaPipe keypoints, runs them through FSLTransformer,
            maps gloss sequences to sentences.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import aiofiles
import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("/tmp/fsl_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/webm", "video/quicktime"}


class InterpretResponse(BaseModel):
    glosses: list[str]
    sentence_fil: str
    sentence_en: str
    total_frames: int
    windows_processed: int


@router.post("/interpret/upload", response_model=InterpretResponse)
async def interpret_video(
    file: UploadFile = File(...),
    lang: str = Query("fil", pattern="^(fil|en)$"),
) -> InterpretResponse:
    """Accept an MP4/WebM/MOV upload and return the FSL interpretation."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use MP4, WebM, or MOV.",
        )

    max_bytes = settings.max_video_size_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds {settings.max_video_size_mb} MB limit.",
        )

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        if settings.use_ai_interpreter and settings.anthropic_api_key:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _process_video_ai, tmp_path, lang
            )
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _process_video_model, tmp_path, lang
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


# ── AI path ───────────────────────────────────────────────────────────────────

def _process_video_ai(video_path: Path, lang: str) -> InterpretResponse:
    """Extract frames and interpret via Claude Vision."""
    import io
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open video file.")

    raw_frames: list[bytes] = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        # Convert BGR → RGB → JPEG bytes (at reduced quality for token efficiency)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # Resize to max 640px wide to reduce payload size
        if pil.width > 640:
            ratio = 640 / pil.width
            pil = pil.resize((640, int(pil.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        raw_frames.append(buf.getvalue())

    cap.release()

    if not raw_frames:
        return InterpretResponse(
            glosses=[], sentence_fil="", sentence_en="",
            total_frames=total_frames, windows_processed=0,
        )

    from app.ai.vision_interpreter import VisionInterpreter
    interpreter = VisionInterpreter.get()
    result = interpreter.interpret(raw_frames, max_frames=settings.ai_max_frames, lang=lang)

    return InterpretResponse(
        glosses=result["glosses"],
        sentence_fil=result["sentence_fil"],
        sentence_en=result["sentence_en"],
        total_frames=total_frames,
        windows_processed=len(raw_frames),
    )


# ── Model path (fallback) ─────────────────────────────────────────────────────

def _process_video_model(video_path: Path, lang: str) -> InterpretResponse:
    """Blocking video processing via MediaPipe + FSLTransformer — runs in thread pool."""
    import numpy as np
    from app.extraction.feature_builder import SlidingWindowBuffer, build_feature_vector
    from app.extraction.mediapipe_extractor import get_extractor
    from app.model.predictor import FSLPredictor
    from app.translation.sentence_mapper import GlossBuffer, SentenceMapper

    extractor = get_extractor(static_image_mode=False, model_complexity=1)
    predictor = FSLPredictor.get()
    mapper = SentenceMapper()
    window_buf = SlidingWindowBuffer(settings.window_size, settings.stride)
    gloss_buf = GlossBuffer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open video file.")

    frame_features: list = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        result = extractor.process_frame(frame)
        feature = build_feature_vector(result)
        frame_features.append(feature)

    cap.release()

    windows = SlidingWindowBuffer.extract_windows_from_sequence(
        frame_features, settings.window_size, settings.stride
    )

    if not windows:
        return InterpretResponse(
            glosses=[], sentence_fil="", sentence_en="",
            total_frames=total_frames, windows_processed=0,
        )

    window_arrays = [np.stack(w) if isinstance(w, list) else w for w in windows]
    predictions = predictor.batch_predict(window_arrays, settings.min_confidence)

    for pred in predictions:
        if pred["committed"]:
            gloss_buf.push(pred["gloss"])

    glosses = gloss_buf.flush()
    sentence_fil = mapper.map(glosses, lang="fil")
    sentence_en = mapper.map(glosses, lang="en")

    if settings.enable_llm_rewrite and settings.anthropic_api_key:
        from app.translation.language_model import LLMRewriter
        rewriter = LLMRewriter(settings.anthropic_api_key)
        if rewriter.available:
            if lang == "fil":
                sentence_fil = rewriter.rewrite(sentence_fil, lang="fil")
            else:
                sentence_en = rewriter.rewrite(sentence_en, lang="en")

    return InterpretResponse(
        glosses=glosses,
        sentence_fil=sentence_fil,
        sentence_en=sentence_en,
        total_frames=total_frames,
        windows_processed=len(windows),
    )
