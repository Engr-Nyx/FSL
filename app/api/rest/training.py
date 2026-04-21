"""REST endpoints for user-trained sign management and translation history.

POST   /train/sign          — save one training capture for a sign
GET    /train/signs         — list all user-trained signs
DELETE /train/sign/{gloss}  — remove a user-trained sign
GET    /db/signs            — same as /train/signs (public alias for the frontend)
POST   /db/log              — log a committed translation to history
GET    /db/history          — fetch translation history (optional ?user_id=...)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.engine import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["train"])


class TrainSignRequest(BaseModel):
    gloss: str = Field(..., description="Uppercase gloss label, e.g. 'TATAY'")
    fil: str   = Field(..., description="Filipino/Tagalog word")
    en: str    = Field(..., description="English translation")
    lm_snapshots: list = Field(
        ...,
        description=(
            "List of per-frame landmark snapshots collected from the browser. "
            "Each item is a list of hand dicts: "
            "[{wrist_y, wrist_x, label, lms:[21×{x,y}], tips, knuckles}, ...]"
        ),
    )


class SignMeta(BaseModel):
    gloss: str
    fil: str
    en: str
    sample_count: int
    trained_at: str


class LogTranslationRequest(BaseModel):
    glosses: list = Field(default_factory=list)
    sentence_fil: str = Field(default="")
    sentence_en: str = Field(default="")
    user_id: Optional[str] = Field(default=None)
    source: str = Field(default="ws")


# ── Training endpoints ────────────────────────────────────────────────────────

@router.post("/train/sign", summary="Save a training capture for a sign")
async def save_training_capture(req: TrainSignRequest):
    from app.ai.user_classifier import train_sign
    if not req.gloss.strip():
        raise HTTPException(status_code=400, detail="gloss must not be empty")
    if not req.lm_snapshots:
        raise HTTPException(status_code=400, detail="lm_snapshots must not be empty")

    success = train_sign(req.gloss, req.fil, req.en, req.lm_snapshots)
    if not success:
        raise HTTPException(
            status_code=422,
            detail="No valid hand landmarks found in the provided snapshots.",
        )
    return {"success": True, "gloss": req.gloss.upper().strip()}


@router.get("/train/signs", response_model=List[SignMeta], summary="List all user-trained signs")
async def get_trained_signs():
    from app.ai.user_classifier import list_signs
    return list_signs()


@router.delete("/train/sign/{gloss}", summary="Delete a user-trained sign")
async def remove_trained_sign(gloss: str):
    from app.ai.user_classifier import delete_sign
    success = delete_sign(gloss)
    if not success:
        raise HTTPException(status_code=404, detail=f"Sign '{gloss}' not found")
    return {"success": True, "gloss": gloss.upper().strip()}


# ── Database / history endpoints ──────────────────────────────────────────────

@router.get("/db/signs", response_model=List[SignMeta], summary="All user-trained signs (alias)")
async def db_list_signs():
    """Public alias for /train/signs — used by the frontend dictionary."""
    from app.ai.user_classifier import list_signs
    return list_signs()


@router.post("/db/log", summary="Log a committed translation to history")
async def db_log_translation(
    req: LogTranslationRequest,
    session: Session = Depends(get_session),
):
    from app.database.crud import log_translation
    entry = log_translation(
        session,
        glosses=req.glosses,
        sentence_fil=req.sentence_fil,
        sentence_en=req.sentence_en,
        user_id=req.user_id,
        source=req.source,
    )
    return {"success": True, "id": entry.id}


@router.get("/db/history", summary="Fetch translation history")
async def db_get_history(
    limit: int = Query(50, ge=1, le=200),
    user_id: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    from app.database.crud import get_recent_translations
    return get_recent_translations(session, limit=limit, user_id=user_id)
