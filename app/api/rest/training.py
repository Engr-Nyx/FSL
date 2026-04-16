"""REST endpoints for user-trained sign management.

POST  /train/sign          — save one training capture for a sign
GET   /train/signs         — list all user-trained signs
DELETE /train/sign/{gloss} — remove a user-trained sign
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train", tags=["train"])


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


@router.post("/sign", summary="Save a training capture for a sign")
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


@router.get("/signs", response_model=List[SignMeta], summary="List all user-trained signs")
async def get_trained_signs():
    from app.ai.user_classifier import list_signs
    return list_signs()


@router.delete("/sign/{gloss}", summary="Delete a user-trained sign")
async def remove_trained_sign(gloss: str):
    from app.ai.user_classifier import delete_sign
    success = delete_sign(gloss)
    if not success:
        raise HTTPException(status_code=404, detail=f"Sign '{gloss}' not found")
    return {"success": True, "gloss": gloss.upper().strip()}
