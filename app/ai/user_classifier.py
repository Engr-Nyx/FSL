"""User-trained sign classifier.

Stores personalised FSL sign samples in the SQLite database (data/fsl.db).
Priority in the recognition pipeline:
  1. UserClassifier  ← this file (highest — user corrections override everything)
  2. LandmarkClassifier (rule-based)
  3. VisionInterpreter (Claude Vision, optional)

Each trained sign stores the averaged feature vector from multiple captures so
the match is robust to per-frame noise.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(lm_snapshots: list) -> Optional[dict]:
    """Summarise a sequence of landmark snapshots into a feature vector.

    Prefers face-relative Y (relative_y) when available in ≥50 % of frames —
    this makes position-based signs (TATAY, NANAY, …) robust to camera angle.
    Falls back to raw wrist_y otherwise.

    Returns None when there are no hand observations.
    """
    from app.ai.landmark_classifier import _get_finger_extensions

    hand_obs = [h for snap in lm_snapshots for h in (snap or [])]
    if not hand_obs:
        return None

    n = len(hand_obs)

    rel_ys = [h["relative_y"] for h in hand_obs if "relative_y" in h]
    use_relative = len(rel_ys) >= n * 0.5
    if use_relative:
        avg_y = sum(rel_ys) / len(rel_ys)
    else:
        avg_y = sum(h.get("wrist_y", 0.5) for h in hand_obs) / n

    avg_x = sum(h.get("wrist_x", 0.5) for h in hand_obs) / n

    ext_lists = [_get_finger_extensions(h) for h in hand_obs]
    avg_ext = [sum(e[i] for e in ext_lists) / n for i in range(5)]
    open_ratio = sum(1 for e in ext_lists if sum(e[1:]) >= 3) / n

    return {
        "avg_y":        avg_y,
        "avg_x":        avg_x,
        "extensions":   avg_ext,
        "open_ratio":   open_ratio,
        "use_relative": use_relative,
    }


def _distance(f1: dict, f2: dict) -> float:
    """Weighted Euclidean distance between two feature vectors."""
    same_coord = f1.get("use_relative", False) == f2.get("use_relative", False)
    y_weight = 2.5 if same_coord else 0.5

    dy = (f1["avg_y"] - f2["avg_y"]) * y_weight
    dx = (f1["avg_x"] - f2["avg_x"]) * 0.4
    do = (f1["open_ratio"] - f2["open_ratio"]) * 0.8
    de = sum(
        (a - b) ** 2 * 1.4
        for a, b in zip(f1["extensions"], f2["extensions"])
    )
    return math.sqrt(dy ** 2 + dx ** 2 + do ** 2 + de)


# ── Database helpers ──────────────────────────────────────────────────────────

def _get_session():
    from app.database.engine import SessionLocal
    return SessionLocal()


def _load_db() -> dict:
    """Load all user signs from SQLite into a dict keyed by gloss."""
    try:
        from app.database.crud import list_signs
        session = _get_session()
        try:
            signs = list_signs(session)
            return {s["gloss"]: s for s in signs}
        finally:
            session.close()
    except Exception as exc:
        logger.warning("user_classifier: could not read DB: %s", exc)
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def train_sign(
    gloss: str,
    fil: str,
    en: str,
    lm_snapshots: list,
) -> bool:
    """Add one training sample for a sign.

    Stores up to 15 samples per sign; older ones are evicted.
    Returns True on success.
    """
    features = _extract_features(lm_snapshots)
    if features is None:
        logger.warning("train_sign(%s): no hand data in snapshots", gloss)
        return False

    gloss_key = gloss.upper().strip()

    try:
        from app.database.crud import get_sign, save_sign
        session = _get_session()
        try:
            import json
            existing = get_sign(session, gloss_key)
            if existing:
                try:
                    samples = json.loads(existing.samples_json or "[]")
                except Exception:
                    samples = []
            else:
                samples = []

            samples.append(features)
            samples = samples[-15:]   # keep newest 15

            save_sign(session, gloss_key, fil, en, samples)
            logger.info(
                "train_sign: saved %s (%d samples total)", gloss_key, len(samples)
            )
            return True
        finally:
            session.close()
    except Exception as exc:
        logger.error("train_sign: DB error: %s", exc)
        return False


def classify(lm_snapshots: list) -> Optional[dict]:
    """Find the closest user-trained sign.

    Returns {glosses, sentence_fil, sentence_en, confidence} or None.
    """
    features = _extract_features(lm_snapshots)
    if features is None:
        return None

    db = _load_db()
    if not db:
        return None

    best_dist = float("inf")
    best_entry = None

    for entry in db.values():
        for sample in entry.get("samples", []):
            d = _distance(features, sample)
            if d < best_dist:
                best_dist = d
                best_entry = entry

    if best_entry is None or best_dist > 1.0:
        return None

    conf = "high" if best_dist < 0.30 else "medium" if best_dist < 0.60 else "low"

    logger.info(
        "user_classifier matched: %s (dist=%.3f conf=%s)",
        best_entry["gloss"], best_dist, conf,
    )
    return {
        "glosses":      [best_entry["gloss"]],
        "sentence_fil": best_entry["fil"],
        "sentence_en":  best_entry["en"],
        "confidence":   conf,
        "source":       "user",
    }


def list_signs() -> list[dict]:
    """Return all user-trained signs with metadata."""
    db = _load_db()
    return [
        {
            "gloss":        v["gloss"],
            "fil":          v["fil"],
            "en":           v["en"],
            "sample_count": len(v.get("samples", [])),
            "trained_at":   v.get("trained_at", ""),
        }
        for v in db.values()
    ]


def delete_sign(gloss: str) -> bool:
    """Remove a user-trained sign.  Returns True if it existed."""
    try:
        from app.database.crud import delete_sign as db_delete
        session = _get_session()
        try:
            result = db_delete(session, gloss)
            return result
        finally:
            session.close()
    except Exception as exc:
        logger.error("delete_sign: DB error: %s", exc)
        return False
