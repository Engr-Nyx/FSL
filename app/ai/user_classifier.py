"""User-trained sign classifier.

Stores personalized FSL sign samples in models/user_signs.json.
Priority in the recognition pipeline:
  1. UserClassifier  ← this file (highest — user corrections override everything)
  2. LandmarkClassifier (rule-based)
  3. VisionInterpreter (Claude Vision, optional)

Each trained sign stores the averaged feature vector from multiple captures so
the match is robust to per-frame noise.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join("models", "user_signs.json")


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

    # Prefer face-relative Y when the holistic detector provided it
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
        "extensions":   avg_ext,      # [thumb, index, middle, ring, pinky] 0–1
        "open_ratio":   open_ratio,
        "use_relative": use_relative, # True = face-relative Y scale
    }


def _distance(f1: dict, f2: dict) -> float:
    """Weighted Euclidean distance between two feature vectors.

    Weights reflect how discriminative each dimension is for FSL:
    • Y-position is the primary family-sign distinguisher (TATAY vs NANAY).
    • Finger extensions are the primary shape distinguisher (letters, numbers).

    When both vectors use the same Y coordinate system (both face-relative or
    both absolute) the full Y weight (2.5) is applied.  When they differ the
    weight is reduced to 0.5 because the two scales are not directly comparable.
    """
    same_coord = f1.get("use_relative", False) == f2.get("use_relative", False)
    y_weight = 2.5 if same_coord else 0.5

    dy  = (f1["avg_y"] - f2["avg_y"]) * y_weight
    dx  = (f1["avg_x"] - f2["avg_x"]) * 0.4
    do  = (f1["open_ratio"] - f2["open_ratio"]) * 0.8
    de  = sum(
        (a - b) ** 2 * 1.4
        for a, b in zip(f1["extensions"], f2["extensions"])
    )
    return math.sqrt(dy ** 2 + dx ** 2 + do ** 2 + de)


# ── Database helpers ──────────────────────────────────────────────────────────

def _load_db() -> dict:
    if os.path.exists(_DB_PATH):
        try:
            with open(_DB_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("user_classifier: could not read DB: %s", exc)
    return {}


def _save_db(db: dict) -> None:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with open(_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


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
    db = _load_db()

    if gloss_key not in db:
        db[gloss_key] = {
            "gloss":      gloss_key,
            "fil":        fil,
            "en":         en,
            "samples":    [],
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

    entry = db[gloss_key]
    entry["fil"] = fil
    entry["en"]  = en
    entry["samples"].append(features)
    entry["samples"] = entry["samples"][-15:]   # keep newest 15
    entry["trained_at"] = datetime.now(timezone.utc).isoformat()

    _save_db(db)
    logger.info(
        "train_sign: saved %s (%d samples total)", gloss_key, len(entry["samples"])
    )
    return True


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
        for sample in entry["samples"]:
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
            "sample_count": len(v["samples"]),
            "trained_at":   v.get("trained_at", ""),
        }
        for v in db.values()
    ]


def delete_sign(gloss: str) -> bool:
    """Remove a user-trained sign.  Returns True if it existed."""
    gloss_key = gloss.upper().strip()
    db = _load_db()
    if gloss_key in db:
        del db[gloss_key]
        _save_db(db)
        logger.info("user_classifier: deleted %s", gloss_key)
        return True
    return False
