"""CRUD helpers for FSL database models."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database.models import TranslationLog, UserSign

logger = logging.getLogger(__name__)


# ── UserSign ──────────────────────────────────────────────────────────────────

def get_sign(session: Session, gloss: str) -> UserSign | None:
    return session.get(UserSign, gloss.upper().strip())


def save_sign(
    session: Session,
    gloss: str,
    fil: str,
    en: str,
    samples: list,
) -> UserSign:
    gloss_key = gloss.upper().strip()
    now = datetime.now(timezone.utc).isoformat()
    existing = session.get(UserSign, gloss_key)

    if existing:
        existing.fil = fil
        existing.en = en
        existing.samples_json = json.dumps(samples, ensure_ascii=False)
        existing.trained_at = now
    else:
        existing = UserSign(
            gloss=gloss_key,
            fil=fil,
            en=en,
            samples_json=json.dumps(samples, ensure_ascii=False),
            trained_at=now,
        )
        session.add(existing)

    session.commit()
    logger.info("db: saved sign %s (%d samples)", gloss_key, len(samples))
    return existing


def list_signs(session: Session) -> list[dict]:
    rows = session.query(UserSign).order_by(UserSign.trained_at.desc()).all()
    result = []
    for row in rows:
        try:
            samples = json.loads(row.samples_json or "[]")
        except Exception:
            samples = []
        result.append(
            {
                "gloss": row.gloss,
                "fil": row.fil,
                "en": row.en,
                "samples": samples,
                "trained_at": row.trained_at,
                "sample_count": len(samples),
            }
        )
    return result


def delete_sign(session: Session, gloss: str) -> bool:
    gloss_key = gloss.upper().strip()
    existing = session.get(UserSign, gloss_key)
    if existing:
        session.delete(existing)
        session.commit()
        logger.info("db: deleted sign %s", gloss_key)
        return True
    return False


# ── TranslationLog ────────────────────────────────────────────────────────────

def log_translation(
    session: Session,
    glosses: list,
    sentence_fil: str,
    sentence_en: str,
    user_id: str | None = None,
    source: str = "ws",
) -> TranslationLog:
    entry = TranslationLog(
        glosses_json=json.dumps(glosses, ensure_ascii=False),
        sentence_fil=sentence_fil or "",
        sentence_en=sentence_en or "",
        user_id=user_id,
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    session.add(entry)
    session.commit()
    return entry


def get_recent_translations(
    session: Session,
    limit: int = 50,
    user_id: str | None = None,
) -> list[dict]:
    q = (
        session.query(TranslationLog)
        .order_by(TranslationLog.created_at.desc())
    )
    if user_id:
        q = q.filter(TranslationLog.user_id == user_id)
    rows = q.limit(limit).all()

    result = []
    for row in rows:
        try:
            glosses = json.loads(row.glosses_json or "[]")
        except Exception:
            glosses = []
        result.append(
            {
                "id": row.id,
                "glosses": glosses,
                "sentence_fil": row.sentence_fil,
                "sentence_en": row.sentence_en,
                "user_id": row.user_id,
                "source": row.source,
                "created_at": row.created_at,
            }
        )
    return result
