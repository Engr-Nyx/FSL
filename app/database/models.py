"""ORM models for FSL database."""

from sqlalchemy import Column, Integer, String, Text

from app.database.engine import Base


class UserSign(Base):
    """Stores user-trained FSL signs (replaces models/user_signs.json)."""

    __tablename__ = "user_signs"

    gloss = Column(String(100), primary_key=True, index=True)
    fil = Column(String(300), nullable=False, default="")
    en = Column(String(300), nullable=False, default="")
    # JSON-serialised list of feature-vector dicts
    samples_json = Column(Text, nullable=False, default="[]")
    trained_at = Column(String(50), nullable=False, default="")


class TranslationLog(Base):
    """Records every committed sentence for history and analytics."""

    __tablename__ = "translation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # JSON-serialised list of gloss strings
    glosses_json = Column(Text, nullable=False, default="[]")
    sentence_fil = Column(String(500), nullable=True)
    sentence_en = Column(String(500), nullable=True)
    # Firebase UID — nullable so server-side uploads still get logged
    user_id = Column(String(200), nullable=True, index=True)
    # "ws" | "upload" | "user"
    source = Column(String(50), nullable=True, default="ws")
    created_at = Column(String(50), nullable=False, default="")
