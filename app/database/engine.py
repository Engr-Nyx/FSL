"""SQLAlchemy engine and session factory for SQLite."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Store DB in the data/ directory so it is persisted with Docker volumes
_DB_DIR = "data"
os.makedirs(_DB_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{_DB_DIR}/fsl.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_session():
    """Yield a database session (use as FastAPI dependency)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
