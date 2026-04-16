"""Central configuration — all settings read from environment / .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_weights_path: str = "models/weights/fsl_transformer_v1.pt"
    vocabulary_path: str = "models/weights/vocabulary.json"
    model_device: str = "cpu"

    # ── Inference ─────────────────────────────────────────────────────────────
    window_size: int = 30
    stride: int = 10
    min_confidence: float = 0.60

    # ── Streaming ─────────────────────────────────────────────────────────────
    pause_gap_ms: int = 800
    max_video_size_mb: int = 100

    # ── AI Vision Interpreter ─────────────────────────────────────────────────
    # When true (and ANTHROPIC_API_KEY is set), Claude Vision is used for sign
    # recognition — no trained model weights or dataset required.
    use_ai_interpreter: bool = True
    ai_model: str = "claude-opus-4-6"
    ai_max_frames: int = 15           # max frames sampled per Claude call
    # Rolling frame buffer for WebSocket sessions (frames kept before flush)
    ai_frame_buffer_size: int = 150   # ~5 s at 30 fps

    # ── Optional LLM rewrite ──────────────────────────────────────────────────
    enable_llm_rewrite: bool = False
    anthropic_api_key: str = ""

    # ── Feature dims (derived, not configurable) ─────────────────────────────
    feature_dim: int = 468           # 132 pose + 63 left + 63 right + 210 face
    pose_dim: int = 132              # 33 landmarks × 4 (x, y, z, vis)
    hand_dim: int = 126              # 21 × 3 × 2 hands
    face_dim: int = 210              # 70 × 3

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def weights_path_obj(self) -> Path:
        return Path(self.model_weights_path)

    @property
    def vocab_path_obj(self) -> Path:
        return Path(self.vocabulary_path)


settings = Settings()
