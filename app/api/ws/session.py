"""Per-connection WebSocket stream session state.

Two interpretation paths:
  AI path   (use_ai_interpreter=True + ANTHROPIC_API_KEY set):
            Accumulates raw JPEG frames in a rolling buffer.
            Frontend detects signing pauses and sends {"type":"flush"}.
            Backend then calls Claude Vision and returns the sentence.

  Model path (fallback):
            Uses MediaPipe + FSLTransformer sliding-window inference.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class StreamSession:
    """State for a single WebSocket stream connection."""

    session_id: str
    lang: str = "fil"
    mode: str = "all"   # "all" | "numbers" | "letters" | "words"

    _last_activity_ts: float = field(default_factory=time.monotonic, init=False)

    # AI path — rolling buffer of raw JPEG bytes + landmark summaries
    _frame_buffer: deque = field(init=False)
    _lm_buffer: list = field(default_factory=list, init=False)  # landmark snapshots

    # Model path
    _window_buf: object = field(default=None, init=False)
    _gloss_buf: object = field(default=None, init=False)
    _mapper: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._frame_buffer = deque(maxlen=settings.ai_frame_buffer_size)
        if not (settings.use_ai_interpreter and settings.anthropic_api_key):
            self._init_model_state()

    def _init_model_state(self) -> None:
        from app.extraction.feature_builder import SlidingWindowBuffer
        from app.translation.sentence_mapper import GlossBuffer, SentenceMapper
        self._window_buf = SlidingWindowBuffer(settings.window_size, settings.stride)
        self._gloss_buf = GlossBuffer()
        self._mapper = SentenceMapper()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, jpeg_bytes: bytes, lm_data=None) -> Optional[dict]:
        """Buffer a JPEG frame + optional landmark data. Returns None (frontend controls flush)."""
        if settings.use_ai_interpreter and settings.anthropic_api_key:
            self._frame_buffer.append(jpeg_bytes)
            if lm_data:
                self._lm_buffer.append(lm_data)
            if len(self._frame_buffer) % 10 == 1:
                logger.info("Session %s: buffer=%d frames (%d with lm), size=%d bytes",
                            self.session_id, len(self._frame_buffer),
                            len(self._lm_buffer), len(jpeg_bytes))
            return None
        return self._process_frame_model(jpeg_bytes)

    def flush_sentence(self) -> dict:
        """Flush the current buffer into a translated sentence."""
        if settings.use_ai_interpreter and settings.anthropic_api_key:
            return self._flush_ai()
        return self._flush_model()

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._lm_buffer.clear()
        self._last_activity_ts = time.monotonic()
        if self._gloss_buf is not None:
            from app.translation.sentence_mapper import GlossBuffer
            self._gloss_buf = GlossBuffer()
        if self._window_buf is not None:
            self._window_buf.reset()

    # ── AI path ───────────────────────────────────────────────────────────────

    def _flush_ai(self) -> dict:
        frames = list(self._frame_buffer)
        lm_snapshots = list(self._lm_buffer)
        self._frame_buffer.clear()
        self._lm_buffer.clear()

        if not frames and not lm_snapshots:
            return {
                "type": "sentence",
                "glosses": [],
                "sentence_fil": "",
                "sentence_en": "",
                "confidence": "low",
                "current_glosses": [],
            }

        # ── 1. User-trained signs (highest priority — personal corrections) ─────
        from app.ai.user_classifier import classify as user_classify
        user_result = user_classify(lm_snapshots)
        if user_result:
            user_result = self._apply_mode_filter(user_result)
        if user_result and user_result["confidence"] in ("high", "medium"):
            logger.info("User classifier: %s (%s)", user_result["glosses"], user_result["confidence"])
            return {"type": "sentence", **user_result, "current_glosses": []}

        # ── 2. Rule-based landmark classifier (no API needed) ─────────────────
        from app.ai.landmark_classifier import classify_from_landmarks
        rule_result = classify_from_landmarks(lm_snapshots)

        if rule_result:
            # Apply mode filter: discard results that don't match the active mode
            rule_result = self._apply_mode_filter(rule_result)

        if rule_result:
            logger.info("Rule classifier: %s (%s)", rule_result["glosses"], rule_result["confidence"])

            # If rule result is high/medium, return it immediately
            if rule_result["confidence"] in ("high", "medium"):
                return {"type": "sentence", **rule_result, "current_glosses": []}

            # For low-confidence: try Claude Vision as enrichment (if available)
            if frames:
                from app.ai.vision_interpreter import VisionInterpreter
                interpreter = VisionInterpreter.get()
                if interpreter.available:
                    try:
                        result = interpreter.interpret(
                            frames,
                            max_frames=settings.ai_max_frames,
                            lang=self.lang,
                            lm_snapshots=lm_snapshots,
                        )
                        if result.get("glosses"):
                            return {
                                "type": "sentence",
                                "glosses": result["glosses"],
                                "sentence_fil": result["sentence_fil"],
                                "sentence_en": result["sentence_en"],
                                "confidence": result.get("confidence", "low"),
                                "current_glosses": [],
                            }
                    except Exception as exc:
                        logger.debug("Claude Vision skipped: %s", exc)

            # Return rule result regardless of confidence
            return {"type": "sentence", **rule_result, "current_glosses": []}

        # ── No landmarks at all — try Claude Vision from raw frames ──────────
        if frames:
            from app.ai.vision_interpreter import VisionInterpreter
            interpreter = VisionInterpreter.get()
            if interpreter.available:
                result = interpreter.interpret(
                    frames,
                    max_frames=settings.ai_max_frames,
                    lang=self.lang,
                    lm_snapshots=lm_snapshots,
                )
                if result.get("glosses"):
                    return {
                        "type": "sentence",
                        "glosses": result["glosses"],
                        "sentence_fil": result["sentence_fil"],
                        "sentence_en": result["sentence_en"],
                        "confidence": result.get("confidence", "low"),
                        "current_glosses": [],
                    }

        return {
            "type": "sentence",
            "glosses": [],
            "sentence_fil": "",
            "sentence_en": "",
            "confidence": "low",
            "current_glosses": [],
        }

    def _apply_mode_filter(self, result: dict) -> Optional[dict]:
        """Return result only if it matches the current mode; else None."""
        if self.mode == "all":
            return result
        gloss = (result.get("glosses") or [""])[0]
        is_number = gloss.isdigit()
        is_letter = len(gloss) == 1 and gloss.isalpha() and gloss.isupper()
        if self.mode == "numbers" and not is_number:
            return None
        if self.mode == "letters" and not is_letter:
            return None
        if self.mode == "words" and (is_number or is_letter):
            return None
        return result

    # ── Model path (fallback) ─────────────────────────────────────────────────

    def _process_frame_model(self, jpeg_bytes: bytes) -> Optional[dict]:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        from app.extraction.feature_builder import build_feature_vector
        from app.extraction.mediapipe_extractor import get_extractor
        from app.model.predictor import FSLPredictor

        extractor = get_extractor(static_image_mode=False)
        result = extractor.process_frame(frame)
        feature = build_feature_vector(result)

        window = self._window_buf.push(feature)
        if window is None:
            return self._check_pause_model()

        predictor = FSLPredictor.get()
        window_arr = np.stack(window)
        pred = predictor.predict_window(window_arr, settings.min_confidence)

        now = time.monotonic()
        response: dict = {
            "type": "prediction",
            "gloss": pred["gloss"],
            "confidence": pred["confidence"],
            "top5": pred["top5"],
            "committed": pred["committed"],
            "current_glosses": self._gloss_buf.current,
            "sentence": "",
            "has_hands": result.has_left_hand or result.has_right_hand,
            "has_pose": result.has_pose,
        }

        if pred["committed"]:
            accepted = self._gloss_buf.push(pred["gloss"])
            if accepted:
                self._last_activity_ts = now
            response["current_glosses"] = self._gloss_buf.current

        pause_event = self._check_pause_model()
        if pause_event:
            response.update(pause_event)

        return response

    def _check_pause_model(self) -> Optional[dict]:
        if not self._gloss_buf.current:
            return None
        elapsed_ms = (time.monotonic() - self._last_activity_ts) * 1000
        if elapsed_ms >= settings.pause_gap_ms:
            glosses = self._gloss_buf.flush()
            return self._build_model_sentence_response(glosses)
        return None

    def _flush_model(self) -> dict:
        glosses = self._gloss_buf.flush()
        return self._build_model_sentence_response(glosses)

    def _build_model_sentence_response(self, glosses: list[str]) -> dict:
        sentence_fil = self._mapper.map(glosses, lang="fil")
        sentence_en = self._mapper.map(glosses, lang="en")

        if settings.enable_llm_rewrite and settings.anthropic_api_key:
            from app.translation.language_model import LLMRewriter
            rewriter = LLMRewriter(settings.anthropic_api_key)
            if rewriter.available:
                sentence_fil = rewriter.rewrite(sentence_fil, lang="fil")
                sentence_en = rewriter.rewrite(sentence_en, lang="en")

        return {
            "type": "sentence",
            "glosses": glosses,
            "sentence_fil": sentence_fil,
            "sentence_en": sentence_en,
            "current_glosses": [],
        }
