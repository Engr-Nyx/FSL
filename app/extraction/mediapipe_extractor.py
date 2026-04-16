"""Thread-local MediaPipe extractor — supports both legacy (0.10.x solutions)
and modern (Tasks API) MediaPipe installations.

MediaPipe is NOT thread-safe — each thread must own its own instance.
Use ``get_extractor()`` to get the thread-local singleton.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_local = threading.local()

# Detect which MediaPipe API is available
try:
    import mediapipe as mp
    _HAS_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
except ImportError:
    _HAS_SOLUTIONS = False

_HAS_TASKS = False
try:
    from mediapipe.tasks import python as mp_tasks  # noqa: F401
    _HAS_TASKS = True
except ImportError:
    pass


@dataclass
class ExtractionResult:
    pose: Optional[np.ndarray] = None       # (33, 4) or None
    left_hand: Optional[np.ndarray] = None  # (21, 3) or None
    right_hand: Optional[np.ndarray] = None # (21, 3) or None
    face: Optional[np.ndarray] = None       # (70, 3) or None
    has_left_hand: bool = False
    has_right_hand: bool = False
    has_pose: bool = False
    has_face: bool = False


# ── Legacy (mp.solutions) ─────────────────────────────────────────────────────

class _LegacyExtractor:
    def __init__(self, static_image_mode=False, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=True,
        )
        from app.extraction.landmark_indices import FSL_FACE_INDICES
        self._face_idx = FSL_FACE_INDICES

    def process_frame(self, bgr_frame: np.ndarray) -> ExtractionResult:
        import cv2 as _cv2
        rgb = _cv2.cvtColor(bgr_frame, _cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        r = self._holistic.process(rgb)
        rgb.flags.writeable = True

        out = ExtractionResult()
        if r.pose_landmarks:
            out.pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                  for lm in r.pose_landmarks.landmark], dtype=np.float32)
            out.has_pose = True
        if r.left_hand_landmarks:
            out.left_hand = np.array([[lm.x, lm.y, lm.z]
                                       for lm in r.left_hand_landmarks.landmark], dtype=np.float32)
            out.has_left_hand = True
        if r.right_hand_landmarks:
            out.right_hand = np.array([[lm.x, lm.y, lm.z]
                                        for lm in r.right_hand_landmarks.landmark], dtype=np.float32)
            out.has_right_hand = True
        if r.face_landmarks:
            all_face = np.array([[lm.x, lm.y, lm.z]
                                   for lm in r.face_landmarks.landmark], dtype=np.float32)
            if len(all_face) > max(self._face_idx):
                out.face = all_face[self._face_idx]
                out.has_face = True
        return out

    def close(self):
        self._holistic.close()


# ── Modern Tasks API (separate hand + pose detectors) ─────────────────────────

class _TasksExtractor:
    """Uses the new MediaPipe Tasks API (available in mp >= 0.10.14 without solutions)."""

    def __init__(self, **_kwargs):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as bo

        # Hand Landmarker
        hand_opts = vision.HandLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=self._model_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hand_det = vision.HandLandmarker.create_from_options(hand_opts)

    @staticmethod
    def _model_path(name: str) -> str:
        """Return path to bundled model or raise informative error."""
        import os
        paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "mediapipe", name),
            name,
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"MediaPipe model '{name}' not found. "
            "Download from https://storage.googleapis.com/mediapipe-models/ "
            "and place in models/mediapipe/"
        )

    def process_frame(self, bgr_frame: np.ndarray) -> ExtractionResult:
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._hand_det.detect(mp_image)

        out = ExtractionResult()
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            lm_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
            # Determine handedness
            handedness = result.handedness[i][0].category_name if result.handedness else "Right"
            if handedness == "Left":
                out.left_hand = lm_arr
                out.has_left_hand = True
            else:
                out.right_hand = lm_arr
                out.has_right_hand = True
        return out

    def close(self):
        self._hand_det.close()


# ── Null extractor (returns empty result — used when MediaPipe not installed) ──

class _NullExtractor:
    def process_frame(self, bgr_frame: np.ndarray) -> ExtractionResult:
        return ExtractionResult()
    def close(self): pass


# ── Public alias ──────────────────────────────────────────────────────────────

class MediaPipeExtractor:
    """Auto-selects the best available MediaPipe backend."""

    def __init__(self, **kwargs):
        if _HAS_SOLUTIONS:
            self._impl = _LegacyExtractor(**kwargs)
        elif _HAS_TASKS:
            self._impl = _TasksExtractor(**kwargs)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "MediaPipe not available — extraction will return empty results."
            )
            self._impl = _NullExtractor()

    def process_frame(self, bgr_frame: np.ndarray) -> ExtractionResult:
        return self._impl.process_frame(bgr_frame)

    def close(self):
        self._impl.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_extractor(**kwargs) -> MediaPipeExtractor:
    if not hasattr(_local, "extractor"):
        _local.extractor = MediaPipeExtractor(**kwargs)
    return _local.extractor
