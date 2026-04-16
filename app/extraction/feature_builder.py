"""Build normalised 468-float feature vectors from MediaPipe ExtractionResult.

Normalisation strategy
----------------------
* Pose landmarks — subtract shoulder midpoint, divide by inter-shoulder distance.
* Hand landmarks — subtract wrist (landmark 0), divide by middle-finger MCP
  distance (landmark 9) to achieve scale invariance.
* Face landmarks — subtract nose tip (landmark 1), divide by the
  ear-to-ear distance for scale.
* Absent components → zero-fill.
"""

import numpy as np

from app.extraction.mediapipe_extractor import ExtractionResult
from app.extraction.landmark_indices import (
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
    FACE_LANDMARKS,
    FEATURE_DIM,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
)

# ── Hand landmark indices used for normalisation ──────────────────────────────
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9   # landmark 9 = middle finger MCP — good scale reference

# ── Face landmark indices used for normalisation ─────────────────────────────
NOSE_TIP_IDX = 0     # index 0 within the 70-subset (mapped from MP 1)
# Approx left / right ear positions within the 70-subset
LEFT_EAR_IDX = 68    # mapped from MP 234
RIGHT_EAR_IDX = 69   # mapped from MP 454


def _normalise_hand(hand: np.ndarray) -> np.ndarray:
    """Normalise a (21, 3) hand landmark array.

    Returns a (21, 3) array with:
    - wrist at origin
    - scale: distance from wrist to middle-finger MCP == 1
    """
    origin = hand[WRIST_IDX].copy()
    centred = hand - origin
    scale = float(np.linalg.norm(centred[MIDDLE_MCP_IDX]) + 1e-8)
    return centred / scale


def _normalise_pose(pose: np.ndarray) -> np.ndarray:
    """Normalise a (33, 4) pose landmark array (x, y, z, vis).

    Returns (33, 4) with shoulder midpoint at origin and inter-shoulder
    distance == 1.
    """
    left_sh = pose[LEFT_SHOULDER_IDX, :3]
    right_sh = pose[RIGHT_SHOULDER_IDX, :3]
    midpoint = (left_sh + right_sh) / 2.0
    scale = float(np.linalg.norm(left_sh - right_sh) + 1e-8)

    normalised = pose.copy()
    normalised[:, :3] = (pose[:, :3] - midpoint) / scale
    return normalised


def _normalise_face(face: np.ndarray) -> np.ndarray:
    """Normalise a (70, 3) face landmark array.

    Returns (70, 3) with nose tip at origin and ear-to-ear == 1.
    """
    origin = face[NOSE_TIP_IDX].copy()
    centred = face - origin
    if len(face) > max(LEFT_EAR_IDX, RIGHT_EAR_IDX):
        scale = float(np.linalg.norm(centred[LEFT_EAR_IDX] - centred[RIGHT_EAR_IDX]) + 1e-8)
    else:
        scale = 1.0
    return centred / scale


def build_feature_vector(result: ExtractionResult) -> np.ndarray:
    """Convert an ExtractionResult into a flat 468-float feature vector.

    Missing components are zero-filled.

    Returns:
        np.ndarray of shape (468,) and dtype float32.
    """
    parts: list[np.ndarray] = []

    # ── Pose (132) ────────────────────────────────────────────────────────────
    if result.has_pose and result.pose is not None:
        norm_pose = _normalise_pose(result.pose)     # (33, 4)
        parts.append(norm_pose.flatten())            # 132
    else:
        parts.append(np.zeros(POSE_LANDMARKS * 4, dtype=np.float32))

    # ── Left hand (63) ────────────────────────────────────────────────────────
    if result.has_left_hand and result.left_hand is not None:
        norm_lh = _normalise_hand(result.left_hand)  # (21, 3)
        parts.append(norm_lh.flatten())              # 63
    else:
        parts.append(np.zeros(HAND_LANDMARKS * 3, dtype=np.float32))

    # ── Right hand (63) ───────────────────────────────────────────────────────
    if result.has_right_hand and result.right_hand is not None:
        norm_rh = _normalise_hand(result.right_hand)  # (21, 3)
        parts.append(norm_rh.flatten())               # 63
    else:
        parts.append(np.zeros(HAND_LANDMARKS * 3, dtype=np.float32))

    # ── Face (210) ────────────────────────────────────────────────────────────
    if result.has_face and result.face is not None:
        norm_face = _normalise_face(result.face)     # (70, 3)
        parts.append(norm_face.flatten())            # 210
    else:
        parts.append(np.zeros(FACE_LANDMARKS * 3, dtype=np.float32))

    vec = np.concatenate(parts).astype(np.float32)
    assert vec.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {vec.shape}"
    return vec


class SlidingWindowBuffer:
    """Accumulates per-frame feature vectors and emits inference windows.

    Args:
        window_size: Number of frames per window.
        stride: Frames to advance between consecutive windows.
    """

    def __init__(self, window_size: int = 30, stride: int = 10) -> None:
        self.window_size = window_size
        self.stride = stride
        self._buffer: list[np.ndarray] = []
        self._frames_since_last_window = 0

    def push(self, feature: np.ndarray) -> list[np.ndarray] | None:
        """Add a feature vector.  Returns a window (list[np.ndarray]) when
        enough frames have accumulated, else None."""
        self._buffer.append(feature)
        self._frames_since_last_window += 1

        if (
            len(self._buffer) >= self.window_size
            and self._frames_since_last_window >= self.stride
        ):
            window = self._buffer[-self.window_size:]
            self._frames_since_last_window = 0
            return window
        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_last_window = 0

    @staticmethod
    def extract_windows_from_sequence(
        sequence: list[np.ndarray],
        window_size: int = 30,
        stride: int = 10,
    ) -> list[np.ndarray]:
        """Extract all non-overlapping windows from a complete sequence.

        Returns a list of (window_size, FEATURE_DIM) numpy arrays.
        """
        windows = []
        for start in range(0, len(sequence) - window_size + 1, stride):
            w = np.stack(sequence[start: start + window_size])  # (W, 468)
            windows.append(w)
        return windows
