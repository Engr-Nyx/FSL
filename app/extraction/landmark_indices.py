"""Defines the flat feature vector layout and the 70 face landmark subset.

Feature layout (468 floats total per frame):
  [0 : 132]   pose   — 33 landmarks × (x, y, z, visibility)
  [132: 195]  left hand  — 21 × (x, y, z)
  [195: 258]  right hand — 21 × (x, y, z)
  [258: 468]  face   — 70 key landmarks × (x, y, z)

Coordinates are normalised to shoulder midpoint and inter-shoulder scale before
being passed to the model.  Absent hands → zero-fill.
"""

# ── Pose ──────────────────────────────────────────────────────────────────────
POSE_START = 0
POSE_END = 132      # 33 × 4
POSE_LANDMARKS = 33
POSE_DIM = 4        # x, y, z, visibility

# MediaPipe Holistic pose landmark indices used for normalisation
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# ── Hands ─────────────────────────────────────────────────────────────────────
LEFT_HAND_START = 132
LEFT_HAND_END = 195    # 21 × 3
RIGHT_HAND_START = 195
RIGHT_HAND_END = 258   # 21 × 3
HAND_LANDMARKS = 21
HAND_DIM = 3

# ── Face ──────────────────────────────────────────────────────────────────────
FACE_START = 258
FACE_END = 468         # 70 × 3
FACE_LANDMARKS = 70
FACE_DIM = 3

# The 70 MediaPipe FaceMesh indices that best encode FSL non-manual signals
# (eyebrows, eyes, mouth corners, nose tip, chin).
FSL_FACE_INDICES: list[int] = [
    # Eyebrows (left / right arch) — 10
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Eye lids — 10
    33, 160, 158, 133, 153, 362, 385, 387, 263, 373,
    # Iris approximate — 2
    468, 473,
    # Nose — 6
    1, 2, 5, 4, 19, 94,
    # Lips outer — 14
    61, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 14, 317,
    # Lips inner — 12
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 312, 13,
    # Chin / jaw — 8
    199, 175, 152, 148, 176, 149, 150, 136,
    # Cheeks — 8
    234, 93, 454, 323, 172, 58, 132, 380,
]

assert len(FSL_FACE_INDICES) == FACE_LANDMARKS, (
    f"FSL_FACE_INDICES must have exactly {FACE_LANDMARKS} entries, "
    f"got {len(FSL_FACE_INDICES)}"
)

# Total feature vector length
FEATURE_DIM = POSE_END + (RIGHT_HAND_END - LEFT_HAND_START) + (FACE_END - FACE_START)
assert FEATURE_DIM == 468
