# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Filipino Sign Language (FSL) interpretation API.  It accepts MP4 video uploads and real-time webcam streams (WebSocket), extracts skeletal keypoints via MediaPipe Holistic, runs them through a multi-branch Spatial-Temporal Transformer, and returns Tagalog + English translations.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server (development)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run with Docker
docker compose up --build

# Extract keypoints from raw video dataset
python scripts/extract_dataset_keypoints.py --raw-dir data/raw --out-dir data/processed

# Train the model
python -m training.train --data-dir data/processed --vocab-path models/weights/vocabulary.json --device cuda

# Evaluate a trained checkpoint
python -m training.evaluate --weights models/weights/fsl_transformer_v1.pt --split test

# Run tests
pytest tests/
pytest tests/unit/test_feature_builder.py   # single file
```

## Architecture

### Feature pipeline (per frame → 468 floats)

`app/extraction/mediapipe_extractor.py` → `app/extraction/feature_builder.py`

Feature layout (defined in `app/extraction/landmark_indices.py`):
- `[0:132]`   — pose (33 landmarks × 4: x, y, z, visibility)
- `[132:195]` — left hand (21 × 3)
- `[195:258]` — right hand (21 × 3)
- `[258:468]` — 70 key face landmarks (70 × 3)

All coordinates are normalised to the shoulder midpoint and scale.  The 70 face indices are chosen specifically to encode FSL facial grammar signals (eyebrows, eyes, mouth, nose, chin).

### ML model

`app/model/architecture/multi_branch_transformer.py` — `FSLTransformer`

Three parallel `BranchEncoder` modules (temporal Transformers) process pose, hand (left+right concat), and face features independently.  A `FusionLayer` applies cross-attention (hands query, pose+face as context, motivated by FSL grammar: hand shape is the primary lexical signal modulated by face/body).  A `ClassifierHead` pools over time and outputs sign logits.

Use `FSLTransformer.small()` for CPU/dev, `.base()` for training, `.large()` for high-accuracy production.

### Real-time WebSocket pipeline

```
Client (base64 JPEG) → FrameQueue → [thread pool] MediaPipe + FeatureBuilder
  → SlidingWindowBuffer (window=30 frames, stride=10)
  → [thread pool] FSLTransformer inference
  → GlossBuffer (dedup + pause detection)
  → SentenceMapper (rule-based FSL grammar)
  → optional LLM rewrite (set ENABLE_LLM_REWRITE=true)
  → JSON response to client
```

Per-connection state lives in `app/api/ws/session.py:StreamSession`.  MediaPipe is thread-local (not thread-safe) — see `mediapipe_extractor.py`.

### Video upload (REST)

`POST /interpret/upload` in `app/api/rest/upload.py` — saves to `/tmp/fsl_uploads`, iterates all frames, extracts windows via `SlidingWindowBuffer.extract_windows_from_sequence()`, runs `batch_predict()`, deduplicates, maps to sentence.

### Translation

`app/translation/sentence_mapper.py` encodes FSL grammar rules (topic-comment word order, WH-question front-placement, BA marker for yes/no questions, HINDI negation) and maps gloss sequences to Tagalog/English.  Optional LLM rewrite via `app/translation/language_model.py` using the Anthropic API.

## Key configuration

All settings are in `app/config.py` (pydantic-settings).  Copy `.env.example` → `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_WEIGHTS_PATH` | `models/weights/fsl_transformer_v1.pt` | Trained checkpoint |
| `VOCABULARY_PATH` | `models/weights/vocabulary.json` | Gloss index map |
| `MODEL_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WINDOW_SIZE` | `30` | Frames per inference window |
| `STRIDE` | `10` | Frames between inference passes |
| `MIN_CONFIDENCE` | `0.60` | Minimum confidence to commit a gloss |
| `PAUSE_GAP_MS` | `800` | Silence gap (ms) that triggers sentence flush |
| `ENABLE_LLM_REWRITE` | `false` | LLM post-processing for fluent output |

## Dataset structure

```
data/
  raw/
    KUMAIN/001.mp4  002.mp4  ...   ← one sign per clip
    UMINOM/001.mp4  ...
  processed/
    KUMAIN/001.npy  ...            ← (T, 468) float32 arrays
    splits.json                    ← {"001": "train", "002": "val", ...}
models/weights/
  fsl_transformer_v1.pt
  vocabulary.json                  ← {"0": "<BLANK>", "1": "KUMAIN", ...}
```

## FSL-specific notes

- FSL uses **non-manual signals (NMS)** extensively — eyebrow position, mouth patterns, and head tilt change sign meaning.  The 70-landmark face subset captures these.
- Word order: **Topic-Comment**, not SVO.  `sentence_mapper.py` handles this.
- There is no standardised open-source FSL dataset.  Refer to KWF FSL dictionary and SPED curriculum for vocabulary.  Aim for 30-50 samples per sign with signer/lighting diversity.
- When a hand is not detected, its 63-float block is zero-filled — the model learns absence as a valid feature.
