"""Extract MediaPipe keypoints from raw video clips and save as .npy arrays.

Usage
-----
    python scripts/extract_dataset_keypoints.py \
        --raw-dir data/raw \
        --out-dir data/processed

Dataset layout expected under --raw-dir:
    data/raw/
        KUMAIN/001.mp4  002.mp4 ...
        UMINOM/001.mp4  ...

Output under --out-dir:
    data/processed/
        KUMAIN/001.npy   (shape: T × 468, dtype float32)
        UMINOM/001.npy
        splits.json      ({"KUMAIN/001": "train", "KUMAIN/002": "val", ...})
"""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_video(video_path: Path) -> np.ndarray | None:
    """Extract feature vectors for every frame in the video.

    Returns: (T, 468) float32 array, or None if no frames extracted.
    """
    from app.extraction.feature_builder import build_feature_vector
    from app.extraction.mediapipe_extractor import MediaPipeExtractor

    extractor = MediaPipeExtractor(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(str(video_path))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = extractor.process_frame(frame)
        features = build_feature_vector(result)
        frames.append(features)

    cap.release()
    extractor.close()

    if not frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FSL keypoints from raw video clips.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Root dir with sign subdirs.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for .npy files.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    sign_dirs = sorted([d for d in args.raw_dir.iterdir() if d.is_dir()])
    if not sign_dirs:
        logger.error("No sign subdirectories found in %s", args.raw_dir)
        return

    splits: dict[str, str] = {}
    total_ok = 0
    total_skip = 0

    for sign_dir in tqdm(sign_dirs, desc="Signs"):
        sign_label = sign_dir.name
        clips = sorted(sign_dir.glob("*.mp4")) + sorted(sign_dir.glob("*.mov")) + sorted(sign_dir.glob("*.webm"))

        if not clips:
            logger.warning("No clips in %s", sign_dir)
            continue

        random.shuffle(clips)
        n = len(clips)
        n_train = max(1, int(n * args.train_ratio))
        n_val = max(1, int(n * args.val_ratio))

        def split_for(i):
            if i < n_train:
                return "train"
            elif i < n_train + n_val:
                return "val"
            return "test"

        out_sign_dir = args.out_dir / sign_label
        out_sign_dir.mkdir(parents=True, exist_ok=True)

        for i, clip in enumerate(tqdm(clips, desc=sign_label, leave=False)):
            arr = extract_video(clip)
            if arr is None or len(arr) == 0:
                logger.warning("Skipped (no frames): %s", clip)
                total_skip += 1
                continue

            out_path = out_sign_dir / (clip.stem + ".npy")
            np.save(str(out_path), arr)
            splits[f"{sign_label}/{clip.stem}"] = split_for(i)
            total_ok += 1

    splits_path = args.out_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info("Done — %d extracted, %d skipped. Splits saved to %s", total_ok, total_skip, splits_path)


if __name__ == "__main__":
    main()
