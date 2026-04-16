"""Evaluate a trained FSLTransformer checkpoint on the test split.

Usage
-----
    python -m training.evaluate \
        --weights models/weights/fsl_transformer_v1.pt \
        --data-dir data/processed \
        --vocab-path models/weights/vocabulary.json \
        --split test
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.model.architecture.multi_branch_transformer import FSLTransformer
from training.train import FSLDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    with open(args.vocab_path) as f:
        vocab_raw: dict = json.load(f)
    idx_to_gloss = {int(k): v for k, v in vocab_raw.items()}
    gloss_to_idx = {v: int(k) for k, v in vocab_raw.items()}
    num_classes = len(vocab_raw)

    with open(args.data_dir / "splits.json") as f:
        splits = json.load(f)

    ds = FSLDataset(args.data_dir, gloss_to_idx, args.split, splits, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = FSLTransformer.base(num_classes=num_classes)
    state = torch.load(args.weights, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels, mask in tqdm(loader, desc="Evaluating"):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            logits = model(features, padding_mask=mask)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    logger.info("Accuracy on [%s]: %.2f%%", args.split, accuracy * 100)

    label_names = [idx_to_gloss.get(i, str(i)) for i in sorted(set(all_labels))]
    print("\n" + classification_report(all_labels, all_preds, target_names=label_names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FSLTransformer.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--vocab-path", default="models/weights/vocabulary.json", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
