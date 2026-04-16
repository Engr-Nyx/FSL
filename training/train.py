"""Train the FSLTransformer on extracted keypoint data.

Usage
-----
    python -m training.train \
        --data-dir data/processed \
        --vocab-path models/weights/vocabulary.json \
        --device cuda \
        --epochs 50 \
        --batch-size 32

Checkpoints are saved to:
    models/weights/fsl_transformer_v1.pt   (best val accuracy)
    models/weights/fsl_transformer_last.pt (last epoch)
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.model.architecture.multi_branch_transformer import FSLTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FSLDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        vocab: dict[str, int],        # gloss → class index
        split: str,
        splits: dict[str, str],
        window_size: int = 30,
        stride: int = 10,
        augment: bool = False,
    ) -> None:
        self.window_size = window_size
        self.vocab = vocab
        self.augment = augment
        self.samples: list[tuple[np.ndarray, int]] = []

        for key, sp in splits.items():
            if sp != split:
                continue
            sign_label, stem = key.split("/", 1)
            if sign_label not in vocab:
                continue
            npy_path = data_dir / sign_label / (stem + ".npy")
            if not npy_path.exists():
                continue

            seq = np.load(str(npy_path))          # (T, 468)
            class_idx = vocab[sign_label]

            # Extract all windows from this sequence
            for start in range(0, len(seq) - window_size + 1, stride):
                w = seq[start: start + window_size]
                self.samples.append((w.copy(), class_idx))

        logger.info("Dataset [%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, label = self.samples[idx]
        tensor = torch.from_numpy(window).float()   # (T, 468)

        if self.augment:
            # Temporal jitter: randomly drop/repeat up to 2 frames
            if random.random() < 0.3:
                drop = random.randint(1, 2)
                indices = sorted(random.sample(range(len(tensor)), len(tensor) - drop))
                tensor = tensor[indices]
                # Pad back to window_size
                pad = self.window_size - len(tensor)
                if pad > 0:
                    tensor = torch.cat([tensor, tensor[-1:].expand(pad, -1)])

            # Small Gaussian noise on hand landmarks
            if random.random() < 0.5:
                noise = torch.randn_like(tensor) * 0.005
                tensor = tensor + noise

        return tensor, label


def collate_fn(batch):
    tensors, labels = zip(*batch)
    max_t = max(t.shape[0] for t in tensors)
    padded = torch.zeros(len(tensors), max_t, tensors[0].shape[-1])
    mask = torch.ones(len(tensors), max_t, dtype=torch.bool)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t
        mask[i, :t.shape[0]] = False
    return padded, torch.tensor(labels), mask


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    with open(args.vocab_path) as f:
        vocab_raw: dict = json.load(f)
    # vocab_raw: idx_str → gloss. Invert to gloss → idx.
    gloss_to_idx = {v: int(k) for k, v in vocab_raw.items()}
    num_classes = len(vocab_raw)

    with open(args.data_dir / "splits.json") as f:
        splits: dict[str, str] = json.load(f)

    train_ds = FSLDataset(args.data_dir, gloss_to_idx, "train", splits, args.window_size, args.stride, augment=True)
    val_ds = FSLDataset(args.data_dir, gloss_to_idx, "val", splits, args.window_size, args.stride, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model_cls = {"small": FSLTransformer.small, "base": FSLTransformer.base, "large": FSLTransformer.large}
    model = model_cls[args.model_size](num_classes=num_classes).to(device)
    logger.info("Model: %s — %d parameters", args.model_size, sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for features, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            logits = model(features, padding_mask=mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += len(labels)

        train_acc = total_correct / total_n
        avg_loss = total_loss / total_n
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)
                logits = model(features, padding_mask=mask)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_n += len(labels)

        val_acc = val_correct / val_n if val_n > 0 else 0.0
        logger.info(
            "Epoch %d/%d — loss=%.4f  train_acc=%.1f%%  val_acc=%.1f%%  lr=%.2e",
            epoch, args.epochs, avg_loss, train_acc * 100, val_acc * 100,
            optimizer.param_groups[0]["lr"],
        )

        # ── Checkpoints ────────────────────────────────────────────────────────
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc}
        torch.save(checkpoint, weights_dir / "fsl_transformer_last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, weights_dir / "fsl_transformer_v1.pt")
            logger.info("  ✓ New best val_acc=%.1f%% — checkpoint saved", val_acc * 100)

    logger.info("Training complete. Best val accuracy: %.1f%%", best_val_acc * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FSLTransformer.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--vocab-path", default="models/weights/vocabulary.json", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="base")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
