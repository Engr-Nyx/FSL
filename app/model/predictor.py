"""Model loader and batch inference helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.config import settings
from app.model.architecture.multi_branch_transformer import FSLTransformer

logger = logging.getLogger(__name__)

_predictor_instance: Optional["FSLPredictor"] = None


class FSLPredictor:
    """Loads and runs the FSL transformer for inference.

    Call ``FSLPredictor.get()`` to obtain the shared singleton — the model is
    loaded once and reused across requests.
    """

    def __init__(
        self,
        weights_path: Path,
        vocab_path: Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # ── Load vocabulary ───────────────────────────────────────────────────
        with open(vocab_path) as f:
            self.vocab: dict[str, str] = json.load(f)          # "0" → "<BLANK>"
        self.idx_to_gloss = {int(k): v for k, v in self.vocab.items()}
        self.num_classes = len(self.vocab)
        logger.info("Vocabulary: %d classes", self.num_classes)

        # ── Build model ───────────────────────────────────────────────────────
        self.model = FSLTransformer.base(num_classes=self.num_classes)
        self._weights_loaded = False

        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self._weights_loaded = True
            logger.info("Loaded weights from %s", weights_path)
        else:
            logger.warning(
                "Weights not found at %s — model will return random predictions "
                "until trained weights are provided.",
                weights_path,
            )
            self.model.to(self.device)
            self.model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_window(
        self,
        window: np.ndarray,
        min_confidence: float = 0.60,
    ) -> dict:
        """Predict a single (T, 468) feature window.

        Returns:
            {
                "gloss": str,          # top-1 gloss or "<BLANK>"
                "confidence": float,   # softmax probability
                "top5": [...],         # list of {gloss, confidence}
                "committed": bool,     # True if confidence >= min_confidence
            }
        """
        tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, T, 468)
        logits = self.model(tensor)                                       # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0]                             # (num_classes,)

        top5_vals, top5_idx = torch.topk(probs, k=min(5, self.num_classes))

        top1_gloss = self.idx_to_gloss.get(int(top5_idx[0]), "<BLANK>")
        top1_conf = float(top5_vals[0])

        top5 = [
            {"gloss": self.idx_to_gloss.get(int(idx), "<BLANK>"), "confidence": float(val)}
            for idx, val in zip(top5_idx.tolist(), top5_vals.tolist())
        ]

        return {
            "gloss": top1_gloss,
            "confidence": top1_conf,
            "top5": top5,
            "committed": top1_conf >= min_confidence and top1_gloss != "<BLANK>",
        }

    @torch.no_grad()
    def batch_predict(
        self,
        windows: list[np.ndarray],
        min_confidence: float = 0.60,
    ) -> list[dict]:
        """Batch-predict multiple windows.

        Args:
            windows: List of (T, 468) numpy arrays.
            min_confidence: Minimum confidence to commit a prediction.

        Returns:
            List of prediction dicts in the same order as *windows*.
        """
        if not windows:
            return []

        tensors = [torch.from_numpy(w) for w in windows]
        # Pad to the same time dimension if windows have different lengths
        max_t = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_t, tensors[0].shape[-1])
        mask = torch.ones(len(tensors), max_t, dtype=torch.bool)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
            mask[i, :t.shape[0]] = False   # False = valid

        padded = padded.to(self.device)
        mask = mask.to(self.device)

        logits = self.model(padded, padding_mask=mask)   # (B, num_classes)
        probs = F.softmax(logits, dim=-1)                # (B, num_classes)

        results = []
        for i in range(len(windows)):
            p = probs[i]
            top5_vals, top5_idx = torch.topk(p, k=min(5, self.num_classes))
            top1_gloss = self.idx_to_gloss.get(int(top5_idx[0]), "<BLANK>")
            top1_conf = float(top5_vals[0])
            results.append({
                "gloss": top1_gloss,
                "confidence": top1_conf,
                "top5": [
                    {"gloss": self.idx_to_gloss.get(int(idx), "<BLANK>"), "confidence": float(val)}
                    for idx, val in zip(top5_idx.tolist(), top5_vals.tolist())
                ],
                "committed": top1_conf >= min_confidence and top1_gloss != "<BLANK>",
            })
        return results

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "FSLPredictor":
        global _predictor_instance
        if _predictor_instance is None:
            _predictor_instance = cls(
                weights_path=settings.weights_path_obj,
                vocab_path=settings.vocab_path_obj,
                device=settings.model_device,
            )
        return _predictor_instance
