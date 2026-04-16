"""Multi-branch Spatial-Temporal Transformer for FSL recognition.

Architecture overview
---------------------
Three parallel BranchEncoders process independent feature streams:
  • pose   — full-body motion cues
  • hands  — primary lexical signal (left + right concatenated)
  • face   — non-manual signals (FSL grammar markers)

A FusionLayer applies cross-attention where *hands* queries over
pose+face context, reflecting FSL grammar (hand shape is primary,
modulated by body/face).

A ClassifierHead temporal-pools the fused representation and outputs
sign logits.

Usage
-----
    model = FSLTransformer.small()    # CPU / development
    model = FSLTransformer.base()     # general training
    model = FSLTransformer.large()    # high-accuracy production
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding ──────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Branch Encoder ────────────────────────────────────────────────────────────

class BranchEncoder(nn.Module):
    """Project a single feature stream into d_model, then apply a Transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, T, d_model)"""
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


# ── Fusion Layer ──────────────────────────────────────────────────────────────

class FusionLayer(nn.Module):
    """Cross-attention fusion: hands attend over (pose + face) context.

    Motivation: in FSL hand shape is the primary lexical signal; facial
    non-manual signals and body movement modulate meaning.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        hands: torch.Tensor,       # (B, T, D) — query
        context: torch.Tensor,     # (B, T, D) — key/value (pose + face cat)
    ) -> torch.Tensor:
        q = self.norm_q(hands)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        fused = hands + attn_out
        fused = fused + self.ff(self.norm_out(fused))
        return fused


# ── Classifier Head ───────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """Temporal mean-pool + MLP classifier."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, num_classes) logits"""
        x = self.norm(x)
        x = x.mean(dim=1)           # temporal mean pool
        return self.classifier(x)


# ── Full Model ────────────────────────────────────────────────────────────────

class FSLTransformer(nn.Module):
    """Filipino Sign Language recognition transformer.

    Args:
        num_classes: Size of the gloss vocabulary (including <BLANK>).
        pose_dim: Flattened pose feature dimension (default 132).
        hand_dim: Combined left+right hand feature dimension (default 126).
        face_dim: Face feature dimension (default 210).
        d_model: Transformer hidden size.
        nhead: Number of attention heads.
        num_layers: Layers per branch encoder.
        dim_feedforward: FFN hidden size.
        dropout: Dropout rate.
    """

    POSE_DIM = 132
    HAND_DIM = 126
    FACE_DIM = 210

    def __init__(
        self,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pose_branch = BranchEncoder(self.POSE_DIM, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.hand_branch = BranchEncoder(self.HAND_DIM, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.face_branch = BranchEncoder(self.FACE_DIM, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.fusion = FusionLayer(d_model, nhead, dropout)
        # Combine pose + face into one context stream
        self.context_proj = nn.Linear(d_model * 2, d_model)
        self.head = ClassifierHead(d_model, num_classes, dropout)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, 468) float32.
            padding_mask: (B, T) bool — True where frame is padding.

        Returns:
            logits: (B, num_classes).
        """
        pose = features[:, :, :132]            # (B, T, 132)
        left = features[:, :, 132:195]         # (B, T,  63)
        right = features[:, :, 195:258]        # (B, T,  63)
        hands = torch.cat([left, right], dim=-1)  # (B, T, 126)
        face = features[:, :, 258:]            # (B, T, 210)

        pose_enc = self.pose_branch(pose, padding_mask)
        hand_enc = self.hand_branch(hands, padding_mask)
        face_enc = self.face_branch(face, padding_mask)

        context = self.context_proj(torch.cat([pose_enc, face_enc], dim=-1))
        fused = self.fusion(hand_enc, context)

        return self.head(fused)

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def small(cls, num_classes: int = 51) -> "FSLTransformer":
        """Compact model for CPU / development (≈500 K params)."""
        return cls(num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)

    @classmethod
    def base(cls, num_classes: int = 51) -> "FSLTransformer":
        """Balanced model for training (≈2 M params)."""
        return cls(num_classes, d_model=128, nhead=4, num_layers=4, dim_feedforward=256)

    @classmethod
    def large(cls, num_classes: int = 51) -> "FSLTransformer":
        """High-accuracy production model (≈8 M params)."""
        return cls(num_classes, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024)
