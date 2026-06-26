"""
SAM + LoRA adapter (Phase 3).

Architecture:
  - Frozen SAM image encoder (ViT-H by default)
  - LoRA injected into Q and V projections of every transformer block
  - Lightweight CNN mask decoder (replaces SAM's prompt-based decoder)
  - Only LoRA params + decoder are trained → ~1-3% of total parameters

Supports three backbone options (set via configs/base.yaml → sam.model_type):
  - SAM  (vit_h / vit_l / vit_b)   — Kirillov et al. 2023   [primary]
  - MedSAM (vit_b)                  — Ma et al. 2024         [comparison]
  - SAM2  (hiera_large)             — Ravi et al. 2024       [comparison]

This file is Phase 3 scope; the stub below wires up the LoRA injection
and decoder so parameter counts are visible now.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA layer (injected into attention Q/V projections)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    def __init__(
        self,
        original: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original = original
        self.r = r
        self.scale = alpha / r

        d_in = original.in_features
        d_out = original.out_features

        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # freeze original weight
        for p in self.original.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


# ---------------------------------------------------------------------------
# Lightweight mask decoder (replaces SAM's prompt-based decoder)
# ---------------------------------------------------------------------------

class LightDecoder(nn.Module):
    """
    Simple 4-layer CNN that upsamples SAM image embeddings to a binary mask.
    Input:  (B, C, H/16, W/16)  — SAM ViT-H outputs 256×22×22 for 352×352 input
    Output: (B, 1, H, W)
    """

    def __init__(self, in_channels: int = 256, img_size: int = 352):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(x)
        # bilinear resize to exact target in case of off-by-one
        if out.shape[-1] != self.img_size:
            out = nn.functional.interpolate(
                out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        return out


# ---------------------------------------------------------------------------
# SAM-LoRA model wrapper
# ---------------------------------------------------------------------------

class SAMLoRA(nn.Module):
    """
    Frozen SAM image encoder + LoRA adapters + lightweight mask decoder.
    Only LoRA params and decoder are in requires_grad=True.
    """

    def __init__(self, sam_model, lora_r: int = 4, lora_alpha: float = 8.0,
                 lora_dropout: float = 0.1, img_size: int = 352):
        super().__init__()
        self.encoder = sam_model.image_encoder

        # freeze entire encoder first
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # SAM pos_embed is sized for 1024×1024 (64×64 patch grid).
        # Interpolate to our training resolution so x + pos_embed doesn't shape-mismatch.
        self._resize_pos_embed(img_size)

        # inject LoRA into Q and V projections of each attention block
        self._inject_lora(lora_r, lora_alpha, lora_dropout)

        embed_dim = next(
            m.out_channels
            for m in reversed(list(self.encoder.neck.children()))
            if isinstance(m, nn.Conv2d)
        )
        self.decoder = LightDecoder(in_channels=embed_dim, img_size=img_size)

    def _resize_pos_embed(self, img_size: int) -> None:
        pe = self.encoder.pos_embed
        if pe is None:
            return
        h = w = img_size // 16  # SAM always uses patch_size=16
        if pe.shape[1] == h and pe.shape[2] == w:
            return
        # pe: (1, H_orig, W_orig, C) — permute to NCHW for interpolate, then back
        pe_resized = nn.functional.interpolate(
            pe.permute(0, 3, 1, 2).float(),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        self.encoder.pos_embed = nn.Parameter(pe_resized, requires_grad=False)

    def _inject_lora(self, r: int, alpha: float, dropout: float):
        for module in self.encoder.modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                # SAM's attention merges QKV into one projection; split via custom wrapper
                # For now inject LoRA on the combined qkv projection
                module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, dropout=dropout)
            elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                module.q_proj = LoRALinear(module.q_proj, r=r, alpha=alpha, dropout=dropout)
            if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
                module.v_proj = LoRALinear(module.v_proj, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)   # (B, embed_dim, H', W')
        return self.decoder(features)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_medsam_lora(
    medsam_checkpoint: str,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.1,
    img_size: int = 352,
    device: str = "cuda",
) -> SAMLoRA:
    """
    Load MedSAM (SAM ViT-B fine-tuned on SA-Med2D-20M by Ma et al. 2024)
    and wrap it with LoRA + a lightweight decoder.

    MedSAM shares the exact SAM ViT-B architecture; only the weights differ.
    Checkpoint: wget https://huggingface.co/bowang-lab/MedSAM/resolve/main/medsam_vit_b.pth
    """
    return build_sam_lora(
        sam_checkpoint=medsam_checkpoint,
        model_type="vit_b",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        img_size=img_size,
        device=device,
    )


def build_sam_lora(
    sam_checkpoint: str,
    model_type: str = "vit_h",
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.1,
    img_size: int = 352,
    device: str = "cuda",
) -> SAMLoRA:
    """
    Load a SAM checkpoint and wrap it with LoRA + a lightweight decoder.
    Requires the segment-anything package: pip install git+https://github.com/facebookresearch/segment-anything.git
    """
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    model = SAMLoRA(
        sam_model=sam,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        img_size=img_size,
    )
    return model
