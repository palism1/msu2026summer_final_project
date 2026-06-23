"""
U-Net specialist baseline using segmentation-models-pytorch.
encoder: ResNet-34 with ImageNet weights (matches PraNet encoder family).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,  # raw logits; sigmoid applied inside loss / at inference
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
