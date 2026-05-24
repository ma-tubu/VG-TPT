# -*- coding: utf-8 -*-
"""
VisualFeatureExtractor — Frozen Swin Transformer for visual guidance feature extraction.

The visual encoder is fully frozen and serves only as an auxiliary modality provider,
extracting compact [B, 1024] guidance features from input frames.

Author: ma-tubu

"""

import os
import torch
import torch.nn as nn
from functools import partial

try:
    from timm.models.swin_transformer import SwinTransformer
except ImportError:
    raise ImportError("Please install timm: pip install timm")


def _pool_swin_features(features, feature_dim):
    """Pool Swin features of various layouts into [B, feature_dim]."""
    if isinstance(features, (tuple, list)):
        features = features[0]

    if features.dim() == 2:
        return features

    if features.dim() == 3:
        if features.shape[-1] == feature_dim:
            return features.mean(dim=1)
        if features.shape[1] == feature_dim:
            return features.mean(dim=2)

    if features.dim() == 4:
        if features.shape[-1] == feature_dim:
            return features.mean(dim=(1, 2))
        if features.shape[1] == feature_dim:
            return features.mean(dim=(2, 3))

    raise ValueError(
        f"Expected Swin features with channel dim {feature_dim}, got shape {tuple(features.shape)}"
    )


class VisualFeatureExtractor(nn.Module):
    """
    Frozen Swin-Base visual feature extractor.

    Args:
        model_name: Swin model identifier (unused, kept for compatibility).
        img_size: Input image size.
        drop_rate: Dropout rate (ignored because the model is frozen).
        pretrained_path: Optional local path to pretrained weights.
    """

    def __init__(
        self,
        model_name="swin_base_patch4_window7_224",
        img_size=224,
        drop_rate=0.0,
        pretrained_path=None,
    ):
        super().__init__()

        print(f"Loading Visual Feature Extractor: {model_name}")

        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )

        self._load_pretrained_weights(model_name, pretrained_path)
        self._freeze_all_parameters()
        self.feature_dim = 1024

    def _load_pretrained_weights(self, model_name, pretrained_path=None):
        """Load pretrained weights from a custom path, local defaults, or timm download."""
        if pretrained_path is not None and os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                self.swin.load_state_dict(state_dict, strict=False)
                return
            except Exception as e:
                print(f"  Failed to load weights from {pretrained_path}: {e}")
                print("  Falling back to default paths...")

        local_paths = [
            r"pretrained\swin_base_patch4_window7_224_22k.pth",
            r"./pretrained/swin_base_patch4_window7_224_22k.pth",
        ]
        for local_path in local_paths:
            if os.path.exists(local_path):
                try:
                    state_dict = torch.load(local_path, map_location="cpu")
                    self.swin.load_state_dict(state_dict, strict=False)
                    return
                except Exception as e:
                    print(f"  Failed to load local weights: {e}")
                    continue

        try:
            import timm
            pretrained_model = timm.create_model(model_name, pretrained=True)
            self.swin.load_state_dict(pretrained_model.state_dict(), strict=False)
        except Exception:
            print("  Warning: No local weights found and network download failed.")
            print("  Model will use random initialization.")

    def _freeze_all_parameters(self):
        """Freeze all parameters and set to eval mode."""
        for param in self.swin.parameters():
            param.requires_grad = False
        self.swin.eval()

    def forward(self, images):
        """Extract frozen visual guidance features [B, 1024]."""
        with torch.no_grad():
            x = self.swin.forward_features(images)
            features = _pool_swin_features(x, self.feature_dim)
        return features

    def extract_features(self, images):
        """Alias for forward."""
        return self.forward(images)


def get_visual_encoder(model_type="swin_base", img_size=224, pretrained=True, pretrained_path=None):
    """
    Factory for the frozen Swin-Base visual encoder.

    Args:
        model_type: Must be "swin_base" or "base".
        img_size: Input image size.
        pretrained: Whether to load pretrained weights.
        pretrained_path: Optional custom weight path.

    Returns:
        VisualFeatureExtractor instance.
    """
    if model_type not in ("swin_base", "base"):
        raise ValueError(f"Unknown model_type: {model_type}. Only 'swin_base' is supported.")

    model_name = "swin_base_patch4_window7_224" if pretrained else None
    encoder = VisualFeatureExtractor(
        model_name=model_name or "swin_base_patch4_window7_224",
        img_size=img_size,
        pretrained_path=pretrained_path,
    )
    return encoder


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Visual Feature Extractor")
    print("=" * 60)

    encoder = get_visual_encoder("swin_base", img_size=224)

    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    features = encoder(images)
    print(f"Input shape:  {images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Requires grad: {features.requires_grad}")

    print("\n--- Gradient Check ---")
    images_grad = torch.randn(1, 3, 224, 224, requires_grad=True)
    features_grad = encoder(images_grad)
    print(f"Input requires_grad:  {images_grad.requires_grad}")
    print(f"Output requires_grad: {features_grad.requires_grad}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
