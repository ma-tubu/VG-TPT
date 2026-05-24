# -*- coding: utf-8 -*-
"""
VisualTextClassifier — VG-TPT multimodal sentiment classifier.

Integrates a frozen visual encoder (auxiliary modality) with a
vision-guided BERT encoder (primary modality) for sentiment prediction.

Author: ma-tubu

"""

import torch
import torch.nn as nn


class VisualTextClassifier(nn.Module):
    """
    VG-TPT visual-text classifier.

    Args:
        text_model: VisualGuidedBertClassifier instance.
        visual_model: Frozen VisualFeatureExtractor instance.
        num_classes: Number of sentiment classes (1 for regression).
        fusion_method: "vg_tpt", "text_only", "visual_only", or "late_concat".
        prompt_bank_size: Number of prompt bases in the trainable prompt basis bank.
        visual_feature_dim: Dimension of the frozen visual guidance feature.
    """

    def __init__(
        self,
        text_model,
        visual_model,
        num_classes,
        fusion_method="vg_tpt",
        prompt_bank_size=8,
        moe_top_k=1,
        visual_feature_dim=None,
    ):
        super().__init__()
        self.text_model = text_model
        self.visual_model = visual_model
        self.fusion_method = fusion_method
        self.prompt_bank_size = prompt_bank_size
        self.moe_top_k = moe_top_k
        self.visual_feature_dim = visual_feature_dim or getattr(visual_model, "feature_dim", 1024)

        self.fusion_head = None
        if fusion_method == "late_concat":
            self.fusion_head = nn.Linear(768 + self.visual_feature_dim, num_classes)

        self.visual_classifier = nn.Linear(self.visual_feature_dim, num_classes)

        # Freeze the visual encoder; it serves only as an auxiliary guidance provider.
        if self.visual_model is not None:
            for param in self.visual_model.parameters():
                param.requires_grad = False
            self.visual_model.eval()

    def forward(self, text_input, visual_input=None, attention_mask=None, return_features=False, blind_prompt=False):
        """Forward pass. Returns (logits, extra_dict) or features if return_features=True."""
        if return_features:
            return self.forward_features(text_input, visual_input, attention_mask)

        if self.fusion_method == "vg_tpt":
            return self.forward_vg_tpt(text_input, visual_input, attention_mask, blind_prompt)
        elif self.fusion_method == "text_only":
            return self.forward_text_only(text_input)
        elif self.fusion_method == "visual_only":
            return self.forward_visual_only(visual_input)
        elif self.fusion_method == "late_concat":
            return self.forward_late_concat(text_input, visual_input, attention_mask)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

    def forward_vg_tpt(self, text_input, visual_input, attention_mask=None, blind_prompt=False):
        """VG-TPT: text-anchored encoding with frozen visual guidance."""
        with torch.no_grad():
            visual_feature = self._extract_visual_feature(visual_input)

        logits, extra = self.text_model(
            text_input,
            visual_feature=visual_feature,
            blind_prompt=blind_prompt,
        )
        return logits, extra

    def forward_text_only(self, text_input):
        """Text-only baseline without visual guidance."""
        logits, extra = self.text_model(text_input, visual_feature=None)
        return logits, extra

    def forward_visual_only(self, visual_input):
        """Visual-only baseline."""
        with torch.no_grad():
            visual_feature = self._extract_visual_feature(visual_input)

        logits = self.visual_classifier(visual_feature)
        return logits, None

    def forward_late_concat(self, text_input, visual_input, attention_mask=None):
        """Late fusion baseline: concatenate text and visual features before classification."""
        with torch.no_grad():
            text_feature = self.text_model(
                text_input,
                visual_feature=None,
                return_features=True,
            )

        with torch.no_grad():
            visual_feature = self._extract_visual_feature(visual_input)

        concat_feature = torch.cat([text_feature, visual_feature], dim=-1)
        logits = self.fusion_head(concat_feature)

        extra_out = {"cls_": concat_feature}
        return logits, extra_out

    def forward_features(self, text_input, visual_input, attention_mask=None):
        """Extract vision-guided text features and frozen visual features."""
        self.eval()

        with torch.no_grad():
            visual_feature = self._extract_visual_feature(visual_input)
            text_fused_feature = self.text_model.extract_fused_features(
                text_input,
                visual_feature,
            )

        return text_fused_feature, visual_feature

    def _extract_visual_feature(self, visual_input):
        """Extract frozen visual guidance features [B, visual_feature_dim]."""
        if visual_input is None or self.visual_model is None:
            return None

        if hasattr(self.visual_model, "extract_features"):
            visual_feature = self.visual_model.extract_features(visual_input)
        else:
            visual_feature = self.visual_model(visual_input)
        return visual_feature


def get_visual_text_classifier(
    num_classes=7,
    bert_pretrained_path="bert-base-uncased",
    img_size=224,
    freeze_layers=8,
    unfreeze_layers=0,
    prompt_bank_size=8,
    prompt_length=10,
    guide_tokens_length=128,
    main_tokens_length=128,
    fusion_method="vg_tpt",
    pretrained_visual=True,
    swin_pretrained_path=None,
    blind_prompt=False,
    guidance_mode="visual",
    text_model=None,
    visual_model=None,
):
    """
    Factory function for VisualTextClassifier.

    Args:
        freeze_layers: First layer index to receive VG-TPT prompts.
        unfreeze_layers: Number of BERT backbone layers to unfreeze.
        prompt_bank_size: Number of prompt bases in the trainable prompt basis bank.
        prompt_length: Length of static and adaptive prompt tokens.
        guide_tokens_length: Visual guidance routing subspace dimension.
        main_tokens_length: Text-side routing subspace dimension.
        fusion_method: "vg_tpt", "text_only", "visual_only", or "late_concat".
        pretrained_visual: Whether to load Swin pretrained weights.
        swin_pretrained_path: Optional path to Swin pretrained weights.
        blind_prompt: If True, text tokens cannot attend to prompt tokens.
        guidance_mode: Prompt generation mode ("visual", "text_only_prompt", etc.).
        text_model: Optional pre-built VisualGuidedBertClassifier.
        visual_model: Optional pre-built frozen visual encoder.

    Returns:
        VisualTextClassifier instance.
    """
    if visual_model is None:
        from .visual_encoder import get_visual_encoder
        visual_model = get_visual_encoder(
            model_type="swin_base",
            img_size=img_size,
            pretrained=pretrained_visual,
            pretrained_path=swin_pretrained_path,
        )

    visual_feature_dim = getattr(visual_model, "feature_dim", 1024)

    if text_model is None:
        from .bert_visual_guided import get_visual_guided_bert
        text_model = get_visual_guided_bert(
            num_classes=num_classes,
            bert_pretrained_path=bert_pretrained_path,
            freeze_layers=freeze_layers,
            unfreeze_layers=unfreeze_layers,
            prompt_bank_size=prompt_bank_size,
            prompt_length=prompt_length,
            guide_tokens_length=guide_tokens_length,
            main_tokens_length=main_tokens_length,
            instructor_dim=visual_feature_dim,
            use_instruct=True,
            blind_prompt=blind_prompt,
            guidance_mode=guidance_mode,
        )

    model = VisualTextClassifier(
        text_model=text_model,
        visual_model=visual_model,
        num_classes=num_classes,
        fusion_method=fusion_method,
        prompt_bank_size=prompt_bank_size,
        visual_feature_dim=visual_feature_dim,
    )

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing VisualTextClassifier")
    print("=" * 60)

    model = get_visual_text_classifier(
        num_classes=7,
        fusion_method="vg_tpt",
        freeze_layers=8,
        prompt_bank_size=8,
    )

    batch_size = 2
    texts = ["This movie is great", "I feel sad today"]
    images = torch.randn(batch_size, 3, 224, 224)

    print(f"\nInput text: {texts}")
    print(f"Input image shape: {images.shape}")

    print("\n--- VG-TPT Mode ---")
    logits, extra = model(texts, images)
    print(f"Output logits shape: {logits.shape}")
    print(f"Extra keys: {extra.keys()}")
    print(f"Balance loss: {extra['balance_loss']:.4f}")

    print("\n--- Text Only Mode ---")
    model_text_only = get_visual_text_classifier(
        num_classes=7,
        fusion_method="text_only",
    )
    logits_text, _ = model_text_only(texts)
    print(f"Output logits shape: {logits_text.shape}")

    print("\n--- Late Concat Mode ---")
    model_concat = get_visual_text_classifier(
        num_classes=7,
        fusion_method="late_concat",
    )
    logits_concat, _ = model_concat(texts, images)
    print(f"Output logits shape: {logits_concat.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
