# -*- coding : utf-8 -*-
# @FileName  : bert_visual_guided.py
# @Description: VG-TPT visual-guided BERT
# @Author: ma-tubu
# Architecture:
# - Base: Pretrained BERT (12 layers, 768 dim)
# - Visual features from a frozen visual encoder guide BERT via VG-TPT prompts
# - Prompts are inserted after [CLS] and removed before next layer (independent per layer)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, logging as transformers_logging


class VisualGuidedBertClassifier(nn.Module):
    """
    VG-TPT visual-guided BERT classifier.

    Prompt structure (per layer, after [CLS]):
        [static_prompt(10) | routed_prompt(10) | visual_instruct(1)] = 21 tokens

    Prompt routing:
        - Visual/default: [current CLS route | visual route] -> prompt basis weights
        - Text-only ablation: current CLS route -> prompt basis weights
        - Visual-only-route ablation: visual route -> prompt basis weights
        - Weighted composition of prompt bases

    Args:
        num_classes: Number of classification classes.
        bert_pretrained_path: BERT pretrained path.
        freeze_layers: Number of BERT layers to freeze without VG-TPT prompts.
        unfreeze_layers: Number of BERT backbone layers to unfreeze.
        prompt_bank_size: Number of adaptive prompt bases in the prompt bank.
        prompt_length: Length of static and routed prompt tokens.
        guide_tokens_length: Visual guidance routing length.
        main_tokens_length: Text routing length.
        instructor_dim: Dimension of frozen visual encoder output. Swin-Base is 1024.
        use_instruct: Whether to use visual instruction prompt.
        blind_prompt: If True, text tokens cannot attend to prompt tokens.
        guidance_mode: Prompt ablation mode: visual, text_only_prompt,
            fixed_prompt, shared_generated_prompt, visual_only_route, or
            uniform_prompt_fusion.
    """

    def __init__(
        self,
        num_classes=7,
        bert_pretrained_path="bert-base-uncased",
        freeze_layers=8,
        unfreeze_layers=0,
        prompt_bank_size=8,
        prompt_length=10,
        guide_tokens_length=128,
        main_tokens_length=128,
        instructor_dim=1024,
        use_instruct=True,
        blind_prompt=False,
        guidance_mode="visual",
    ):
        super(VisualGuidedBertClassifier, self).__init__()

        if guidance_mode not in {
            "visual",
            "text_only_prompt",
            "fixed_prompt",
            "shared_generated_prompt",
            "visual_only_route",
            "uniform_prompt_fusion",
        }:
            raise ValueError(f"Unknown guidance_mode: {guidance_mode}")

        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.unfreeze_layers = unfreeze_layers
        self.prompt_bank_size = prompt_bank_size
        self.prompt_length = prompt_length
        self.guide_tokens_length = guide_tokens_length
        self.main_tokens_length = main_tokens_length
        self.instructor_dim = instructor_dim
        self.guidance_mode = guidance_mode
        self.use_adaptive_prompt = guidance_mode in {
            "visual",
            "text_only_prompt",
            "shared_generated_prompt",
            "visual_only_route",
        }
        self.use_uniform_prompt_fusion = guidance_mode == "uniform_prompt_fusion"
        self.use_prompt_bank = self.use_adaptive_prompt or self.use_uniform_prompt_fusion
        self.use_text_route = guidance_mode in {"visual", "text_only_prompt", "shared_generated_prompt"}
        self.use_visual_route = guidance_mode in {"visual", "shared_generated_prompt", "visual_only_route"}
        self.use_visual_instruct = use_instruct and guidance_mode in {
            "visual",
            "fixed_prompt",
            "shared_generated_prompt",
            "visual_only_route",
            "uniform_prompt_fusion",
        }
        self.use_instruct = self.use_visual_instruct
        self.blind_prompt = blind_prompt
        self.num_prompts = prompt_length + prompt_length + (1 if self.use_visual_instruct else 0)
        self.num_prompt_entries = 1 + prompt_bank_size

        # Load pretrained BERT
        print(f"Loading BERT: {bert_pretrained_path}")
        previous_transformers_verbosity = transformers_logging.get_verbosity()
        transformers_progress_enabled = None
        if hasattr(transformers_logging, "is_progress_bar_enabled"):
            transformers_progress_enabled = transformers_logging.is_progress_bar_enabled()
        transformers_logging.set_verbosity_error()
        if hasattr(transformers_logging, "disable_progress_bar"):
            transformers_logging.disable_progress_bar()
        try:
            self.bert_encoder = BertModel.from_pretrained(bert_pretrained_path)
            self.tokenizer = BertTokenizerFast.from_pretrained(bert_pretrained_path)
        finally:
            transformers_logging.set_verbosity(previous_transformers_verbosity)
            if transformers_progress_enabled and hasattr(transformers_logging, "enable_progress_bar"):
                transformers_logging.enable_progress_bar()
        self.dim = self.bert_encoder.config.hidden_size
        self.n_layers = self.bert_encoder.config.num_hidden_layers

        # Freeze BERT backbone by default
        self._freeze_bert_backbone()

        # Prompt routing projections
        if self.use_adaptive_prompt:
            if self.use_text_route:
                self.main_route_proj = nn.Linear(self.dim, main_tokens_length)
            if self.use_visual_route:
                self.guide_route_proj = nn.Linear(instructor_dim, guide_tokens_length)

            # Frozen routing anchors [route_dim, prompt_bank_size].
            route_dim = (
                (main_tokens_length if self.use_text_route else 0)
                + (guide_tokens_length if self.use_visual_route else 0)
            )
            self.frozen_routing_anchors = nn.Parameter(
                torch.zeros(route_dim, prompt_bank_size)
            )
            nn.init.orthogonal_(self.frozen_routing_anchors)
            self.frozen_routing_anchors.requires_grad = False

        # Visual instruction projection: visual [B, instructor_dim] -> [B, 768]
        if self.use_visual_instruct:
            self.visual_instruction_proj = nn.Sequential(
                nn.Linear(instructor_dim, self.dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )

        self.prompt_dropout = nn.Dropout(0.1)

        if self.use_prompt_bank:
            # Entry 0 stores the static prompt; entries 1..K form the prompt basis bank.
            self.prompt_bank_embeddings = nn.Parameter(
                torch.zeros(
                    self.num_prompt_entries,
                    self.n_layers,
                    prompt_length,
                    self.dim,
                )
            )
            nn.init.trunc_normal_(self.prompt_bank_embeddings, std=0.02)
        else:
            # Layer-wise fixed prompt: no sample-adaptive routing, but each layer has
            # its own learnable prompt with the same token count as [static|routed].
            self.fixed_prompt_embeddings = nn.Parameter(
                torch.zeros(
                    self.n_layers,
                    prompt_length * 2,
                    self.dim,
                )
            )
            nn.init.trunc_normal_(self.fixed_prompt_embeddings, std=0.02)

        self.classifier = nn.Linear(self.dim, num_classes)
        self._print_parameter_stats()

    def _freeze_bert_backbone(self):
        """Freeze all BERT parameters."""
        for param in self.bert_encoder.parameters():
            param.requires_grad = False

        if self.unfreeze_layers > 0:
            start_layer = self.n_layers - self.unfreeze_layers
            for layer_idx in range(start_layer, self.n_layers):
                for param in self.bert_encoder.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
            print(f"  Unfrozen backbone layers {start_layer}-{self.n_layers - 1}")

    def _print_parameter_stats(self):
        """Print trainable/frozen parameter statistics."""
        frozen = 0
        trainable = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        total = frozen + trainable
        print(f"\nVisualGuidedBert Parameter Statistics:")
        print(f"  Frozen:    {frozen:,} ({100 * frozen / total:.1f}%)")
        print(f"  Trainable: {trainable:,} ({100 * trainable / total:.1f}%)")
        print(f"  Total:     {total:,}")

    def _check_visual_feature(self, visual_feature):
        """Validate that the auxiliary visual feature is already pooled to [B, C]."""
        if visual_feature is None:
            return None

        if visual_feature.dim() == 2 and visual_feature.shape[-1] == self.instructor_dim:
            return visual_feature

        raise ValueError(
            f"VisualGuidedBert expects pooled visual_feature with shape "
            f"[batch, {self.instructor_dim}], got {tuple(visual_feature.shape)}. "
            "Pool image tokens in VisualFeatureExtractor before calling BERT."
        )

    def _extract_layer_hidden_states(self, layer_output, layer_idx):
        """Return the [B, seq_len, hidden] tensor from a BertLayer output."""
        if torch.is_tensor(layer_output):
            hidden_states = layer_output
        elif hasattr(layer_output, "last_hidden_state"):
            hidden_states = layer_output.last_hidden_state
        else:
            hidden_states = layer_output[0]

        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"BERT layer {layer_idx} returned hidden_states with shape "
                f"{tuple(hidden_states.shape)}; expected [batch, seq_len, hidden_dim]."
            )

        return hidden_states

    def _get_extended_attention_mask(self, attention_mask, dtype, num_prompts=0, blind_prompt=False):
        """
        Convert 2D attention mask to BERT extended format.

        Standard mode: [B, 1, 1, seq_len]
        Blind prompt mode: [B, 1, seq_len, seq_len] asymmetric mask where
            - prompts can attend to all positions
            - text tokens cannot attend to prompts

        Valid positions: 0.0, Padding/Blind positions: -10000.0
        """
        if attention_mask is None:
            return None

        batch_size, seq_len = attention_mask.shape

        if not blind_prompt or num_prompts == 0:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.to(dtype=dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0
            return extended_mask

        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.to(dtype=dtype)
        padding_mask = (1.0 - padding_mask) * -10000.0

        blind_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=attention_mask.device)
        text_query_indices = [0] + list(range(num_prompts + 1, seq_len))
        prompt_key_indices = list(range(1, num_prompts + 1))

        for q_idx in text_query_indices:
            for k_idx in prompt_key_indices:
                blind_mask[q_idx, k_idx] = -10000.0

        blind_mask = blind_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        combined_mask = padding_mask + blind_mask
        return combined_mask

    def _compose_layer_prompt(self, hidden_states, visual_feature, layer_idx):
        """
        Compose the prompt for a specific layer from routing conditions.

        Args:
            hidden_states: [B, seq_len, C] current BERT hidden states.
            visual_feature: [B, instructor_dim] features from frozen visual encoder.
            layer_idx: int, layer index.

        Returns:
            prompt_emb: [B, num_prompts, dim] concatenated prompt embeddings.
            balance_loss: scalar.
        """
        batch_size = hidden_states.shape[0]

        if self.guidance_mode == "fixed_prompt":
            fixed_prompt = self.fixed_prompt_embeddings[layer_idx]
            fixed_prompt = fixed_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            prompt_parts = [fixed_prompt]

            if self.use_visual_instruct and visual_feature is not None:
                visual_feature = self._check_visual_feature(visual_feature)
                visual_instruct = self.visual_instruction_proj(visual_feature)
                visual_instruct = visual_instruct.unsqueeze(1)
                prompt_parts.append(visual_instruct)

            prompt_emb = torch.cat(prompt_parts, dim=1)
            prompt_emb = self.prompt_dropout(prompt_emb)
            balance_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
            return prompt_emb, balance_loss

        if self.guidance_mode == "uniform_prompt_fusion":
            static_prompt = self.prompt_bank_embeddings[0, layer_idx]
            static_prompt = static_prompt.unsqueeze(0).expand(batch_size, -1, -1)

            routed_prompt = self.prompt_bank_embeddings[1:, layer_idx].mean(dim=0)
            routed_prompt = routed_prompt.unsqueeze(0).expand(batch_size, -1, -1)

            prompt_parts = [static_prompt, routed_prompt]

            if self.use_visual_instruct and visual_feature is not None:
                visual_feature = self._check_visual_feature(visual_feature)
                visual_instruct = self.visual_instruction_proj(visual_feature)
                visual_instruct = visual_instruct.unsqueeze(1)
                prompt_parts.append(visual_instruct)

            prompt_emb = torch.cat(prompt_parts, dim=1)
            prompt_emb = self.prompt_dropout(prompt_emb)
            balance_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
            return prompt_emb, balance_loss

        route_parts = []
        if self.use_text_route:
            # Use the current [CLS] token as the text-side routing state.
            bert_pooled = hidden_states[:, 0, :]
            main_route = self.main_route_proj(bert_pooled)
            route_parts.append(main_route)

        if self.use_visual_route:
            visual_feature = self._check_visual_feature(visual_feature)
            guide_route = self.guide_route_proj(visual_feature)
            route_parts.append(guide_route)

        if len(route_parts) == 0:
            raise RuntimeError(f"No routing source is enabled for guidance_mode={self.guidance_mode}")

        routing_condition = route_parts[0] if len(route_parts) == 1 else torch.cat(route_parts, dim=1)

        routing_logits = routing_condition @ self.frozen_routing_anchors
        temperature = 0.1
        noise = torch.randn(routing_logits.shape).to(routing_logits.device) / (16 ** 2)
        routing_weights = F.softmax(routing_logits / temperature + noise, dim=-1)

        sum_scores = torch.sum(routing_weights, dim=0)
        std_scores = torch.std(sum_scores, dim=-1)
        mean_scores = torch.mean(sum_scores, dim=-1)
        threshold = 0.1
        zero_tensor = torch.tensor(0.0, device=routing_weights.device, dtype=routing_weights.dtype)
        raw_balance_loss = (std_scores / mean_scores) ** 2 if mean_scores > 0 else zero_tensor
        balance_loss = raw_balance_loss if raw_balance_loss > threshold else zero_tensor

        static_prompt = self.prompt_bank_embeddings[0, layer_idx]
        static_prompt = static_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        prompt_bases = self.prompt_bank_embeddings[1:, layer_idx]
        prompt_bases_batch = prompt_bases.unsqueeze(0).expand(batch_size, -1, -1, -1)
        routed_prompt = torch.einsum(
            "bk,bktd->btd", routing_weights, prompt_bases_batch
        )

        prompt_parts = [static_prompt, routed_prompt]

        if self.use_visual_instruct:
            visual_instruct = self.visual_instruction_proj(visual_feature)
            visual_instruct = visual_instruct.unsqueeze(1)
            prompt_parts.append(visual_instruct)

        prompt_emb = torch.cat(prompt_parts, dim=1)
        prompt_emb = self.prompt_dropout(prompt_emb)

        return prompt_emb, balance_loss

    def forward(self, text_input, visual_feature=None, return_features=False, blind_prompt=None):
        """
        Forward pass.

        Args:
            text_input: List[str] text strings.
            visual_feature: [B, instructor_dim] visual features from a frozen visual encoder.
            return_features: If True, return BERT [CLS] features instead of logits.
            blind_prompt: Optional runtime override for blind prompt attention.

        Returns:
            If return_features=False:
                logits: [B, num_classes]
                extra: dict with balance_loss
            If return_features=True:
                cls_feature: [B, 768]
        """
        tokenizer_output = self.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokenizer_output["input_ids"].to(self.bert_encoder.device)
        token_type_ids = tokenizer_output["token_type_ids"].to(self.bert_encoder.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.bert_encoder.device)
        batch_size = input_ids.shape[0]

        hidden_states = self.bert_encoder.embeddings(input_ids, token_type_ids)
        total_balance_loss = torch.tensor(0.0, device=hidden_states.device)
        if (self.use_visual_route or self.use_visual_instruct) and visual_feature is not None:
            visual_feature = self._check_visual_feature(visual_feature).to(hidden_states.device)
        elif not self.use_visual_route and not self.use_visual_instruct:
            visual_feature = None

        shared_prompt_emb = None
        shared_balance_loss = None

        for layer_idx in range(self.n_layers):
            use_blind = self.blind_prompt if blind_prompt is None else blind_prompt
            use_prompt = layer_idx >= self.freeze_layers and (
                (self.guidance_mode == "visual" and visual_feature is not None)
                or self.guidance_mode == "text_only_prompt"
                or self.guidance_mode == "fixed_prompt"
                or (self.guidance_mode == "shared_generated_prompt" and visual_feature is not None)
                or (self.guidance_mode == "visual_only_route" and visual_feature is not None)
                or (self.guidance_mode == "uniform_prompt_fusion" and visual_feature is not None)
            )
            if use_prompt:
                if self.guidance_mode == "shared_generated_prompt":
                    if shared_prompt_emb is None:
                        prompt_emb, balance_loss = self._compose_layer_prompt(
                            hidden_states, visual_feature, layer_idx
                        )
                        shared_prompt_emb = prompt_emb
                        shared_balance_loss = balance_loss
                    else:
                        prompt_emb = shared_prompt_emb
                        balance_loss = shared_balance_loss
                else:
                    prompt_emb, balance_loss = self._compose_layer_prompt(
                        hidden_states, visual_feature, layer_idx
                    )
                total_balance_loss += balance_loss

                actual_num_prompts = prompt_emb.shape[1]
                prompt_mask = torch.ones(
                    batch_size, actual_num_prompts, device=attention_mask.device
                )
                layer_attention_mask = torch.cat([
                    attention_mask[:, :1],
                    prompt_mask,
                    attention_mask[:, 1:],
                ], dim=1)
                layer_extended_mask = self._get_extended_attention_mask(
                    layer_attention_mask, hidden_states.dtype,
                    num_prompts=actual_num_prompts, blind_prompt=use_blind
                )

                hidden_states = torch.cat([
                    hidden_states[:, :1, :],
                    prompt_emb,
                    hidden_states[:, 1:, :],
                ], dim=1)

                layer_output = self.bert_encoder.encoder.layer[layer_idx](
                    hidden_states, layer_extended_mask
                )
                hidden_states = self._extract_layer_hidden_states(layer_output, layer_idx)

                hidden_states = torch.cat([
                    hidden_states[:, :1, :],
                    hidden_states[:, 1 + actual_num_prompts:, :],
                ], dim=1)
            else:
                orig_extended_mask = self._get_extended_attention_mask(
                    attention_mask, hidden_states.dtype
                )
                layer_output = self.bert_encoder.encoder.layer[layer_idx](
                    hidden_states, orig_extended_mask
                )
                hidden_states = self._extract_layer_hidden_states(layer_output, layer_idx)

        cls_feature = hidden_states[:, 0, :]

        if return_features:
            return cls_feature

        logits = self.classifier(cls_feature)

        prompt_layer_count = max(1, self.n_layers - self.freeze_layers)
        avg_balance_loss = total_balance_loss / prompt_layer_count

        extra = {
            "balance_loss": avg_balance_loss,
        }

        return logits, extra

    def extract_fused_features(self, text_input, visual_feature):
        """
        Extract VG-TPT text features.

        Args:
            text_input: List[str] text strings.
            visual_feature: [B, instructor_dim] visual features from a frozen visual encoder.

        Returns:
            fused_text_feature: [B, 768]
        """
        self.eval()
        with torch.no_grad():
            cls_feature = self.forward(
                text_input, visual_feature=visual_feature, return_features=True
            )
        return cls_feature


# =============================================================================
# Factory function
# =============================================================================

def get_visual_guided_bert(
    num_classes=7,
    bert_pretrained_path="bert-base-uncased",
    freeze_layers=8,
    unfreeze_layers=0,
    prompt_bank_size=8,
    prompt_length=10,
    guide_tokens_length=128,
    main_tokens_length=128,
    instructor_dim=1024,
    use_instruct=True,
    blind_prompt=False,
    guidance_mode="visual",
):
    """
    Factory function for VisualGuidedBertClassifier.

    Args:
        freeze_layers: VG-TPT prompt start layer (layers N-11 have prompts).
        unfreeze_layers: Number of BERT backbone layers to unfreeze.
        instructor_dim: Frozen visual encoder output dimension. Swin-Base is 1024.
    """
    model = VisualGuidedBertClassifier(
        num_classes=num_classes,
        bert_pretrained_path=bert_pretrained_path,
        freeze_layers=freeze_layers,
        unfreeze_layers=unfreeze_layers,
        prompt_bank_size=prompt_bank_size,
        prompt_length=prompt_length,
        guide_tokens_length=guide_tokens_length,
        main_tokens_length=main_tokens_length,
        instructor_dim=instructor_dim,
        use_instruct=use_instruct,
        blind_prompt=blind_prompt,
        guidance_mode=guidance_mode,
    )
    return model


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing VisualGuidedBertClassifier")
    print("=" * 60)

    model = get_visual_guided_bert(
        num_classes=7,
        freeze_layers=8,
        unfreeze_layers=0,
        prompt_bank_size=8,
        instructor_dim=1024,
    )

    batch_size = 2
    texts = ["This is a test sentence", "Another test sentence"]
    visual_features = torch.randn(batch_size, 1024)

    print("\n--- Forward Pass ---")
    logits, extra = model(texts, visual_feature=visual_features)
    print(f"Logits shape: {logits.shape}")
    print(f"Balance loss: {extra['balance_loss']:.4f}")

    print("\n--- Feature Extraction ---")
    features = model.extract_fused_features(texts, visual_features)
    print(f"Feature shape: {features.shape}")

    print("\n--- Without Visual Feature (Baseline) ---")
    logits_no_visual, _ = model(texts, visual_feature=None)
    print(f"Logits shape: {logits_no_visual.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
