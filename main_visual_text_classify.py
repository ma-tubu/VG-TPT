# -*- coding: utf-8 -*-
"""
Vision-Guided Text Prompt Tuning (VG-TPT) — Training Script for Visual-Text Sentiment Analysis

Supported tasks:
    - Regression: continuous sentiment intensity in [-3, +3] with MAE loss.
    - Classification: 7-class or 2-class (Acc-2) discrete sentiment labels.
Usage (regression):
    python main_visual_text_classify.py \
        --data_root ./moseidata \
        --dataset mosei \
        --fusion_method vg_tpt \
        --max_epoch 20

Author: ma-tubu

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from types import ModuleType

# Compatibility patch for older PyTorch versions.
if "torch._six" not in sys.modules:
    m = ModuleType("torch._six")
    m.string_classes = (str, bytes)
    import collections.abc
    m.container_abcs = collections.abc
    sys.modules["torch._six"] = m

import argparse
import json
import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything

from models.visual_text_classifier import get_visual_text_classifier


# ---------------------------------------------------------------------------
# Dataset-specific constants
# ---------------------------------------------------------------------------
# Image normalization statistics computed on the CMU-MOSEI visual frames.
IMG_MEAN = [0.46777044, 0.44531429, 0.40661017]
IMG_STD = [0.12221994, 0.12145835, 0.14380469]

SUPPORTED_DATASETS = ("mosei", "mosi")


# ---------------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------------
ARGUMENT_GROUPS = (
    ("Dataset", (
        ("--dataset", dict(type=str, default="mosei", choices=SUPPORTED_DATASETS,
                           help="Dataset name (must provide text + image).")),
        ("--data_root", dict(type=str, required=True, help="Root directory of dataset.")),
        ("--batch-size", dict(type=int, default=32, help="Batch size.")),
        ("--num-workers", dict(type=int, default=8, help="Dataloader workers.")),
        ("--use_acc2", dict(action="store_true", default=False,
                            help="Acc-2 sentiment mode (binary classification).")),
    )),
    ("Model", (
        ("--fusion_method", dict(type=str, default="vg_tpt",
                                 choices=["vg_tpt", "text_only", "visual_only", "late_concat"],
                                 help="Fusion method: vg_tpt (ours) or baselines.")),
        ("--bert_pretrained_path", dict(type=str, default="pretrained/bert-base-uncased",
                                        help="BERT pretrained path.")),
        ("--img_size", dict(type=int, default=224, help="Input image size.")),
        ("--swin_pretrained_path", dict(type=str, default=None,
                                        help="Path to Swin Transformer pretrained weights (.pth).")),
        ("--no_pretrained_visual", dict(action="store_true", default=False,
                                        help="Do not load visual pretrained weights.")),
    )),
    ("VG-TPT Prompts", (
        ("--prompt_bank_size", dict(type=int, default=16,
                                    help="Number of prompt bases in the trainable prompt basis bank.")),
        ("--text_prompt_length", dict(type=int, default=10,
                                      help="Length of static and adaptive prompt tokens per layer.")),
        ("--text_freeze_layers", dict(type=int, default=0,
                                      help="Number of lower BERT layers without prompts. "
                                           "0 means every BERT layer receives prompts.")),
        ("--text_unfreeze_layers", dict(type=int, default=0,
                                        help="Number of BERT backbone layers to unfreeze. "
                                             "0 keeps the whole backbone frozen.")),
        ("--blind_prompt", dict(action="store_true", default=False,
                                help="Prevent text tokens from attending to prompt tokens.")),
        ("--prompt_guidance", dict(type=str, default="visual",
                                   choices=[
                                       "visual",
                                       "text_only_prompt",
                                       "fixed_prompt",
                                       "shared_generated_prompt",
                                       "visual_only_route",
                                       "uniform_prompt_fusion",
                                   ],
                                   help="Prompt generation mode for ablation studies.")),
        ("--guide_tokens_length", dict(type=int, default=32,
                                       help="Dimension of the visual guidance routing subspace.")),
        ("--main_tokens_length", dict(type=int, default=64,
                                      help="Dimension of the text-side routing subspace.")),
    )),
    ("Training", (
        ("--max_epoch", dict(type=int, default=20, help="Max epochs.")),
        ("--random-seed", dict(type=int, default=42, help="Random seed.")),
        ("--exp_name", dict(type=str, default="visual_text", help="Experiment name.")),
        ("--ckpt", dict(type=str, default="", help="Resume or evaluate from checkpoint.")),
        ("--evaluate", dict(action="store_true", default=False, help="Evaluation mode only.")),
        ("--save_predictions", dict(action="store_true", default=False,
                                    help="Save per-sample predictions to a text file.")),
        ("--pred_output", dict(type=str, default="",
                               help="Prediction output path (relative to logger dir if not absolute).")),
        ("--pred_limit", dict(type=int, default=0,
                              help="Max prediction rows to save. 0 means save all.")),
    )),
    ("Optimizer", (
        ("--lr_text", dict(type=float, default=1e-4,
                           help="Learning rate for prompt modules and prediction head.")),
        ("--lr_backbone", dict(type=float, default=2e-5,
                               help="Learning rate for unfrozen BERT backbone layers.")),
        ("--wd_text", dict(type=float, default=1e-2, help="Weight decay.")),
    )),
    ("Loss", (
        ("--balance_loss_weight", dict(type=float, default=0.01,
                                       help="Weight for the prompt activation balance loss.")),
        ("--ignore_label", dict(type=int, default=-100, help="Ignore index for classification.")),
        ("--exp_note", dict(type=str, default="", help="Experiment note for logging.")),
    )),
)


# Task mode: regression (default) or classification.
TASK_MODES = (
    ("--regression", dict(dest="regression", action="store_true",
                          help="Continuous sentiment score regression with MAE loss.")),
    ("--classification", dict(dest="regression", action="store_false",
                              help="Discrete classification with cross-entropy loss.")),
)


def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VG-TPT Visual-Text Sentiment Analysis Training"
    )
    for title, arguments in ARGUMENT_GROUPS:
        group = parser.add_argument_group(title)
        for flag, options in arguments:
            group.add_argument(flag, **options)

    task_group = parser.add_mutually_exclusive_group()
    for flag, options in TASK_MODES:
        task_group.add_argument(flag, **options)
    parser.set_defaults(regression=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data collate functions
# ---------------------------------------------------------------------------
def _visual_text_collate_fn(batch):
    """Collate a batch of (text, image, label) samples."""
    texts = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    labels = [item[2] for item in batch]
    labels = torch.cat(labels) if labels[0].dim() == 1 else torch.stack(labels)
    return texts, images, labels


def _acc2_visual_text_collate_fn(batch):
    """Collate function for Acc-2 mode that filters out neutral samples (label == -100)."""
    filtered = [item for item in batch if item[2].item() != -100]
    if len(filtered) == 0:
        return None, None, None
    return _visual_text_collate_fn(filtered)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _multiclass_acc(y_pred, y_true):
    """Compute multi-class accuracy after rounding predictions and labels."""
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


def compute_score_regression_metrics(y_pred, y_true):
    """
    Compute CMU-MOSEI / CMU-MOSI regression metrics from continuous predictions.

    Metrics:
        MAE    : Mean Absolute Error.
        Corr   : Pearson correlation coefficient.
        Acc-2  : Binary accuracy (positive vs. negative, neutral excluded).
        F1     : Weighted F1 score (binary).
        Acc-7  : Seven-class accuracy after clipping to [-3, +3].
    """
    if torch.is_tensor(y_pred):
        y_pred = y_pred.view(-1).detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred).reshape(-1)

    if torch.is_tensor(y_true):
        y_true = y_true.view(-1).detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true).reshape(-1)

    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.float64)

    pred_a7 = np.clip(y_pred, a_min=-3.0, a_max=3.0)
    true_a7 = np.clip(y_true, a_min=-3.0, a_max=3.0)

    mae = np.mean(np.abs(y_pred - y_true)).astype(np.float64)
    if len(y_pred) > 1 and np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        corr = np.corrcoef(y_pred, y_true)[0][1]
    else:
        corr = 0.0

    acc_7 = _multiclass_acc(pred_a7, true_a7)

    non_zeros = np.array([i for i, score in enumerate(y_true) if score != 0])
    if len(non_zeros) > 0:
        binary_truth = y_true[non_zeros] > 0
        binary_preds = y_pred[non_zeros] > 0
        acc_2 = accuracy_score(binary_truth, binary_preds)
        f1 = f1_score(binary_truth, binary_preds, average="weighted", zero_division=0)
    else:
        acc_2 = 0.0
        f1 = 0.0

    return {
        "MAE": float(mae),
        "Corr": float(corr),
        "Acc_2": float(acc_2),
        "F1_score": float(f1),
        "Acc_7": float(acc_7),
    }


# ---------------------------------------------------------------------------
# Model parameter summary
# ---------------------------------------------------------------------------
def _format_parameter_count(count):
    """Format a parameter count into a human-readable string."""
    if count >= 100_000_000:
        return f"{int(count / 1_000_000)} M"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f} M"
    if count >= 1_000:
        return f"{count / 1_000:.1f} K"
    return str(count)


def print_model_parameter_summary(model):
    """Print the number of trainable, frozen, and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params:     {_format_parameter_count(trainable)}")
    print(f"Non-trainable params: {_format_parameter_count(non_trainable)}")
    print(f"Total params:         {_format_parameter_count(trainable + non_trainable)}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class VisualTextRegressionDataset(Dataset):
    """
    Visual-text dataset for sentiment intensity regression.

    Reads a JSONL file where each line contains:
        - text: the utterance text.
        - img : relative path to the corresponding frame image.
        - meta.score: continuous sentiment score in [-3, +3] (optional).
        - label: fallback discrete label if meta.score is absent.
    """

    def __init__(self, data_root, split="train", transforms=None):
        data_path = os.path.join(data_root, f"{split}.jsonl")
        self.data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
        self.data_dir = os.path.dirname(data_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence = item["text"]

        # Load the visual frame; use a gray placeholder on failure.
        if item.get("img"):
            try:
                image = Image.open(os.path.join(self.data_dir, item["img"])).convert("RGB")
            except Exception as exc:
                print(f"[Warning] Failed to load image: {item['img']}, error: {exc}")
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        if self.transforms is not None:
            image = self.transforms(image)

        # Sentiment score: prefer meta.score, fall back to (label - 3).
        score = None
        meta = item.get("meta", {})
        if isinstance(meta, dict):
            score = meta.get("score", None)
        if score is None:
            raw_label = item.get("label", 3)
            score = float(raw_label) - 3.0

        return sentence, image, torch.FloatTensor([float(score)])


class VisualTextClassificationDataset(Dataset):
    """
    Visual-text dataset for sentiment classification.

    Supports both 7-class and Acc-2 (binary) classification modes.
    """

    def __init__(self, data_root, split="train", transforms=None, use_acc2=False):
        data_path = os.path.join(data_root, f"{split}.jsonl")
        self.data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
        self.data_dir = os.path.dirname(data_path)
        self.transforms = transforms
        self.use_acc2 = use_acc2

        if use_acc2:
            self.labels = ["0", "1"]
        else:
            with open(os.path.join(self.data_dir, "labels.txt"), encoding="utf-8") as label_file:
                self.labels = [line.strip() for line in label_file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence = item["text"]

        if item.get("img"):
            try:
                image = Image.open(os.path.join(self.data_dir, item["img"])).convert("RGB")
            except Exception as exc:
                print(f"[Warning] Failed to load image: {item['img']}, error: {exc}")
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        if self.transforms is not None:
            image = self.transforms(image)

        # Build label tensor.
        raw_label = item["label"]
        if self.use_acc2:
            raw_label = int(raw_label)
            if raw_label <= 2:
                label = torch.LongTensor([0])
            elif raw_label >= 4:
                label = torch.LongTensor([1])
            else:
                label = torch.LongTensor([-100])  # neutral samples are ignored
        else:
            label_key = raw_label if raw_label in self.labels else str(raw_label)
            label = torch.LongTensor([self.labels.index(label_key)])

        return sentence, image, label


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def create_visual_text_loaders(data_root, batch_size, num_workers, use_acc2=False, use_regression=False):
    """
    Create train/validation/test data loaders for visual-text sentiment analysis.

    Args:
        data_root: Root directory containing train.jsonl, val.jsonl, test.jsonl.
        batch_size: Batch size for all splits.
        num_workers: Number of DataLoader worker processes.
        use_acc2: Whether to use binary (Acc-2) classification.
        use_regression: Whether to use continuous score regression.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Training augmentation: RandAugment + resize + crop + normalize.
    transforms_train = transforms.Compose([
        transforms.RandAugment(2, 7),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    # Validation / test: no augmentation.
    transforms_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    if use_regression:
        trainset = VisualTextRegressionDataset(data_root, "train", transforms_train)
        validset = VisualTextRegressionDataset(data_root, "val", transforms_val)
        testset = VisualTextRegressionDataset(data_root, "test", transforms_test)
    else:
        trainset = VisualTextClassificationDataset(data_root, "train", transforms_train, use_acc2=use_acc2)
        validset = VisualTextClassificationDataset(data_root, "val", transforms_val, use_acc2=use_acc2)
        testset = VisualTextClassificationDataset(data_root, "test", transforms_test, use_acc2=use_acc2)

    print(f"Train: {len(trainset)}, Val: {len(validset)}, Test: {len(testset)}")

    collate_fn = _acc2_visual_text_collate_fn if use_acc2 and not use_regression else _visual_text_collate_fn
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(trainset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(validset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(testset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class VisualTextModel(LightningModule):
    """
    PyTorch Lightning module for VG-TPT visual-text sentiment analysis.

    This module wraps the VG-TPT classifier, handles forward passes,
    computes the training loss (task loss + prompt balance loss),
    and evaluates regression or classification metrics.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.validation_step_outputs = []

        if isinstance(self.args, dict):
            self.args = argparse.Namespace(**self.args)
        self.save_hyperparameters(vars(self.args))
        self.is_regression = bool(self.args.regression)
        self.prediction_save_path = None
        self.prediction_save_limit = 0
        self.evaluation_split_name = "val"

        if self.args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        # Determine output dimension.
        if self.is_regression:
            if self.args.use_acc2:
                raise ValueError("--regression and --use_acc2 are mutually exclusive.")
            self.num_classes = 1
        else:
            self.num_classes = 2 if self.args.use_acc2 else 7

        self.final_act = lambda x: F.log_softmax(x, dim=1)
        self.loss = nn.L1Loss() if self.is_regression else nn.NLLLoss(ignore_index=self.args.ignore_label)

        # Build the VG-TPT classifier.
        self.classifier = get_visual_text_classifier(
            num_classes=self.num_classes,
            bert_pretrained_path=self.args.bert_pretrained_path,
            img_size=self.args.img_size,
            freeze_layers=self.args.text_freeze_layers,
            unfreeze_layers=self.args.text_unfreeze_layers,
            prompt_bank_size=self.args.prompt_bank_size,
            prompt_length=self.args.text_prompt_length,
            guide_tokens_length=self.args.guide_tokens_length,
            main_tokens_length=self.args.main_tokens_length,
            fusion_method=self.args.fusion_method,
            pretrained_visual=not self.args.no_pretrained_visual,
            swin_pretrained_path=self.args.swin_pretrained_path,
            blind_prompt=self.args.blind_prompt,
            guidance_mode=self.args.prompt_guidance,
        )

    def forward(self, text_input, visual_input=None):
        """Forward pass through the VG-TPT classifier."""
        logits, extra = self.classifier(
            text_input=text_input,
            visual_input=visual_input,
            blind_prompt=self.args.blind_prompt,
        )
        return logits, extra

    def set_prediction_output(self, path=None, limit=0):
        """Set the path and row limit for saving prediction outputs."""
        self.prediction_save_path = path
        self.prediction_save_limit = max(0, int(limit or 0))

    def set_evaluation_split(self, split_name="val"):
        """Set the split name used for metric logging prefixes."""
        self.evaluation_split_name = split_name

    def _save_regression_predictions(self, preds, labels, texts):
        """Save regression predictions to a tab-separated text file."""
        if not self.prediction_save_path:
            return

        save_path = self.prediction_save_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        preds_np = preds.view(-1).detach().cpu().numpy()
        labels_np = labels.view(-1).detach().cpu().numpy()
        limit = self.prediction_save_limit if self.prediction_save_limit > 0 else len(preds_np)
        limit = min(limit, len(preds_np))

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("idx\ttrue_score\tpred_score\tabs_error\ttext\n")
                for idx in range(limit):
                    text = texts[idx] if idx < len(texts) else ""
                    text = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
                    error = abs(float(preds_np[idx]) - float(labels_np[idx]))
                    f.write(f"{idx}\t{labels_np[idx]:.6f}\t{preds_np[idx]:.6f}\t{error:.6f}\t{text}\n")
        except OSError as exc:
            print(f"[Warning] Failed to save predictions to {save_path}: {exc}")
            return

        print(f"[Prediction Dump] Saved {limit} regression predictions to: {save_path}")

    def training_step(self, batch, batch_idx):
        """Single training step: compute task loss + prompt balance loss."""
        if batch is None or batch[0] is None:
            return None

        texts, images, labels = batch
        if images.dim() == 5:
            images = images.squeeze(1)

        logits, extra_out = self.forward(texts, images)

        if self.is_regression:
            preds = logits.view(-1)
            targets = labels.float().view(-1)
            valid_mask = torch.isfinite(targets)
            if valid_mask.sum() == 0:
                return None

            loss_val = self.loss(preds[valid_mask], targets[valid_mask])
        else:
            outputs = self.final_act(logits)
            labels_sq = labels.view(-1)

            if (labels_sq == -100).all():
                return None

            loss_val = self.loss(outputs, labels_sq)

        # Add the prompt activation balance loss (regularizes prompt basis utilization).
        if extra_out is not None and "balance_loss" in extra_out:
            balance_loss = extra_out["balance_loss"]
            self.log("train_balance_loss", balance_loss)
            loss_val += balance_loss * self.args.balance_loss_weight

        # Log learning rate for prompt modules.
        optimizer = self.optimizers()
        if optimizer is not None and len(optimizer.param_groups) > 0:
            prompt_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]
            self.log("text_prompt_lr", prompt_lr, prog_bar=True)
        self.log("train_loss", loss_val, prog_bar=True)

        # Log auxiliary training metrics.
        with torch.no_grad():
            if self.is_regression:
                mae = torch.abs(preds[valid_mask] - targets[valid_mask]).mean()
                self.log("train_mae", mae, prog_bar=True)
                return loss_val

            pred_label = torch.argmax(outputs, dim=1)
            valid_mask = (labels_sq != -100)

            if valid_mask.sum() > 0:
                valid_pred = pred_label[valid_mask]
                valid_labels = labels_sq[valid_mask]

                acc = (valid_pred == valid_labels).float().mean()
                f1_micro = f1_score(valid_labels.cpu().numpy(), valid_pred.cpu().numpy(), average="micro", zero_division=0)
                f1_macro = f1_score(valid_labels.cpu().numpy(), valid_pred.cpu().numpy(), average="macro", zero_division=0)

                self.log("train_acc", acc, prog_bar=True)
                self.log("train_f1_micro", f1_micro)
                self.log("train_f1_macro", f1_macro)

        return loss_val

    def validation_step(self, batch, batch_idx):
        """Single validation step: collect predictions for epoch-end aggregation."""
        if batch is None or batch[0] is None:
            return None

        texts, images, labels = batch
        if images.dim() == 5:
            images = images.squeeze(1)

        logits, extra_out = self.forward(texts, images)
        if self.is_regression:
            pred_score = logits.view(-1)
            gt_score = labels.float().view(-1)
            valid_mask = torch.isfinite(gt_score)
            valid_texts = [text for text, keep in zip(texts, valid_mask.detach().cpu().tolist()) if keep]
            ret_dict = {
                "pred_score": pred_score[valid_mask],
                "gt_score": gt_score[valid_mask],
                "texts": valid_texts,
            }
        else:
            outputs = self.final_act(logits)
            pred_label = torch.argmax(outputs, dim=1)

            labels_sq = labels.view(-1)
            valid_mask = (labels_sq != -100)

            ret_dict = {
                "pred_label": pred_label[valid_mask],
                "gt_label": labels_sq[valid_mask],
            }

        if extra_out is not None and "balance_loss" in extra_out:
            ret_dict["balance_loss"] = extra_out["balance_loss"]

        self.validation_step_outputs.append(ret_dict)
        return ret_dict

    def on_validation_epoch_end(self) -> None:
        """Aggregate validation outputs and log final metrics."""
        if self.is_regression:
            all_preds, all_labels, all_texts = [], [], []

            for step_out in self.validation_step_outputs:
                if step_out["gt_score"].numel() > 0:
                    all_preds.append(step_out["pred_score"])
                    all_labels.append(step_out["gt_score"])
                    all_texts.extend(step_out.get("texts", []))

            if len(all_preds) == 0:
                print("[Warning] No valid validation samples in this epoch.")
                self.validation_step_outputs.clear()
                return

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = compute_score_regression_metrics(all_preds, all_labels)

            metric_prefix = self.evaluation_split_name
            self.log(f"{metric_prefix}_mae", metrics["MAE"], prog_bar=True)
            self.log(f"{metric_prefix}_corr", metrics["Corr"], prog_bar=True)
            self.log(f"{metric_prefix}_acc_2", metrics["Acc_2"], prog_bar=True)
            self.log(f"{metric_prefix}_f1_score", metrics["F1_score"], prog_bar=True)
            self.log(f"{metric_prefix}_acc_7", metrics["Acc_7"], prog_bar=True)

            if metric_prefix == "test":
                print(
                    "Final test results: "
                    f"MAE={metrics['MAE']:.4f}, "
                    f"Corr={metrics['Corr']:.4f}, "
                    f"Acc_2={metrics['Acc_2']:.4f}, "
                    f"F1={metrics['F1_score']:.4f}, "
                    f"Acc_7={metrics['Acc_7']:.4f}"
                )
            else:
                print("\n" + "=" * 40)
                print(f"VG-TPT Score Regression | Epoch: {self.current_epoch}")
                print(
                    "Validation -> "
                    f"MAE: {metrics['MAE']:.4f}, "
                    f"Corr: {metrics['Corr']:.4f}, "
                    f"Acc_2: {metrics['Acc_2']:.4f}, "
                    f"F1_score: {metrics['F1_score']:.4f}, "
                    f"Acc_7: {metrics['Acc_7']:.4f}"
                )
                print("=" * 40 + "\n")

            self._save_regression_predictions(all_preds, all_labels, all_texts)
            self.validation_step_outputs.clear()
            return

        # Classification branch.
        all_preds, all_labels = [], []

        for step_out in self.validation_step_outputs:
            if step_out["gt_label"].numel() > 0:
                all_preds.append(step_out["pred_label"])
                all_labels.append(step_out["gt_label"])

        if len(all_preds) == 0:
            print("[Warning] No valid validation samples in this epoch.")
            self.validation_step_outputs.clear()
            return

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        acc = (all_preds == all_labels).mean()

        metric_prefix = self.evaluation_split_name
        self.log(f"{metric_prefix}_f1_macro", f1_macro, prog_bar=True)
        self.log(f"{metric_prefix}_f1_micro", f1_micro, prog_bar=True)
        self.log(f"{metric_prefix}_acc", acc, prog_bar=True)

        if metric_prefix == "test":
            print(
                "Final test results: "
                f"F1_macro={f1_macro:.4f}, "
                f"F1_micro={f1_micro:.4f}, "
                f"Acc={acc:.4f}"
            )

        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Any:
        """
        Configure the optimizer with separate parameter groups:
            - backbone: unfrozen BERT layers (lower LR).
            - prompts : prompt basis bank, routing projections, visual instruction projection.
            - head    : sentiment prediction head.
        """
        backbone_params = []
        prompt_params = []
        head_params = []

        for name, param in self.classifier.named_parameters():
            if not param.requires_grad:
                continue

            if "text_model" in name:
                if any(key in name for key in ["prompt", "route_proj", "routing_", "instruction_proj", "visual_instruction_proj"]):
                    prompt_params.append(param)
                elif "classifier" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            elif "fusion_head" in name or "visual_classifier" in name:
                head_params.append(param)
            else:
                prompt_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": self.args.lr_backbone,
                "weight_decay": self.args.wd_text,
                "name": "backbone",
            })
        if prompt_params:
            param_groups.append({
                "params": prompt_params,
                "lr": self.args.lr_text,
                "weight_decay": self.args.wd_text,
                "name": "prompts",
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": self.args.lr_text,
                "weight_decay": self.args.wd_text,
                "name": "head",
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10, 15], gamma=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Strip frozen parameters from the checkpoint to reduce disk usage.
        Only trainable parameters (prompt modules + prediction head) are retained.
        """
        trainable_keys = {k for k, v in self.named_parameters() if v.requires_grad}

        keys_to_delete = []
        for k in checkpoint["state_dict"].keys():
            if k not in trainable_keys:
                keys_to_delete.append(k)

        for k in keys_to_delete:
            del checkpoint["state_dict"][k]

        return super().on_save_checkpoint(checkpoint)


# ---------------------------------------------------------------------------
# Prediction output configuration
# ---------------------------------------------------------------------------
def configure_prediction_output(model, args, logger, split_name="test"):
    """Configure the model to save predictions for a given split."""
    model.set_evaluation_split(split_name)
    if not getattr(args, "save_predictions", False):
        model.set_prediction_output(None)
        return

    if args.pred_output:
        pred_path = args.pred_output
        if not os.path.isabs(pred_path):
            pred_path = os.path.join(logger.log_dir, pred_path)
    else:
        pred_path = os.path.join(logger.log_dir, f"{args.dataset}_{split_name}_predictions.txt")

    model.set_prediction_output(pred_path, args.pred_limit)
    print(f"[Prediction Dump] Will save {split_name} predictions to: {pred_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Main training/evaluation routine."""
    args = get_arguments()
    for logger_name in ("pytorch_lightning", "lightning", "lightning_fabric"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    seed_everything(args.random_seed)

    if args.regression and args.use_acc2:
        raise ValueError("--regression and --use_acc2 are mutually exclusive.")

    print(f"Loading dataset from: {args.data_root}")
    train_loader, val_loader, test_loader = create_visual_text_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_acc2=args.use_acc2,
        use_regression=args.regression,
    )

    model = VisualTextModel(args)
    print_model_parameter_summary(model)

    # Checkpointing: monitor MAE for regression, F1-micro for classification.
    monitor_metric = "val_mae" if args.regression else "val_f1_micro"
    monitor_mode = "min" if args.regression else "max"
    ckpt_filename = "{epoch:02d}-{val_mae:.4f}" if args.regression else "{epoch:02d}-{val_f1_micro:.4f}"

    save_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_metric,
        filename=ckpt_filename,
        save_top_k=3,
        save_last=True,
        mode=monitor_mode,
        save_weights_only=True,
    )

    log_save_dir = os.environ.get("MOPE_LOG_DIR", "logs")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_save_dir,
        name=args.exp_name,
        version=args.exp_note if args.exp_note else None,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[save_callback],
        precision=16,
        logger=logger,
        enable_model_summary=False,
        max_epochs=args.max_epoch,
        val_check_interval=0.33,
        gradient_clip_val=1.0,
    )

    if args.evaluate:
        # Evaluation-only mode: load checkpoint and run on test set.
        if args.ckpt:
            model = VisualTextModel.load_from_checkpoint(args.ckpt, strict=False, args=args)
            model.eval()
            print(">>> Evaluating on test set...")
            configure_prediction_output(model, args, logger, split_name="test")
            trainer.validate(model, test_loader)
        else:
            raise ValueError("--evaluate requires --ckpt")
    else:
        # Training mode: optionally resume from checkpoint.
        if args.ckpt:
            model = VisualTextModel.load_from_checkpoint(args.ckpt, strict=False, args=args)

        # Backup source code to the logger directory for reproducibility.
        os.makedirs(logger.log_dir, exist_ok=True)
        if os.name == "nt":
            os.system(f"copy *.py {logger.log_dir}\\ >nul 2>&1")
            os.system(f"xcopy models {logger.log_dir}\\models\\ /E /I /Y >nul 2>&1")
            os.system(f"xcopy utils {logger.log_dir}\\utils\\ /E /I /Y >nul 2>&1")
        else:
            os.system(f"cp *.py {logger.log_dir}/")
            os.system(f"cp -r models {logger.log_dir}/")
            os.system(f"cp -r utils {logger.log_dir}/")

        trainer.fit(model, train_loader, val_loader)
        print(">>> Final evaluation on test set...")
        configure_prediction_output(model, args, logger, split_name="test")
        trainer.validate(model, test_loader)


if __name__ == "__main__":
    main()
