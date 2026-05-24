# -*- coding: utf-8 -*-
"""
Visual-Text datasets for CMU-MOSEI and CMU-MOSI sentiment analysis.

vision + text (VT) modalities are supported.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_MEAN = [0.46777044, 0.44531429, 0.40661017]
IMG_STD = [0.12221994, 0.12145835, 0.14380469]


class JsonlDataset(Dataset):
    """Visual-text dataset for MOSEI / MOSI loaded from jsonl files."""

    def __init__(self, data_root, split="train", transforms=None, use_acc2=False):
        data_path = os.path.join(data_root, f"{split}.jsonl")
        self.data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
        self.data_dir = os.path.dirname(data_path)
        self.transforms = transforms
        self.use_acc2 = use_acc2

        if use_acc2:
            self.labels = ["0", "1"]
        else:
            with open(os.path.join(self.data_dir, "labels.txt"), encoding="utf-8") as f:
                self.labels = [line.strip() for line in f]
        self.n_classes = len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence = item["text"]

        if item.get("img"):
            try:
                image = Image.open(
                    os.path.join(self.data_dir, item["img"])
                ).convert("RGB")
            except Exception as e:
                print(f"[Warning] Failed to load image: {item['img']}, error: {e}")
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        if self.transforms is not None:
            image = self.transforms(image)

        original_label = item["label"]
        if self.use_acc2:
            label_int = int(original_label)
            if label_int <= 2:
                label = torch.LongTensor([0])
            elif label_int >= 4:
                label = torch.LongTensor([1])
            else:
                label = torch.LongTensor([-100])
        else:
            label = torch.LongTensor([self.labels.index(str(original_label))])

        return sentence, image, label


def _visual_text_collate_fn(batch):
    """Collate a batch of (text, image, label) samples."""
    texts = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    labels = [item[2] for item in batch]
    labels = torch.cat(labels) if labels[0].dim() == 1 else torch.stack(labels)
    return texts, images, labels


def _acc2_visual_text_collate_fn(batch):
    """Collate for Acc-2 mode: filter out neutral samples (label == -100)."""
    filtered = [item for item in batch if item[2].item() != -100]
    if len(filtered) == 0:
        return None, None, None
    return _visual_text_collate_fn(filtered)


def create_loaders(data_root, batch_size, num_workers, use_acc2=False):
    """Create train/val/test DataLoaders for MOSEI or MOSI visual-text."""
    transforms_train = transforms.Compose([
        transforms.RandAugment(2, 7),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

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

    trainset = JsonlDataset(data_root, "train", transforms_train, use_acc2=use_acc2)
    validset = JsonlDataset(data_root, "val", transforms_val, use_acc2=use_acc2)
    testset = JsonlDataset(data_root, "test", transforms_test, use_acc2=use_acc2)

    collate_fn = _acc2_visual_text_collate_fn if use_acc2 else _visual_text_collate_fn
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
