# -*- coding: utf-8 -*-
"""
MOSI visual-text dataset backed by precomputed feature packages.

Note: this loader uses mosi_raw.pkl (precomputed visual features)
and mosi.hdf5 (word tokens), not raw images. It is kept for
compatibility with legacy MOSI feature releases.

For raw-image-based VT loading, use utils.datasets.JsonlDataset instead.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _decode_word(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _words_to_text(word_features):
    words = []
    for item in np.asarray(word_features).reshape(-1):
        word = _decode_word(item).strip()
        if word and word.lower() != "sp":
            words.append(word)
    return " ".join(words)


def score_to_7class(score):
    """Map MOSI/MOSEI continuous score [-3, 3] to class id [0, 6]."""
    return int(np.clip(np.round(float(score)) + 3, 0, 6))


class MosiVisualTextDataset(Dataset):
    """MOSI VT dataset from mosi_raw.pkl and mosi.hdf5."""

    SPLIT_ALIASES = {
        "train": "train",
        "val": "valid",
        "valid": "valid",
        "dev": "valid",
        "test": "test",
    }

    def __init__(
        self,
        data_root,
        split="train",
        pkl_name="mosi_raw.pkl",
        hdf5_name="mosi.hdf5",
        label_mode="regression",
        pkl_data=None,
    ):
        self.data_root = data_root
        self.split = self.SPLIT_ALIASES.get(split, split)
        self.pkl_name = pkl_name
        self.hdf5_name = hdf5_name
        self.label_mode = label_mode

        if self.label_mode not in {"regression", "classification", "acc2"}:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        if pkl_data is None:
            pkl_path = os.path.join(data_root, pkl_name)
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)

        if self.split not in pkl_data:
            raise KeyError(
                f"Split '{self.split}' not found in {pkl_name}. "
                f"Available: {list(pkl_data.keys())}"
            )

        split_data = pkl_data[self.split]
        self.ids = list(split_data["id"])
        self.vision = np.asarray(split_data["vision"], dtype=np.float32)
        self.scores = np.asarray(split_data["labels"], dtype=np.float32).reshape(-1)
        self.feature_dim = int(self.vision.shape[-1])
        self.max_steps = int(self.vision.shape[1]) if self.vision.ndim >= 3 else 1

        self.texts = self._load_texts_from_hdf5()
        self.indices = self._build_indices()

        print(
            f"  [MosiVisualTextDataset] {self.split}: {len(self.indices)} samples, "
            f"vision={self.vision.shape}, label_mode={self.label_mode}"
        )

    def _load_texts_from_hdf5(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "MosiVisualTextDataset needs h5py to read mosi.hdf5 words. "
                "Install it: pip install h5py"
            ) from exc

        hdf5_path = os.path.join(self.data_root, self.hdf5_name)
        texts = []
        with h5py.File(hdf5_path, "r") as h5:
            words_group = h5["words"]
            for segment_id in self.ids:
                if segment_id in words_group:
                    text = _words_to_text(words_group[segment_id]["features"][:])
                else:
                    text = str(segment_id)
                    print(f"[Warning] Missing MOSI words for segment id: {segment_id}")
                texts.append(text)
        return texts

    def _build_indices(self):
        if self.label_mode != "acc2":
            return list(range(len(self.ids)))
        return [idx for idx, score in enumerate(self.scores) if float(score) != 0.0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]
        text = self.texts[real_index]
        visual_seq = torch.from_numpy(self.vision[real_index]).float()
        score = float(self.scores[real_index])

        if self.label_mode == "regression":
            label = torch.FloatTensor([score])
        elif self.label_mode == "classification":
            label = torch.LongTensor([score_to_7class(score)])
        else:
            label = torch.LongTensor([1 if score > 0 else 0])

        return text, visual_seq, label


def _mosi_visual_text_collate_fn(batch):
    texts = [item[0] for item in batch]
    visual = torch.stack([item[1] for item in batch])
    labels = [item[2] for item in batch]
    labels = torch.cat(labels) if labels[0].dim() == 1 else torch.stack(labels)
    return texts, visual, labels


def load_mosi_pkl(data_root, pkl_name="mosi_raw.pkl"):
    pkl_path = os.path.join(data_root, pkl_name)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def create_mosi_visual_text_loaders(
    data_root,
    batch_size,
    num_workers,
    pkl_name="mosi_raw.pkl",
    hdf5_name="mosi.hdf5",
    label_mode="regression",
):
    pkl_data = load_mosi_pkl(data_root, pkl_name)

    trainset = MosiVisualTextDataset(
        data_root=data_root, split="train", pkl_name=pkl_name,
        hdf5_name=hdf5_name, label_mode=label_mode, pkl_data=pkl_data,
    )
    validset = MosiVisualTextDataset(
        data_root=data_root, split="valid", pkl_name=pkl_name,
        hdf5_name=hdf5_name, label_mode=label_mode, pkl_data=pkl_data,
    )
    testset = MosiVisualTextDataset(
        data_root=data_root, split="test", pkl_name=pkl_name,
        hdf5_name=hdf5_name, label_mode=label_mode, pkl_data=pkl_data,
    )

    if trainset.feature_dim != validset.feature_dim or trainset.feature_dim != testset.feature_dim:
        raise ValueError("MOSI split visual feature dimensions do not match.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": _mosi_visual_text_collate_fn,
    }

    train_loader = DataLoader(trainset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(validset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(testset, shuffle=False, **loader_kwargs)

    print("Created MOSI Visual-Text feature loaders:")
    print(f"  Train: {len(trainset)}, Val: {len(validset)}, Test: {len(testset)}")
    print(f"  Visual feature dim: {trainset.feature_dim}, max steps: {trainset.max_steps}")

    return train_loader, val_loader, test_loader, trainset.feature_dim
