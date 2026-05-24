# -*- coding: utf-8 -*-
"""
Build train/val/test JSONL splits from a full-sample JSONL and physical file checks.

Filters samples by physical file intersection (e.g., images and audio),
then splits by the 'split' field. Unknown splits are randomly divided.

Usage:
    python build_dataset_splits.py \
        --jsonl_input all_sample.jsonl \
        --image_dir images/ \
        --output_dir final_dataset/
"""

import argparse
import json
import os
import random


def build_splits(jsonl_input_path, image_dir, output_dir, image_ext=".jpg", seed=42):
    """Build dataset splits with physical file filtering."""
    os.makedirs(output_dir, exist_ok=True)

    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(image_ext)}
    print(f"Physical files: {len(image_files)} images found.")

    useable_ids = image_files

    train_list, val_list, test_list, unknown_list = [], [], [], []

    with open(jsonl_input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            vid = data.get("meta", {}).get("vid", "")
            seg = str(data.get("meta", {}).get("seg", ""))
            sample_key = f"{vid}_{seg}"

            if sample_key not in useable_ids:
                continue

            split_label = data.get("split", "unknown").lower()
            if split_label == "train":
                train_list.append(data)
            elif split_label in ("val", "valid", "validation"):
                val_list.append(data)
            elif split_label == "test":
                test_list.append(data)
            else:
                unknown_list.append(data)

    # Randomly divide unknown samples (e.g., 9:1:2 ratio).
    random.seed(seed)
    random.shuffle(unknown_list)
    total_unk = len(unknown_list)
    p1 = int(total_unk * (9 / 12))
    p2 = int(total_unk * (10 / 12))

    train_list.extend(unknown_list[:p1])
    val_list.extend(unknown_list[p1:p2])
    test_list.extend(unknown_list[p2:])

    datasets = {
        "train.jsonl": train_list,
        "val.jsonl": val_list,
        "test.jsonl": test_list,
    }

    print("\n" + "=" * 40)
    print("Dataset split report")
    print("-" * 40)
    for name, data_list in datasets.items():
        save_path = os.path.join(output_dir, name)
        with open(save_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name:<12} : {len(data_list)} samples")
    print("=" * 40)
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build dataset splits from JSONL")
    parser.add_argument("--jsonl_input", required=True, help="Full sample JSONL")
    parser.add_argument("--image_dir", required=True, help="Directory with image files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_splits(args.jsonl_input, args.image_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
