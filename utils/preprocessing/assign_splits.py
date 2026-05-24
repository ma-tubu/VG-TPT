# -*- coding: utf-8 -*-
"""
Assign train/val/test split labels from a CSV into an existing JSONL file.

Usage:
    python assign_splits.py \
        --csv_input mosei_processed_with_text.csv \
        --jsonl_input old.jsonl \
        --jsonl_output new_with_split.jsonl
"""

import argparse
import json
import os

import pandas as pd


def assign_splits(csv_input_path, jsonl_input_path, jsonl_output_path):
    """Read split info from CSV and inject it into a JSONL file."""
    print(f"Loading CSV: {csv_input_path} ...")
    df = pd.read_csv(csv_input_path)

    split_map = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        try:
            c_id = str(int(float(row["clip_id"])))
        except ValueError:
            c_id = str(row["clip_id"])
        split_val = str(row["split"]) if not pd.isna(row["split"]) else "unknown"
        split_map[f"{vid}_{c_id}"] = split_val

    print(f"Built split map with {len(split_map)} entries.\n")

    if not os.path.exists(jsonl_input_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_input_path}")

    success_count = 0
    unknown_count = 0

    with open(jsonl_input_path, "r", encoding="utf-8") as infile, \
         open(jsonl_output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            vid = data.get("meta", {}).get("vid", "")
            seg = str(data.get("meta", {}).get("seg", ""))
            assigned = split_map.get(f"{vid}_{seg}", "unknown")

            if assigned == "unknown":
                unknown_count += 1
            else:
                success_count += 1

            data["split"] = assigned
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Done! Saved to: {jsonl_output_path}")
    print(f"  Matched: {success_count}, Unknown: {unknown_count}")


def main():
    parser = argparse.ArgumentParser(description="Assign split labels to JSONL")
    parser.add_argument("--csv_input", required=True, help="CSV with video_id, clip_id, split columns")
    parser.add_argument("--jsonl_input", required=True, help="Input JSONL without split")
    parser.add_argument("--jsonl_output", required=True, help="Output JSONL with split")
    args = parser.parse_args()

    assign_splits(args.csv_input, args.jsonl_input, args.jsonl_output)


if __name__ == "__main__":
    main()
