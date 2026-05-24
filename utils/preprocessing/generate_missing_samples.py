# -*- coding: utf-8 -*-
"""
Generate a JSONL of missing samples by comparing a CSV against an existing JSONL.

Useful for extracting samples that are present in the CSV but missing from the JSONL.

Usage:
    python generate_missing_samples.py \
        --csv_input mosei_processed_with_text.csv \
        --jsonl_existing existing.jsonl \
        --jsonl_output missing.jsonl
"""

import argparse
import json
import os

import pandas as pd


def generate_missing(csv_input_path, existing_jsonl_path, output_jsonl_path):
    """Generate JSONL of samples present in CSV but not in existing JSONL."""
    processed_set = set()
    if os.path.exists(existing_jsonl_path):
        print(f"Reading existing JSONL: {existing_jsonl_path} ...")
        with open(existing_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                vid = data.get("meta", {}).get("vid", "")
                seg = str(data.get("meta", {}).get("seg", ""))
                if vid and seg:
                    processed_set.add(f"{vid}_{seg}")
        print(f"Found {len(processed_set)} existing samples.\n")
    else:
        print("[Warning] Existing JSONL not found; treating all CSV rows as missing.")

    print(f"Loading CSV: {csv_input_path} ...")
    df = pd.read_csv(csv_input_path)

    missing_count = 0
    with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
        for _, row in df.iterrows():
            vid = str(row["video_id"])
            try:
                seg_int = int(float(row["clip_id"]))
                seg_str = str(seg_int)
            except ValueError:
                seg_str = str(row["clip_id"])
                seg_int = row["clip_id"]

            if f"{vid}_{seg_str}" in processed_set:
                continue

            text_content = str(row["text_content"]) if not pd.isna(row["text_content"]) else ""
            label_val = str(row["label_7class"]) if not pd.isna(row["label_7class"]) else "0"
            start_val = float(row["start_time"]) if not pd.isna(row["start_time"]) else 0.0
            end_val = float(row["end_time"]) if not pd.isna(row["end_time"]) else 0.0
            score_val = float(row["sentiment"]) if not pd.isna(row["sentiment"]) else 0.0
            split_val = str(row["split"]) if not pd.isna(row["split"]) else "unknown"

            missing_data = {
                "text": text_content,
                "label": label_val,
                "img": f"images/{vid}_{seg_str}.jpg",
                "speaker": "speaker1",
                "meta": {
                    "vid": vid,
                    "seg": seg_int,
                    "start": start_val,
                    "end": end_val,
                    "score": score_val,
                },
                "audio": f"audio/{vid}_{seg_str}.wav",
                "split": split_val,
            }

            outfile.write(json.dumps(missing_data, ensure_ascii=False) + "\n")
            missing_count += 1

    print(f"Done! Saved to: {output_jsonl_path}")
    print(f"  CSV total: {len(df)}, Missing extracted: {missing_count}")


def main():
    parser = argparse.ArgumentParser(description="Generate missing sample JSONL")
    parser.add_argument("--csv_input", required=True, help="Source CSV with all samples")
    parser.add_argument("--jsonl_existing", required=True, help="Existing JSONL to compare against")
    parser.add_argument("--jsonl_output", required=True, help="Output JSONL for missing samples")
    args = parser.parse_args()

    generate_missing(args.csv_input, args.jsonl_existing, args.jsonl_output)


if __name__ == "__main__":
    main()
