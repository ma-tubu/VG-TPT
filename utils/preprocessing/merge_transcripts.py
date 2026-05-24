# -*- coding: utf-8 -*-
"""
Merge MOSEI CSV metadata with segmented transcript TXT files.

Outputs a CSV with start_time, end_time, and text_content for each clip.

Usage:
    python merge_transcripts.py \
        --csv_input mosei_statistics.csv \
        --txt_dir Transcript/Segmented/Combined \
        --csv_output mosei_processed_with_text.csv
"""

import argparse
import os

import pandas as pd


def merge_transcripts(csv_input_path, txt_folder_path, csv_output_path):
    """Merge CSV metadata with transcript TXT files."""
    print(f"Reading CSV: {csv_input_path} ...")
    df = pd.read_csv(csv_input_path)

    df["start_time"] = None
    df["end_time"] = None
    df["text_content"] = None

    unique_videos = df["video_id"].dropna().unique()
    total_videos = len(unique_videos)
    print(f"Found {total_videos} unique video_ids. Matching TXT files...\n")

    processed_count = 0
    missing_txt_count = 0

    for v_id in unique_videos:
        txt_file = os.path.join(txt_folder_path, f"{v_id}.txt")

        if not os.path.exists(txt_file):
            print(f"[Warning] Missing transcript: {txt_file}")
            missing_txt_count += 1
            continue

        clip_data_map = {}
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("___")
                if len(parts) >= 5:
                    c_id = str(parts[1])
                    start = parts[2]
                    end = parts[3]
                    text = "___".join(parts[4:])
                    clip_data_map[c_id] = (start, end, text)

        mask = df["video_id"] == v_id
        for idx, row in df[mask].iterrows():
            try:
                curr_clip_id = str(int(float(row["clip_id"])))
            except ValueError:
                curr_clip_id = str(row["clip_id"])

            if curr_clip_id in clip_data_map:
                start, end, text = clip_data_map[curr_clip_id]
                df.at[idx, "start_time"] = start
                df.at[idx, "end_time"] = end
                df.at[idx, "text_content"] = text

        processed_count += 1
        if processed_count % 100 == 0 or processed_count == total_videos:
            print(f"Progress: {processed_count}/{total_videos} videos processed...")

    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"\nDone! Saved to: {csv_output_path}")
    print(f"  Processed: {processed_count} videos, Missing TXTs: {missing_txt_count}")


def main():
    parser = argparse.ArgumentParser(description="Merge MOSEI CSV with transcript TXTs")
    parser.add_argument("--csv_input", required=True, help="Input CSV with video_id and clip_id")
    parser.add_argument("--txt_dir", required=True, help="Folder containing {video_id}.txt transcripts")
    parser.add_argument("--csv_output", required=True, help="Output CSV path")
    args = parser.parse_args()

    merge_transcripts(args.csv_input, args.txt_dir, args.csv_output)


if __name__ == "__main__":
    main()
