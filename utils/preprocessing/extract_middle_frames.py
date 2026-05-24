# -*- coding: utf-8 -*-
"""
Extract middle frames from video clips for visual-text sentiment analysis.

Reads a JSONL file containing vid and seg metadata, locates the corresponding
video files, and saves the middle frame as a JPEG image.

Usage:
    python extract_middle_frames.py \
        --jsonl_input samples.jsonl \
        --video_dir clipped_videos/ \
        --output_dir images/
"""

import argparse
import json
import os

import cv2


def extract_middle_frames(jsonl_input_path, video_dir, output_dir, video_ext=".mp4", quality=95):
    """Extract middle frame from each video listed in the JSONL."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(jsonl_input_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_input_path}")

    with open(jsonl_input_path, "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in f)

    success_count = 0
    fail_count = 0

    with open(jsonl_input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            data = json.loads(line)
            vid = data.get("meta", {}).get("vid", "")
            seg = str(data.get("meta", {}).get("seg", ""))
            sample_name = f"{vid}_{seg}"

            video_path = os.path.join(video_dir, f"{sample_name}{video_ext}")
            output_path = os.path.join(output_dir, f"{sample_name}.jpg")

            if not os.path.exists(video_path):
                fail_count += 1
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                fail_count += 1
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                fail_count += 1
                continue

            mid_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()

            if ret:
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                success_count += 1
            else:
                fail_count += 1

            cap.release()

            if (i + 1) % 100 == 0 or (i + 1) == total_samples:
                print(f"Progress: {i + 1}/{total_samples} (success: {success_count}, fail: {fail_count})")

    print("\n" + "=" * 40)
    print("Frame extraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Output:  {output_dir}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Extract middle frames from videos")
    parser.add_argument("--jsonl_input", required=True, help="JSONL with vid/seg metadata")
    parser.add_argument("--video_dir", required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", required=True, help="Output directory for images")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (0-100)")
    args = parser.parse_args()

    extract_middle_frames(args.jsonl_input, args.video_dir, args.output_dir, quality=args.quality)


if __name__ == "__main__":
    main()
