#!/usr/bin/env python3
"""SkySentry v1 — YOLO bird detection on video with range profiles."""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="skysentry",
        description="Run YOLO bird detection on a video with a selected range profile.",
    )
    parser.add_argument("input", type=Path, help="Path to input video file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path for annotated output video (default: <input>_<profile>.mp4)",
    )
    parser.add_argument(
        "--profile", "-p",
        choices=["near", "medium", "far"],
        default="medium",
        help="Range profile controlling detection thresholds (default: medium)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_stem(f"{args.input.stem}_{args.profile}")

    from code.range_profile import get_profile
    from code.video_pipeline import process_video

    profile = get_profile(args.profile)

    print(f"SkySentry v1")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {output}")
    print(f"  Profile : {profile.name.upper()} — {profile.description}")
    print()

    summary = process_video(args.input, output, profile, show_progress=not args.quiet)

    print(f"Done.")
    print(f"  Frames processed : {summary['frames_processed']}")
    print(f"  Total detections : {summary['total_detections']}")
    print(f"  Output written   : {summary['output_path']}")


if __name__ == "__main__":
    main()
