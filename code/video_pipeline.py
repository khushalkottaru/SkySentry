from pathlib import Path

import cv2

from .detector import run_inference
from .range_profile import RangeProfile
from .visualizer import draw_detections


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    profile: RangeProfile,
    show_progress: bool = True,
) -> dict:
    """Run bird detection on every frame of input_path and write annotated video
    to output_path. Returns a summary dict with frame counts and total detections."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_index = 0
    total_detections = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = run_inference(frame, profile)
            total_detections += len(detections)
            annotated = draw_detections(frame, detections, profile, frame_index)
            writer.write(annotated)

            if show_progress and frame_index % 30 == 0:
                pct = (frame_index / total_frames * 100) if total_frames > 0 else 0
                print(f"\r  [{pct:5.1f}%] frame {frame_index}/{total_frames}  birds so far: {total_detections}", end="", flush=True)

            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if show_progress:
        print()  # newline after progress line

    return {
        "frames_processed": frame_index,
        "total_detections": total_detections,
        "output_path": str(output_path),
        "profile": profile.name,
    }
