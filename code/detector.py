from pathlib import Path
import numpy as np
from ultralytics import YOLO

from .range_profile import RangeProfile

_MODEL_PATH = Path(__file__).parent.parent / "yolo11n.pt"
_model = None

COCO_BIRD_CLASS_ID = 14


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(_MODEL_PATH))
    return _model


def run_inference(frame: np.ndarray, profile: RangeProfile) -> list[dict]:
    """Run YOLO inference on a single BGR frame, returning only bird detections
    that pass the given range profile's confidence and size filters."""
    results = _get_model()(
        frame,
        conf=profile.conf_threshold,
        iou=profile.iou_threshold,
        classes=[COCO_BIRD_CLASS_ID],
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)

        if area < profile.min_bbox_area:
            continue
        if profile.max_bbox_area is not None and area > profile.max_bbox_area:
            continue

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(box.conf[0]),
            "class_id": int(box.cls[0]),
            "class_name": results.names[int(box.cls[0])],
            "area": area,
        })

    return detections
