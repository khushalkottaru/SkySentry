from pathlib import Path
import numpy as np
from ultralytics import YOLO

_MODEL_PATH = Path(__file__).parent.parent / "yolo11n.pt"
_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(_MODEL_PATH))
    return _model


def run_inference(frame: np.ndarray) -> list[dict]:
    """Run YOLO inference on a single BGR frame. Returns a list of detections."""
    results = _get_model()(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(box.conf[0]),
            "class_id": int(box.cls[0]),
            "class_name": results.names[int(box.cls[0])],
        })
    return detections
