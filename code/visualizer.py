import cv2
import numpy as np

from .range_profile import RangeProfile

_BBOX_COLOR = (0, 200, 50)     # green
_TEXT_COLOR = (255, 255, 255)  # white
_LABEL_BG = (0, 140, 35)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS = 2


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    profile: RangeProfile,
    frame_index: int,
) -> np.ndarray:
    """Return a copy of frame with bounding boxes and HUD overlay drawn."""
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), _BBOX_COLOR, _THICKNESS)

        label = f"bird {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), _LABEL_BG, -1)
        cv2.putText(out, label, (x1 + 2, y1 - baseline - 2), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS)

    _draw_hud(out, profile, len(detections), frame_index)
    return out


def _draw_hud(frame: np.ndarray, profile: RangeProfile, count: int, frame_index: int) -> None:
    h, w = frame.shape[:2]
    lines = [
        f"Profile : {profile.name.upper()}",
        f"Conf    : >={profile.conf_threshold:.2f}",
        f"Birds   : {count}",
        f"Frame   : {frame_index}",
    ]
    x, y0 = 10, 24
    for line in lines:
        cv2.putText(frame, line, (x, y0), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y0), _FONT, 0.5, (200, 230, 200), 1, cv2.LINE_AA)
        y0 += 20
