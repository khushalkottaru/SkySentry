from dataclasses import dataclass


@dataclass(frozen=True)
class RangeProfile:
    name: str
    conf_threshold: float
    min_bbox_area: float   # pixels²
    max_bbox_area: float | None  # None = no upper limit
    iou_threshold: float
    description: str


PROFILES: dict[str, RangeProfile] = {
    "near": RangeProfile(
        name="near",
        conf_threshold=0.50,
        min_bbox_area=2500.0,   # ~50×50 px — bird fills a large portion of frame
        max_bbox_area=None,
        iou_threshold=0.45,
        description="Near-range: large birds, high confidence required",
    ),
    "medium": RangeProfile(
        name="medium",
        conf_threshold=0.35,
        min_bbox_area=400.0,    # ~20×20 px
        max_bbox_area=None,
        iou_threshold=0.45,
        description="Medium-range: moderate size, standard confidence",
    ),
    "far": RangeProfile(
        name="far",
        conf_threshold=0.20,
        min_bbox_area=16.0,     # ~4×4 px — small dots at distance
        max_bbox_area=10000.0,  # oversized boxes are unlikely to be distant birds
        iou_threshold=0.45,
        description="Far-range: small birds, permissive confidence threshold",
    ),
}


def get_profile(name: str) -> RangeProfile:
    name = name.lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown range profile '{name}'. Choose from: {list(PROFILES)}")
    return PROFILES[name]
