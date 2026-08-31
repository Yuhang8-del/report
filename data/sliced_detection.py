"""Deterministic sliced inference primitives for validation screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from PIL import Image

from fruit_ssod.detection.types import DetectionRecord


@dataclass(frozen=True, order=True)
class SliceWindow:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.left < 0 or self.top < 0 or self.right <= self.left or self.bottom <= self.top:
            raise ValueError("SliceWindow requires non-negative coordinates with positive area")


def _starts(length: int, size: int, overlap: float) -> tuple[int, ...]:
    if length <= size:
        return (0,)
    stride = max(1, int(round(size * (1.0 - overlap))))
    values = list(range(0, length - size + 1, stride))
    final = length - size
    if values[-1] != final:
        values.append(final)
    return tuple(values)


def generate_slice_windows(width: int, height: int, *, slice_size: int = 640, overlap: float = 0.2) -> tuple[SliceWindow, ...]:
    """Cover the image completely, including non-stride-aligned edges."""
    if width <= 0 or height <= 0 or slice_size <= 0:
        raise ValueError("image dimensions and slice_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    crop_width, crop_height = min(width, slice_size), min(height, slice_size)
    return tuple(
        SliceWindow(left, top, left + crop_width, top + crop_height)
        for top in _starts(height, crop_height, overlap)
        for left in _starts(width, crop_width, overlap)
    )


def _iou(left: DetectionRecord, right: DetectionRecord) -> float:
    ax1, ay1, ax2, ay2 = left.xyxy
    bx1, by1, bx2, by2 = right.xyxy
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    left_area, right_area = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    return intersection / (left_area + right_area - intersection)


def merge_sliced_detections(detections: Iterable[DetectionRecord], *, iou_threshold: float = 0.5) -> tuple[DetectionRecord, ...]:
    """Apply deterministic class-aware NMS to projected slice detections."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(
        detections,
        key=lambda item: (-item.confidence, item.class_id, item.xyxy, item.source_model),
    )
    kept: list[DetectionRecord] = []
    for candidate in ordered:
        if any(candidate.class_id == accepted.class_id and _iou(candidate, accepted) > iou_threshold for accepted in kept):
            continue
        kept.append(candidate)
    return tuple(kept)


SlicePredictor = Callable[[Image.Image, SliceWindow], Iterable[DetectionRecord]]


def predict_sliced(
    image: Image.Image,
    predictor: SlicePredictor,
    *,
    slice_size: int = 640,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
) -> tuple[DetectionRecord, ...]:
    """Run an injected predictor on slices and project boxes to full-image XYXY."""
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a decoded PIL Image")
    projected: list[DetectionRecord] = []
    for window in generate_slice_windows(image.width, image.height, slice_size=slice_size, overlap=overlap):
        crop = image.crop((window.left, window.top, window.right, window.bottom))
        for detection in predictor(crop, window):
            x1, y1, x2, y2 = detection.xyxy
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(crop.width), x2), min(float(crop.height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            projected.append(
                DetectionRecord(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    xyxy=(x1 + window.left, y1 + window.top, x2 + window.left, y2 + window.top),
                    is_unknown=False,
                    source_model=detection.source_model,
                )
            )
    return merge_sliced_detections(projected, iou_threshold=nms_iou)
