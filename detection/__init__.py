"""Inference-facing detector abstractions for Fruit SSOD."""

from fruit_ssod.detection.adapter import DetectorAdapter, DetectorAdapterError, validate_nms_iou_threshold
from fruit_ssod.detection.types import DetectionRecord, DetectionValidationError
from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter

__all__ = [
    "DetectionRecord",
    "DetectionValidationError",
    "DetectorAdapter",
    "DetectorAdapterError",
    "validate_nms_iou_threshold",
    "UltralyticsDetectorAdapter",
]
