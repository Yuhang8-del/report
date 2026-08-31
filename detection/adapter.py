"""Backend-neutral detector interface and validation helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.class_mapping import ClassRegistry, DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.types import (
    DetectionRecord,
    DetectionValidationError,
    canonical_class_mapping,
)


class DetectorAdapterError(RuntimeError):
    """Raised when an inference backend is missing, incompatible, or malformed."""


def actionable_error(problem: str, cause: str, remediation: str) -> DetectorAdapterError:
    """Build uniform backend errors that a CLI or GUI can display directly."""
    return DetectorAdapterError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


def validate_class_mapping(
    names: object, registry: ClassRegistry = DEFAULT_CLASS_REGISTRY
) -> dict[int, str]:
    """Require a checkpoint's complete ID/name mapping to equal the registry exactly."""
    if isinstance(names, Mapping):
        raw_mapping = dict(names)
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        raw_mapping = dict(enumerate(names))
    else:
        raise actionable_error(
            "model class mapping is missing or malformed",
            "the model does not expose a names mapping or sequence",
            "save the checkpoint with all five canonical class IDs and names",
        )

    try:
        expected = canonical_class_mapping(registry)
    except DetectionValidationError as error:
        raise actionable_error(
            "configured registry does not match the canonical detector registry",
            str(error),
            "use exactly IDs 0-4 for Apple, Banana, Orange, Strawberry, and Pineapple",
        ) from error
    if (
        any(isinstance(key, bool) or not isinstance(key, int) for key in raw_mapping)
        or any(not isinstance(value, str) or not value.strip() for value in raw_mapping.values())
        or raw_mapping != expected
    ):
        raise actionable_error(
            "model class mapping does not match the canonical registry",
            f"expected {expected!r}, received {raw_mapping!r}",
            "use a checkpoint ordered as Apple, Banana, Orange, Strawberry, Pineapple with IDs 0-4",
        )
    return expected


def validate_confidence_threshold(confidence: object) -> float | None:
    """Validate an optional inference confidence threshold before calling a backend."""
    if confidence is None:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        raise actionable_error(
            "confidence threshold must be a finite number",
            "the requested threshold is not numeric or is NaN/infinite",
            "provide a number from 0 to 1, for example 0.4",
        )
    threshold = float(confidence)
    if not 0.0 <= threshold <= 1.0:
        raise actionable_error(
            "confidence threshold must be between 0 and 1",
            "the requested threshold is outside probability bounds",
            "provide a number from 0 to 1, for example 0.4",
        )
    return threshold


def validate_nms_iou_threshold(nms_iou: object) -> float | None:
    """Validate an optional NMS IoU threshold shared by file-inference clients.

    The value is deliberately part of the backend-neutral detector contract rather
    than a GUI-only post-processing knob.  This lets the GUI report precisely which
    NMS policy was passed to the checkpoint backend and avoids applying a second,
    potentially different NMS implementation after the model has returned boxes.
    """
    if nms_iou is None:
        return None
    if isinstance(nms_iou, bool) or not isinstance(nms_iou, (int, float)) or not math.isfinite(nms_iou):
        raise actionable_error(
            "NMS IoU threshold must be a finite number",
            "the requested NMS setting is not numeric or is NaN/infinite",
            "provide a number from 0 to 1, for example 0.5",
        )
    threshold = float(nms_iou)
    if not 0.0 <= threshold <= 1.0:
        raise actionable_error(
            "NMS IoU threshold must be between 0 and 1",
            "the requested NMS setting is outside probability bounds",
            "provide a number from 0 to 1, for example 0.5",
        )
    return threshold


class DetectorAdapter(ABC):
    """Stable inference interface used by training, evaluation, and PySide6 layers."""

    def initialize(self) -> None:
        """Eagerly validate optional backend state when a consumer needs it.

        Adapters that are already backed by an injected in-memory model need no setup.
        Lazy backends may override this hook so GUI model selection reports checkpoint
        compatibility errors before an inference worker is started.
        """
        return None

    @abstractmethod
    def predict(
        self,
        image: str | Path | Any,
        *,
        confidence: float | None = None,
        nms_iou: float | None = None,
    ) -> tuple[DetectionRecord, ...]:
        """Return validated records using optional confidence and backend NMS controls."""
