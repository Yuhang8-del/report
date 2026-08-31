"""Optional, dependency-injected Ultralytics inference implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from fruit_ssod.data.class_mapping import ClassRegistry, DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import (
    DetectorAdapter,
    DetectorAdapterError,
    actionable_error,
    validate_class_mapping,
    validate_confidence_threshold,
    validate_nms_iou_threshold,
)
from fruit_ssod.detection.types import DetectionRecord, DetectionValidationError


def _as_python(value: object) -> object:
    """Convert common tensor-like values without importing torch or numpy."""
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


class UltralyticsDetectorAdapter(DetectorAdapter):
    """Adapt a YOLO-style model while keeping Ultralytics optional at import time."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        weights_path: str | Path | None = None,
        source_model: str | None = None,
        registry: ClassRegistry = DEFAULT_CLASS_REGISTRY,
    ) -> None:
        """Use an injected model for tests, or lazily build one from a weights path."""
        if model is None and weights_path is None:
            raise actionable_error(
                "no detector model was configured",
                "neither an injected model nor a weights path was supplied",
                "provide model=... for testing or weights_path='path/to/best.pt' for inference",
            )
        if model is not None and weights_path is not None:
            raise actionable_error(
                "detector model configuration is ambiguous",
                "both an injected model and a weights path were supplied",
                "provide exactly one of model=... or weights_path='path/to/best.pt'",
            )
        self._model = model
        self._weights_path = Path(weights_path) if weights_path is not None else None
        # validate_class_mapping verifies equivalence, then detach from caller-owned state.
        self._registry = DEFAULT_CLASS_REGISTRY
        self._source_model = source_model or (str(weights_path) if weights_path is not None else type(model).__name__)
        if model is not None:
            validate_class_mapping(getattr(model, "names", None), registry)

    def _build_model(self) -> Any:
        """Import the optional dependency only when real model construction is requested."""
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise actionable_error(
                "Ultralytics backend could not be initialized",
                "the optional 'ultralytics' package is not installed in the active Conda environment",
                "install the documented inference dependencies, then retry",
            ) from error
        assert self._weights_path is not None
        return YOLO(str(self._weights_path))

    def _get_model(self) -> Any:
        """Create and validate a lazy model exactly once, with context on failures."""
        if self._model is None:
            try:
                candidate_model = self._build_model()
                validate_class_mapping(getattr(candidate_model, "names", None), self._registry)
                self._model = candidate_model
            except DetectorAdapterError:
                raise
            except Exception as error:
                raise actionable_error(
                    "Ultralytics backend could not be initialized",
                    str(error),
                    "verify the weights path and the optional Ultralytics installation",
                ) from error
        return self._model

    def initialize(self) -> None:
        """Eagerly load and validate a checkpoint for GUI model selection."""
        self._get_model()

    def predict(
        self,
        image: str | Path | Any,
        *,
        confidence: float | None = None,
        nms_iou: float | None = None,
    ) -> tuple[DetectionRecord, ...]:
        """Run the model for a path or array-like image and normalize its result objects."""
        if image is None:
            raise actionable_error(
                "image input is missing",
                "predict received None instead of an image path or image array",
                "provide a pathlib.Path, string path, or decoded image array",
            )
        threshold = validate_confidence_threshold(confidence)
        iou_threshold = validate_nms_iou_threshold(nms_iou)
        kwargs: dict[str, object] = {"verbose": False}
        if threshold is not None:
            kwargs["conf"] = threshold
        if iou_threshold is not None:
            kwargs["iou"] = iou_threshold
        model = self._get_model()
        try:
            raw_results = model(image, **kwargs)
        except DetectorAdapterError:
            raise
        except Exception as error:
            raise actionable_error(
                "detector inference failed",
                str(error),
                "verify the input image format and model compatibility",
            ) from error
        return self._convert_results(raw_results)

    def _convert_results(self, raw_results: object) -> tuple[DetectionRecord, ...]:
        """Convert Ultralytics-style result/boxes fields into public record values."""
        if not isinstance(raw_results, Iterable) or isinstance(raw_results, (str, bytes, dict)):
            raise actionable_error(
                "model returned malformed results",
                "the backend result is not an iterable of result objects",
                "return the standard Ultralytics prediction result sequence",
            )
        detections: list[DetectionRecord] = []
        for result in raw_results:
            validate_class_mapping(getattr(result, "names", None), self._registry)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_values = _as_python(getattr(boxes, "xyxy", None))
            confidence_values = _as_python(getattr(boxes, "conf", None))
            class_values = _as_python(getattr(boxes, "cls", None))
            if not all(isinstance(value, (list, tuple)) for value in (xyxy_values, confidence_values, class_values)):
                raise actionable_error(
                    "model returned malformed box fields",
                    "boxes.xyxy, boxes.conf, or boxes.cls is missing or not sequence-like",
                    "return standard Ultralytics Boxes tensors for every result",
                )
            if not (len(xyxy_values) == len(confidence_values) == len(class_values)):
                raise actionable_error(
                    "model returned malformed box fields",
                    "box coordinates, scores, and classes have different lengths",
                    "ensure every predicted box has one confidence and one class ID",
                )
            for raw_xyxy, raw_confidence, raw_class_id in zip(xyxy_values, confidence_values, class_values):
                row = _as_python(raw_xyxy)
                raw_class_id = _as_python(raw_class_id)
                raw_confidence = _as_python(raw_confidence)
                if (
                    not isinstance(row, (list, tuple))
                    or len(row) != 4
                    or isinstance(raw_class_id, bool)
                    or not isinstance(raw_class_id, (int, float))
                    or not math.isfinite(raw_class_id)
                    or int(raw_class_id) != raw_class_id
                ):
                    raise actionable_error(
                        "model returned malformed box",
                        "a box does not have four coordinates or an integer class ID",
                        "return XYXY boxes and integral canonical class IDs from the backend",
                    )
                class_id = int(raw_class_id)
                class_name = next((item.name for item in self._registry.classes if item.id == class_id), None)
                if class_name is None:
                    raise actionable_error(
                        "model returned an unapproved class ID",
                        f"prediction class ID {class_id} is outside the canonical registry",
                        "use a checkpoint whose output classes match the fixed five-fruit registry",
                    )
                # Some Ultralytics post-processing paths can emit a
                # zero-width/height box at an image border under a very low
                # confidence threshold.  It has no valid detection geometry
                # and cannot contribute to inference, pseudo labels, or PR
                # matching, so discard only this finite degenerate case.
                # Malformed/non-finite coordinates still go through the
                # strict DetectionRecord validation below.
                if (
                    all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in row)
                    and (float(row[0]) >= float(row[2]) or float(row[1]) >= float(row[3]))
                ):
                    continue
                try:
                    detections.append(
                        DetectionRecord(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=raw_confidence,  # type: ignore[arg-type]
                            xyxy=tuple(row),  # type: ignore[arg-type]
                            is_unknown=False,
                            source_model=self._source_model,
                            registry=self._registry,
                        )
                    )
                except DetectionValidationError as error:
                    raise actionable_error(
                        "model returned malformed box",
                        str(error),
                        "return finite confidence scores and positive-area XYXY boxes",
                    ) from error
        return tuple(detections)
