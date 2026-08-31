"""Detector adapter for a reviewed append-only open-world class registry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def _python(value: object) -> object:
    for name in ("detach", "cpu"):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def load_reviewed_mapping(path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = sorted(payload["classes"], key=lambda item: int(item["id"]))
    mapping = {int(item["id"]): str(item["name"]) for item in classes}
    if list(mapping) != list(range(len(mapping))) or any(not name.strip() for name in mapping.values()):
        raise ValueError("reviewed class registry must contain contiguous zero-based IDs and nonempty names")
    return mapping


@dataclass(frozen=True)
class ReviewedDetectionRecord:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    is_unknown: bool
    source_model: str


class ReviewedUltralyticsDetectorAdapter:
    """Keep the immutable five-class adapter intact while serving reviewed additions."""

    def __init__(
        self,
        *,
        registry_path: str | Path,
        weights_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        if (weights_path is None) == (model is None):
            raise ValueError("provide exactly one of weights_path or model")
        self.mapping = load_reviewed_mapping(registry_path)
        self._weights = Path(weights_path).resolve() if weights_path is not None else None
        self._model = model
        self.source_model = str(self._weights) if self._weights is not None else type(model).__name__
        if model is not None:
            self._validate_names(getattr(model, "names", None))

    def _validate_names(self, names: object) -> None:
        if isinstance(names, Mapping):
            received = {int(key): str(value) for key, value in names.items()}
        elif isinstance(names, (list, tuple)):
            received = {index: str(value) for index, value in enumerate(names)}
        else:
            raise ValueError("incremental model does not expose a class-name mapping")
        if received != self.mapping:
            raise ValueError(f"incremental model mapping mismatch: expected {self.mapping}, received {received}")

    def _get_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            assert self._weights is not None
            self._model = YOLO(str(self._weights))
            self._validate_names(getattr(self._model, "names", None))
        return self._model

    def initialize(self) -> None:
        self._get_model()

    def predict(
        self,
        image: str | Path | Any,
        *,
        confidence: float | None = None,
        nms_iou: float | None = None,
    ) -> tuple[ReviewedDetectionRecord, ...]:
        kwargs: dict[str, object] = {"verbose": False}
        if confidence is not None:
            kwargs["conf"] = float(confidence)
        if nms_iou is not None:
            kwargs["iou"] = float(nms_iou)
        raw_results = self._get_model()(image, **kwargs)
        if not isinstance(raw_results, Iterable):
            raise RuntimeError("incremental model returned non-iterable results")
        detections = []
        for result in raw_results:
            self._validate_names(getattr(result, "names", None))
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = _python(getattr(boxes, "xyxy", None))
            confidences = _python(getattr(boxes, "conf", None))
            classes = _python(getattr(boxes, "cls", None))
            if not all(isinstance(value, (list, tuple)) for value in (coordinates, confidences, classes)):
                raise RuntimeError("incremental model returned malformed box fields")
            for raw_box, raw_confidence, raw_class in zip(coordinates, confidences, classes):
                class_id = int(raw_class)
                box = tuple(float(value) for value in raw_box)
                score = float(raw_confidence)
                if class_id not in self.mapping or len(box) != 4:
                    raise RuntimeError("incremental model returned an unregistered class or malformed box")
                if not all(math.isfinite(value) for value in (*box, score)):
                    raise RuntimeError("incremental model returned non-finite values")
                if not (box[0] < box[2] and box[1] < box[3] and 0.0 <= score <= 1.0):
                    raise RuntimeError("incremental model returned invalid box geometry or confidence")
                detections.append(
                    ReviewedDetectionRecord(
                        class_id=class_id,
                        class_name=self.mapping[class_id],
                        confidence=score,
                        xyxy=box,  # type: ignore[arg-type]
                        is_unknown=False,
                        source_model=self.source_model,
                    )
                )
        return tuple(detections)
