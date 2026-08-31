"""Off-GUI-thread image inference for the desktop demonstrator.

The worker owns no Qt widgets and receives a stable detector reference captured by
the page when work begins.  It is intentionally file-only: camera capture and
open-world classification are outside the first delivered scope.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from PySide6.QtCore import QObject, Signal, Slot

from fruit_ssod.detection import DetectionRecord, DetectorAdapter
from fruit_ssod.detection.adapter import validate_confidence_threshold, validate_nms_iou_threshold


SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class ImageInferenceError(RuntimeError):
    """An actionable file-inference setup or execution failure."""


def _error(problem: str, cause: str, remediation: str) -> ImageInferenceError:
    return ImageInferenceError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


@dataclass(frozen=True)
class ImageInferenceSettings:
    """Validated controls passed as one auditable snapshot to a worker."""

    confidence: float = 0.25
    nms_iou: float = 0.50

    def __post_init__(self) -> None:
        confidence = validate_confidence_threshold(self.confidence)
        nms_iou = validate_nms_iou_threshold(self.nms_iou)
        # The validators only return None for None, which this concrete settings
        # object does not accept.  Retain explicit checks for a clear public error.
        if confidence is None or nms_iou is None:
            raise _error(
                "image inference controls are incomplete",
                "confidence or NMS IoU was omitted",
                "provide both a confidence and NMS IoU value from 0 to 1",
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "nms_iou", nms_iou)


@dataclass(frozen=True)
class ImageInferenceResult:
    """One completed image prediction and the display/export facts derived from it."""

    image_path: Path
    detections: tuple[DetectionRecord, ...]
    latency_ms: float
    confidence: float
    nms_iou: float

    def __post_init__(self) -> None:
        if not isinstance(self.image_path, Path) or not self.image_path.is_file():
            raise _error(
                "inference result has no readable image file",
                "the worker received a missing or non-file image path",
                "select an existing supported image file before running inference",
            )
        if not isinstance(self.latency_ms, (int, float)) or isinstance(self.latency_ms, bool) or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise _error(
                "inference latency is invalid",
                "the worker timer produced a non-finite or negative value",
                "rerun inference and report the error if it persists",
            )
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    @property
    def class_counts(self) -> dict[str, int]:
        """Return canonical class counts in first-detection order for the result pane."""
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe export evidence including the applied controls."""
        return {
            "image_path": str(self.image_path),
            "confidence_threshold": self.confidence,
            "nms_iou_threshold": self.nms_iou,
            "latency_ms": self.latency_ms,
            "class_counts": self.class_counts,
            "detections": [record.to_dict() for record in self.detections],
        }


def resolve_image_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    """Canonicalize a selected ordered file list and reject unsafe input early."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise _error(
                "selected image file was not found",
                f"{path} is not an existing regular file",
                "select one or more existing image files",
            )
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise _error(
                "selected file is not a supported image",
                f"{path.name} has extension {path.suffix or 'none'}",
                "select BMP, JPEG, PNG, TIFF, or WEBP image files",
            )
        canonical = path.resolve()
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    if not resolved:
        raise _error(
            "no images were selected",
            "the selected file list was empty",
            "choose one image or a folder containing supported image files",
        )
    return tuple(resolved)


class ImageInferenceWorker(QObject):
    """Run one ordered image list sequentially on a dedicated ``QThread``."""

    started = Signal(int)
    image_completed = Signal(object, int, int)
    image_failed = Signal(str, str, int, int)
    progress_changed = Signal(int, int)
    cancelled = Signal(int, int)
    finished = Signal(int, int, bool)

    def __init__(
        self,
        *,
        adapter: DetectorAdapter,
        image_paths: Iterable[str | Path],
        settings: ImageInferenceSettings,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, DetectorAdapter):
            raise _error(
                "no compatible detector is active",
                f"received {type(adapter).__name__}",
                "load a compatible five-fruit .pt model before running image inference",
            )
        self._adapter = adapter
        self._image_paths = resolve_image_paths(image_paths)
        self._settings = settings
        self._cancel_requested = threading.Event()

    @property
    def total_images(self) -> int:
        """Return the immutable work-list size for UI progress initialization."""
        return len(self._image_paths)

    def request_cancel(self) -> None:
        """Request cooperative cancellation safely from the GUI thread."""
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        """Predict sequentially, emitting only value objects back to the GUI thread."""
        completed = 0
        was_cancelled = False
        total = len(self._image_paths)
        self.started.emit(total)
        try:
            for path in self._image_paths:
                if self._cancel_requested.is_set():
                    was_cancelled = True
                    break
                started_at = perf_counter()
                try:
                    detections = tuple(
                        self._adapter.predict(
                            path,
                            confidence=self._settings.confidence,
                            nms_iou=self._settings.nms_iou,
                        )
                    )
                    result = ImageInferenceResult(
                        image_path=path,
                        detections=detections,
                        latency_ms=(perf_counter() - started_at) * 1000.0,
                        confidence=self._settings.confidence,
                        nms_iou=self._settings.nms_iou,
                    )
                    completed += 1
                    self.image_completed.emit(result, completed, total)
                except Exception as error:
                    completed += 1
                    self.image_failed.emit(str(path), str(error), completed, total)
                self.progress_changed.emit(completed, total)
                if self._cancel_requested.is_set():
                    was_cancelled = True
                    break
            if was_cancelled:
                self.cancelled.emit(completed, total)
        finally:
            self.finished.emit(completed, total, was_cancelled)
