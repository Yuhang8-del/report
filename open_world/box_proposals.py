"""Box-level unknown proposal generation behind the stable open-world contract."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.open_world.contracts import UnknownProposal, UnknownProposalRequest


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _python(value: object) -> object:
    for name in ("detach", "cpu"):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


class UltralyticsObjectnessProposalProvider:
    """Use a one-class Fruit detector to localize proposals not claimed by known classes."""

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        model: Any | None = None,
        objectness_threshold: float = 0.20,
        known_iou_threshold: float = 0.35,
        minimum_area: float = 64.0,
        nms_iou: float = 0.6,
        image_size: int = 768,
        device: str | int = 0,
    ) -> None:
        if (weights_path is None) == (model is None):
            raise ValueError("provide exactly one of weights_path or model")
        for name, value in {
            "objectness_threshold": objectness_threshold,
            "known_iou_threshold": known_iou_threshold,
            "nms_iou": nms_iou,
        }.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if minimum_area <= 0.0:
            raise ValueError("minimum_area must be positive")
        self._weights_path = Path(weights_path).resolve() if weights_path is not None else None
        self._model = model
        self.objectness_threshold = float(objectness_threshold)
        self.known_iou_threshold = float(known_iou_threshold)
        self.minimum_area = float(minimum_area)
        self.nms_iou = float(nms_iou)
        self.image_size = int(image_size)
        self.device = device

    @property
    def source_model(self) -> str:
        return str(self._weights_path) if self._weights_path is not None else type(self._model).__name__

    def _get_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            assert self._weights_path is not None
            self._model = YOLO(str(self._weights_path))
        return self._model

    def _predict_boxes(self, image_path: Path) -> Iterable[tuple[tuple[float, float, float, float], float]]:
        results = self._get_model().predict(
            source=str(image_path),
            conf=self.objectness_threshold,
            iou=self.nms_iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = _python(getattr(boxes, "xyxy", None))
            confidences = _python(getattr(boxes, "conf", None))
            if not isinstance(coordinates, (list, tuple)) or not isinstance(confidences, (list, tuple)):
                raise RuntimeError("objectness model returned malformed boxes")
            for raw_box, raw_confidence in zip(coordinates, confidences):
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                    continue
                box = tuple(float(value) for value in raw_box)
                confidence = float(raw_confidence)
                if all(math.isfinite(value) for value in (*box, confidence)):
                    yield box, confidence  # type: ignore[misc]

    def propose_unknowns(self, request: UnknownProposalRequest) -> tuple[UnknownProposal, ...]:
        image_path = Path(request.image_path).resolve(strict=True)
        image_id = request.evidence.get("image_id") or hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:16]
        proposals: list[UnknownProposal] = []
        for box, objectness in self._predict_boxes(image_path):
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) < self.minimum_area:
                continue
            max_known_iou = max((box_iou(box, known.xyxy) for known in request.known_detections), default=0.0)
            if max_known_iou >= self.known_iou_threshold:
                continue
            novelty = max(0.0, min(1.0, objectness * (1.0 - max_known_iou)))
            signature = f"{image_id}|{','.join(f'{value:.3f}' for value in box)}|{self.source_model}"
            proposal_id = "unknown-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
            proposals.append(
                UnknownProposal(
                    proposal_id=proposal_id,
                    image_id=image_id,
                    xyxy=box,
                    novelty_score=novelty,
                    source_model=self.source_model,
                    evidence={
                        "source_run_id": request.source_run_id,
                        "image_path": str(image_path),
                        "objectness_score": f"{objectness:.8f}",
                        "max_known_iou": f"{max_known_iou:.8f}",
                        "proposal_policy": "known-exclusion-v1",
                    },
                )
            )
        return tuple(sorted(proposals, key=lambda item: item.novelty_score, reverse=True))


def known_overlap_count(
    known_detections: Iterable[DetectionRecord],
    unknown_boxes: Iterable[tuple[float, float, float, float]],
    *,
    iou_threshold: float = 0.5,
) -> int:
    """Count unknown ground-truth boxes incorrectly claimed by a known-class detector."""
    known = tuple(known_detections)
    return sum(max((box_iou(box, item.xyxy) for item in known), default=0.0) >= iou_threshold for box in unknown_boxes)
