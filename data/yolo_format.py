"""Deterministic conversion of normalized Open Images boxes into YOLO labels.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import math


class YoloFormatError(ValueError):
    """Raised when a normalized box cannot safely become a YOLO label."""


def _problem(problem: str, cause: str, remediation: str) -> str:
    """Format errors consistently for data-curation operators."""
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def xyxy_normalized_to_yolo(x_min: float, y_min: float, x_max: float, y_max: float) -> tuple[float, float, float, float]:
    """Clamp finite normalized XYXY values, then return YOLO center/size values.

    Open Images coordinates are normalized.  The deterministic policy is to clamp
    each finite endpoint to ``[0, 1]`` and reject the row when clamping leaves a
    zero- or negative-area box.
    """
    values = (x_min, y_min, x_max, y_max)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise YoloFormatError(
            _problem(
                "normalized box coordinates must be finite numbers",
                "the annotation contains missing, text, or non-finite coordinates",
                "repair the source CSV row before conversion",
            )
        )
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(value))) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise YoloFormatError(
            _problem(
                "normalized box has no non-zero area after clamping",
                "the source box lies outside the image or has inverted endpoints",
                "correct the source box; invalid rows are intentionally excluded",
            )
        )
    width = x2 - x1
    height = y2 - y1
    return (x1 + width / 2, y1 + height / 2, width, height)


def format_yolo_label(class_id: int, box: tuple[float, float, float, float]) -> str:
    """Render a stable six-decimal YOLO text row without modifying class IDs."""
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise YoloFormatError(
            _problem(
                "class ID must be a non-negative integer",
                "the source class mapping returned an invalid type or ID",
                "resolve labels through the approved source-aware class registry",
            )
        )
    return f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
