"""Geometry-preserving image transforms used by pseudo-label generation."""

from __future__ import annotations

import math
from typing import Sequence


class TransformError(ValueError):
    """Raised when a geometric transform would make candidate evidence ambiguous."""


def _problem(problem: str, cause: str, remediation: str) -> TransformError:
    return TransformError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _dimension(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise _problem(f"{name} must be a positive finite number", "the image dimensions are missing or invalid", "provide decoded positive image dimensions")
    return float(value)


def _box(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise _problem("xyxy must contain four coordinates", "the detector result is not an XYXY box", "return a four-element finite numeric XYXY box")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise _problem("xyxy coordinates must be finite numbers", "the detector result contains a boolean, NaN, infinity, or text coordinate", "return finite numeric XYXY coordinates")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not x1 < x2 or not y1 < y2:
        raise _problem("xyxy must have positive area", "the supplied box is inverted or empty", "return x1 < x2 and y1 < y2")
    return x1, y1, x2, y2


def horizontal_flip_xyxy(xyxy: object, *, width: object) -> tuple[float, float, float, float]:
    """Map an XYXY box between original and horizontally flipped image spaces.

    Horizontal reflection is its own inverse.  The same function is therefore
    deliberately used to map flip-view predictions back to original pixels and
    to test the reverse mapping property.
    """
    image_width = _dimension(width, "width")
    x1, y1, x2, y2 = _box(xyxy)
    if x1 < 0 or x2 > image_width:
        raise _problem("xyxy lies outside image width", "the detector or transform used incompatible dimensions", "use the manifest width for the exact image passed to the detector")
    mapped = (image_width - x2, y1, image_width - x1, y2)
    # _box provides a useful final invariant even if floating point values were
    # supplied close to an edge.
    return _box(mapped)


def horizontal_flip_image(image: object) -> object:
    """Return a horizontally mirrored Pillow image without touching source files."""
    try:
        from PIL import ImageOps

        return ImageOps.mirror(image)  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError) as error:
        raise _problem("image cannot be horizontally flipped", str(error), "provide a Pillow-readable raster image to the pseudo-label generator") from error
