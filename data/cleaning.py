"""Non-destructive image decoding and canonical annotation cleaning.

This module deliberately records every excluded input.  It never moves, deletes,
or rewrites source images or source manifests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from fruit_ssod.data.schema import AnnotationValidationError, CanonicalAnnotation


class DataCleaningError(ValueError):
    """Raised when a cleaning invocation itself is malformed."""


@dataclass(frozen=True)
class QuarantineRecord:
    """A machine-readable reason an input image or annotation was not accepted."""

    scope: str
    source_image_id: str
    file_path: str
    reason_code: str
    details: Mapping[str, str]

    def as_mapping(self) -> dict[str, Any]:
        """Render a JSON-compatible quarantine row."""
        return {
            "scope": self.scope,
            "source_image_id": self.source_image_id,
            "file_path": self.file_path,
            "reason_code": self.reason_code,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DecodedImage:
    """Decoded image facts used to bound canonical annotations."""

    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CleaningResult:
    """Accepted canonical annotations and all quarantined source rows."""

    accepted: tuple[CanonicalAnnotation, ...]
    rejected: tuple[QuarantineRecord, ...]


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _details(problem: str, cause: str, remediation: str) -> dict[str, str]:
    return {"problem": problem, "likely_cause": cause, "remediation": remediation}


def _field(row: Mapping[str, Any], name: str, default: str = "<unknown>") -> str:
    value = row.get(name, default)
    return value if isinstance(value, str) and value else default


def _image_path(file_path: str, image_root: Path | None) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() or image_root is None else image_root / path


def decode_image(path: Path) -> DecodedImage:
    """Fully decode a Pillow-supported image and return its positive dimensions."""
    if not path.is_file():
        raise DataCleaningError(
            _problem(
                f"image {path} is missing",
                "the manifest references a file that does not exist",
                "restore the local image or correct file_path before curation",
            )
        )
    try:
        # verify detects truncated/corrupt streams; reopening and load checks pixels.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DataCleaningError(
            _problem(
                f"image {path} cannot be decoded",
                str(error),
                "replace the file with a Pillow-readable image or quarantine this source record",
            )
        ) from error
    if width <= 0 or height <= 0:
        raise DataCleaningError(
            _problem(
                f"image {path} has non-positive dimensions",
                "the decoded image header is invalid",
                "replace the source image with a valid raster image",
            )
        )
    return DecodedImage(path=path, width=width, height=height)


def _finite_box(raw_box: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)) or len(raw_box) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in raw_box):
        return None
    return tuple(float(value) for value in raw_box)  # type: ignore[return-value]


def _quarantine(row: Mapping[str, Any], scope: str, reason_code: str, problem: str, cause: str, remediation: str) -> QuarantineRecord:
    return QuarantineRecord(
        scope=scope,
        source_image_id=_field(row, "source_image_id"),
        file_path=_field(row, "file_path"),
        reason_code=reason_code,
        details=_details(problem, cause, remediation),
    )


def clean_manifest_rows(rows: Iterable[Mapping[str, Any]], *, image_root: Path | None = None) -> CleaningResult:
    """Decode images, clamp finite boxes, and return canonical records plus quarantine rows.

    The input iteration order is preserved.  ``image_root`` only resolves relative
    ``file_path`` values; it is never created or modified.
    """
    accepted: list[CanonicalAnnotation] = []
    rejected: list[QuarantineRecord] = []
    decoded: dict[Path, DecodedImage | DataCleaningError] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise DataCleaningError(
                _problem(
                    "manifest row is not an object",
                    "the input manifest contains a scalar or array where an annotation is required",
                    "supply JSON objects following the canonical annotation schema",
                )
            )
        row = dict(raw_row)
        file_path = _field(row, "file_path", "")
        if not file_path:
            rejected.append(_quarantine(row, "annotation", "ANNOTATION_SCHEMA_INVALID", "file_path is missing", "the annotation does not identify a local image", "provide a nonempty file_path"))
            continue
        path = _image_path(file_path, image_root)
        decoded_image = decoded.get(path)
        if decoded_image is None:
            try:
                decoded_image = decode_image(path)
            except DataCleaningError as error:
                decoded_image = error
            decoded[path] = decoded_image
        if isinstance(decoded_image, DataCleaningError):
            missing = not path.is_file()
            rejected.append(_quarantine(
                row,
                "image",
                "IMAGE_MISSING" if missing else "IMAGE_UNDECODABLE",
                f"image {path} is {'missing' if missing else 'not Pillow-decodable'}",
                str(decoded_image),
                "restore or replace the local image; source files are left untouched",
            ))
            continue
        box = _finite_box(row.get("xyxy"))
        if box is None:
            rejected.append(_quarantine(row, "annotation", "BOX_NON_FINITE", "xyxy is not four finite numeric coordinates", "the source box contains text, booleans, missing values, or NaN/Infinity", "repair the source annotation before rerunning"))
            continue
        x1, y1, x2, y2 = box
        clamped = (
            min(float(decoded_image.width), max(0.0, x1)),
            min(float(decoded_image.height), max(0.0, y1)),
            min(float(decoded_image.width), max(0.0, x2)),
            min(float(decoded_image.height), max(0.0, y2)),
        )
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            rejected.append(_quarantine(row, "annotation", "BOX_NON_POSITIVE_AREA", "xyxy has zero or negative area after clamping", "the box is inverted, empty, or entirely outside the decoded image", "correct the source coordinates; this row remains in the quarantine manifest"))
            continue
        row["xyxy"] = list(clamped)
        row["width"] = decoded_image.width
        row["height"] = decoded_image.height
        try:
            accepted.append(CanonicalAnnotation.from_mapping(row))
        except (AnnotationValidationError, ValueError, TypeError) as error:
            rejected.append(_quarantine(row, "annotation", "ANNOTATION_SCHEMA_INVALID", "annotation cannot enter the canonical schema", str(error), "repair the reported canonical fields without changing source provenance or class identity"))
    return CleaningResult(tuple(accepted), tuple(rejected))


def annotation_mapping(annotation: CanonicalAnnotation) -> dict[str, Any]:
    """Convert an accepted annotation to a deterministic JSON-compatible row."""
    return {
        "source": annotation.source,
        "source_category": annotation.source_category,
        "source_image_id": annotation.source_image_id,
        "file_path": annotation.file_path,
        "width": annotation.width,
        "height": annotation.height,
        "class_id": annotation.class_id,
        "xyxy": list(annotation.xyxy),
        "split": annotation.split,
        "label_status": annotation.label_status,
        "license_metadata": {"name": annotation.license_metadata.name, "url": annotation.license_metadata.url, "attribution": annotation.license_metadata.attribution},
    }


def write_quarantine_manifest(path: Path, records: Iterable[QuarantineRecord]) -> None:
    """Write a caller-designated JSONL quarantine manifest; source paths are untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is imported lazily here to keep the record model independent of I/O.
    import json

    path.write_text("".join(json.dumps(record.as_mapping(), sort_keys=True) + "\n" for record in records), encoding="utf-8")
