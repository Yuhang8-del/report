"""Import local Berremangra YOLO-segmentation labels as Orange boxes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from fruit_ssod.data.class_mapping import resolve_class_id
from fruit_ssod.data.fruits360 import SourceMetadata, _license_mapping, _problem
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "berremangra_orange"


class BerremangraOrangeImportError(ValueError):
    """Raised when a local Orange segmentation source is unsafe to import."""


@dataclass(frozen=True)
class BerremangraOrangeImportResult:
    records: tuple[CanonicalAnnotation, ...]
    manifest: Mapping[str, Any]


def _fail(problem: str, cause: str, remediation: str) -> BerremangraOrangeImportError:
    return BerremangraOrangeImportError(_problem(problem, cause, remediation))


def _image_id(image: Path) -> str:
    digest = hashlib.sha256()
    try:
        with image.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _fail("source image cannot be hashed", str(image), "restore the readable downloaded image") from error
    return digest.hexdigest()


def _dimensions(image: Path) -> tuple[int, int]:
    try:
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
    except (OSError, ValueError) as error:
        raise _fail("source image cannot be decoded", str(image), "restore a valid image paired with the label") from error
    if width <= 0 or height <= 0:
        raise _fail("source image has invalid dimensions", str(image), "restore an image with positive dimensions")
    return width, height


def _polygon_box(line: str, label: Path, line_number: int) -> tuple[float, float, float, float]:
    values = line.split()
    if len(values) < 7 or (len(values) - 1) % 2:
        raise _fail("YOLO segmentation row is malformed", f"{label}:{line_number}", "use class_id followed by at least three normalized x/y points")
    try:
        class_id = int(values[0])
        points = [float(value) for value in values[1:]]
    except ValueError as error:
        raise _fail("YOLO segmentation row is non-numeric", f"{label}:{line_number}", "use finite normalized numeric points") from error
    if class_id != 0 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in points):
        raise _fail("YOLO segmentation row has unsupported class or out-of-range point", f"{label}:{line_number}", "use Orange class 0 and normalized points within [0, 1]")
    xs, ys = points[::2], points[1::2]
    if min(xs) >= max(xs) or min(ys) >= max(ys):
        raise _fail("YOLO segmentation polygon has zero area", f"{label}:{line_number}", "restore a polygon enclosing nonzero area")
    return min(xs), min(ys), max(xs), max(ys)


def import_berremangra_orange(dataset_root: Path, *, source_version: str, source_page: str, license_metadata: LicenseMetadata) -> BerremangraOrangeImportResult:
    """Convert public one-class polygon labels to canonical Orange rectangles.

    Original polygons are not discarded: their enclosing rectangle is the
    canonical detection geometry and source label paths remain in the manifest.
    The original source's train/val/test split is deliberately ignored so a
    fresh project-wide split can prevent inter-source evaluation leakage.
    """
    try:
        metadata = SourceMetadata(source_version, source_page, license_metadata)
    except ValueError as error:
        raise _fail("source metadata is invalid", str(error), "provide source version, page and licence metadata") from error
    if not dataset_root.is_dir():
        raise _fail("dataset root is missing", str(dataset_root), "point --dataset-root at the extracted Berremangra Orange directory")
    images = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.parent.name == "images")
    if not images:
        raise _fail("dataset contains no images", str(dataset_root), "restore the extracted train/valid/test images directories")
    class_id = resolve_class_id(SOURCE_NAME, "orange")
    records: list[CanonicalAnnotation] = []
    for image in images:
        partition = image.parent.parent
        label = partition / "labels" / image.with_suffix(".txt").name
        if not label.is_file():
            raise _fail("image lacks its YOLO segmentation label", str(image), "restore the matching labels/<image>.txt file")
        width, height = _dimensions(image)
        image_id = _image_id(image)
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise _fail("YOLO segmentation label cannot be read", str(label), "restore a readable UTF-8 label file") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            x1, y1, x2, y2 = _polygon_box(line, label, line_number)
            records.append(CanonicalAnnotation(source=SOURCE_NAME, source_category="orange", source_image_id=image_id, file_path=image.resolve().relative_to(dataset_root.resolve()).as_posix(), width=width, height=height, class_id=class_id, xyxy=(x1 * width, y1 * height, x2 * width, y2 * height), split="train_pool", label_status="labeled", license_metadata=license_metadata))
    if not records:
        raise _fail("dataset contains no Orange polygons", str(dataset_root), "restore nonempty YOLO segmentation labels")
    rows = [{"record_type": "canonical_annotation", "source": record.source, "source_image_id": record.source_image_id, "source_category": record.source_category, "file_path": record.file_path, "source_label_path": str((Path(record.file_path).parent.parent / "labels" / Path(record.file_path).with_suffix(".txt").name).as_posix()), "width": record.width, "height": record.height, "class_id": record.class_id, "xyxy": list(record.xyxy), "split": record.split, "label_status": record.label_status, "license_metadata": _license_mapping(record.license_metadata)} for record in records]
    manifest = {"manifest_version": "1.0", "source": {"name": SOURCE_NAME, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)}, "conversion": "YOLO polygon -> enclosing XYXY rectangle", "split": "train_pool", "label_status": "labeled", "records": rows, "record_count": len(rows), "rejection_count": 0}
    return BerremangraOrangeImportResult(tuple(records), manifest)
