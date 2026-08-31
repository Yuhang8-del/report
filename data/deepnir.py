"""Import reviewed deepNIR single-fruit YOLO directories as canonical boxes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from fruit_ssod.data.class_mapping import resolve_class_id
from fruit_ssod.data.fruits360 import SourceMetadata, _license_mapping, _problem
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "deepnir"
_APPROVED_DIRECTORIES = ("apple", "orange", "strawberry")


class DeepNIRImportError(ValueError):
    """Raised when the reviewed deepNIR archive cannot be imported safely."""


@dataclass(frozen=True)
class DeepNIRImportResult:
    records: tuple[CanonicalAnnotation, ...]
    manifest: Mapping[str, Any]


def _fail(problem: str, cause: str, remediation: str) -> DeepNIRImportError:
    return DeepNIRImportError(_problem(problem, cause, remediation))


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


def _box(line: str, label: Path, line_number: int) -> tuple[float, float, float, float]:
    fields = line.split()
    if len(fields) != 5:
        raise _fail("YOLO label row is malformed", f"{label}:{line_number}", "use class_id x_center y_center width height")
    try:
        source_id = int(fields[0])
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError as error:
        raise _fail("YOLO label row is non-numeric", f"{label}:{line_number}", "use finite normalized numeric values") from error
    if source_id != 0:
        raise _fail("YOLO label row has an unsupported class", f"{label}:{line_number} class={source_id}", "restore deepNIR single-class labels using class 0")
    values = (x, y, width, height)
    if any(not math.isfinite(value) for value in values) or width <= 0 or height <= 0:
        raise _fail("YOLO label box is invalid", f"{label}:{line_number}", "use finite positive normalized dimensions")
    x1, y1, x2, y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise _fail("YOLO label box is out of bounds", f"{label}:{line_number}", "restore normalized boxes strictly within the image")
    return x1, y1, x2, y2


def import_deepnir(dataset_root: Path, *, source_version: str, source_page: str, license_metadata: LicenseMetadata) -> DeepNIRImportResult:
    """Import only approved source directories, then force a fresh project split.

    The archive has one source directory per fruit and inconsistent placeholder
    values in its YAML ``names`` fields.  The reviewed directory name, not that
    placeholder, is therefore the auditable source category.
    """
    try:
        metadata = SourceMetadata(source_version, source_page, license_metadata)
    except ValueError as error:
        raise _fail("source metadata is invalid", str(error), "provide source version, page and licence metadata") from error
    source_root = dataset_root / "yolov5"
    if not source_root.is_dir():
        raise _fail("dataset root is missing expected yolov5 directory", str(source_root), "point --dataset-root at the extracted deepNIR archive root")

    records: list[CanonicalAnnotation] = []
    for category in _APPROVED_DIRECTORIES:
        category_root = source_root / category
        if not category_root.is_dir():
            raise _fail("reviewed source directory is missing", str(category_root), "restore the extracted deepNIR archive before importing")
        class_id = resolve_class_id(SOURCE_NAME, category)
        images = sorted(path for path in category_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.parent.name == "images")
        if not images:
            raise _fail("reviewed source directory has no images", str(category_root), "restore its train/valid image directories")
        for image in images:
            label = image.parent.parent / "labels" / image.with_suffix(".txt").name
            if not label.is_file():
                raise _fail("image lacks its YOLO label", str(image), "restore the matching labels/<image>.txt file")
            width, height = _dimensions(image)
            image_id = _image_id(image)
            try:
                lines = label.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as error:
                raise _fail("YOLO label cannot be read", str(label), "restore a readable UTF-8 label file") from error
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                x1, y1, x2, y2 = _box(line, label, line_number)
                records.append(CanonicalAnnotation(source=SOURCE_NAME, source_category=category, source_image_id=image_id, file_path=image.resolve().relative_to(dataset_root.resolve()).as_posix(), width=width, height=height, class_id=class_id, xyxy=(x1 * width, y1 * height, x2 * width, y2 * height), split="train_pool", label_status="labeled", license_metadata=license_metadata))
    if not records:
        raise _fail("dataset contains no approved fruit boxes", str(source_root), "restore nonempty approved deepNIR labels")

    rows = [{"record_type": "canonical_annotation", "source": record.source, "source_image_id": record.source_image_id, "source_category": record.source_category, "file_path": record.file_path, "source_label_path": str((Path(record.file_path).parent.parent / "labels" / Path(record.file_path).with_suffix(".txt").name).as_posix()), "width": record.width, "height": record.height, "class_id": record.class_id, "xyxy": list(record.xyxy), "split": record.split, "label_status": record.label_status, "license_metadata": _license_mapping(record.license_metadata)} for record in records]
    manifest = {"manifest_version": "1.0", "source": {"name": SOURCE_NAME, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)}, "source_category_policy": "reviewed directory name", "approved_source_directories": list(_APPROVED_DIRECTORIES), "split": "train_pool", "label_status": "labeled", "records": rows, "record_count": len(rows), "rejection_count": 0}
    return DeepNIRImportResult(tuple(records), manifest)
