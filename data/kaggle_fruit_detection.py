"""Strict local importer core for reviewed YOLO fruit-detection sources.

Network acquisition is intentionally outside this module. Callers must pass a
downloaded, locally reviewed archive plus an explicit source namespace whose
aliases exist in the immutable class registry.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from PIL import Image

from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.fruits360 import SourceMetadata, _license_mapping, _problem
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "kaggle_fruit_detection"


class KaggleFruitDetectionImportError(ValueError):
    """Raised when a local Kaggle-style YOLO source cannot be safely indexed."""


@dataclass(frozen=True)
class KaggleFruitDetectionImportResult:
    records: tuple[CanonicalAnnotation, ...]
    rejections: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def _fail(problem: str, cause: str, remediation: str) -> KaggleFruitDetectionImportError:
    return KaggleFruitDetectionImportError(_problem(problem, cause, remediation))


def _source_classes(data_yaml: Path) -> dict[int, str]:
    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _fail("YOLO data YAML cannot be read", str(error), "provide a readable UTF-8 YOLO data.yaml with a names field") from error
    if not isinstance(payload, Mapping):
        raise _fail("YOLO data YAML must be an object", repr(payload), "provide a mapping with a names field")
    names = payload.get("names")
    if isinstance(names, list):
        indexed = {index: name for index, name in enumerate(names)}
    elif isinstance(names, Mapping):
        indexed = {key: value for key, value in names.items()}
    else:
        raise _fail("YOLO data YAML names is malformed", repr(names), "use a list or integer-keyed mapping of source class names")
    normalized: dict[int, str] = {}
    for raw_id, name in indexed.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise _fail("YOLO names has a non-integer class ID", repr(raw_id), "use zero-based integer class IDs") from error
        if class_id < 0 or class_id in normalized or not isinstance(name, str) or not name.strip():
            raise _fail("YOLO names has an invalid class definition", repr({raw_id: name}), "use unique nonnegative IDs and nonempty class names")
        normalized[class_id] = name.strip()
    if not normalized:
        raise _fail("YOLO data YAML has no classes", repr(names), "provide at least one source class in names")
    return normalized


def _label_for_image(dataset_root: Path, image: Path) -> Path:
    relative = image.resolve().relative_to(dataset_root.resolve())
    parts = list(relative.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise _fail("image is outside an images directory", str(relative), "use a YOLO layout containing images/<partition>") from error
    parts[index] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


def _source_image_id(image: Path) -> str:
    digest = hashlib.sha256()
    try:
        with image.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise _fail("source image cannot be hashed", str(image), "restore a readable regular image file") from error
    return digest.hexdigest()


def _dimensions(image: Path) -> tuple[int, int]:
    try:
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
    except (OSError, ValueError) as error:
        raise _fail("source image cannot be decoded", str(image), "restore a valid image referenced by the YOLO label") from error
    if width <= 0 or height <= 0:
        raise _fail("source image has invalid dimensions", str(image), "restore an image with positive width and height")
    return width, height


def _source_class_id(line: str, label: Path, line_number: int) -> int:
    """Read the leading YOLO class ID before deciding whether geometry is in scope."""
    parts = line.split()
    if not parts:
        raise _fail("YOLO label row is empty", f"{label}:{line_number}", "remove blank rows or provide a source class ID")
    try:
        class_id = int(parts[0])
    except ValueError as error:
        raise _fail("YOLO label class ID is non-numeric", f"{label}:{line_number}", "use a nonnegative integer class ID") from error
    if class_id < 0:
        raise _fail("YOLO label class ID is invalid", f"{label}:{line_number}", "use a nonnegative class ID")
    return class_id


def _parse_label(line: str, label: Path, line_number: int, *, allow_polygons: bool) -> tuple[int, float, float, float, float, str]:
    parts = line.split()
    if len(parts) == 5:
        geometry_kind = "box"
    elif allow_polygons and len(parts) >= 7 and len(parts) % 2 == 1:
        geometry_kind = "polygon"
    else:
        remediation = "use class_id x_center y_center width height"
        if allow_polygons:
            remediation += " or class_id followed by at least three normalized x y polygon pairs"
        raise _fail("YOLO label row has an unsupported field count", f"{label}:{line_number}", remediation)
    try:
        class_id = int(parts[0])
        values = tuple(float(value) for value in parts[1:])
    except ValueError as error:
        raise _fail("YOLO label row is non-numeric", f"{label}:{line_number}", "use finite numeric normalized YOLO values") from error
    if class_id < 0 or any(not math.isfinite(value) for value in values):
        raise _fail("YOLO label row is invalid", f"{label}:{line_number}", "use a nonnegative class ID and finite values")
    if geometry_kind == "box":
        x, y, width, height = values
        if width <= 0 or height <= 0 or not (0 <= x - width / 2 < x + width / 2 <= 1 and 0 <= y - height / 2 < y + height / 2 <= 1):
            raise _fail("YOLO label box is out of bounds or empty", f"{label}:{line_number}", "repair normalized boxes so they are strictly within the image")
        return class_id, x, y, width, height, geometry_kind
    points = tuple(zip(values[0::2], values[1::2]))
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
        raise _fail("YOLO polygon is out of bounds", f"{label}:{line_number}", "keep every normalized polygon coordinate within [0, 1]")
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    y_min, y_max = min(y for _, y in points), max(y for _, y in points)
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        raise _fail("YOLO polygon is empty", f"{label}:{line_number}", "provide at least three non-collinear polygon points")
    return class_id, x_min + width / 2, y_min + height / 2, width, height, geometry_kind


def import_yolo_fruit_detection(
    dataset_root: Path,
    data_yaml: Path,
    *,
    source_name: str,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
    allow_polygons: bool = False,
) -> KaggleFruitDetectionImportResult:
    """Import one reviewed YOLO source under an explicit registry namespace."""
    try:
        metadata = SourceMetadata(source_version, source_page, license_metadata)
    except ValueError as error:
        raise _fail("source metadata is invalid", str(error), "provide source version, page and licence metadata") from error
    if not isinstance(source_name, str) or not source_name.strip():
        raise _fail("source namespace is invalid", repr(source_name), "pass a nonempty reviewed class-registry source key")
    if not dataset_root.is_dir() or not data_yaml.is_file():
        raise _fail("local YOLO source is missing", f"dataset_root={dataset_root}, data_yaml={data_yaml}", "extract the dataset locally and pass its data.yaml")
    names = _source_classes(data_yaml)
    images = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and "images" in path.parts)
    if not images:
        raise _fail("local YOLO source has no images", str(dataset_root), "provide the extracted YOLO images/<partition> layout")
    records: list[CanonicalAnnotation] = []
    rejections: list[dict[str, str]] = []
    geometry_counts = {"box": 0, "polygon_to_box": 0}
    for image in images:
        label = _label_for_image(dataset_root, image)
        if not label.is_file():
            raise _fail("YOLO image lacks its label file", str(image), "provide a matching labels/<partition>/<stem>.txt file")
        width, height = _dimensions(image)
        image_id = _source_image_id(image)
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise _fail("YOLO label file cannot be read", str(label), "restore a readable UTF-8 label file") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            source_id = _source_class_id(line, label, line_number)
            category = names.get(source_id)
            if category is None:
                raise _fail("YOLO label references an absent class", f"{label}:{line_number} class={source_id}", "align label IDs with data.yaml names")
            try:
                canonical_id = resolve_class_id(source_name, category)
            except ClassMappingError as error:
                rejections.append({"source_image_id": image_id, "source_category": category, "source_class_id": str(source_id), "reason": str(error)})
                continue
            parsed_id, x, y, box_width, box_height, geometry_kind = _parse_label(line, label, line_number, allow_polygons=allow_polygons)
            if parsed_id != source_id:  # pragma: no cover - defensive invariant against parser regressions
                raise _fail("YOLO class parsing is inconsistent", f"{label}:{line_number}", "report the source label file and do not import it")
            geometry_counts["box" if geometry_kind == "box" else "polygon_to_box"] += 1
            x1, y1 = (x - box_width / 2) * width, (y - box_height / 2) * height
            x2, y2 = (x + box_width / 2) * width, (y + box_height / 2) * height
            record = CanonicalAnnotation(source=source_name, source_category=category, source_image_id=image_id, file_path=image.resolve().relative_to(dataset_root.resolve()).as_posix(), width=width, height=height, class_id=canonical_id, xyxy=(x1, y1, x2, y2), split="train_pool", label_status="labeled", license_metadata=license_metadata)
            records.append(record)
    if not records:
        raise _fail("local YOLO source contains no approved fruit boxes", str(dataset_root), "review names and add only explicit canonical aliases before importing")
    rows = [{"record_type": "canonical_annotation", "source": row.source, "source_image_id": row.source_image_id, "source_category": row.source_category, "file_path": row.file_path, "width": row.width, "height": row.height, "class_id": row.class_id, "xyxy": list(row.xyxy), "split": row.split, "label_status": row.label_status, "license_metadata": _license_mapping(row.license_metadata)} for row in records]
    manifest = {"manifest_version": "1.0", "source": {"name": source_name, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)}, "source_data_yaml": str(data_yaml.resolve()), "split": "train_pool", "label_status": "labeled", "geometry_conversion": {"source_box_count": geometry_counts["box"], "polygon_to_enclosing_box_count": geometry_counts["polygon_to_box"]}, "records": rows, "rejections": rejections, "record_count": len(rows), "rejection_count": len(rejections)}
    return KaggleFruitDetectionImportResult(tuple(records), tuple(rejections), manifest)


def import_kaggle_fruit_detection(dataset_root: Path, data_yaml: Path, *, source_version: str, source_page: str, license_metadata: LicenseMetadata) -> KaggleFruitDetectionImportResult:
    """Import the reviewed Kaggle YOLO source under its fixed namespace."""
    return import_yolo_fruit_detection(
        dataset_root,
        data_yaml,
        source_name=SOURCE_NAME,
        source_version=source_version,
        source_page=source_page,
        license_metadata=license_metadata,
    )
