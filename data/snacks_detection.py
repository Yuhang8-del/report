"""Strict local importer for the CC-BY Snacks Detection supplementary labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.fruits360 import read_image_dimensions
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "snacks_detection"
_SOURCE_SPLITS = ("train", "val", "test")
_REQUIRED_FIELDS = frozenset({"image_id", "x_min", "x_max", "y_min", "y_max", "class_name", "folder"})


class SnacksDetectionImportError(ValueError):
    """Raised for malformed local supplementary source data."""


@dataclass(frozen=True)
class SnacksDetectionImportResult:
    records: tuple[CanonicalAnnotation, ...]
    rejections: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _safe_component(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or Path(value).name != value or value in {".", ".."}:
        raise SnacksDetectionImportError(_problem(f"Snacks Detection {field} is unsafe", repr(value), "use a nonempty path component without separators or traversal"))
    return value


def _unit(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SnacksDetectionImportError(_problem(f"Snacks Detection {field} is not numeric", repr(value), "provide finite normalized coordinates")) from error
    if not 0.0 <= parsed <= 1.0:
        raise SnacksDetectionImportError(_problem(f"Snacks Detection {field} is outside [0, 1]", repr(value), "repair the normalized box coordinates"))
    return parsed


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(_REQUIRED_FIELDS - set(reader.fieldnames or ()))
            if missing:
                raise SnacksDetectionImportError(_problem("Snacks Detection CSV has missing columns", repr(missing), "restore the documented image_id, coordinates, class_name, and folder columns"))
            return [{key: value or "" for key, value in row.items()} for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise SnacksDetectionImportError(_problem(f"Snacks Detection CSV {path} cannot be read", str(error), "restore a readable UTF-8 CSV")) from error


def _mapping(record: CanonicalAnnotation, source_partition: str) -> dict[str, Any]:
    return {
        "source": record.source,
        "source_partition": source_partition,
        "source_category": record.source_category,
        "source_image_id": record.source_image_id,
        "file_path": record.file_path,
        "width": record.width,
        "height": record.height,
        "class_id": record.class_id,
        "xyxy": list(record.xyxy),
        "split": record.split,
        "label_status": record.label_status,
        "license_metadata": {"name": record.license_metadata.name, "url": record.license_metadata.url, "attribution": record.license_metadata.attribution},
    }


def import_snacks_detection(
    images_root: Path,
    *,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
) -> SnacksDetectionImportResult:
    """Import only five reviewed labels as project-level train-pool annotations.

    The source's own partitions are retained as provenance but are deliberately
    not reused as this project's validation or test sets.  The caller must run
    the normal cleaning, deduplication, and deterministic project split stages.
    """
    if not images_root.is_dir():
        raise SnacksDetectionImportError(_problem("Snacks Detection images root is unavailable", str(images_root), "extract the reviewed image archive and pass its root"))
    if not isinstance(source_version, str) or not source_version.strip() or not isinstance(source_page, str) or not source_page.strip():
        raise SnacksDetectionImportError(_problem("Snacks Detection provenance is incomplete", "source_version or source_page is empty", "record the reviewed release and dataset-card URL"))
    root = images_root.resolve()
    inputs = {"train": train_csv, "val": val_csv, "test": test_csv}
    records: list[CanonicalAnnotation] = []
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    for source_partition in _SOURCE_SPLITS:
        for row in _csv_rows(inputs[source_partition]):
            category = row["class_name"].strip().lower()
            try:
                class_id = resolve_class_id(SOURCE_NAME, category)
            except ClassMappingError:
                continue
            try:
                image_id = _safe_component(row["image_id"].strip(), "image_id")
                folder = _safe_component(row["folder"].strip(), "folder")
                relative = Path("data") / source_partition / folder / f"{image_id}.jpg"
                try:
                    image_path = (root / relative).resolve(strict=True)
                    image_path.relative_to(root)
                except (FileNotFoundError, OSError, ValueError) as error:
                    raise SnacksDetectionImportError(
                        _problem(
                            "Snacks Detection image is unavailable",
                            f"{relative}: {error}",
                            "restore the missing extracted image or review the source row before import",
                        )
                    ) from error
                if image_path.is_symlink() or not image_path.is_file():
                    raise SnacksDetectionImportError(_problem("Snacks Detection image is unavailable", str(relative), "use a regular extracted image beneath --images-root"))
                width, height = read_image_dimensions(image_path)
                x1, x2, y1, y2 = (_unit(row[key], key) for key in ("x_min", "x_max", "y_min", "y_max"))
                if not (x1 < x2 and y1 < y2):
                    raise SnacksDetectionImportError(_problem("Snacks Detection box has no positive area", f"{image_id}: {(x1, y1, x2, y2)!r}", "repair x_min/x_max and y_min/y_max"))
                record = CanonicalAnnotation(
                    source=SOURCE_NAME,
                    source_category=category,
                    source_image_id=f"snacks-{source_partition}-{folder}-{image_id}",
                    file_path=relative.as_posix(),
                    width=width,
                    height=height,
                    class_id=class_id,
                    xyxy=(x1 * width, y1 * height, x2 * width, y2 * height),
                    split="train_pool",
                    label_status="labeled",
                    license_metadata=license_metadata,
                )
            except (SnacksDetectionImportError, ValueError) as error:
                rejections.append({"source_partition": source_partition, "source_image_id": row.get("image_id", "<unknown>"), "reason": str(error)})
                continue
            records.append(record)
            rows.append(_mapping(record, source_partition))
    if not rows:
        raise SnacksDetectionImportError(_problem("Snacks Detection import contains no approved fruit records", "no Apple, Banana, Orange, Strawberry, or Pineapple records survived validation", "verify the supplied CSVs and reviewed class aliases"))
    manifest = {
        "manifest_version": "1.0",
        "source": {"name": SOURCE_NAME, "version": source_version, "page": source_page, "license": {"name": license_metadata.name, "url": license_metadata.url, "attribution": license_metadata.attribution}},
        "project_split": "train_pool_before_deterministic_resplit",
        "records": rows,
        "rejections": rejections,
        "record_count": len(rows),
        "rejection_count": len(rejections),
    }
    return SnacksDetectionImportResult(tuple(records), tuple(rejections), manifest)
