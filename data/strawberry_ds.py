"""Import the public Strawberry-DS Parquet release as one Strawberry class.

The source annotates six maturity stages of the same fruit.  They are retained
in ``source_category`` for provenance but deliberately collapsed to the
project's canonical ``Strawberry`` detector class (ID 3).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.data.fruits360 import read_image_dimensions
from fruit_ssod.data.schema import AnnotationValidationError, CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "strawberry_ds"
_CATEGORIES = ("Early-Turning", "Green", "Late-Turning", "Red", "Turning", "White")
_STRAWBERRY_CLASS_ID = 3


class StrawberryDSImportError(ValueError):
    """Raised when a local Strawberry-DS Parquet release is malformed."""


@dataclass(frozen=True)
class StrawberryDSImportResult:
    records: tuple[CanonicalAnnotation, ...]
    rejections: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]
    image_root: Path


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise StrawberryDSImportError(_problem("Strawberry-DS Parquet cannot be hashed", str(error), "restore a readable local raw Parquet file")) from error
    return digest.hexdigest()


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


def _image_bytes(row: Mapping[str, Any]) -> bytes:
    image = row.get("image")
    if not isinstance(image, Mapping):
        raise StrawberryDSImportError(_problem("Strawberry-DS image field is malformed", repr(image), "restore the documented Hugging Face Image Parquet release"))
    payload = image.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise StrawberryDSImportError(_problem("Strawberry-DS image bytes are unavailable", repr(type(payload).__name__), "download the complete raw Parquet release with embedded image bytes"))
    return payload


def _object_rows(row: Mapping[str, Any]) -> list[tuple[str, tuple[float, float, float, float]]]:
    objects = row.get("objects")
    if not isinstance(objects, Mapping):
        raise StrawberryDSImportError(_problem("Strawberry-DS objects field is malformed", repr(objects), "restore the documented bbox/categories structure"))
    boxes, categories = objects.get("bbox"), objects.get("categories")
    if not isinstance(boxes, list) or not isinstance(categories, list) or len(boxes) != len(categories):
        raise StrawberryDSImportError(_problem("Strawberry-DS bbox/categories fields disagree", f"boxes={type(boxes).__name__}, categories={type(categories).__name__}", "restore a complete source row with equal-length bbox and category lists"))
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, (box, category) in enumerate(zip(boxes, categories, strict=True)):
        if isinstance(category, bool) or not isinstance(category, int) or category < 0 or category >= len(_CATEGORIES):
            raise StrawberryDSImportError(_problem("Strawberry-DS category is unsupported", f"index={index}, value={category!r}", "use the reviewed six maturity categories only"))
        if not isinstance(box, list) or len(box) != 4:
            raise StrawberryDSImportError(_problem("Strawberry-DS bbox is malformed", f"index={index}, value={box!r}", "use four-element pixel xywh boxes"))
        try:
            x, y, width, height = (float(value) for value in box)
        except (TypeError, ValueError) as error:
            raise StrawberryDSImportError(_problem("Strawberry-DS bbox is nonnumeric", f"index={index}, value={box!r}", "use finite pixel xywh coordinates")) from error
        if not width > 0 or not height > 0:
            raise StrawberryDSImportError(_problem("Strawberry-DS bbox has no positive area", f"index={index}, value={box!r}", "repair the source annotation before import"))
        result.append((_CATEGORIES[category], (x, y, x + width, y + height)))
    return result


def import_strawberry_ds(
    parquet: Path,
    output_root: Path,
    *,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
) -> StrawberryDSImportResult:
    """Extract a fresh, local image tree and canonical records from raw Parquet."""
    if not isinstance(source_version, str) or not source_version.strip() or not isinstance(source_page, str) or not source_page.strip():
        raise StrawberryDSImportError(_problem("Strawberry-DS provenance is incomplete", "source_version or source_page is empty", "record the reviewed release revision and dataset-card URL"))
    try:
        raw = parquet.resolve(strict=True)
    except OSError as error:
        raise StrawberryDSImportError(_problem("Strawberry-DS Parquet is unavailable", str(error), "download the reviewed raw Parquet file before import")) from error
    if not raw.is_file() or raw.is_symlink():
        raise StrawberryDSImportError(_problem("Strawberry-DS Parquet is not a regular file", str(raw), "use the downloaded regular Parquet source file"))
    root = output_root.resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise StrawberryDSImportError(_problem("Strawberry-DS output already exists", str(root), "choose a fresh output directory; imported source trees are immutable"))
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise StrawberryDSImportError(_problem("PyArrow is unavailable", str(error), "install the project dependencies before importing the Parquet source")) from error
    records: list[CanonicalAnnotation] = []
    rejections: list[dict[str, str]] = []
    source_partitions: dict[str, str] = {}
    temporary: Path | None = None
    image_count = 0
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        images = temporary / "images"
        row_index = 0
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                source_image_id = f"strawberry-ds-{row_index:06d}"
                row_index += 1
                target: Path | None = None
                try:
                    if not isinstance(row, Mapping):
                        raise StrawberryDSImportError(_problem("Strawberry-DS row is malformed", repr(row), "restore the raw Parquet release"))
                    payload = _image_bytes(row)
                    source_partition = row.get("split")
                    if source_partition not in {"train", "val", "valid", "test"}:
                        raise StrawberryDSImportError(_problem("Strawberry-DS source split is unsupported", repr(source_partition), "restore a row with the documented train, val/valid, or test source split"))
                    target = images / f"{source_image_id}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    width, height = read_image_dimensions(target)
                    annotations = _object_rows(row)
                    if not annotations:
                        raise StrawberryDSImportError(_problem("Strawberry-DS image has no annotations", source_image_id, "review the source row before using it as labeled training data"))
                    for category, xyxy in annotations:
                        records.append(CanonicalAnnotation(
                            source=SOURCE_NAME, source_category=category, source_image_id=source_image_id,
                            file_path=(Path("images") / target.name).as_posix(), width=width, height=height,
                            class_id=_STRAWBERRY_CLASS_ID, xyxy=xyxy, split="train_pool", label_status="labeled",
                            license_metadata=license_metadata,
                        ))
                    source_partitions[source_image_id] = source_partition
                    image_count += 1
                except (StrawberryDSImportError, AnnotationValidationError, ValueError) as error:
                    rejections.append({"source_image_id": source_image_id, "reason": str(error)})
                    if target is not None and target.exists():
                        target.unlink()
        if not records or not image_count:
            raise StrawberryDSImportError(_problem("Strawberry-DS import contains no valid boxes", "all source rows were rejected", "inspect raw Parquet structure and source annotations"))
        os.replace(temporary, root)
        temporary = None
    except StrawberryDSImportError:
        raise
    except (OSError, ValueError) as error:
        raise StrawberryDSImportError(_problem("Strawberry-DS extraction failed", str(error), "verify writable storage and the complete raw Parquet source")) from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    rows = [_mapping(record, source_partitions[record.source_image_id]) for record in records]
    manifest = {
        "manifest_version": "1.0",
        "source": {"name": SOURCE_NAME, "version": source_version, "page": source_page, "license": {"name": license_metadata.name, "url": license_metadata.url, "attribution": license_metadata.attribution}},
        "raw_parquet": {"path": str(raw), "bytes": raw.stat().st_size, "sha256": _sha256(raw)},
        "project_split": "train_pool_before_deterministic_resplit",
        "category_mapping": {name: "Strawberry" for name in _CATEGORIES},
        "records": rows,
        "rejections": rejections,
        "record_count": len(rows),
        "rejection_count": len(rejections),
        "image_count": image_count,
    }
    return StrawberryDSImportResult(tuple(records), tuple(rejections), manifest, root)
