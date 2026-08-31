"""Safely materialize several labeled image sources beneath one controlled root.

The cleaner and split writer intentionally accept one image root.  This module
therefore copies already-local source images into a new immutable-style source
tree and rewrites only each canonical row's relative ``file_path``.  Source
identity, category, license, and original manifest digest remain in the
published provenance evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.cleaning import annotation_mapping
from fruit_ssod.data.schema import AnnotationValidationError, CanonicalAnnotation


class LabeledSourceMergeError(ValueError):
    """Raised when a local labeled-source materialization is unsafe or invalid."""


@dataclass(frozen=True)
class LabeledSourceInput:
    """One canonical manifest and its local root of relative image paths."""

    source: str
    manifest: Path
    image_root: Path


@dataclass(frozen=True)
class LabeledSourceMergeResult:
    """Published merged manifest and image-root facts."""

    root: Path
    manifest: Path
    image_count: int
    record_count: int


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MIXED_SOURCE = "*"


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    try:
        return _digest_bytes(path.read_bytes())
    except OSError as error:
        raise LabeledSourceMergeError(_problem(f"file {path} cannot be hashed", str(error), "restore the source file and retry materialization")) from error


def _safe_component(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise LabeledSourceMergeError(_problem(f"{field} is not a safe path component", repr(value), "use a source name made of letters, digits, dots, underscores, or hyphens"))
    return value


def _declared_source(value: str) -> str:
    """Validate one source selector; ``*`` preserves a prior materialized union."""
    return value if value == _MIXED_SOURCE else _safe_component(value, "source")


def _resolve_source_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not relative or ".." in candidate.parts:
        raise LabeledSourceMergeError(_problem("source manifest has an unsafe file_path", repr(relative), "use canonical relative paths beneath the declared source image root"))
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise LabeledSourceMergeError(_problem("source image is unavailable or escapes its root", f"{relative}: {error}", "restore a regular image beneath the declared source image root")) from error
    if resolved.is_symlink() or not resolved.is_file():
        raise LabeledSourceMergeError(_problem("source image is not a regular local file", str(resolved), "replace the path with a regular image file beneath the declared source root"))
    return resolved


def _load_records(item: LabeledSourceInput) -> tuple[list[CanonicalAnnotation], dict[str, str]]:
    source = _declared_source(item.source)
    try:
        manifest = item.manifest.resolve(strict=True)
        image_root = item.image_root.resolve(strict=True)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LabeledSourceMergeError(_problem(f"source manifest {item.manifest} cannot be read", str(error), "supply a readable canonical JSON manifest")) from error
    rows = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise LabeledSourceMergeError(_problem(f"source manifest {manifest} has no canonical records", "records is absent, empty, or malformed", "supply a successful importer or cleaner output"))
    records: list[CanonicalAnnotation] = []
    for index, row in enumerate(rows):
        try:
            record = CanonicalAnnotation.from_mapping(row)
        except (AnnotationValidationError, TypeError, ValueError) as error:
            raise LabeledSourceMergeError(_problem("source manifest record is invalid", f"{manifest} record {index}: {error}", "rerun the source importer or cleaner before materializing")) from error
        if source != _MIXED_SOURCE and record.source != source:
            raise LabeledSourceMergeError(_problem("declared source does not match canonical record source", f"expected {source!r}, found {record.source!r}", "pass the matching --source name and source manifest"))
        _resolve_source_path(image_root, record.file_path)
        records.append(record)
    provenance = {"source": source, "manifest": str(manifest), "manifest_sha256": _file_digest(manifest), "image_root": str(image_root)}
    return records, provenance


def _record_sort_key(record: CanonicalAnnotation) -> tuple[Any, ...]:
    return (record.source, record.source_image_id, record.file_path, record.class_id, record.xyxy)


def materialize_labeled_sources(inputs: Sequence[LabeledSourceInput], output_root: Path) -> LabeledSourceMergeResult:
    """Copy multiple canonical sources into a fresh, non-overlapping image root."""
    if len(inputs) < 2:
        raise LabeledSourceMergeError(_problem("at least two labeled sources are required", "the merge protocol is for a supplementary source plus the primary source", "pass --source for both canonical local sources"))
    for item in inputs:
        _declared_source(item.source)
    root = output_root.resolve(strict=False)
    if root.exists():
        raise LabeledSourceMergeError(_problem(f"merged source root {root} already exists", "published source snapshots are immutable", "choose a fresh output root"))

    loaded = [(item, *_load_records(item)) for item in inputs]
    all_records = sorted(
        ((index, record) for index, (_, records, _) in enumerate(loaded) for record in records),
        key=lambda pair: (*_record_sort_key(pair[1]), pair[0]),
    )
    provenance = sorted((entry for _, _, entry in loaded), key=lambda entry: (entry["source"], entry["manifest"]))
    temporary: Path | None = None
    copied: dict[tuple[int, str, str, str], str] = {}
    rewritten: list[dict[str, Any]] = []
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        for source_index, record in all_records:
            item = loaded[source_index][0]
            key = (source_index, record.source, record.source_image_id, record.file_path)
            relative = copied.get(key)
            if relative is None:
                source = _resolve_source_path(item.image_root.resolve(strict=True), record.file_path)
                stable_id = _digest_bytes("\x1f".join(str(part) for part in key).encode("utf-8"))
                suffix = source.suffix.lower() if source.suffix else ".jpg"
                destination_relative = Path("images") / record.source / f"{stable_id}{suffix}"
                destination = temporary / destination_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _file_digest(source) != _file_digest(destination):
                    raise LabeledSourceMergeError(_problem("materialized image digest differs from source", str(record.source_image_id), "retry from stable local storage before using this data protocol"))
                relative = destination_relative.as_posix()
                copied[key] = relative
            row = annotation_mapping(record)
            row["file_path"] = relative
            rewritten.append(row)
        payload = {
            "manifest_version": "1.0",
            "artifact_type": "materialized_labeled_source_union",
            "records": rewritten,
            "sources": provenance,
            "summary": {"source_count": len(inputs), "image_count": len(copied), "record_count": len(rewritten)},
        }
        manifest = temporary / "combined_annotations.json"
        manifest.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root)
        temporary = None
        return LabeledSourceMergeResult(root=root, manifest=root / "combined_annotations.json", image_count=len(copied), record_count=len(rewritten))
    except LabeledSourceMergeError:
        raise
    except OSError as error:
        raise LabeledSourceMergeError(_problem("labeled source materialization could not be written", str(error), "choose a new writable output root and verify local source images")) from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
