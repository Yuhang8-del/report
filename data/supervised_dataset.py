"""Materialize immutable supervised YOLO snapshots from sealed split records."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY


class SupervisedDatasetError(ValueError):
    """Raised when sealed split records cannot safely form a YOLO snapshot."""


@dataclass(frozen=True)
class SupervisedDatasetResult:
    root: Path
    dataset_yaml: Path
    membership: Path
    image_count: int


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["records"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SupervisedDatasetError(_problem(f"sealed split file {path} cannot be read", str(error), "use unmodified create_splits output")) from error
    if not isinstance(records, list) or not records or any(not isinstance(record, Mapping) for record in records):
        raise SupervisedDatasetError(_problem(f"sealed split file {path} has no valid records", "records is empty or malformed", "regenerate deterministic split outputs"))
    return records


def _safe_source(source_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SupervisedDatasetError(_problem("sealed source file path is unsafe", repr(relative), "regenerate split outputs from controlled relative source paths"))
    candidate = (source_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(source_root)
    except ValueError as error:
        raise SupervisedDatasetError(_problem("sealed source image escapes source root", str(candidate), "restore source images beneath the configured source root")) from error
    if not candidate.is_file() or candidate.is_symlink():
        raise SupervisedDatasetError(_problem("sealed source image is unavailable", str(candidate), "restore a regular source image and rerun data preparation"))
    return candidate


def _yolo_lines(record: Mapping[str, Any]) -> str:
    width, height, labels = record.get("width"), record.get("height"), record.get("labels")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0 or not isinstance(labels, list) or not labels:
        raise SupervisedDatasetError(_problem("sealed annotation record has invalid dimensions or labels", repr(record.get("source_image_id")), "regenerate deterministic split outputs"))
    output: list[str] = []
    for label in labels:
        if not isinstance(label, Mapping) or not isinstance(label.get("class_id"), int) or label["class_id"] not in range(len(DEFAULT_CLASS_REGISTRY.class_names)) or not isinstance(label.get("xyxy"), list) or len(label["xyxy"]) != 4:
            raise SupervisedDatasetError(_problem("sealed annotation label is malformed", repr(record.get("source_image_id")), "regenerate deterministic split outputs"))
        try:
            x1, y1, x2, y2 = (float(value) for value in label["xyxy"])
        except (TypeError, ValueError) as error:
            raise SupervisedDatasetError(_problem("sealed annotation box is nonnumeric", repr(record.get("source_image_id")), "regenerate deterministic split outputs")) from error
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise SupervisedDatasetError(_problem("sealed annotation box is out of bounds", repr(record.get("source_image_id")), "regenerate deterministic split outputs"))
        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        box_w, box_h = (x2 - x1) / width, (y2 - y1) / height
        output.append(f"{label['class_id']} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
    return "\n".join(output) + "\n"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_supervised_dataset(split_root: Path, source_root: Path, output_root: Path, *, budget: int) -> SupervisedDatasetResult:
    """Copy only the three explicit supervised partitions into an immutable snapshot."""
    if budget not in {10, 20, 40, 100}:
        raise SupervisedDatasetError(_problem("label budget is unsupported", str(budget), "use one of 10, 20, 40, or 100"))
    root = output_root.resolve(strict=False)
    if root.exists():
        raise SupervisedDatasetError(_problem(f"supervised snapshot {root} already exists", "published dataset snapshots are immutable", "choose a fresh output root"))
    split = split_root.resolve(strict=True)
    source = source_root.resolve(strict=True)
    groups = {
        "train": _load_records(split / "budgets" / str(budget) / "labels.json"),
        "val": _load_records(split / "protected_splits" / "validation_labels.json"),
        "test": _load_records(split / "protected_splits" / "test_labels.json"),
    }
    seen: set[str] = set()
    for name, records in groups.items():
        ids = {str(record.get("source_image_id", "")) for record in records}
        if not ids or "" in ids or ids & seen:
            raise SupervisedDatasetError(_problem("supervised partitions overlap or have invalid image IDs", name, "regenerate the sealed split output before materializing"))
        seen.update(ids)
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        membership: dict[str, list[dict[str, str]]] = {}
        for partition, records in groups.items():
            image_dir, label_dir = temporary / "images" / partition, temporary / "labels" / partition
            image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
            entries: list[str] = []
            members: list[dict[str, str]] = []
            for record in sorted(records, key=lambda value: str(value["source_image_id"])):
                image_id = record["source_image_id"]
                original = _safe_source(source, record.get("file_path"))
                extension = original.suffix.lower() or ".jpg"
                image_out, label_out = image_dir / f"{image_id}{extension}", label_dir / f"{image_id}.txt"
                shutil.copy2(original, image_out)
                if _digest(original) != _digest(image_out):
                    raise SupervisedDatasetError(_problem("snapshot image digest differs from source", str(image_id), "retry from stable source storage"))
                label_out.write_text(_yolo_lines(record), encoding="utf-8", newline="\n")
                # Ultralytics on Windows does not reliably resolve relative
                # entries inside a text image list against dataset.yaml's
                # ``path``.  Absolute snapshot paths preserve the standard
                # images/ -> labels/ conversion and make the data contract
                # independent of the caller's current working directory.
                # The snapshot is built in a same-parent temporary directory
                # and atomically renamed into ``root``.  Image lists must
                # therefore name the *published* location, not ``temporary``;
                # otherwise a valid snapshot contains absolute paths that
                # disappear immediately after publication.
                entries.append(str((root / image_out.relative_to(temporary)).resolve()))
                members.append({"source_image_id": image_id, "source_file_path": str(record["file_path"]), "snapshot_image": image_out.relative_to(temporary).as_posix(), "image_sha256": _digest(image_out), "snapshot_label": label_out.relative_to(temporary).as_posix()})
            (temporary / f"{partition}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
            membership[partition] = members
        dataset = {"path": str(temporary), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        evidence = {"schema_version": "1.0", "artifact_type": "sealed_supervised_dataset", "label_budget_percent": budget, "split_root": str(split), "split_manifest_sha256": _digest(split / "split_manifest.json"), "members": membership}
        (temporary / "membership.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root); temporary = None
        published = {"path": str(root), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (root / "dataset.yaml").write_text(yaml.safe_dump(published, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        return SupervisedDatasetResult(root, root / "dataset.yaml", root / "membership.json", len(seen))
    except SupervisedDatasetError:
        raise
    except OSError as error:
        raise SupervisedDatasetError(_problem("supervised snapshot could not be written", str(error), "choose a new writable output root and verify source files")) from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
