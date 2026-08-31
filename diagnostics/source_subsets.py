"""Evaluate sealed held-out members by their public-data source.

This module is deliberately diagnostic-only.  Its output is outside a run
directory and cannot replace the one fixed-test result accepted by the
aggregation and report pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapterError, validate_class_mapping
from fruit_ssod.training.run_record import read_run_record
from fruit_ssod.training.supervised import (
    SupervisedTrainingError,
    _dataset_evidence,
    _serialize_metric_object,
    file_evidence,
)


class SourceSubsetDiagnosticError(RuntimeError):
    """Raised when source-subset diagnostic evidence cannot be established."""


_SOURCE_NAME = re.compile(r"^[a-z0-9_]+$")


def _problem(problem: str, cause: str, remediation: str) -> SourceSubsetDiagnosticError:
    return SourceSubsetDiagnosticError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _source_from_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _problem("test member source path is missing", repr(value), "restore the sealed membership.json generated with the dataset snapshot")
    parts = Path(value.replace("\\", "/")).parts
    try:
        source = parts[parts.index("images") + 1]
    except (ValueError, IndexError) as error:
        raise _problem("test member source path is malformed", value, "restore source_file_path with an images/<source>/ layout") from error
    if not _SOURCE_NAME.fullmatch(source):
        raise _problem("test member has an invalid public source", value, "use a lowercase source identifier containing only letters, digits and underscores")
    return source


def source_members(membership: Mapping[str, Any], snapshot_root: Path, *, split: str) -> dict[str, list[Path]]:
    """Return validated held-out image paths for one sealed split by source."""
    if split not in {"test", "val"}:
        raise _problem("diagnostic split is unsupported", repr(split), "use the sealed test or validation split")
    try:
        members = membership["members"][split]
    except (KeyError, TypeError) as error:
        raise _problem(f"sealed membership lacks {split} members", str(error), "restore membership.json from the materialized supervised snapshot") from error
    if not isinstance(members, list) or not members:
        raise _problem(f"sealed membership has no {split} members", repr(members), "materialize a nonempty held-out snapshot")
    grouped: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for row in members:
        if not isinstance(row, Mapping):
            raise _problem("sealed membership test row is malformed", repr(row), "restore membership.json from the materialized supervised snapshot")
        source = _source_from_path(row.get("source_file_path"))
        relative = row.get("snapshot_image")
        label_relative = row.get("snapshot_label")
        if not isinstance(relative, str) or not isinstance(label_relative, str):
            raise _problem("sealed membership lacks snapshot paths", repr(row), "restore membership.json from the materialized supervised snapshot")
        image = (snapshot_root / relative).resolve()
        label = (snapshot_root / label_relative).resolve()
        if image in seen:
            raise _problem("sealed membership contains a duplicate image", str(image), "restore a snapshot with one row per held-out image")
        if not image.is_file() or not label.is_file():
            raise _problem("sealed snapshot member is missing", f"image={image}, label={label}", "restore the immutable supervised dataset snapshot")
        seen.add(image)
        grouped.setdefault(source, []).append(image)
    return grouped


def source_test_members(membership: Mapping[str, Any], snapshot_root: Path) -> dict[str, list[Path]]:
    """Return validated fixed-test image paths grouped by public source."""
    return source_members(membership, snapshot_root, split="test")


def _sealed_membership_snapshot(dataset_root: Path) -> tuple[Path, Mapping[str, Any], Path]:
    """Resolve the full snapshot behind an optional balanced training view."""
    candidate = dataset_root / "membership.json"
    try:
        view_membership = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("dataset membership cannot be read", str(error), "restore membership.json beside the recorded dataset YAML") from error
    if isinstance(view_membership, Mapping) and isinstance(view_membership.get("members"), Mapping):
        return dataset_root, view_membership, candidate
    source_snapshot = view_membership.get("source_snapshot") if isinstance(view_membership, Mapping) else None
    if not isinstance(source_snapshot, str) or not source_snapshot:
        raise _problem("balanced view lacks source snapshot", repr(source_snapshot), "restore the full-label snapshot reference in membership.json")
    snapshot_root = Path(source_snapshot).resolve()
    membership_path = snapshot_root / "membership.json"
    try:
        membership = json.loads(membership_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("source snapshot membership cannot be read", str(error), "restore the full-label snapshot referenced by the balanced view") from error
    if not isinstance(membership, Mapping) or not isinstance(membership.get("members"), Mapping):
        raise _problem("source snapshot has no sealed members", str(membership_path), "restore the materialized full-label snapshot")
    return snapshot_root, membership, membership_path


def write_source_dataset(*, snapshot_root: Path, images: list[Path], output: Path) -> tuple[Path, Path]:
    """Materialize a diagnostic-only test view without touching the snapshot.

    Ultralytics writes ``labels/*.cache`` beside the paths it validates.  A
    list that points directly at the sealed snapshot would therefore mutate a
    cache inside immutable evidence.  Images are hard-linked when possible
    (copied only when the filesystem forbids it); labels are copied into the
    diagnostic root, where any framework cache is also contained.
    """
    if output.exists() or output.is_symlink():
        raise _problem("diagnostic dataset output already exists", str(output), "choose a fresh diagnostics output directory to preserve previous evidence")
    output.mkdir(parents=True)
    materialized_images: list[Path] = []
    for image in images:
        try:
            relative_image = image.resolve().relative_to(snapshot_root.resolve())
        except ValueError as error:
            raise _problem("source image is outside snapshot root", str(image), "use image paths listed by the sealed snapshot membership") from error
        if len(relative_image.parts) < 3 or relative_image.parts[0] != "images":
            raise _problem("source image has an unexpected snapshot location", str(relative_image), "restore the canonical images/<split>/<name> snapshot layout")
        relative_label = Path("labels", *relative_image.parts[1:]).with_suffix(".txt")
        label = snapshot_root / relative_label
        target_image = output / relative_image
        target_label = output / relative_label
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(image, target_image)
        except OSError:
            shutil.copyfile(image, target_image)
        shutil.copyfile(label, target_label)
        materialized_images.append(target_image.resolve())
    image_list = output / "test.txt"
    image_list.write_text("\n".join(str(image) for image in materialized_images) + "\n", encoding="utf-8")
    dataset = {
        "path": str(output),
        "train": str(image_list),
        "val": str(image_list),
        "test": str(image_list),
        "names": list(DEFAULT_CLASS_REGISTRY.class_names),
    }
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return image_list, dataset_yaml


def _serialize_subset_metrics(metrics: Any) -> dict[str, Any]:
    """Serialize a source subset without inventing AP for absent classes.

    Formal five-class evaluation deliberately requires AP50 for every
    canonical class.  A source-specific diagnostic can legitimately contain
    only a subset (for example Strawberry-DS), so its observed class IDs are
    recorded explicitly instead of padding absent classes with zeroes.
    """
    box = getattr(metrics, "box", None)
    try:
        all_ap = getattr(box, "all_ap")
        rows = all_ap.tolist() if hasattr(all_ap, "tolist") else list(all_ap)
        class_indices = getattr(box, "ap_class_index")
        class_ids = class_indices.tolist() if hasattr(class_indices, "tolist") else list(class_indices)
        if len(rows) != len(class_ids) or not rows:
            raise ValueError("per-class AP50 rows and observed class IDs differ")
        normalized_ids = [int(class_id) for class_id in class_ids]
        if len(set(normalized_ids)) != len(normalized_ids) or any(class_id not in DEFAULT_CLASS_REGISTRY.class_ids for class_id in normalized_ids):
            raise ValueError("observed class IDs are not a unique canonical subset")
        precision = float(getattr(box, "mp"))
        recall = float(getattr(box, "mr"))
        return {
            "map50": float(getattr(box, "map50")),
            "map50_95": float(getattr(box, "map")),
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "reported_class_ids": normalized_ids,
            "per_class_ap50": {str(class_id): float(row[0]) for class_id, row in zip(normalized_ids, rows)},
        }
    except (AttributeError, TypeError, ValueError, IndexError) as error:
        raise _problem("Ultralytics subset validation output is incompatible", str(error), "use a supported Ultralytics detection version with observed-class AP50 metrics") from error


def run_source_subset_diagnostic(*, run_dir: Path, output: Path, device: str | None = None, batch: int = 4, image_size: int | None = None, split: str = "test") -> Path:
    """Run a non-authoritative source breakdown for a sealed test or val split."""
    if output.exists() or output.is_symlink():
        raise _problem("diagnostic output already exists", str(output), "choose a fresh output directory; diagnostics are never overwritten")
    if batch <= 0:
        raise _problem("batch must be positive", repr(batch), "set --batch to a positive integer suitable for the available GPU")
    if split not in {"test", "val"}:
        raise _problem("diagnostic split is unsupported", repr(split), "use 'test' or 'val'")
    try:
        record = read_run_record(run_dir / "run_record.json")
    except Exception as error:
        raise _problem("run record cannot be read", str(error), "pass a completed supervised run directory") from error
    if record.status != "complete":
        raise _problem("run is not complete", f"status={record.status!r}", "run this diagnostic only after the recorded training run completes")
    snapshot_value = record.config_snapshot.get("dataset_yaml")
    if not isinstance(snapshot_value, str):
        raise _problem("run record lacks dataset YAML", repr(snapshot_value), "use a run created by train_supervised")
    dataset_yaml = Path(snapshot_value).resolve()
    try:
        effective_dataset, dataset_digest = _dataset_evidence(dataset_yaml)
    except SupervisedTrainingError as error:
        raise _problem("recorded dataset YAML is invalid", str(error), "restore the canonical dataset snapshot YAML") from error
    dataset_root = Path(str(effective_dataset["path"])).resolve()
    snapshot_root, membership, membership_path = _sealed_membership_snapshot(dataset_root)
    grouped = source_members(membership, snapshot_root, split=split)
    weights = run_dir / "weights" / "best.pt"
    try:
        checkpoint = file_evidence(weights, description="diagnostic best checkpoint")
        from ultralytics import YOLO
        model = YOLO(str(weights))
        validate_class_mapping(getattr(model, "names", None), DEFAULT_CLASS_REGISTRY)
    except (ModuleNotFoundError, DetectorAdapterError, SupervisedTrainingError) as error:
        raise _problem("checkpoint cannot be prepared for diagnostic evaluation", str(error), "restore the completed canonical checkpoint and active training environment") from error
    output.mkdir(parents=True)
    results: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "fruit_ssod_diagnostic_source_subset_v1",
        "diagnostic_only": True,
        "evaluated_split": split,
        "run_id": record.run_id,
        "checkpoint": checkpoint,
        "recorded_dataset_yaml": str(dataset_yaml),
        "recorded_dataset_yaml_sha256": dataset_digest,
        "membership_path": str(membership_path),
        "image_size": image_size if image_size is not None else record.config_snapshot.get("image_size"),
        "batch": batch,
        "device": device or record.config_snapshot.get("device"),
        "sources": {},
    }
    try:
        effective_image_size = int(image_size if image_size is not None else record.config_snapshot["image_size"])
        if effective_image_size <= 0:
            raise ValueError("image_size must be positive")
        all_images = [image for images in grouped.values() for image in images]
        all_list, all_yaml = write_source_dataset(snapshot_root=snapshot_root, images=all_images, output=output / "datasets" / f"all_{split}")
        all_metric = model.val(
            data=str(all_yaml), split="test", imgsz=effective_image_size, device=device or record.config_snapshot.get("device"),
            batch=batch, workers=0, verbose=False, plots=False,
            project=str(output / "ultralytics"), name=f"all_{split}", exist_ok=False,
        )
        results["fixed_test" if split == "test" else "validation"] = {
            "image_count": len(all_images), "image_list": str(all_list), "dataset_yaml": str(all_yaml),
            "metrics": _serialize_metric_object(all_metric).mapping(),
        }
        for source, images in grouped.items():
            subset_dir = output / "datasets" / source
            image_list, source_yaml = write_source_dataset(snapshot_root=snapshot_root, images=images, output=subset_dir)
            metric = model.val(
                data=str(source_yaml), split="test", imgsz=effective_image_size, device=device or record.config_snapshot.get("device"),
                batch=batch, workers=0, verbose=False, plots=False,
                project=str(output / "ultralytics"), name=source, exist_ok=False,
            )
            results["sources"][source] = {
                "image_count": len(images),
                "image_list": str(image_list),
                "dataset_yaml": str(source_yaml),
                "metrics": _serialize_subset_metrics(metric),
            }
    except Exception as error:
        raise _problem("source-subset evaluation failed", str(error), "check the immutable snapshot images, GPU device, and the active Ultralytics environment") from error
    result_path = output / "source_subset_metrics.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result_path
