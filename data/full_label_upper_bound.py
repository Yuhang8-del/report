"""Materialize a supervised upper bound that restores hidden train labels.

The SSOD split deliberately hides labels for an unlabeled pool.  A credible
fully labelled upper-bound diagnostic may use those already-existing labels,
but it must keep validation, pseudo-audit and test membership unchanged.  This
module creates that separate immutable snapshot without changing the SSOD
budget artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.data.supervised_dataset import (
    SupervisedDatasetError,
    SupervisedDatasetResult,
    _digest,
    _load_records,
    _safe_source,
    _yolo_lines,
)


def _problem(problem: str, cause: str, remediation: str) -> SupervisedDatasetError:
    return SupervisedDatasetError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _load_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the untouched JSON artifact") from error
    if not isinstance(value, Mapping):
        raise _problem(f"{description} is not an object", repr(type(value).__name__), "restore the untouched JSON artifact")
    return value


def _id_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise _problem(f"{field} is malformed", repr(value), "regenerate the deterministic split")
    if len(value) != len(set(value)):
        raise _problem(f"{field} contains duplicate image IDs", repr(value[:5]), "regenerate the deterministic split")
    return tuple(value)


def _record_ids(records: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    values = {str(record.get("source_image_id", "")) for record in records}
    if "" in values or len(values) != len(records):
        raise _problem(f"{field} records have invalid or duplicate image IDs", str(len(records)), "restore the sealed split labels")
    return values


def _candidate_records(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _load_mapping(path, "candidate manifest")
    rows = payload.get("images")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise _problem("candidate manifest has no image records", repr(rows), "use the audited image-level candidate manifest")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        image_id = row.get("source_image_id")
        if not isinstance(image_id, str) or not image_id or image_id in output:
            raise _problem("candidate manifest image IDs are invalid or ambiguous", repr(image_id), "regenerate the candidate manifest")
        output[image_id] = row
    return output


def _publish_image(original: Path, output: Path) -> bool:
    """Copy an image and repair missing JPEG EOI before sealing its digest."""
    shutil.copy2(original, output)
    try:
        with Image.open(output) as image:
            image.verify()
        normalized = False
        if output.suffix.lower() in {".jpg", ".jpeg"}:
            with output.open("rb") as handle:
                handle.seek(-2, 2)
                missing_eoi = handle.read() != b"\xff\xd9"
            if missing_eoi:
                # Ultralytics otherwise re-encodes the image in place during
                # cache creation. Appending the standard JPEG end marker is
                # sufficient, preserves pixels/annotation geometry, and makes
                # the sealed digest stable before training starts.
                with output.open("ab") as handle:
                    handle.write(b"\xff\xd9")
                normalized = True
        with Image.open(output) as image:
            image.verify()
        return normalized
    except (OSError, ValueError) as error:
        raise _problem("source image cannot be normalized", f"{original}: {error}", "repair or quarantine the source image before materialization") from error


def _copy_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_root: Path,
    root: Path,
    temporary: Path,
) -> dict[str, list[dict[str, Any]]]:
    membership: dict[str, list[dict[str, Any]]] = {}
    for partition, records in groups.items():
        image_dir, label_dir = temporary / "images" / partition, temporary / "labels" / partition
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        entries: list[str] = []
        members: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda value: str(value["source_image_id"])):
            image_id = str(record["source_image_id"])
            if Path(image_id).name != image_id or image_id in {".", ".."}:
                raise _problem("source image ID is unsafe for a snapshot filename", repr(image_id), "regenerate canonical source IDs")
            original = _safe_source(source_root, record.get("file_path"))
            extension = original.suffix.lower() or ".jpg"
            image_out, label_out = image_dir / f"{image_id}{extension}", label_dir / f"{image_id}.txt"
            source_digest = _digest(original)
            normalized = _publish_image(original, image_out)
            image_digest = _digest(image_out)
            if not normalized and source_digest != image_digest:
                raise _problem("snapshot image digest differs from source", image_id, "retry from stable local storage")
            label_out.write_text(_yolo_lines(record), encoding="utf-8", newline="\n")
            entries.append(str((root / image_out.relative_to(temporary)).resolve()))
            members.append(
                {
                    "source_image_id": image_id,
                    "source_file_path": str(record["file_path"]),
                    "snapshot_image": image_out.relative_to(temporary).as_posix(),
                    "source_image_sha256": source_digest,
                    "image_sha256": image_digest,
                    "image_normalized": normalized,
                    "snapshot_label": label_out.relative_to(temporary).as_posix(),
                }
            )
        (temporary / f"{partition}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
        membership[partition] = members
    return membership


def materialize_full_label_upper_bound(
    candidate_manifest: Path,
    split_root: Path,
    source_root: Path,
    output_root: Path,
    *,
    expected_train_count: int,
) -> SupervisedDatasetResult:
    """Publish train_pool + hidden unlabeled labels with unchanged val/test."""
    if expected_train_count <= 0:
        raise _problem("expected train count must be positive", str(expected_train_count), "declare the audited upper-bound train count")
    candidate_path = candidate_manifest.resolve(strict=True)
    split = split_root.resolve(strict=True)
    source = source_root.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem(f"full-label snapshot {root} already exists", "published snapshots are immutable", "choose a fresh output root")

    manifest = _load_mapping(split / "split_manifest.json", "split manifest")
    train_pool = _id_sequence(manifest.get("train_pool_image_ids"), "train_pool_image_ids")
    unlabeled = _id_sequence(manifest.get("unlabeled_image_ids"), "unlabeled_image_ids")
    split_ids = manifest.get("split_image_ids")
    if not isinstance(split_ids, Mapping):
        raise _problem("split_image_ids is missing", repr(split_ids), "regenerate the deterministic split")
    validation_ids = _id_sequence(split_ids.get("validation"), "validation split")
    test_ids = _id_sequence(split_ids.get("test"), "test split")
    pseudo_ids = _id_sequence(split_ids.get("pseudo_audit"), "pseudo-audit split")

    partitions = {
        "train_pool": set(train_pool),
        "unlabeled": set(unlabeled),
        "validation": set(validation_ids),
        "test": set(test_ids),
        "pseudo_audit": set(pseudo_ids),
    }
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = partitions[left] & partitions[right]
            if overlap:
                raise _problem("upper-bound partitions overlap", f"{left}/{right}: {sorted(overlap)[:5]}", "regenerate the deterministic split")

    budget_records = _load_records(split / "budgets" / "100" / "labels.json")
    if _record_ids(budget_records, "100% budget") != partitions["train_pool"]:
        raise _problem("100% budget membership differs from train_pool", "the sealed budget was changed", "restore the original split output")
    validation_records = _load_records(split / "protected_splits" / "validation_labels.json")
    test_records = _load_records(split / "protected_splits" / "test_labels.json")
    if _record_ids(validation_records, "validation") != partitions["validation"] or _record_ids(test_records, "test") != partitions["test"]:
        raise _problem("protected label membership differs from split manifest", "validation or test labels changed", "restore the sealed protected artifacts")

    training_ids = partitions["train_pool"] | partitions["unlabeled"]
    if len(training_ids) != expected_train_count:
        raise _problem("full-label training count differs from the declared audit", f"expected {expected_train_count}, found {len(training_ids)}", "review the split manifest before materializing")
    candidates = _candidate_records(candidate_path)
    missing = (training_ids | partitions["validation"] | partitions["test"]) - set(candidates)
    if missing:
        raise _problem("candidate manifest lacks required images", repr(sorted(missing)[:5]), "restore the audited candidate manifest")
    groups: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": [candidates[image_id] for image_id in training_ids],
        "val": validation_records,
        "test": test_records,
    }

    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        members = _copy_groups(groups, source_root=source, root=root, temporary=temporary)
        dataset = {"path": str(root), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        evidence = {
            "schema_version": "1.0",
            "artifact_type": "sealed_full_label_supervised_upper_bound",
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": _digest(candidate_path),
            "split_root": str(split),
            "split_manifest_sha256": _digest(split / "split_manifest.json"),
            "split_protocol_fingerprint": manifest.get("fingerprints", {}).get("split_protocol") if isinstance(manifest.get("fingerprints"), Mapping) else None,
            "train_pool_count": len(train_pool),
            "recovered_hidden_label_count": len(unlabeled),
            "full_label_train_count": len(training_ids),
            "protected_pseudo_audit_count": len(pseudo_ids),
            "members": members,
        }
        (temporary / "membership.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root)
        temporary = None
        return SupervisedDatasetResult(root, root / "dataset.yaml", root / "membership.json", sum(len(value) for value in groups.values()))
    except SupervisedDatasetError:
        raise
    except OSError as error:
        raise _problem("full-label snapshot could not be written", str(error), "choose a fresh writable output root and verify source files") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _audit_label(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise _problem("snapshot label cannot be read", str(error), "restore the sealed snapshot") from error
    if not lines:
        raise _problem("snapshot label is empty", str(path), "restore the audited supervised labels")
    for number, line in enumerate(lines, start=1):
        parts = line.split()
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except (IndexError, ValueError) as error:
            raise _problem("snapshot label is malformed", f"{path}:{number}", "restore the audited supervised labels") from error
        if len(parts) != 5 or class_id not in range(len(DEFAULT_CLASS_REGISTRY.classes)) or any(value < 0.0 or value > 1.0 for value in coordinates):
            raise _problem("snapshot label violates the canonical YOLO contract", f"{path}:{number}", "restore the audited supervised labels")
    return len(lines)


def audit_full_label_upper_bound(snapshot_root: Path) -> dict[str, Any]:
    """Verify every sealed image digest and canonical label without mutation."""
    root = snapshot_root.resolve(strict=True)
    membership_path = root / "membership.json"
    payload = _load_mapping(membership_path, "full-label membership")
    if payload.get("artifact_type") != "sealed_full_label_supervised_upper_bound":
        raise _problem("snapshot artifact type is invalid", repr(payload.get("artifact_type")), "use a sealed v12 full-label snapshot")
    members = payload.get("members")
    if not isinstance(members, Mapping):
        raise _problem("snapshot members are missing", repr(members), "restore the sealed membership")
    counts: dict[str, int] = {}
    seen: set[str] = set()
    image_count = label_count = box_count = 0
    for partition in ("train", "val", "test"):
        rows = members.get(partition)
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise _problem(f"snapshot {partition} membership is malformed", repr(rows), "restore the sealed membership")
        counts[partition] = len(rows)
        for row in rows:
            image_id = row.get("source_image_id")
            if not isinstance(image_id, str) or not image_id or image_id in seen:
                raise _problem("snapshot membership contains duplicate or invalid IDs", repr(image_id), "restore disjoint train, validation and test membership")
            seen.add(image_id)
            image = _safe_source(root, row.get("snapshot_image"))
            label = _safe_source(root, row.get("snapshot_label"))
            expected_digest = row.get("image_sha256")
            if not isinstance(expected_digest, str) or _digest(image) != expected_digest:
                raise _problem("snapshot image digest differs from membership", image_id, "discard the mutated snapshot and rematerialize from the canonical source")
            try:
                with Image.open(image) as loaded:
                    loaded.verify()
                if image.suffix.lower() in {".jpg", ".jpeg"}:
                    with image.open("rb") as handle:
                        handle.seek(-2, 2)
                        if handle.read() != b"\xff\xd9":
                            raise ValueError("JPEG end marker is missing")
            except (OSError, ValueError) as error:
                raise _problem("snapshot image verification failed", f"{image}: {error}", "discard the snapshot and rematerialize normalized images") from error
            box_count += _audit_label(label)
            image_count += 1
            label_count += 1
    return {
        "schema_version": "1.0",
        "artifact_type": "full_label_supervised_upper_bound_audit",
        "snapshot_root": str(root),
        "membership_sha256": _digest(membership_path),
        "partition_counts": counts,
        "verified_image_count": image_count,
        "verified_label_count": label_count,
        "verified_box_count": box_count,
        "critical_findings": 0,
    }
