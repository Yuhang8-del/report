"""Publish a supervised training expansion without changing protected splits.

The v13 recovery protocol may add only newly curated natural-scene images to
training.  Validation and test remain the immutable v12 lists; this module
therefore never copies, rewrites, or re-splits protected examples.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import imagehash
import yaml
from PIL import Image, UnidentifiedImageError

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.data.full_label_upper_bound import _audit_label, _publish_image
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError, _digest, _safe_source, _yolo_lines


@dataclass(frozen=True)
class TrainOnlyAugmentationResult:
    root: Path
    dataset_yaml: Path
    membership: Path
    base_train_exposure_count: int
    added_train_image_count: int


def _problem(problem: str, cause: str, remediation: str) -> SupervisedDatasetError:
    return SupervisedDatasetError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the immutable input artifact") from error
    if not isinstance(payload, Mapping):
        raise _problem(f"{description} is not an object", repr(type(payload).__name__), "restore the immutable input artifact")
    return payload


def _safe_snapshot_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _problem(f"protected membership omits {field}", repr(value), "restore the sealed v12 snapshot")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise _problem(f"protected membership {field} is unsafe", repr(value), "restore the sealed v12 snapshot")
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as error:
        raise _problem(f"protected membership {field} is unavailable", str(error), "restore the sealed v12 snapshot") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise _problem(f"protected membership {field} is not a regular file", str(candidate), "restore the sealed v12 snapshot")
    return candidate


def _members(payload: Mapping[str, Any], partition: str) -> list[Mapping[str, Any]]:
    raw = payload.get("members")
    rows = raw.get(partition) if isinstance(raw, Mapping) else None
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise _problem(f"sealed {partition} membership is malformed", repr(rows), "restore the sealed v12 snapshot")
    ids = [row.get("source_image_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise _problem(f"sealed {partition} membership IDs are invalid", repr(ids[:5]), "restore the sealed v12 snapshot")
    return rows


def _load_base(base_root: Path) -> tuple[Path, list[str], list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]], Path]:
    """Return the sealed source and selected base training exposure list.

    A full snapshot supplies its original ``train.txt``; a deterministic
    balanced view supplies ``train_balanced.txt`` while resolving its protected
    membership back to the full sealed snapshot.
    """
    base_payload = _mapping(base_root / "membership.json", "base training membership")
    artifact_type = base_payload.get("artifact_type")
    if artifact_type == "sealed_full_label_supervised_upper_bound":
        sealed = base_root
        training_list = base_root / "train.txt"
    elif artifact_type == "deterministic_class_balanced_training_view":
        source_value = base_payload.get("source_snapshot")
        if not isinstance(source_value, str) or not source_value:
            raise _problem("balanced training view has no source snapshot", repr(source_value), "restore the immutable balanced-view membership")
        try:
            sealed = Path(source_value).resolve(strict=True)
        except OSError as error:
            raise _problem("balanced training view source snapshot is unavailable", str(error), "restore the v12 full-label snapshot") from error
        training_list = base_root / "train_balanced.txt"
    else:
        raise _problem("base artifact is not a sealed snapshot or balanced view", repr(artifact_type), "use the v12 full-label snapshot or its deterministic balanced view")
    sealed_payload = _mapping(sealed / "membership.json", "sealed v12 membership")
    if sealed_payload.get("artifact_type") != "sealed_full_label_supervised_upper_bound":
        raise _problem("resolved protected source is not a sealed full-label snapshot", repr(sealed_payload.get("artifact_type")), "restore the v12 full-label snapshot")
    try:
        lines = [line.strip() for line in training_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as error:
        raise _problem("base training list cannot be read", str(error), "restore the immutable base training view") from error
    if not lines:
        raise _problem("base training list is empty", str(training_list), "restore the immutable base training view")
    for line in lines:
        try:
            candidate = Path(line).resolve(strict=True)
            candidate.relative_to(sealed)
        except (OSError, ValueError) as error:
            raise _problem("base training list escapes the sealed snapshot", line, "restore the immutable base training view") from error
        if not candidate.is_file() or candidate.is_symlink():
            raise _problem("base training image is unavailable", str(candidate), "restore the immutable base training view")
    return sealed, lines, _members(sealed_payload, "train"), _members(sealed_payload, "val"), _members(sealed_payload, "test"), training_list


def _fingerprint(path: Path) -> tuple[str, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                perceptual = int(str(imagehash.phash(image.convert("RGB"))), 16)
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise _problem("image cannot be fingerprinted", f"{path}: {error}", "repair or quarantine the image before training") from error
    return _digest(path), perceptual


def _protected_fingerprints(sealed: Path, rows: list[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
    output: list[tuple[str, str, int]] = []
    for row in rows:
        image_id = str(row["source_image_id"])
        image = _safe_snapshot_path(sealed, row.get("snapshot_image"), "snapshot_image")
        expected = row.get("image_sha256")
        digest, perceptual = _fingerprint(image)
        if isinstance(expected, str) and expected and expected != digest:
            raise _problem("sealed protected image digest differs from membership", image_id, "restore the v12 snapshot before building v13")
        output.append((image_id, digest, perceptual))
    return output


def _candidate_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = _mapping(path, "added candidate manifest")
    rows = payload.get("images")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise _problem("added candidate manifest has no image records", repr(rows), "use a cleaned image-level candidate manifest")
    output = sorted(rows, key=lambda row: str(row.get("source_image_id", "")))
    ids = [row.get("source_image_id") for row in output]
    if any(not isinstance(value, str) or not value or Path(value).name != value for value in ids) or len(ids) != len(set(ids)):
        raise _problem("added candidate image IDs are invalid or duplicated", repr(ids[:5]), "regenerate the cleaned image-level candidate manifest")
    return output


def materialize_train_only_augmentation(
    base_training_root: Path,
    added_candidate_manifest: Path,
    added_source_root: Path,
    output_root: Path,
    *,
    protected_near_hash_threshold: int = 4,
) -> TrainOnlyAugmentationResult:
    """Add curated images to training while retaining the original val/test lists.

    Exact image identity and perceptual near-duplicates of protected validation
    or test samples are rejected before any output is published.  The caller
    must provide an image-level manifest from the normal cleaning pipeline.
    """
    if isinstance(protected_near_hash_threshold, bool) or not isinstance(protected_near_hash_threshold, int) or not 0 <= protected_near_hash_threshold <= 64:
        raise _problem("protected near-hash threshold is invalid", repr(protected_near_hash_threshold), "use an integer from zero through 64")
    base = base_training_root.resolve(strict=True)
    source = added_source_root.resolve(strict=True)
    candidate_path = added_candidate_manifest.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem(f"training augmentation {root} already exists", "published training artifacts are immutable", "choose a fresh output root")

    sealed, base_lines, base_train, validation, test, training_list = _load_base(base)
    all_base_ids = {str(row["source_image_id"]) for row in base_train + validation + test}
    protected = _protected_fingerprints(sealed, validation + test)
    protected_hashes = {digest for _, digest, _ in protected}
    protected_phashes = [(image_id, value) for image_id, _, value in protected]
    rows = _candidate_rows(candidate_path)

    additions: list[tuple[Mapping[str, Any], Path, str, int]] = []
    excluded_protected: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    seen_perceptual: list[tuple[str, int]] = []
    for row in rows:
        image_id = str(row["source_image_id"])
        if image_id in all_base_ids:
            raise _problem("added training image ID already occurs in the sealed snapshot", image_id, "use a selection excluding all prior source image IDs")
        original = _safe_source(source, row.get("file_path"))
        digest, perceptual = _fingerprint(original)
        if digest in protected_hashes:
            excluded_protected.append({"source_image_id": image_id, "reason": "exact_duplicate_of_protected_member"})
            continue
        matching = next((protected_id for protected_id, value in protected_phashes if (value ^ perceptual).bit_count() <= protected_near_hash_threshold), None)
        if matching is not None:
            excluded_protected.append({"source_image_id": image_id, "reason": "near_duplicate_of_protected_member", "protected_source_image_id": matching})
            continue
        if digest in seen_hashes or any((value ^ perceptual).bit_count() <= protected_near_hash_threshold for _, value in seen_perceptual):
            raise _problem("added candidate manifest contains duplicate or near-duplicate training images", image_id, "deduplicate the incremental source before materialization")
        seen_hashes.add(digest)
        seen_perceptual.append((image_id, perceptual))
        additions.append((row, original, digest, perceptual))
    if not additions:
        raise _problem("no v13 additions remain after protected-duplicate exclusion", str(len(excluded_protected)), "select further source-ID-disjoint natural-scene images before training")

    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        image_dir = temporary / "images" / "added_train"
        label_dir = temporary / "labels" / "added_train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        addition_lines: list[str] = []
        addition_members: list[dict[str, Any]] = []
        for row, original, source_digest, _ in additions:
            image_id = str(row["source_image_id"])
            extension = original.suffix.lower() or ".jpg"
            image_out = image_dir / f"{image_id}{extension}"
            label_out = label_dir / f"{image_id}.txt"
            normalized = _publish_image(original, image_out)
            output_digest = _digest(image_out)
            if not normalized and output_digest != source_digest:
                raise _problem("published added image digest differs from source", image_id, "retry from stable local storage")
            label_out.write_text(_yolo_lines(row), encoding="utf-8", newline="\n")
            addition_lines.append(str((root / image_out.relative_to(temporary)).resolve()))
            addition_members.append(
                {
                    "source_image_id": image_id,
                    "source": row.get("source"),
                    "source_file_path": row.get("file_path"),
                    "snapshot_image": image_out.relative_to(temporary).as_posix(),
                    "snapshot_label": label_out.relative_to(temporary).as_posix(),
                    "source_image_sha256": source_digest,
                    "image_sha256": output_digest,
                    "image_normalized": normalized,
                }
            )
        all_lines = [*base_lines, *addition_lines]
        (temporary / "train_augmented.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8", newline="\n")
        dataset = {
            "path": str(root),
            "train": str((root / "train_augmented.txt").resolve()),
            "val": str((sealed / "val.txt").resolve()),
            "test": str((sealed / "test.txt").resolve()),
            "names": list(DEFAULT_CLASS_REGISTRY.class_names),
        }
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        evidence = {
            "schema_version": "1.0",
            "artifact_type": "train_only_supervised_augmentation",
            "base_training_root": str(base),
            "base_training_membership_sha256": _digest(base / "membership.json"),
            "base_training_list": str(training_list.resolve()),
            "base_training_list_sha256": _digest(training_list),
            "sealed_snapshot": str(sealed),
            "sealed_membership_sha256": _digest(sealed / "membership.json"),
            "added_candidate_manifest": str(candidate_path),
            "added_candidate_manifest_sha256": _digest(candidate_path),
            "added_source_root": str(source),
            "protected_near_hash_threshold": protected_near_hash_threshold,
            "excluded_protected_members": excluded_protected,
            "base_train_exposure_count": len(base_lines),
            "added_train_image_count": len(addition_members),
            "total_train_exposure_count": len(all_lines),
            "train_augmented_list_sha256": _digest(temporary / "train_augmented.txt"),
            "protected_validation_count": len(validation),
            "protected_test_count": len(test),
            "preserved_partitions": {
                "validation": {"list": str((sealed / "val.txt").resolve()), "list_sha256": _digest(sealed / "val.txt"), "member_ids": [str(row["source_image_id"]) for row in validation]},
                "test": {"list": str((sealed / "test.txt").resolve()), "list_sha256": _digest(sealed / "test.txt"), "member_ids": [str(row["source_image_id"]) for row in test]},
            },
            "added_train_members": addition_members,
        }
        (temporary / "membership.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, root)
        temporary = None
        return TrainOnlyAugmentationResult(root, root / "dataset.yaml", root / "membership.json", len(base_lines), len(addition_members))
    except SupervisedDatasetError:
        raise
    except OSError as error:
        raise _problem("training-only augmentation cannot be written", str(error), "choose a fresh writable output root") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def audit_train_only_augmentation(augmentation_root: Path) -> dict[str, Any]:
    """Verify a published v13 training-only view without evaluating its test set."""
    root = augmentation_root.resolve(strict=True)
    membership_path = root / "membership.json"
    payload = _mapping(membership_path, "training-only augmentation membership")
    if payload.get("artifact_type") != "train_only_supervised_augmentation":
        raise _problem("augmentation artifact type is invalid", repr(payload.get("artifact_type")), "use a published v13 train-only augmentation")
    sealed_value = payload.get("sealed_snapshot")
    if not isinstance(sealed_value, str) or not sealed_value:
        raise _problem("augmentation has no sealed snapshot reference", repr(sealed_value), "restore the augmentation membership")
    try:
        sealed = Path(sealed_value).resolve(strict=True)
    except OSError as error:
        raise _problem("referenced sealed snapshot is unavailable", str(error), "restore the v12 snapshot before auditing") from error
    sealed_payload = _mapping(sealed / "membership.json", "sealed v12 membership")
    if sealed_payload.get("artifact_type") != "sealed_full_label_supervised_upper_bound":
        raise _problem("referenced protected source is not a sealed full-label snapshot", repr(sealed_payload.get("artifact_type")), "restore the v12 snapshot")
    base_train, validation, test = (_members(sealed_payload, partition) for partition in ("train", "val", "test"))
    preserved = payload.get("preserved_partitions")
    if not isinstance(preserved, Mapping):
        raise _problem("augmentation preserved partition evidence is malformed", repr(preserved), "restore the augmentation membership")
    for key, filename, rows in (("validation", "val.txt", validation), ("test", "test.txt", test)):
        item = preserved.get(key)
        if not isinstance(item, Mapping):
            raise _problem(f"augmentation has no {key} preservation evidence", repr(item), "restore the augmentation membership")
        expected_digest = item.get("list_sha256")
        declared_path = item.get("list")
        expected_ids = item.get("member_ids")
        actual = sealed / filename
        if not isinstance(expected_digest, str) or expected_digest != _digest(actual):
            raise _problem(f"sealed {key} list digest differs from augmentation evidence", str(actual), "do not train; restore the immutable v12 protected list")
        if declared_path != str(actual.resolve()):
            raise _problem(f"sealed {key} list path differs from augmentation evidence", repr(declared_path), "do not train; restore the immutable augmentation membership")
        if expected_ids != [str(row["source_image_id"]) for row in rows]:
            raise _problem(f"sealed {key} membership differs from augmentation evidence", repr(expected_ids), "do not train; restore the immutable v12 protected membership")
    train_list = root / "train_augmented.txt"
    expected_train_digest = payload.get("train_augmented_list_sha256")
    if not isinstance(expected_train_digest, str) or expected_train_digest != _digest(train_list):
        raise _problem("augmented training list digest differs from membership", str(train_list), "restore the immutable v13 training view")
    try:
        dataset = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _problem("augmentation dataset YAML cannot be read", str(error), "restore the immutable v13 training view") from error
    if not isinstance(dataset, Mapping) or dataset.get("val") != str((sealed / "val.txt").resolve()) or dataset.get("test") != str((sealed / "test.txt").resolve()):
        raise _problem("augmentation dataset YAML no longer references the sealed protected lists", repr(dataset), "restore the immutable v13 dataset YAML")
    additions = payload.get("added_train_members")
    if not isinstance(additions, list) or not additions or any(not isinstance(item, Mapping) for item in additions):
        raise _problem("augmentation added training membership is malformed", repr(additions), "restore the immutable v13 training view")
    known_ids = {str(row["source_image_id"]) for row in base_train + validation + test}
    added_ids: set[str] = set()
    verified_boxes = 0
    for row in additions:
        image_id = row.get("source_image_id")
        if not isinstance(image_id, str) or not image_id or image_id in known_ids or image_id in added_ids:
            raise _problem("augmentation added training IDs overlap or are invalid", repr(image_id), "restore source-ID-disjoint v13 additions")
        added_ids.add(image_id)
        image = _safe_snapshot_path(root, row.get("snapshot_image"), "added snapshot_image")
        label = _safe_snapshot_path(root, row.get("snapshot_label"), "added snapshot_label")
        expected_digest = row.get("image_sha256")
        if not isinstance(expected_digest, str) or _digest(image) != expected_digest:
            raise _problem("augmentation added image digest differs from membership", image_id, "restore the immutable v13 training view")
        verified_boxes += _audit_label(label)
    expected_added = payload.get("added_train_image_count")
    expected_total = payload.get("total_train_exposure_count")
    try:
        actual_lines = [line for line in train_list.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError) as error:
        raise _problem("augmented training list cannot be read", str(error), "restore the immutable v13 training view") from error
    if expected_added != len(additions) or expected_total != len(actual_lines):
        raise _problem("augmentation count evidence differs from files", f"added={len(additions)}, exposure={len(actual_lines)}", "restore the immutable v13 training view")
    return {
        "schema_version": "1.0",
        "artifact_type": "train_only_supervised_augmentation_audit",
        "augmentation_root": str(root),
        "membership_sha256": _digest(membership_path),
        "verified_added_train_image_count": len(additions),
        "verified_added_train_label_count": len(additions),
        "verified_added_train_box_count": verified_boxes,
        "protected_validation_count": len(validation),
        "protected_test_count": len(test),
        "verified_train_exposure_count": len(actual_lines),
        "critical_findings": 0,
    }
