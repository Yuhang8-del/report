"""Create a deterministic class-balanced view over a sealed YOLO snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError, _digest


@dataclass(frozen=True)
class BalancedTrainingResult:
    root: Path
    dataset_yaml: Path
    membership: Path
    base_image_count: int
    exposure_count: int


def _problem(problem: str, cause: str, remediation: str) -> SupervisedDatasetError:
    return SupervisedDatasetError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the sealed snapshot") from error
    if not isinstance(value, Mapping):
        raise _problem(f"{description} is not an object", repr(type(value).__name__), "restore the sealed snapshot")
    return value


def _members(payload: Mapping[str, Any], partition: str) -> list[Mapping[str, Any]]:
    groups = payload.get("members")
    rows = groups.get(partition) if isinstance(groups, Mapping) else None
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise _problem(f"{partition} membership is malformed", repr(rows), "restore the sealed full-label snapshot")
    ids = [row.get("source_image_id") for row in rows]
    if any(not isinstance(image_id, str) or not image_id for image_id in ids) or len(ids) != len(set(ids)):
        raise _problem(f"{partition} membership IDs are invalid", repr(ids[:5]), "restore the sealed full-label snapshot")
    return rows


def _snapshot_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _problem(f"{field} is missing", repr(value), "restore the sealed full-label snapshot")
    candidate = (root / value).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise _problem(f"{field} escapes the snapshot", value, "restore safe relative snapshot paths") from error
    return candidate


def _class_presence(label: Path) -> tuple[int, ...]:
    values: set[int] = set()
    try:
        lines = label.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise _problem("training label cannot be read", str(error), "restore the sealed full-label snapshot") from error
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise _problem("training label row is malformed", f"{label}:{line_number}", "regenerate the audited YOLO snapshot")
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise _problem("training label row is not numeric", f"{label}:{line_number}", "regenerate the audited YOLO snapshot") from error
        if class_id not in range(len(DEFAULT_CLASS_REGISTRY.classes)) or any(value < 0.0 or value > 1.0 for value in coordinates):
            raise _problem("training label row is outside the canonical contract", f"{label}:{line_number}", "regenerate the audited YOLO snapshot")
        values.add(class_id)
    if not values:
        raise _problem("training image has no fruit labels", str(label), "remove background-only records from the supervised snapshot")
    return tuple(sorted(values))


def _stable_key(seed: int, image_id: str, appearance: int) -> str:
    return hashlib.sha256(f"balanced-training-v1:{seed}:{image_id}:{appearance}".encode("utf-8")).hexdigest()


def materialize_balanced_training_view(
    snapshot_root: Path,
    output_root: Path,
    *,
    seed: int,
    max_appearances_per_image: int = 3,
) -> BalancedTrainingResult:
    """Repeat rare-class training paths without changing any sealed image or label."""
    if max_appearances_per_image < 1:
        raise _problem("max appearances must be positive", str(max_appearances_per_image), "use a value from one through three")
    source = snapshot_root.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem(f"balanced training view {root} already exists", "published training views are immutable", "choose a fresh output root")
    membership_path = source / "membership.json"
    payload = _mapping(membership_path, "full-label membership")
    if payload.get("artifact_type") != "sealed_full_label_supervised_upper_bound":
        raise _problem("input is not a sealed full-label upper bound", repr(payload.get("artifact_type")), "use the v12 full-label snapshot")
    train, validation, test = (_members(payload, name) for name in ("train", "val", "test"))
    train_ids = {str(row["source_image_id"]) for row in train}
    protected_ids = {str(row["source_image_id"]) for row in validation + test}
    overlap = train_ids & protected_ids
    if overlap:
        raise _problem("protected images occur in training", repr(sorted(overlap)[:5]), "restore the sealed full-label snapshot")

    image_paths: dict[str, Path] = {}
    class_presence: dict[str, tuple[int, ...]] = {}
    for row in train:
        image_id = str(row["source_image_id"])
        image_paths[image_id] = _snapshot_path(source, row.get("snapshot_image"), "snapshot_image")
        class_presence[image_id] = _class_presence(_snapshot_path(source, row.get("snapshot_label"), "snapshot_label"))
    class_count = len(DEFAULT_CLASS_REGISTRY.classes)
    before = [sum(class_id in classes for classes in class_presence.values()) for class_id in range(class_count)]
    if any(value == 0 for value in before):
        raise _problem("at least one canonical class is absent from training", repr(before), "repair the full-label training source before balancing")
    target = max(before)
    after = before.copy()
    appearances = {image_id: 1 for image_id in image_paths}
    repeated: list[str] = []
    while min(after) < target:
        rare_class = min(range(class_count), key=lambda class_id: (after[class_id], class_id))
        candidates = [
            image_id
            for image_id, classes in class_presence.items()
            if rare_class in classes and appearances[image_id] < max_appearances_per_image
        ]
        if not candidates:
            raise _problem(
                "class balance target cannot be reached within the appearance cap",
                f"class {rare_class}, current {after[rare_class]}, target {target}",
                "increase the cap only after reviewing duplicate exposure risk or add genuine rare-class data",
            )
        chosen = min(
            candidates,
            key=lambda image_id: (
                appearances[image_id],
                sum(after[class_id] / target for class_id in class_presence[image_id]),
                _stable_key(seed, image_id, appearances[image_id] + 1),
            ),
        )
        appearances[chosen] += 1
        repeated.append(chosen)
        for class_id in class_presence[chosen]:
            after[class_id] += 1

    ordered_ids = sorted(image_paths) + repeated
    lines = [str(image_paths[image_id]) for image_id in ordered_ids]
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        (temporary / "train_balanced.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        dataset = {
            "path": str(root),
            "train": str((root / "train_balanced.txt").resolve()),
            "val": str((source / "val.txt").resolve()),
            "test": str((source / "test.txt").resolve()),
            "names": list(DEFAULT_CLASS_REGISTRY.class_names),
        }
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        names = DEFAULT_CLASS_REGISTRY.class_names
        evidence = {
            "schema_version": "1.0",
            "artifact_type": "deterministic_class_balanced_training_view",
            "algorithm": "rare-class-greedy-image-exposure-v1",
            "seed": seed,
            "max_appearances_per_image": max_appearances_per_image,
            "target_class_image_exposure": target,
            "source_snapshot": str(source),
            "source_membership_sha256": _digest(membership_path),
            "output_root": str(root),
            "base_train_image_count": len(train),
            "balanced_training_exposure_count": len(lines),
            "class_image_exposure_before": {names[index]: value for index, value in enumerate(before)},
            "class_image_exposure_after": {names[index]: value for index, value in enumerate(after)},
            "repeated_image_appearances": {image_id: count for image_id, count in sorted(appearances.items()) if count > 1},
            "protected_validation_count": len(validation),
            "protected_test_count": len(test),
        }
        membership_out = temporary / "membership.json"
        membership_out.write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, root)
        temporary = None
        return BalancedTrainingResult(root, root / "dataset.yaml", root / "membership.json", len(train), len(lines))
    except SupervisedDatasetError:
        raise
    except OSError as error:
        raise _problem("balanced training view cannot be written", str(error), "choose a fresh writable output root") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
