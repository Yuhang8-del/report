"""Reviewed class-registry expansion and replay dataset construction."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.open_world.box_protocol import _link_or_copy, label_path_for_image, read_yolo_boxes
from fruit_ssod.open_world.discovery import NOVEL_CLASSES


def _write_label(source_label: Path, destination: Path, class_id: int | None) -> int:
    boxes = read_yolo_boxes(source_label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for box in boxes:
        output_class = box.class_id if class_id is None else class_id
        rows.append(
            f"{output_class} {box.x_center:.10f} {box.y_center:.10f} {box.width:.10f} {box.height:.10f}\n"
        )
    destination.write_text("".join(rows), encoding="utf-8")
    return len(boxes)


def _materialize(
    source_image: Path,
    output_root: Path,
    split: str,
    *,
    class_id: int | None,
) -> tuple[Path, int]:
    identity = hashlib.sha256(str(source_image.resolve()).casefold().encode("utf-8")).hexdigest()[:20]
    destination_image = output_root / "images" / split / f"{identity}{source_image.suffix.casefold()}"
    _link_or_copy(source_image, destination_image)
    destination_label = output_root / "labels" / split / f"{identity}.txt"
    box_count = _write_label(label_path_for_image(source_image), destination_label, class_id)
    return destination_image.resolve(), box_count


def build_incremental_replay_dataset(
    *,
    known_train_list: Path,
    known_validation_list: Path,
    protected_novel_truth: Path,
    confirmed_categories: Sequence[str],
    output_root: Path,
    replay_images: int = 2000,
    novel_validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, object]:
    """Append reviewed classes while replaying old classes and preserving holdout truth."""
    confirmed = tuple(dict.fromkeys(category.strip() for category in confirmed_categories if category.strip()))
    unsupported = sorted(set(confirmed) - set(NOVEL_CLASSES))
    if not confirmed or unsupported:
        raise ValueError(f"confirmed categories must be a nonempty subset of {NOVEL_CLASSES}; invalid={unsupported}")
    if not 0.0 < novel_validation_fraction < 0.5:
        raise ValueError("novel_validation_fraction must be between 0 and 0.5")
    known_names = [item.name for item in DEFAULT_CLASS_REGISTRY.classes]
    names = known_names + list(confirmed)
    class_ids = {name: index for index, name in enumerate(names)}
    train_sources = sorted(
        set(Path(line).resolve() for line in known_train_list.read_text(encoding="utf-8").splitlines() if line.strip()),
        key=lambda path: str(path).casefold(),
    )
    rng = random.Random(seed)
    rng.shuffle(train_sources)
    replay = train_sources[: min(replay_images, len(train_sources))]
    known_validation = [
        Path(line).resolve() for line in known_validation_list.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    truth = json.loads(protected_novel_truth.read_text(encoding="utf-8"))["records"]
    if any(record["split"] == "holdout" and record["category"] in confirmed for record in truth):
        # Presence is expected, but those records must never be materialized below.
        pass
    discovery_by_category = {
        category: sorted(
            (record for record in truth if record["split"] == "discovery" and record["category"] == category),
            key=lambda record: record["image_id"],
        )
        for category in confirmed
    }
    train_destinations: list[Path] = []
    val_destinations: list[Path] = []
    box_counts = {"known_replay": 0, "known_validation": 0, "novel_train": 0, "novel_validation": 0}
    for source in replay:
        destination, count = _materialize(source, output_root, "train", class_id=None)
        train_destinations.append(destination)
        box_counts["known_replay"] += count
    for source in known_validation:
        destination, count = _materialize(source, output_root, "val", class_id=None)
        val_destinations.append(destination)
        box_counts["known_validation"] += count
    novel_train_counts: dict[str, int] = {}
    novel_validation_counts: dict[str, int] = {}
    for category, records in discovery_by_category.items():
        validation_count = max(1, round(len(records) * novel_validation_fraction))
        validation_ids = {record["image_id"] for record in records[:validation_count]}
        for record in records:
            split = "val" if record["image_id"] in validation_ids else "train"
            destination, count = _materialize(
                Path(record["image_path"]), output_root, split, class_id=class_ids[category]
            )
            if split == "train":
                train_destinations.append(destination)
                novel_train_counts[category] = novel_train_counts.get(category, 0) + 1
                box_counts["novel_train"] += count
            else:
                val_destinations.append(destination)
                novel_validation_counts[category] = novel_validation_counts.get(category, 0) + 1
                box_counts["novel_validation"] += count
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.txt"
    val_path = output_root / "val.txt"
    train_path.write_text("".join(f"{path}\n" for path in train_destinations), encoding="utf-8")
    val_path.write_text("".join(f"{path}\n" for path in val_destinations), encoding="utf-8")
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_root.as_posix()}\ntrain: {train_path.as_posix()}\nval: {val_path.as_posix()}\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(names)),
        encoding="utf-8",
    )
    registry = {
        "schema_version": "2.0",
        "base_registry": known_names,
        "confirmed_additions": list(confirmed),
        "classes": [{"id": index, "name": name} for index, name in enumerate(names)],
        "review_status": "confirmed-for-incremental-training",
    }
    (output_root / "class_registry_v2.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    membership = {
        "schema_version": "1.0",
        "artifact_type": "open_world_incremental_replay_dataset",
        "seed": seed,
        "class_count": len(names),
        "classes": names,
        "known_replay_images": len(replay),
        "known_validation_images": len(known_validation),
        "novel_train_images": novel_train_counts,
        "novel_validation_images": novel_validation_counts,
        "protected_holdout_images_used": 0,
        "train_images": len(train_destinations),
        "validation_images": len(val_destinations),
        "box_counts": box_counts,
        "dataset_yaml": str(yaml_path.resolve()),
    }
    (output_root / "membership.json").write_text(
        json.dumps(membership, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return membership


def build_protected_holdout_eval_dataset(
    *,
    protected_novel_truth: Path,
    class_registry: Path,
    output_root: Path,
) -> dict[str, object]:
    """Materialize the never-trained-on novel holdout as a YOLO evaluation view.

    The source annotations remain read-only. Images are hard-linked when possible
    and labels are rewritten only to map the source-local class 0 to the reviewed
    11-class registry IDs.
    """
    registry = json.loads(class_registry.read_text(encoding="utf-8"))
    classes = sorted(registry["classes"], key=lambda item: int(item["id"]))
    if [int(item["id"]) for item in classes] != list(range(len(classes))):
        raise ValueError("class registry IDs must be contiguous and zero-based")
    names = [str(item["name"]) for item in classes]
    class_ids = {str(item["name"]): int(item["id"]) for item in classes}
    truth = json.loads(protected_novel_truth.read_text(encoding="utf-8"))
    records = [record for record in truth["records"] if record["split"] == "holdout"]
    if not records:
        raise ValueError("protected truth contains no holdout records")
    unsupported = sorted({str(record["category"]) for record in records} - set(class_ids))
    if unsupported:
        raise ValueError(f"holdout categories missing from class registry: {unsupported}")

    destinations: list[Path] = []
    counts: dict[str, dict[str, int]] = {}
    for record in sorted(records, key=lambda item: str(item["image_id"])):
        category = str(record["category"])
        destination, box_count = _materialize(
            Path(record["image_path"]), output_root, "val", class_id=class_ids[category]
        )
        destinations.append(destination)
        item = counts.setdefault(category, {"images": 0, "boxes": 0})
        item["images"] += 1
        item["boxes"] += box_count

    output_root.mkdir(parents=True, exist_ok=True)
    val_path = output_root / "val.txt"
    val_path.write_text("".join(f"{path}\n" for path in destinations), encoding="utf-8")
    # Ultralytics requires a train key even for val-only invocations. Point it
    # to an intentionally empty file so this protected view cannot silently be
    # used as a viable training dataset.
    empty_train_path = output_root / "TRAINING_PROHIBITED_EMPTY.txt"
    empty_train_path.write_text("", encoding="utf-8")
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_root.as_posix()}\ntrain: {empty_train_path.as_posix()}\nval: {val_path.as_posix()}\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(names)),
        encoding="utf-8",
    )
    membership = {
        "schema_version": "1.0",
        "artifact_type": "protected_novel_holdout_evaluation_dataset",
        "training_use_permitted": False,
        "source_truth": str(protected_novel_truth.resolve(strict=True)),
        "class_registry": str(class_registry.resolve(strict=True)),
        "classes": names,
        "images": len(destinations),
        "boxes": sum(item["boxes"] for item in counts.values()),
        "per_category": counts,
        "dataset_yaml": str(yaml_path.resolve()),
    }
    (output_root / "membership.json").write_text(
        json.dumps(membership, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return membership
