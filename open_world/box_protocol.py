"""Deterministic box-level protocol for the fruit open-world experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from fruit_ssod.open_world.discovery import NovelImage, discover_images


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = next(index for index, part in enumerate(parts) if part.casefold() == "images")
    except StopIteration as error:
        raise ValueError(f"image path has no images directory: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_boxes(path: Path) -> tuple[YoloBox, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"YOLO label file does not exist: {path}")
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label row at {path}:{line_number}")
        class_value, *coordinates = fields
        class_id = int(class_value)
        values = [float(value) for value in coordinates]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"normalized box is outside [0, 1] at {path}:{line_number}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"box has non-positive area at {path}:{line_number}")
        boxes.append(YoloBox(class_id, *values))
    return tuple(boxes)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _record_payload(record: NovelImage, *, include_truth: bool) -> dict[str, object]:
    label_path = label_path_for_image(record.path)
    boxes = read_yolo_boxes(label_path)
    payload: dict[str, object] = {
        "image_id": record.image_id,
        "image_path": str(record.path.resolve()),
        "image_sha256": sha256_file(record.path),
        "split": record.split,
        "box_count": len(boxes),
    }
    if include_truth:
        payload.update(
            {
                "category": record.category,
                "label_path": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
                "boxes": [asdict(box) for box in boxes],
            }
        )
    return payload


def build_novel_box_protocol(
    source_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    holdout_fraction: float = 0.2,
) -> dict[str, object]:
    """Seal public image membership separately from protected box/category truth."""
    records = discover_images(source_root, seed=seed, holdout_fraction=holdout_fraction)
    public_records = [_record_payload(record, include_truth=False) for record in records]
    protected_records = [_record_payload(record, include_truth=True) for record in records]
    output_root.mkdir(parents=True, exist_ok=True)
    public_path = output_root / "novel_public_manifest.json"
    protected_path = output_root / "protected_novel_box_truth.json"
    _write_json(
        public_path,
        {
            "schema_version": "1.0",
            "artifact_type": "open_world_novel_public_membership",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "records": public_records,
        },
    )
    _write_json(
        protected_path,
        {
            "schema_version": "1.0",
            "artifact_type": "protected_open_world_box_truth",
            "purpose": "evaluation and post-discovery naming only; never initial training",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "records": protected_records,
        },
    )
    discovery = [item for item in public_records if item["split"] == "discovery"]
    holdout = [item for item in public_records if item["split"] == "holdout"]
    (output_root / "novel_discovery_images.txt").write_text(
        "".join(f"{item['image_path']}\n" for item in discovery), encoding="utf-8"
    )
    (output_root / "novel_holdout_images.txt").write_text(
        "".join(f"{item['image_path']}\n" for item in holdout), encoding="utf-8"
    )
    summary = {
        "schema_version": "1.0",
        "artifact_type": "open_world_box_protocol_summary",
        "source_root": str(source_root.resolve()),
        "seed": seed,
        "discovery_images": len(discovery),
        "holdout_images": len(holdout),
        "total_images": len(records),
        "public_manifest": str(public_path.resolve()),
        "public_manifest_sha256": sha256_file(public_path),
        "protected_truth": str(protected_path.resolve()),
        "protected_truth_sha256": sha256_file(protected_path),
    }
    _write_json(output_root / "protocol_summary.json", summary)
    return summary


def _read_image_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"image list does not exist: {path}")
    return [Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _collapsed_label_text(boxes: Sequence[YoloBox]) -> str:
    return "".join(
        f"0 {box.x_center:.10f} {box.y_center:.10f} {box.width:.10f} {box.height:.10f}\n" for box in boxes
    )


def _snapshot_split(
    image_paths: Iterable[Path],
    output_root: Path,
    split: str,
) -> tuple[list[Path], dict[str, int]]:
    destinations: list[Path] = []
    counts = {"hardlink": 0, "copy": 0, "existing": 0}
    path_to_destination: dict[Path, Path] = {}
    for source in image_paths:
        if source in path_to_destination:
            destinations.append(path_to_destination[source])
            continue
        if not source.is_file():
            raise FileNotFoundError(f"known training image does not exist: {source}")
        source_label = label_path_for_image(source)
        boxes = read_yolo_boxes(source_label)
        identity = hashlib.sha256(str(source).casefold().encode("utf-8")).hexdigest()[:16]
        destination = output_root / "images" / split / f"{identity}{source.suffix.casefold()}"
        mode = _link_or_copy(source, destination)
        counts[mode] += 1
        destination_label = output_root / "labels" / split / f"{identity}.txt"
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        collapsed = _collapsed_label_text(boxes)
        if not destination_label.exists() or destination_label.read_text(encoding="utf-8") != collapsed:
            destination_label.write_text(collapsed, encoding="utf-8")
        path_to_destination[source] = destination.resolve()
        destinations.append(destination.resolve())
    return destinations, counts


def build_known_objectness_dataset(
    train_list: Path,
    validation_list: Path,
    output_root: Path,
) -> dict[str, object]:
    """Collapse all five known fruit IDs into one Fruit objectness class."""
    train_sources = _read_image_list(train_list)
    validation_sources = _read_image_list(validation_list)
    protected_overlap = set(train_sources) & set(validation_sources)
    if protected_overlap:
        raise ValueError(f"known train/validation overlap contains {len(protected_overlap)} images")
    output_root.mkdir(parents=True, exist_ok=True)
    train_destinations, train_modes = _snapshot_split(train_sources, output_root, "train")
    validation_destinations, validation_modes = _snapshot_split(validation_sources, output_root, "val")
    train_output = output_root / "train.txt"
    val_output = output_root / "val.txt"
    train_output.write_text("".join(f"{path}\n" for path in train_destinations), encoding="utf-8")
    val_output.write_text("".join(f"{path}\n" for path in validation_destinations), encoding="utf-8")
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output_root.as_posix()}\ntrain: {train_output.as_posix()}\nval: {val_output.as_posix()}\nnames:\n  0: Fruit\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "known_only_class_agnostic_objectness_dataset",
        "rule": "all canonical known fruit class IDs are collapsed to class 0; no novel fruit enters training",
        "train_occurrences": len(train_destinations),
        "train_unique_images": len(set(train_destinations)),
        "validation_images": len(validation_destinations),
        "protected_overlap_count": 0,
        "train_materialization": train_modes,
        "validation_materialization": validation_modes,
        "source_train_list": str(train_list.resolve()),
        "source_validation_list": str(validation_list.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "train_list_sha256": sha256_file(train_output),
        "validation_list_sha256": sha256_file(val_output),
    }
    _write_json(output_root / "membership.json", payload)
    return payload
