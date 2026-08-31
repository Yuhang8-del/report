"""Deterministic object-centric training tiles with parent provenance."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError, _digest


@dataclass(frozen=True)
class ObjectCentricTileResult:
    root: Path
    dataset_yaml: Path
    membership: Path
    tile_count: int
    exposure_count: int


@dataclass(frozen=True)
class _Box:
    class_id: int
    left: float
    top: float
    right: float
    bottom: float

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)


def _problem(problem: str, cause: str, remediation: str) -> SupervisedDatasetError:
    return SupervisedDatasetError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _load_membership(root: Path) -> tuple[Path, Mapping[str, Any]]:
    path = root / "membership.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("snapshot membership cannot be read", str(error), "restore the sealed full-label snapshot") from error
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "sealed_full_label_supervised_upper_bound":
        raise _problem("input is not a sealed full-label upper bound", repr(getattr(payload, "get", lambda *_: None)("artifact_type")), "use the v12 full-label snapshot")
    return path, payload


def _members(payload: Mapping[str, Any], partition: str) -> list[Mapping[str, Any]]:
    groups = payload.get("members")
    rows = groups.get(partition) if isinstance(groups, Mapping) else None
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise _problem(f"{partition} membership is malformed", repr(rows), "restore the sealed full-label snapshot")
    return rows


def _path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _problem(f"{field} is missing", repr(value), "restore the sealed full-label snapshot")
    candidate = (root / value).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise _problem(f"{field} escapes the snapshot", value, "restore safe relative snapshot paths") from error
    return candidate


def _boxes(path: Path, width: int, height: int) -> list[_Box]:
    output: list[_Box] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise _problem("snapshot label cannot be read", str(error), "restore the sealed snapshot") from error
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise _problem("snapshot label is malformed", f"{path}:{line_number}", "regenerate the audited YOLO snapshot")
        try:
            class_id = int(parts[0])
            cx, cy, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError as error:
            raise _problem("snapshot label is not numeric", f"{path}:{line_number}", "regenerate the audited YOLO snapshot") from error
        if class_id not in range(len(DEFAULT_CLASS_REGISTRY.classes)) or not all(0.0 <= value <= 1.0 for value in (cx, cy, box_width, box_height)):
            raise _problem("snapshot label is outside the canonical contract", f"{path}:{line_number}", "regenerate the audited YOLO snapshot")
        left, right = (cx - box_width / 2.0) * width, (cx + box_width / 2.0) * width
        top, bottom = (cy - box_height / 2.0) * height, (cy + box_height / 2.0) * height
        left, top, right, bottom = max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)
        if right <= left or bottom <= top:
            raise _problem("snapshot label has a non-positive box", f"{path}:{line_number}", "repair the source annotation")
        output.append(_Box(class_id, left, top, right, bottom))
    if not output:
        raise _problem("snapshot training label is empty", str(path), "remove background-only records from supervised training")
    return output


def _crop_for(box: _Box, width: int, height: int, tile_size: int) -> tuple[int, int, int, int]:
    crop_width, crop_height = min(tile_size, width), min(tile_size, height)
    center_x, center_y = (box.left + box.right) / 2.0, (box.top + box.bottom) / 2.0
    left = min(max(int(round(center_x - crop_width / 2.0)), 0), width - crop_width)
    top = min(max(int(round(center_y - crop_height / 2.0)), 0), height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _clipped_rows(boxes: Sequence[_Box], crop: tuple[int, int, int, int], minimum_visibility: float) -> list[str]:
    left, top, right, bottom = crop
    crop_width, crop_height = right - left, bottom - top
    rows: list[str] = []
    for box in boxes:
        clip_left, clip_top = max(box.left, left), max(box.top, top)
        clip_right, clip_bottom = min(box.right, right), min(box.bottom, bottom)
        if clip_right <= clip_left or clip_bottom <= clip_top:
            continue
        visibility = ((clip_right - clip_left) * (clip_bottom - clip_top)) / box.area
        if visibility < minimum_visibility:
            continue
        cx = ((clip_left + clip_right) / 2.0 - left) / crop_width
        cy = ((clip_top + clip_bottom) / 2.0 - top) / crop_height
        box_width, box_height = (clip_right - clip_left) / crop_width, (clip_bottom - clip_top) / crop_height
        if box_width <= 0.0 or box_height <= 0.0:
            raise _problem("clipped tile box is non-positive", repr((box_width, box_height)), "review the tile boundary implementation")
        rows.append(f"{box.class_id} {cx:.10f} {cy:.10f} {box_width:.10f} {box_height:.10f}")
    return rows


def _base_lines(snapshot: Path, train_ids: set[str], source: Path | None) -> list[str]:
    path = (snapshot / "train.txt") if source is None else source.resolve(strict=True)
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as error:
        raise _problem("base training list cannot be read", str(error), "use the sealed train list or deterministic balanced view") from error
    if not lines:
        raise _problem("base training list is empty", str(path), "restore the sealed training view")
    # Paths can be repeated by the balanced view, but every path must still be
    # a training member of the sealed snapshot.
    for line in lines:
        image_id = Path(line).stem
        if image_id not in train_ids:
            raise _problem("base training list contains a non-training image", line, "use only the sealed v12 train membership")
    return lines


def materialize_object_centric_tiles(
    snapshot_root: Path,
    output_root: Path,
    *,
    base_training_list: Path | None = None,
    tile_size: int = 512,
    small_object_area: float = 0.01,
    minimum_visibility: float = 0.5,
    max_tiles_per_image: int = 3,
) -> ObjectCentricTileResult:
    """Add train-only crops centred on small objects and retain all full images."""
    if tile_size <= 0 or max_tiles_per_image <= 0:
        raise _problem("tile size and tile cap must be positive", repr((tile_size, max_tiles_per_image)), "use positive integer settings")
    if not 0.0 < small_object_area <= 1.0 or not 0.0 < minimum_visibility <= 1.0:
        raise _problem("tile thresholds are outside (0, 1]", repr((small_object_area, minimum_visibility)), "use normalized fractional thresholds")
    snapshot = snapshot_root.resolve(strict=True)
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem(f"tile view {root} already exists", "published tile views are immutable", "choose a fresh output root")
    membership_path, payload = _load_membership(snapshot)
    train, validation, test = (_members(payload, name) for name in ("train", "val", "test"))
    train_ids = {str(row.get("source_image_id")) for row in train}
    protected_ids = {str(row.get("source_image_id")) for row in validation + test}
    if "" in train_ids or len(train_ids) != len(train) or train_ids & protected_ids:
        raise _problem("snapshot split membership is invalid", "duplicate, empty or protected IDs occur in training", "restore the sealed v12 snapshot")
    base_lines = _base_lines(snapshot, train_ids, base_training_list)

    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        image_dir, label_dir = temporary / "images" / "train_tiles", temporary / "labels" / "train_tiles"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        tiles: list[dict[str, Any]] = []
        tile_paths: list[str] = []
        for row in sorted(train, key=lambda value: str(value["source_image_id"])):
            image_id = str(row["source_image_id"])
            image_path = _path(snapshot, row.get("snapshot_image"), "snapshot_image")
            label_path = _path(snapshot, row.get("snapshot_label"), "snapshot_label")
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
                width, height = image.size
                boxes = _boxes(label_path, width, height)
                if width <= tile_size and height <= tile_size:
                    continue
                small = sorted(
                    (box for box in boxes if box.area / (width * height) < small_object_area),
                    key=lambda box: (box.area, box.class_id, box.left, box.top),
                )
                crops: list[tuple[int, int, int, int]] = []
                for box in small:
                    crop = _crop_for(box, width, height, tile_size)
                    if crop not in crops:
                        crops.append(crop)
                    if len(crops) >= max_tiles_per_image:
                        break
                for index, crop in enumerate(crops):
                    label_rows = _clipped_rows(boxes, crop, minimum_visibility)
                    if not label_rows:
                        raise _problem("object-centric tile became empty", f"{image_id}:{crop}", "review visibility and crop geometry")
                    left, top, right, bottom = crop
                    tile_id = f"{image_id}__tile_{index:02d}_{left}_{top}_{right}_{bottom}"
                    image_out, label_out = image_dir / f"{tile_id}.jpg", label_dir / f"{tile_id}.txt"
                    image.crop(crop).save(image_out, format="JPEG", quality=95, subsampling=0)
                    label_out.write_text("\n".join(label_rows) + "\n", encoding="utf-8", newline="\n")
                    published_image = root / image_out.relative_to(temporary)
                    tile_paths.append(str(published_image.resolve()))
                    tiles.append(
                        {
                            "tile_id": tile_id,
                            "parent_image_id": image_id,
                            "parent_snapshot_image": str(row["snapshot_image"]),
                            "crop_xyxy": [left, top, right, bottom],
                            "source_size": [width, height],
                            "tile_image": image_out.relative_to(temporary).as_posix(),
                            "tile_label": label_out.relative_to(temporary).as_posix(),
                            "retained_box_count": len(label_rows),
                        }
                    )
        training_lines = base_lines + tile_paths
        (temporary / "train_with_tiles.txt").write_text("\n".join(training_lines) + "\n", encoding="utf-8", newline="\n")
        dataset = {
            "path": str(root),
            "train": str((root / "train_with_tiles.txt").resolve()),
            "val": str((snapshot / "val.txt").resolve()),
            "test": str((snapshot / "test.txt").resolve()),
            "names": list(DEFAULT_CLASS_REGISTRY.class_names),
        }
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        evidence = {
            "schema_version": "1.0",
            "artifact_type": "deterministic_object_centric_training_tiles",
            "algorithm": "small-object-centred-crop-v1",
            "source_snapshot": str(snapshot),
            "source_membership_sha256": _digest(membership_path),
            "base_training_list": str((snapshot / "train.txt") if base_training_list is None else base_training_list.resolve(strict=True)),
            "base_training_exposure_count": len(base_lines),
            "tile_size": tile_size,
            "small_object_area": small_object_area,
            "minimum_visibility": minimum_visibility,
            "max_tiles_per_image": max_tiles_per_image,
            "tile_count": len(tiles),
            "combined_training_exposure_count": len(training_lines),
            "protected_validation_count": len(validation),
            "protected_test_count": len(test),
            "tiles": tiles,
        }
        (temporary / "membership.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, root)
        temporary = None
        return ObjectCentricTileResult(root, root / "dataset.yaml", root / "membership.json", len(tiles), len(training_lines))
    except SupervisedDatasetError:
        raise
    except OSError as error:
        raise _problem("object-centric tile view cannot be written", str(error), "choose a fresh writable output root and verify source images") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
