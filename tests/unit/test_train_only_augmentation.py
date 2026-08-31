"""Tests for immutable training-only supervised data expansion."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from PIL import Image

from fruit_ssod.data.supervised_dataset import SupervisedDatasetError, _digest
from fruit_ssod.data.train_only_augmentation import audit_train_only_augmentation, materialize_train_only_augmentation


def _image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64))
    image.putdata(
        [
            (value, value, value)
            for y in range(64)
            for x in range(64)
            for value in (sha256(f"{color}:{x}:{y}".encode("utf-8")).digest()[0],)
        ]
    )
    image.save(path)


def _member(root: Path, partition: str, image_id: str, color: str) -> dict[str, str]:
    image = root / "images" / partition / f"{image_id}.jpg"
    label = root / "labels" / partition / f"{image_id}.txt"
    _image(image, color)
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return {
        "source_image_id": image_id,
        "snapshot_image": image.relative_to(root).as_posix(),
        "snapshot_label": label.relative_to(root).as_posix(),
        "image_sha256": _digest(image),
    }


def _snapshot(root: Path) -> None:
    members = {
        "train": [_member(root, "train", "base-train", "red")],
        "val": [_member(root, "val", "base-val", "green")],
        "test": [_member(root, "test", "base-test", "blue")],
    }
    for partition in members:
        lines = [str((root / row["snapshot_image"]).resolve()) for row in members[partition]]
        (root / f"{partition}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "membership.json").write_text(
        json.dumps({"artifact_type": "sealed_full_label_supervised_upper_bound", "members": members}),
        encoding="utf-8",
    )


def _candidate(source: Path, image_id: str, color: str) -> Path:
    _image(source / "images" / f"{image_id}.jpg", color)
    path = source / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "source": "open_images_v7",
                        "source_image_id": image_id,
                        "file_path": f"images/{image_id}.jpg",
                        "width": 20,
                        "height": 10,
                        "labels": [{"class_id": 1, "xyxy": [2, 1, 12, 6]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_train_only_augmentation_preserves_base_protected_lists_and_adds_only_train_images(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _snapshot(base)
    source = tmp_path / "addition-source"
    candidate = _candidate(source, "fresh", "yellow")
    validation_before = _digest(base / "val.txt")
    test_before = _digest(base / "test.txt")

    result = materialize_train_only_augmentation(base, candidate, source, tmp_path / "augmented")

    dataset = yaml.safe_load(result.dataset_yaml.read_text(encoding="utf-8"))
    evidence = json.loads(result.membership.read_text(encoding="utf-8"))
    assert dataset["val"] == str((base / "val.txt").resolve())
    assert dataset["test"] == str((base / "test.txt").resolve())
    assert _digest(base / "val.txt") == validation_before
    assert _digest(base / "test.txt") == test_before
    assert (result.root / "images" / "added_train" / "fresh.jpg").is_file()
    assert not (result.root / "images" / "val").exists()
    assert (result.root / "train_augmented.txt").read_text(encoding="utf-8").splitlines() == [
        str((base / "images" / "train" / "base-train.jpg").resolve()),
        str((result.root / "images" / "added_train" / "fresh.jpg").resolve()),
    ]
    assert evidence["base_train_exposure_count"] == 1
    assert evidence["added_train_image_count"] == 1
    assert evidence["preserved_partitions"]["validation"]["list_sha256"] == _digest(base / "val.txt")
    assert evidence["preserved_partitions"]["test"]["list_sha256"] == _digest(base / "test.txt")
    audit = audit_train_only_augmentation(result.root)
    assert audit["verified_added_train_image_count"] == 1
    assert audit["protected_validation_count"] == 1
    assert audit["protected_test_count"] == 1


def test_train_only_augmentation_rejects_addition_visually_matching_protected_validation(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _snapshot(base)
    source = tmp_path / "addition-source"
    candidate = _candidate(source, "fresh", "green")

    with pytest.raises(SupervisedDatasetError, match="no v13 additions"):
        materialize_train_only_augmentation(base, candidate, source, tmp_path / "augmented")


def test_train_only_augmentation_retains_balanced_base_exposures(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _snapshot(base)
    view = tmp_path / "balanced-view"
    view.mkdir()
    base_train = str((base / "images" / "train" / "base-train.jpg").resolve())
    (view / "train_balanced.txt").write_text(f"{base_train}\n{base_train}\n", encoding="utf-8")
    (view / "membership.json").write_text(
        json.dumps({"artifact_type": "deterministic_class_balanced_training_view", "source_snapshot": str(base)}),
        encoding="utf-8",
    )
    source = tmp_path / "addition-source"
    candidate = _candidate(source, "fresh", "yellow")

    result = materialize_train_only_augmentation(view, candidate, source, tmp_path / "augmented")

    assert result.base_train_exposure_count == 2
    assert (result.root / "train_augmented.txt").read_text(encoding="utf-8").splitlines().count(base_train) == 2
