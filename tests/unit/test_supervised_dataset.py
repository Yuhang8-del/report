"""Tests for sealed supervised YOLO snapshot materialization."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from fruit_ssod.data.supervised_dataset import materialize_supervised_dataset


def _record(image_id: str, class_id: int) -> dict[str, object]:
    return {"source": "open_images_v7", "source_image_id": image_id, "file_path": f"images/{image_id}.jpg", "width": 20, "height": 10, "labels": [{"class_id": class_id, "xyxy": [2, 1, 12, 6]}], "license_metadata": {"name": "CC"}}


def test_materialize_supervised_dataset_copies_only_sealed_members(tmp_path: Path) -> None:
    source = tmp_path / "source"; (source / "images").mkdir(parents=True)
    for image_id in ("train", "val", "test"):
        Image.new("RGB", (20, 10), "green").save(source / "images" / f"{image_id}.jpg")
    split = tmp_path / "splits"; (split / "budgets" / "20").mkdir(parents=True); (split / "protected_splits").mkdir()
    (split / "budgets" / "20" / "labels.json").write_text(json.dumps({"records": [_record("train", 0)]}), encoding="utf-8")
    (split / "protected_splits" / "validation_labels.json").write_text(json.dumps({"records": [_record("val", 1)]}), encoding="utf-8")
    (split / "protected_splits" / "test_labels.json").write_text(json.dumps({"records": [_record("test", 2)]}), encoding="utf-8")
    (split / "split_manifest.json").write_text("{}", encoding="utf-8")

    result = materialize_supervised_dataset(split, source, tmp_path / "snapshot", budget=20)

    dataset = yaml.safe_load(result.dataset_yaml.read_text(encoding="utf-8"))
    assert result.image_count == 3
    assert dataset["names"] == ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]
    assert (result.root / "labels" / "train" / "train.txt").read_text(encoding="utf-8") == "0 0.350000 0.350000 0.500000 0.500000\n"
    assert (result.root / "train.txt").read_text(encoding="utf-8").strip() == str((result.root / "images" / "train" / "train.jpg").resolve())
    assert not (result.root / "images" / "train" / "val.jpg").exists()
