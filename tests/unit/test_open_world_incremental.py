from __future__ import annotations

import json
from pathlib import Path

from fruit_ssod.open_world.incremental import (
    build_incremental_replay_dataset,
    build_protected_holdout_eval_dataset,
)


def _image(root: Path, category: str, split: str, name: str, class_id: int = 0) -> Path:
    image = root / category / split / "images" / f"{name}.jpg"
    label = root / category / split / "labels" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(name.encode())
    label.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return image


def test_incremental_dataset_appends_class_and_never_uses_holdout(tmp_path: Path) -> None:
    known_train = _image(tmp_path, "known", "train", "apple", 0)
    known_val = _image(tmp_path, "known", "val", "banana", 1)
    novel_train_a = _image(tmp_path, "mango", "train", "mango-a")
    novel_train_b = _image(tmp_path, "mango", "train", "mango-b")
    novel_holdout = _image(tmp_path, "mango", "test", "mango-hidden")
    train_list = tmp_path / "known_train.txt"
    val_list = tmp_path / "known_val.txt"
    train_list.write_text(str(known_train) + "\n", encoding="utf-8")
    val_list.write_text(str(known_val) + "\n", encoding="utf-8")
    protected = tmp_path / "protected.json"
    protected.write_text(
        json.dumps(
            {
                "records": [
                    {"image_id": "a", "image_path": str(novel_train_a), "category": "Mango", "split": "discovery"},
                    {"image_id": "b", "image_path": str(novel_train_b), "category": "Mango", "split": "discovery"},
                    {"image_id": "h", "image_path": str(novel_holdout), "category": "Mango", "split": "holdout"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "incremental"
    result = build_incremental_replay_dataset(
        known_train_list=train_list,
        known_validation_list=val_list,
        protected_novel_truth=protected,
        confirmed_categories=["Mango"],
        output_root=output,
        replay_images=1,
        seed=42,
    )
    assert result["classes"][-1] == "Mango"
    assert result["protected_holdout_images_used"] == 0
    all_members = (output / "train.txt").read_text() + (output / "val.txt").read_text()
    assert "mango-hidden" not in all_members
    novel_labels = list((output / "labels").rglob("*.txt"))
    assert any(path.read_text().startswith("5 ") for path in novel_labels)


def test_protected_holdout_eval_contains_only_holdout_and_maps_registry_id(tmp_path: Path) -> None:
    discovery = _image(tmp_path, "mango", "train", "mango-discovery")
    holdout = _image(tmp_path, "mango", "test", "mango-holdout")
    protected = tmp_path / "protected.json"
    protected.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_id": "d",
                        "image_path": str(discovery),
                        "category": "Mango",
                        "split": "discovery",
                    },
                    {
                        "image_id": "h",
                        "image_path": str(holdout),
                        "category": "Mango",
                        "split": "holdout",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "classes": [
                    {"id": 0, "name": "Apple"},
                    {"id": 1, "name": "Banana"},
                    {"id": 2, "name": "Orange"},
                    {"id": 3, "name": "Strawberry"},
                    {"id": 4, "name": "Pineapple"},
                    {"id": 5, "name": "Mango"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "protected-eval"
    result = build_protected_holdout_eval_dataset(
        protected_novel_truth=protected,
        class_registry=registry,
        output_root=output,
    )
    assert result["training_use_permitted"] is False
    assert result["images"] == 1
    assert result["boxes"] == 1
    members = (output / "val.txt").read_text(encoding="utf-8")
    assert "mango-discovery" not in members
    assert len(list((output / "labels" / "val").glob("*.txt"))) == 1
    assert next((output / "labels" / "val").glob("*.txt")).read_text().startswith("5 ")
