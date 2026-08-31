from __future__ import annotations

from pathlib import Path

from fruit_ssod.open_world.box_protocol import build_known_objectness_dataset, read_yolo_boxes


def _fixture_image(root: Path, split: str, name: str, class_id: int) -> Path:
    image = root / "images" / split / f"{name}.jpg"
    label = root / "labels" / split / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    label.write_text(f"{class_id} 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    return image


def test_build_known_objectness_dataset_collapses_classes_and_keeps_repeats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    first = _fixture_image(source, "train", "first", 4)
    second = _fixture_image(source, "val", "second", 2)
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    train_list.write_text(f"{first}\n{first}\n", encoding="utf-8")
    val_list.write_text(f"{second}\n", encoding="utf-8")
    output = tmp_path / "objectness"
    result = build_known_objectness_dataset(train_list, val_list, output)
    assert result["train_occurrences"] == 2
    assert result["train_unique_images"] == 1
    destination = Path((output / "train.txt").read_text(encoding="utf-8").splitlines()[0])
    boxes = read_yolo_boxes(output / "labels" / "train" / f"{destination.stem}.txt")
    assert boxes[0].class_id == 0
