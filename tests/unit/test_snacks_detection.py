from __future__ import annotations

import csv
from pathlib import Path

from fruit_ssod.data.schema import LicenseMetadata
from fruit_ssod.data.snacks_detection import import_snacks_detection


def _jpeg(width: int = 10, height: int = 8) -> bytes:
    """Return the smallest header accepted by the stdlib JPEG dimension reader."""
    return b"\xff\xd8\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["image_id", "x_min", "x_max", "y_min", "y_max", "class_name", "folder"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_import_retains_source_partition_but_requires_new_project_split(tmp_path: Path) -> None:
    images = tmp_path / "images"
    sources = {"train": "apple", "val": "banana", "test": "pineapple"}
    paths: dict[str, Path] = {}
    for partition, category in sources.items():
        image = images / "data" / partition / category / f"{partition}.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(_jpeg())
        csv_path = tmp_path / f"{partition}.csv"
        _csv(csv_path, [{"image_id": partition, "x_min": "0.1", "x_max": "0.8", "y_min": "0.2", "y_max": "0.9", "class_name": category, "folder": category}, {"image_id": partition, "x_min": "0.1", "x_max": "0.8", "y_min": "0.2", "y_max": "0.9", "class_name": "cake", "folder": category}])
        paths[partition] = csv_path

    result = import_snacks_detection(images, train_csv=paths["train"], val_csv=paths["val"], test_csv=paths["test"], source_version="main", source_page="https://huggingface.co/datasets/Matthijs/snacks-detection", license_metadata=LicenseMetadata(name="CC BY 4.0"))

    assert result.manifest["record_count"] == 3
    assert {row["source_partition"] for row in result.manifest["records"]} == {"train", "val", "test"}
    assert all(row["split"] == "train_pool" and row["label_status"] == "labeled" for row in result.manifest["records"])
    assert {record.class_id for record in result.records} == {0, 1, 4}


def test_import_quarantines_missing_images_without_discarding_valid_rows(tmp_path: Path) -> None:
    images = tmp_path / "images"
    image = images / "data" / "train" / "apple" / "present.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(_jpeg())
    csv_path = tmp_path / "train.csv"
    _csv(csv_path, [
        {"image_id": "present", "x_min": "0.1", "x_max": "0.8", "y_min": "0.2", "y_max": "0.9", "class_name": "apple", "folder": "apple"},
        {"image_id": "missing", "x_min": "0.1", "x_max": "0.8", "y_min": "0.2", "y_max": "0.9", "class_name": "apple", "folder": "apple"},
    ])
    empty = tmp_path / "empty.csv"
    _csv(empty, [])

    result = import_snacks_detection(images, train_csv=csv_path, val_csv=empty, test_csv=empty, source_version="main", source_page="https://huggingface.co/datasets/Matthijs/snacks-detection", license_metadata=LicenseMetadata(name="CC BY 4.0"))

    assert len(result.records) == 1
    assert len(result.rejections) == 1
    assert "image is unavailable" in result.rejections[0]["reason"]
