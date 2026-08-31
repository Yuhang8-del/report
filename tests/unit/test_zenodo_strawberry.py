from pathlib import Path

from PIL import Image

from fruit_ssod.data.schema import LicenseMetadata
from fruit_ssod.data.zenodo_strawberry import import_zenodo_strawberry


def _source(tmp_path: Path) -> tuple[Path, Path]:
    for partition in ("training", "validation"):
        image = tmp_path / partition / f"{partition}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), "white").save(image)
        image.with_suffix(".txt").write_text("0 0.5 0.5 0.4 0.4\n2 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    yaml_path = tmp_path / "strawberries.yaml"
    yaml_path.write_text("names: [ripe, unripe, peduncle]\n", encoding="utf-8")
    return tmp_path, yaml_path


def test_import_zenodo_strawberry_maps_fruit_and_records_peduncles(tmp_path: Path) -> None:
    root, data_yaml = _source(tmp_path)

    result = import_zenodo_strawberry(root, data_yaml, source_version="6126677", source_page="https://zenodo.org/records/6126677", license_metadata=LicenseMetadata("CC BY 4.0"))

    assert len(result.records) == 2
    assert {record.class_id for record in result.records} == {3}
    assert {record.split for record in result.records} == {"train_pool"}
    assert result.manifest["source_partitions_discarded"] == ["training", "validation"]
    assert result.manifest["rejection_count"] == 2
    assert all(row["source_category"] == "peduncle" for row in result.rejections)


def test_import_zenodo_strawberry_clips_border_box_and_rejects_empty_box(tmp_path: Path) -> None:
    root, data_yaml = _source(tmp_path)
    for partition in ("training", "validation"):
        label = root / partition / f"{partition}.txt"
        label.write_text("0 0.02 0.5 0.2 0.4\n1 0.5 0.5 0.0 0.2\n", encoding="utf-8")

    result = import_zenodo_strawberry(root, data_yaml, source_version="6126677", source_page="https://zenodo.org/records/6126677", license_metadata=LicenseMetadata("CC BY 4.0"))

    assert len(result.records) == 2
    assert result.manifest["geometry_conversion"]["clipped_box_count"] == 2
    assert result.manifest["rejection_count"] == 2
    assert all(row["reason"].startswith("YOLO box is empty") for row in result.rejections)
    assert all(record.xyxy[0] == 0.0 for record in result.records)
