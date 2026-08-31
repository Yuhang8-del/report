from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.data.hf_crop_plant import import_hf_crop_plant
from fruit_ssod.data.schema import LicenseMetadata


def test_import_hf_crop_plant_keeps_only_reviewed_five_class_aliases(tmp_path: Path) -> None:
    root = tmp_path / "leaflogic"
    image = root / "train" / "images" / "sample.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image)
    label = root / "train" / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("names: ['apple fruit', grapes]\n", encoding="utf-8")

    result = import_hf_crop_plant(root, data_yaml, source_version="v5", source_page="https://huggingface.co/datasets/devshaheen/100_crops_plants_object_detection_25k_image_dataset", license_metadata=LicenseMetadata("CC BY 4.0"))

    assert result.manifest["source"]["name"] == "hf_crop_plant_25k"
    assert len(result.records) == 1
    assert result.records[0].source == "hf_crop_plant_25k"
    assert result.records[0].class_id == 0
    assert result.manifest["rejection_count"] == 1


def test_import_hf_crop_plant_converts_a_known_polygon_to_an_enclosing_box(tmp_path: Path) -> None:
    root = tmp_path / "leaflogic"
    image = root / "valid" / "images" / "sample.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image)
    label = root / "valid" / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.1 0.2 0.7 0.2 0.7 0.8 0.1 0.8\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("names: ['apple fruit']\n", encoding="utf-8")

    result = import_hf_crop_plant(root, data_yaml, source_version="v5", source_page="https://example.test/hf", license_metadata=LicenseMetadata("CC BY 4.0"))

    assert result.records[0].xyxy == pytest.approx((10.0, 16.0, 70.0, 64.0))
    assert result.manifest["geometry_conversion"] == {"source_box_count": 0, "polygon_to_enclosing_box_count": 1}
