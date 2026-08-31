from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.data.kaggle_fruit_detection import KaggleFruitDetectionImportError, import_kaggle_fruit_detection
from fruit_ssod.data.schema import LicenseMetadata


def _source(tmp_path: Path, *, label: str = "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n") -> tuple[Path, Path]:
    image = tmp_path / "images" / "train" / "sample.jpg"; image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image)
    text = tmp_path / "labels" / "train" / "sample.txt"; text.parent.mkdir(parents=True); text.write_text(label, encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"; yaml_path.write_text("names: [Apple, Grapes]\n", encoding="utf-8")
    return tmp_path, yaml_path


def test_import_kaggle_yolo_preserves_approved_boxes_and_rejections(tmp_path: Path) -> None:
    root, data_yaml = _source(tmp_path)
    result = import_kaggle_fruit_detection(root, data_yaml, source_version="1", source_page="https://example.test/kaggle", license_metadata=LicenseMetadata("CC0"))
    assert len(result.records) == 1
    assert result.records[0].class_id == 0
    assert result.records[0].xyxy == (30.0, 24.0, 70.0, 56.0)
    assert result.manifest["rejection_count"] == 1
    assert result.manifest["records"][0]["file_path"] == "images/train/sample.jpg"


def test_import_kaggle_yolo_rejects_invalid_box(tmp_path: Path) -> None:
    root, data_yaml = _source(tmp_path, label="0 1.0 0.5 0.4 0.4\n")
    with pytest.raises(KaggleFruitDetectionImportError, match="out of bounds"):
        import_kaggle_fruit_detection(root, data_yaml, source_version="1", source_page="https://example.test/kaggle", license_metadata=LicenseMetadata("CC0"))
