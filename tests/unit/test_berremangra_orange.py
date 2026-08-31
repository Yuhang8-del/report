from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.data.berremangra_orange import BerremangraOrangeImportError, import_berremangra_orange
from fruit_ssod.data.schema import LicenseMetadata


def _source(tmp_path: Path, label: str) -> Path:
    image = tmp_path / "train" / "images" / "orange.jpg"; image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "orange").save(image)
    annotation = tmp_path / "train" / "labels" / "orange.txt"; annotation.parent.mkdir(parents=True); annotation.write_text(label, encoding="utf-8")
    return tmp_path


def test_import_berremangra_converts_polygon_to_box(tmp_path: Path) -> None:
    root = _source(tmp_path, "0 0.2 0.3 0.8 0.3 0.7 0.9\n")
    result = import_berremangra_orange(root, source_version="v1", source_page="https://example.test/orange", license_metadata=LicenseMetadata("CC BY 4.0"))
    assert result.records[0].class_id == 2
    assert result.records[0].xyxy == pytest.approx((20.0, 24.0, 80.0, 72.0))
    assert result.manifest["conversion"] == "YOLO polygon -> enclosing XYXY rectangle"


def test_import_berremangra_rejects_out_of_range_polygon(tmp_path: Path) -> None:
    root = _source(tmp_path, "0 0.2 0.3 1.2 0.3 0.7 0.9\n")
    with pytest.raises(BerremangraOrangeImportError, match="out-of-range point"):
        import_berremangra_orange(root, source_version="v1", source_page="https://example.test/orange", license_metadata=LicenseMetadata("CC BY 4.0"))
