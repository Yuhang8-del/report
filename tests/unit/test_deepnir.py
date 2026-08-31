from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.data.deepnir import DeepNIRImportError, import_deepnir
from fruit_ssod.data.schema import LicenseMetadata


def _source(tmp_path: Path, *, label: str = "0 0.5 0.5 0.4 0.4\n") -> Path:
    for category in ("apple", "orange", "strawberry"):
        image = tmp_path / "yolov5" / category / "train" / "images" / "sample.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (100, 80), "white").save(image)
        annotation = image.parent.parent / "labels" / "sample.txt"
        annotation.parent.mkdir(parents=True)
        annotation.write_text(label, encoding="utf-8")
    return tmp_path


def test_import_deepnir_uses_reviewed_directory_as_category(tmp_path: Path) -> None:
    root = _source(tmp_path)

    result = import_deepnir(
        root,
        source_version="v1",
        source_page="https://example.test/deepnir",
        license_metadata=LicenseMetadata("CC BY 4.0"),
    )

    assert len(result.records) == 3
    assert result.records[0].source_category == "apple"
    assert result.records[0].class_id == 0
    assert result.records[0].xyxy == pytest.approx((30.0, 24.0, 70.0, 56.0))
    assert result.manifest["source_category_policy"] == "reviewed directory name"


def test_import_deepnir_rejects_nonzero_source_class(tmp_path: Path) -> None:
    root = _source(tmp_path, label="1 0.5 0.5 0.4 0.4\n")

    with pytest.raises(DeepNIRImportError, match="unsupported class"):
        import_deepnir(
            root,
            source_version="v1",
            source_page="https://example.test/deepnir",
            license_metadata=LicenseMetadata("CC BY 4.0"),
        )
