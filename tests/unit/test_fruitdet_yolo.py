"""Tests for external-only FruitDet YOLO materialization."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fruit_ssod.data.fruitdet_yolo import materialize_fruitdet_yolo
from fruit_ssod.data.schema import LicenseMetadata


def _write_pair(root: Path, category: str, name: str) -> None:
    image = root / "data" / category / "images" / "test" / f"{name}.jpg"
    label = root / "data" / category / "labels" / "test" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), "red").save(image)
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_materialize_fruitdet_yolo_keeps_external_records_and_published_paths(tmp_path: Path) -> None:
    _write_pair(tmp_path / "source", "apple", "apple_0020")
    _write_pair(tmp_path / "source", "banana", "banana_0020")

    result = materialize_fruitdet_yolo(tmp_path / "source", tmp_path / "external", source_version="fixture", source_page="https://example.invalid/fruitdet", license_metadata=LicenseMetadata(name="CC BY 4.0"))

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert (result.image_count, result.annotation_count) == (2, 2)
    assert {row["class_id"] for row in manifest["records"]} == {0, 1}
    assert all(Path(row["file_path"]).is_file() for row in manifest["records"])
    assert "Orange" in {row["source_category"] for row in manifest["rejections"]}
    assert all(Path(line).is_file() for line in (result.root / "test.txt").read_text(encoding="utf-8").splitlines())
