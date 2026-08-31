from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fruit_ssod.data.schema import LicenseMetadata
from fruit_ssod.data.strawberry_ds import import_strawberry_ds


def _jpeg(width: int = 10, height: int = 8) -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"


def test_import_extracts_images_and_collapses_all_maturity_stages(tmp_path: Path) -> None:
    raw = tmp_path / "source.parquet"
    rows = [
        {"image": {"bytes": _jpeg(), "path": "first.jpg"}, "objects": {"bbox": [[1.0, 2.0, 3.0, 4.0]], "categories": [0]}, "split": "train"},
        {"image": {"bytes": _jpeg(), "path": "second.jpg"}, "objects": {"bbox": [[2.0, 1.0, 4.0, 3.0]], "categories": [5]}, "split": "valid"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw)
    result = import_strawberry_ds(raw, tmp_path / "imported", source_version="291e1cd9eec4ae452c13555ca74b2e36ccc6923c", source_page="https://huggingface.co/datasets/Project-AgML/Strawberry-DS_strawberry_detection", license_metadata=LicenseMetadata(name="CC BY 4.0"))
    assert result.manifest["image_count"] == 2
    assert result.manifest["record_count"] == 2
    assert {row["source_partition"] for row in result.manifest["records"]} == {"train", "valid"}
    assert {record.source_category for record in result.records} == {"Early-Turning", "White"}
    assert {record.class_id for record in result.records} == {3}
    assert (result.image_root / "images" / "strawberry-ds-000000.jpg").is_file()
