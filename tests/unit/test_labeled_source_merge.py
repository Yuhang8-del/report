from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.data.labeled_source_merge import LabeledSourceInput, LabeledSourceMergeError, materialize_labeled_sources


def _jpeg(width: int = 10, height: int = 8) -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"


def _manifest(path: Path, source: str, image_id: str, file_path: str, class_id: int) -> Path:
    category = "Apple" if class_id == 0 else "Banana"
    if source == "snacks_detection":
        category = category.lower()
    payload = {"records": [{"source": source, "source_category": category, "source_image_id": image_id, "file_path": file_path, "width": 10, "height": 8, "class_id": class_id, "xyxy": [1.0, 1.0, 8.0, 6.0], "split": "train_pool", "label_status": "labeled", "license_metadata": {"name": "CC BY"}}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_materialize_labeled_sources_rewrites_paths_and_preserves_provenance(tmp_path: Path) -> None:
    oi_root, snacks_root = tmp_path / "oi", tmp_path / "snacks"
    (oi_root / "images").mkdir(parents=True)
    (snacks_root / "data" / "train" / "apple").mkdir(parents=True)
    (oi_root / "images" / "same.jpg").write_bytes(_jpeg())
    (snacks_root / "data" / "train" / "apple" / "same.jpg").write_bytes(_jpeg(12, 9))
    oi_manifest = _manifest(tmp_path / "oi.json", "open_images_v7", "same", "images/same.jpg", 0)
    snacks_manifest = _manifest(tmp_path / "snacks.json", "snacks_detection", "snacks-train-apple-same", "data/train/apple/same.jpg", 1)

    result = materialize_labeled_sources([LabeledSourceInput("open_images_v7", oi_manifest, oi_root), LabeledSourceInput("snacks_detection", snacks_manifest, snacks_root)], tmp_path / "union")

    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert result.image_count == 2 and result.record_count == 2
    assert {row["source"] for row in payload["records"]} == {"open_images_v7", "snacks_detection"}
    assert all(not Path(row["file_path"]).is_absolute() and (result.root / row["file_path"]).is_file() for row in payload["records"])
    assert {item["source"] for item in payload["sources"]} == {"open_images_v7", "snacks_detection"}


def test_materialize_labeled_sources_accepts_distinct_snapshots_of_same_source(tmp_path: Path) -> None:
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    (first_root / "images").mkdir(parents=True)
    (second_root / "images").mkdir(parents=True)
    (first_root / "images" / "first.jpg").write_bytes(_jpeg())
    (second_root / "images" / "second.jpg").write_bytes(_jpeg(12, 9))
    first_manifest = _manifest(tmp_path / "first.json", "open_images_v7", "first", "images/first.jpg", 0)
    second_manifest = _manifest(tmp_path / "second.json", "open_images_v7", "second", "images/second.jpg", 1)

    result = materialize_labeled_sources(
        [
            LabeledSourceInput("open_images_v7", first_manifest, first_root),
            LabeledSourceInput("open_images_v7", second_manifest, second_root),
        ],
        tmp_path / "union",
    )

    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert result.image_count == 2 and result.record_count == 2
    assert len(payload["sources"]) == 2
    assert {row["source_image_id"] for row in payload["records"]} == {"first", "second"}


def test_materialize_labeled_sources_accepts_a_prior_mixed_union(tmp_path: Path) -> None:
    existing_root, fresh_root = tmp_path / "existing", tmp_path / "fresh"
    (existing_root / "images").mkdir(parents=True)
    (fresh_root / "images").mkdir(parents=True)
    (existing_root / "images" / "apple.jpg").write_bytes(_jpeg())
    (existing_root / "images" / "banana.jpg").write_bytes(_jpeg())
    (fresh_root / "images" / "orange.jpg").write_bytes(_jpeg())
    mixed = {
        "records": [
            json.loads(_manifest(tmp_path / "apple.json", "open_images_v7", "apple", "images/apple.jpg", 0).read_text(encoding="utf-8"))["records"][0],
            json.loads(_manifest(tmp_path / "banana.json", "snacks_detection", "banana", "images/banana.jpg", 1).read_text(encoding="utf-8"))["records"][0],
        ]
    }
    mixed_manifest = tmp_path / "mixed.json"
    mixed_manifest.write_text(json.dumps(mixed), encoding="utf-8")
    fresh_manifest = _manifest(tmp_path / "fresh.json", "open_images_v7", "orange", "images/orange.jpg", 0)

    result = materialize_labeled_sources([LabeledSourceInput("*", mixed_manifest, existing_root), LabeledSourceInput("open_images_v7", fresh_manifest, fresh_root)], tmp_path / "union")

    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert result.image_count == 3 and result.record_count == 3
    assert {row["source"] for row in payload["records"]} == {"open_images_v7", "snacks_detection"}


def test_materialize_labeled_sources_refuses_existing_output_root(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "images").mkdir(parents=True)
    (root / "images" / "image.jpg").write_bytes(_jpeg())
    manifest = _manifest(tmp_path / "source.json", "open_images_v7", "image", "images/image.jpg", 0)
    output = tmp_path / "union"
    output.mkdir()

    with pytest.raises(LabeledSourceMergeError, match="already exists"):
        materialize_labeled_sources([LabeledSourceInput("open_images_v7", manifest, root), LabeledSourceInput("second_source", manifest, root)], output)
