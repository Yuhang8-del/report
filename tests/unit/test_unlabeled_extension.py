"""Tests for sealed auxiliary image-only pseudo-label pool extensions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.data.schema import LicenseMetadata, UnlabeledImageRecord
from fruit_ssod.data.unlabeled_extension import UnlabeledExtensionError, extend_unlabeled_pool
from fruit_ssod.pseudo.generator import canonical_unlabeled_fingerprint, load_unlabeled_manifest


def _row(record: UnlabeledImageRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "source_image_id": record.source_image_id,
        "file_path": record.file_path,
        "width": record.width,
        "height": record.height,
        "split": record.split,
        "label_status": record.label_status,
        "license_metadata": {"name": record.license_metadata.name, "url": record.license_metadata.url, "attribution": record.license_metadata.attribution},
    }


def _base_artifacts(root: Path) -> tuple[Path, Path]:
    record = UnlabeledImageRecord("open_images_v7", "base-image", "open_images_v7/images/base-image.jpg", 32, 24, "train_pool", "unlabeled", LicenseMetadata(name="CC"))
    unlabeled = root / "unlabeled.json"
    unlabeled.write_text(json.dumps({"records": [_row(record)]}), encoding="utf-8")
    split = root / "split_manifest.json"
    split.write_text(json.dumps({
        "manifest_version": "1.0",
        "split_image_ids": {"validation": ["val"], "test": ["test"], "pseudo_audit": ["audit"], "external_test": []},
        "budget_image_ids": {"20": ["human"]},
        "unlabeled_image_ids": ["base-image"],
        "fingerprints": {"split_protocol": "a" * 64, "unlabeled": canonical_unlabeled_fingerprint((record,)), "budget/20": "b" * 64, "protected/validation": "c" * 64},
    }), encoding="utf-8")
    return unlabeled, split


def _auxiliary_manifest(path: Path) -> Path:
    record = UnlabeledImageRecord("fruit_360", "Apple 6/r0.jpg", "Apple 6/r0.jpg", 100, 100, "train_pool", "unlabeled", LicenseMetadata(name="CC BY-SA 4.0"))
    path.write_text(json.dumps({"split": "train_pool", "label_status": "unlabeled", "records": [{**_row(record), "source_category": "Apple 6"}]}), encoding="utf-8")
    return path


def test_extension_preserves_primary_membership_and_seals_auxiliary_paths(tmp_path: Path) -> None:
    base_unlabeled, base_split = _base_artifacts(tmp_path)
    auxiliary = _auxiliary_manifest(tmp_path / "fruits360.json")

    result = extend_unlabeled_pool(base_unlabeled, base_split, auxiliary, tmp_path / "extension", raw_prefix="fruits_360/official/Training")

    membership = load_unlabeled_manifest(result.unlabeled_manifest, split_manifest_path=result.split_manifest)
    assert result.record_count == 2 and result.auxiliary_record_count == 1
    assert membership.records[0].source_image_id == "base-image"
    assert membership.records[1].source_image_id.startswith("aux-fruit_360-")
    assert membership.records[1].file_path == "fruits_360/official/Training/Apple 6/r0.jpg"
    split = json.loads(result.split_manifest.read_text(encoding="utf-8"))
    assert split["fingerprints"]["split_protocol"] == result.split_fingerprint
    assert split["unlabeled_extension"]["auxiliary_member_count"] == 1
    mapping = json.loads((result.root / "extension.json").read_text(encoding="utf-8"))["source_id_mapping"]
    assert mapping[0]["auxiliary_source_image_id"] == "Apple 6/r0.jpg"


def test_extension_refuses_unsafe_raw_prefix(tmp_path: Path) -> None:
    base_unlabeled, base_split = _base_artifacts(tmp_path)
    auxiliary = _auxiliary_manifest(tmp_path / "fruits360.json")

    with pytest.raises(UnlabeledExtensionError, match="path prefix is unsafe"):
        extend_unlabeled_pool(base_unlabeled, base_split, auxiliary, tmp_path / "extension", raw_prefix="../outside")
