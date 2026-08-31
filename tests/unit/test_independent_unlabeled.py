"""Tests for independent image-only pseudo-label pool sealing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.data.independent_unlabeled import IndependentUnlabeledError, seal_independent_unlabeled_pool
from fruit_ssod.pseudo.generator import load_unlabeled_manifest


def _image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=color).save(path)


def _base_split(path: Path) -> Path:
    payload = {
        "split_image_ids": {"validation": ["val"], "test": ["test"], "pseudo_audit": ["audit"], "external_test": []},
        "budget_image_ids": {"20": ["human"]},
        "fingerprints": {"split_protocol": "a" * 64, "budget/20": "b" * 64},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_seal_independent_pool_omits_labels_and_is_loadable(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    images = source_root / "aux" / "images"
    _image(images / "fresh.jpg", "red")
    teacher = tmp_path / "teacher"; _image(teacher / "images" / "train" / "seen.jpg", "blue")
    result = seal_independent_unlabeled_pool(base_split_manifest=_base_split(tmp_path / "base.json"), image_directory=images, source_root=source_root, relative_prefix="aux/images", teacher_dataset_root=teacher, output_root=tmp_path / "out")
    membership = load_unlabeled_manifest(result.unlabeled_manifest, split_manifest_path=result.split_manifest)
    assert len(membership.records) == 1
    assert membership.records[0].file_path == "aux/images/fresh.jpg"
    payload = result.unlabeled_manifest.read_text(encoding="utf-8")
    assert "labels" not in payload and "xyxy" not in payload
    evidence = json.loads(result.evidence.read_text(encoding="utf-8"))
    assert evidence["teacher_snapshot_image_count"] == 1


def test_seal_independent_pool_rejects_teacher_image_overlap(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    images = source_root / "aux" / "images"
    _image(images / "same.jpg", "red")
    teacher = tmp_path / "teacher"; _image(teacher / "images" / "train" / "same.jpg", "red")
    with pytest.raises(IndependentUnlabeledError, match="overlaps Teacher"):
        seal_independent_unlabeled_pool(base_split_manifest=_base_split(tmp_path / "base.json"), image_directory=images, source_root=source_root, relative_prefix="aux/images", teacher_dataset_root=teacher, output_root=tmp_path / "out")
