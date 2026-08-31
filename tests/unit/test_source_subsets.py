from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.diagnostics.source_subsets import (
    SourceSubsetDiagnosticError,
    _sealed_membership_snapshot,
    _serialize_subset_metrics,
    source_members,
    source_test_members,
    write_source_dataset,
)


def _member(source: str, image: str) -> dict[str, str]:
    return {
        "source_file_path": f"images/{source}/raw-{image}.jpg",
        "snapshot_image": f"images/test/{image}.jpg",
        "snapshot_label": f"labels/test/{image}.txt",
    }


def test_source_test_members_groups_only_existing_sealed_members(tmp_path: Path) -> None:
    for image in ("oi", "snacks", "strawberry"):
        target = tmp_path / "images" / "test" / f"{image}.jpg"; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"image")
        label = tmp_path / "labels" / "test" / f"{image}.txt"; label.parent.mkdir(parents=True, exist_ok=True); label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    membership = {"members": {"test": [
        _member("open_images_v7", "oi"),
        _member("snacks_detection", "snacks"),
        _member("strawberry_ds", "strawberry"),
    ]}}
    grouped = source_test_members(membership, tmp_path)
    assert {name: [path.name for path in paths] for name, paths in grouped.items()} == {
        "open_images_v7": ["oi.jpg"],
        "snacks_detection": ["snacks.jpg"],
        "strawberry_ds": ["strawberry.jpg"],
    }


def test_source_test_members_rejects_unknown_or_missing_snapshot_member(tmp_path: Path) -> None:
    unknown = {"members": {"test": [_member("../other", "x")]}}
    with pytest.raises(SourceSubsetDiagnosticError, match="invalid public source"):
        source_test_members(unknown, tmp_path)
    missing = {"members": {"test": [_member("open_images_v7", "x"), _member("snacks_detection", "y")]}}
    with pytest.raises(SourceSubsetDiagnosticError, match="member is missing"):
        source_test_members(missing, tmp_path)


def test_source_members_supports_validation_without_falling_back_to_test(tmp_path: Path) -> None:
    image = tmp_path / "images" / "val" / "x.jpg"; image.parent.mkdir(parents=True); image.write_bytes(b"image")
    label = tmp_path / "labels" / "val" / "x.txt"; label.parent.mkdir(parents=True); label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    membership = {"members": {"test": [], "val": [{
        "source_file_path": "images/open_images_v7/raw-x.jpg", "snapshot_image": "images/val/x.jpg", "snapshot_label": "labels/val/x.txt",
    }]}}
    grouped = source_members(membership, tmp_path, split="val")
    assert {name: [path.name for path in paths] for name, paths in grouped.items()} == {"open_images_v7": ["x.jpg"]}
    with pytest.raises(SourceSubsetDiagnosticError, match="unsupported"):
        source_members(membership, tmp_path, split="train")


def test_balanced_view_resolves_the_referenced_full_snapshot_membership(tmp_path: Path) -> None:
    full = tmp_path / "full"; full.mkdir()
    sealed = {"members": {"test": [], "train": [], "val": []}}
    (full / "membership.json").write_text(json.dumps(sealed), encoding="utf-8")
    balanced = tmp_path / "balanced"; balanced.mkdir()
    (balanced / "membership.json").write_text(json.dumps({"source_snapshot": str(full)}), encoding="utf-8")
    root, membership, membership_path = _sealed_membership_snapshot(balanced)
    assert root == full.resolve()
    assert membership == sealed
    assert membership_path == full / "membership.json"


def test_write_source_dataset_is_fresh_and_canonical(tmp_path: Path) -> None:
    image = tmp_path / "images" / "test" / "x.jpg"; image.parent.mkdir(parents=True); image.write_bytes(b"image")
    label = tmp_path / "labels" / "test" / "x.txt"; label.parent.mkdir(parents=True); label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    list_path, yaml_path = write_source_dataset(snapshot_root=tmp_path, images=[image.resolve()], output=tmp_path / "diagnostic")
    materialized = tmp_path / "diagnostic" / "images" / "test" / "x.jpg"
    assert list_path.read_text(encoding="utf-8").strip() == str(materialized.resolve())
    assert (tmp_path / "diagnostic" / "labels" / "test" / "x.txt").read_text(encoding="utf-8") == label.read_text(encoding="utf-8")
    assert "Pineapple" in yaml_path.read_text(encoding="utf-8")
    with pytest.raises(SourceSubsetDiagnosticError, match="already exists"):
        write_source_dataset(snapshot_root=tmp_path, images=[image.resolve()], output=tmp_path / "diagnostic")


def test_subset_metrics_preserve_only_observed_canonical_classes() -> None:
    class Box:
        all_ap = [[0.8, 0.5], [0.6, 0.3]]
        ap_class_index = [1, 3]
        mp = 0.7
        mr = 0.5
        map50 = 0.65
        map = 0.4

    class Metrics:
        box = Box()

    result = _serialize_subset_metrics(Metrics())
    assert result["reported_class_ids"] == [1, 3]
    assert result["per_class_ap50"] == {"1": 0.8, "3": 0.6}
    assert result["f1"] == pytest.approx(0.5833333333333334)
