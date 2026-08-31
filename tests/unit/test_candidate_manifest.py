"""Tests for cleaned-annotation aggregation into split candidates."""

from __future__ import annotations

import json
from pathlib import Path

from fruit_ssod.data.candidate_manifest import build_candidate_manifest


def _row(image_id: str, class_id: int) -> dict[str, object]:
    return {"source": "open_images_v7", "source_image_id": image_id, "file_path": f"images/{image_id}.jpg", "width": 10, "height": 8, "class_id": class_id, "xyxy": [1, 1, 4, 4], "split": "train_pool", "label_status": "labeled", "license_metadata": {"name": "CC"}}


def test_candidate_manifest_groups_multilabel_rows_and_duplicate_components(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.json"
    cleaned.write_text(json.dumps({"records": [_row("one", 0), _row("one", 2), _row("two", 1), _row("three", 3)], "deduplication": {"exact_groups": [{"group_id": "x", "member_image_ids": ["one", "two"]}], "near_groups": [{"group_id": "y", "member_image_ids": ["two", "three"]}]}}, indent=2), encoding="utf-8")

    output = tmp_path / "candidates.json"
    assert build_candidate_manifest(cleaned, output) == 3

    images = json.loads(output.read_text(encoding="utf-8"))["images"]
    grouped = {row["source_image_id"]: row for row in images}
    assert grouped["one"]["class_presence"] == [0, 2]
    assert len(grouped["one"]["labels"]) == 2
    assert {grouped[item]["duplicate_group_id"] for item in ("one", "two", "three")} == {"duplicate:one"}


def test_candidate_manifest_collapses_same_source_image_from_two_snapshots(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.json"
    first = _row("same", 0)
    second = _row("same", 0)
    second["file_path"] = "images/second-snapshot/same.jpg"
    cleaned.write_text(json.dumps({"records": [first, second], "deduplication": {"exact_groups": [], "near_groups": []}}), encoding="utf-8")

    output = tmp_path / "candidates.json"
    assert build_candidate_manifest(cleaned, output) == 1

    image = json.loads(output.read_text(encoding="utf-8"))["images"][0]
    assert image["file_path"] == "images/same.jpg"
    assert image["class_presence"] == [0]
    assert image["labels"] == [{"class_id": 0, "xyxy": [1, 1, 4, 4]}]
