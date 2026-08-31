"""Local-fixture integration coverage for split output leakage safeguards."""

from __future__ import annotations

import json
from pathlib import Path

from fruit_ssod.cli.create_splits import main


def test_local_manifest_never_leaks_duplicate_groups_or_hidden_labels(tmp_path: Path) -> None:
    images = []
    for index in range(30):
        images.append({
            "source": "local_fixture",
            "source_image_id": f"image-{index}",
            "file_path": f"local/{index}.jpg",
            "width": 8,
            "height": 8,
            "class_presence": [index % 3, (index + 1) % 3],
            "labels": [{"class_id": index % 3, "xyxy": [0, 0, 4, 4]}],
            "duplicate_group_id": f"pair-{index // 2}",
            "license_metadata": {"name": "local fixture"},
        })
    source = tmp_path / "local-only.json"
    source.write_text(json.dumps({"images": images}), encoding="utf-8")
    output = tmp_path / "splits"

    assert main(["--input-manifest", str(source), "--output-root", str(output)]) == 0
    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    decisions = json.loads((output / "duplicate_group_decisions.json").read_text(encoding="utf-8"))
    ownership = {}
    for decision in decisions["decisions"]:
        assert decision["group_id"] not in ownership
        ownership[decision["group_id"]] = decision["split"]
    all_ids = [image_id for decision in decisions["decisions"] for image_id in decision["source_image_ids"]]
    assert len(all_ids) == len(set(all_ids)) == len(images)
    assert manifest["fingerprints"] == json.loads((output / "fingerprints.json").read_text(encoding="utf-8"))["fingerprints"]
    unlabeled = json.loads((output / "unlabeled.json").read_text(encoding="utf-8"))
    assert all("labels" not in item and "class_id" not in item and "xyxy" not in item for item in unlabeled["records"])
    assert not set(manifest["split_image_ids"]["pseudo_audit"]) & set(manifest["unlabeled_image_ids"])
