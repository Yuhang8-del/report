from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fruit_ssod.data.full_label_upper_bound import audit_full_label_upper_bound, materialize_full_label_upper_bound


def _record(image_id: str, class_id: int) -> dict[str, object]:
    return {
        "source": "fixture",
        "source_image_id": image_id,
        "file_path": f"images/{image_id}.jpg",
        "width": 20,
        "height": 10,
        "class_presence": [class_id],
        "labels": [{"class_id": class_id, "xyxy": [2, 1, 12, 6]}],
        "duplicate_group_id": f"unique:{image_id}",
        "license_metadata": {"name": "CC"},
    }


def test_materialize_full_label_upper_bound_restores_hidden_training_labels(tmp_path: Path) -> None:
    records = [_record("train", 0), _record("hidden", 1), _record("val", 2), _record("test", 3), _record("audit", 4)]
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    for record in records:
        Image.new("RGB", (20, 10), "green").save(source / str(record["file_path"]))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"images": records}), encoding="utf-8")
    split = tmp_path / "split"
    (split / "budgets" / "100").mkdir(parents=True)
    (split / "protected_splits").mkdir()
    manifest = {
        "train_pool_image_ids": ["train"],
        "unlabeled_image_ids": ["hidden"],
        "split_image_ids": {"validation": ["val"], "test": ["test"], "pseudo_audit": ["audit"], "external_test": []},
        "fingerprints": {"split_protocol": "a" * 64},
    }
    (split / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (split / "budgets" / "100" / "labels.json").write_text(json.dumps({"records": [records[0]]}), encoding="utf-8")
    (split / "protected_splits" / "validation_labels.json").write_text(json.dumps({"records": [records[2]]}), encoding="utf-8")
    (split / "protected_splits" / "test_labels.json").write_text(json.dumps({"records": [records[3]]}), encoding="utf-8")

    result = materialize_full_label_upper_bound(candidate, split, source, tmp_path / "snapshot", expected_train_count=2)

    membership = json.loads(result.membership.read_text(encoding="utf-8"))
    assert result.image_count == 4
    assert membership["artifact_type"] == "sealed_full_label_supervised_upper_bound"
    assert [row["source_image_id"] for row in membership["members"]["train"]] == ["hidden", "train"]
    assert {row["source_image_id"] for row in membership["members"]["val"]} == {"val"}
    assert {row["source_image_id"] for row in membership["members"]["test"]} == {"test"}
    assert not (result.root / "images" / "train" / "audit.jpg").exists()
    assert membership["recovered_hidden_label_count"] == 1


def test_full_label_upper_bound_rejects_changed_budget_membership(tmp_path: Path) -> None:
    records = [_record("train", 0), _record("hidden", 1), _record("val", 2), _record("test", 3)]
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    for record in records:
        Image.new("RGB", (20, 10), "green").save(source / str(record["file_path"]))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"images": records}), encoding="utf-8")
    split = tmp_path / "split"
    (split / "budgets" / "100").mkdir(parents=True)
    (split / "protected_splits").mkdir()
    manifest = {
        "train_pool_image_ids": ["train"],
        "unlabeled_image_ids": ["hidden"],
        "split_image_ids": {"validation": ["val"], "test": ["test"], "pseudo_audit": [], "external_test": []},
        "fingerprints": {"split_protocol": "a" * 64},
    }
    (split / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (split / "budgets" / "100" / "labels.json").write_text(json.dumps({"records": [records[1]]}), encoding="utf-8")
    (split / "protected_splits" / "validation_labels.json").write_text(json.dumps({"records": [records[2]]}), encoding="utf-8")
    (split / "protected_splits" / "test_labels.json").write_text(json.dumps({"records": [records[3]]}), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="100% budget membership"):
        materialize_full_label_upper_bound(candidate, split, source, tmp_path / "snapshot", expected_train_count=2)


def test_snapshot_normalizes_truncated_jpeg_and_detects_later_mutation(tmp_path: Path) -> None:
    records = [_record("train", 0), _record("hidden", 1), _record("val", 2), _record("test", 3)]
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    for record in records:
        path = source / str(record["file_path"])
        Image.new("RGB", (20, 10), "green").save(path)
    truncated = source / "images" / "train.jpg"
    truncated.write_bytes(truncated.read_bytes()[:-2])
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"images": records}), encoding="utf-8")
    split = tmp_path / "split"
    (split / "budgets" / "100").mkdir(parents=True)
    (split / "protected_splits").mkdir()
    manifest = {
        "train_pool_image_ids": ["train"],
        "unlabeled_image_ids": ["hidden"],
        "split_image_ids": {"validation": ["val"], "test": ["test"], "pseudo_audit": [], "external_test": []},
        "fingerprints": {"split_protocol": "a" * 64},
    }
    (split / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (split / "budgets" / "100" / "labels.json").write_text(json.dumps({"records": [records[0]]}), encoding="utf-8")
    (split / "protected_splits" / "validation_labels.json").write_text(json.dumps({"records": [records[2]]}), encoding="utf-8")
    (split / "protected_splits" / "test_labels.json").write_text(json.dumps({"records": [records[3]]}), encoding="utf-8")

    result = materialize_full_label_upper_bound(candidate, split, source, tmp_path / "snapshot", expected_train_count=2)
    published = result.root / "images" / "train" / "train.jpg"
    assert published.read_bytes()[-2:] == b"\xff\xd9"
    audit = audit_full_label_upper_bound(result.root)
    assert audit["verified_image_count"] == 4

    published.write_bytes(published.read_bytes() + b"mutation")
    import pytest

    with pytest.raises(ValueError, match="image digest differs"):
        audit_full_label_upper_bound(result.root)
