"""Fixture-only coverage for the report-ready dataset audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.cli.audit_dataset import main
from fruit_ssod.data.audit import DatasetAuditError, audit_annotations


def _row(*, image_id: str, class_id: int, split: str, source: str = "open_images_v7") -> dict[str, object]:
    return {
        "source": source,
        "source_category": ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")[class_id],
        "source_image_id": image_id,
        "file_path": f"images/{image_id}.png",
        "width": 32,
        "height": 24,
        "class_id": class_id,
        "xyxy": [2, 3, 20, 18],
        "split": split,
        "label_status": "labeled" if split != "pseudo_audit" else "pseudo",
        "license_metadata": {"name": "CC BY 4.0", "url": "https://example.invalid/license"},
    }


def _complete_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train_pool", "validation", "test", "pseudo_audit"):
        for class_id in range(5):
            rows.append(_row(image_id=f"{split}-{class_id}", class_id=class_id, split=split))
    return rows


def test_audit_writes_report_ready_summaries_and_montage(tmp_path: Path) -> None:
    rows = _complete_rows()
    images = tmp_path / "images"
    images.mkdir()
    for index, row in enumerate(rows):
        Image.new("RGB", (32, 24), (index, index * 3 % 255, index * 7 % 255)).save(images / f"{row['source_image_id']}.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"records": rows}), encoding="utf-8")
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(json.dumps({"budget_image_ids": {"20": ["train_pool-0"], "100": [f"train_pool-{index}" for index in range(5)]}}), encoding="utf-8")
    output = tmp_path / "audit"

    assert main(["--annotations", str(annotations), "--split-manifest", str(split_manifest), "--image-root", str(tmp_path), "--output-root", str(output)]) == 0

    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    assert audit["critical_finding_count"] == 0
    assert audit["class_box_counts"]["train_pool"]["open_images_v7"]["0"] == 1
    assert audit["label_budget_membership"]["20"]["image_count"] == 1
    assert audit["source_license_summary"][0]["license_name"] == "CC BY 4.0"
    assert (output / "sample_annotation_montage.png").is_file()
    with (output / "data_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    assert len(manifest_rows) == 20
    assert manifest_rows[0]["source_image_id"]


def test_audit_reports_protocol_violations_as_critical_findings_and_cli_fails(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows = [row for row in rows if not (row["split"] == "validation" and row["class_id"] == 4)]
    rows.append(_row(image_id="same-image", class_id=0, split="train_pool"))
    duplicate = _row(image_id="same-image-copy", class_id=0, split="test")
    duplicate["image_hash"] = "shared-hash"
    rows[-1]["image_hash"] = "shared-hash"
    rows.append(duplicate)
    malformed = _row(image_id="bad-box", class_id=1, split="validation")
    malformed["xyxy"] = [10, 3, 2, 18]
    rows.append(malformed)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "audit"

    assert main(["--annotations", str(annotations), "--output-root", str(output)]) == 1
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    codes = {finding["code"] for finding in audit["findings"]}
    assert {"MISSING_CLASS", "DUPLICATE_HASH_CROSS_SPLIT", "ILLEGAL_BBOX"} <= codes
    assert audit["critical_finding_count"] >= 3


def test_audit_detects_empty_required_split() -> None:
    rows = [row for row in _complete_rows() if row["split"] != "pseudo_audit"]

    result = audit_annotations(rows)

    assert any(finding.code == "EMPTY_SPLIT" and finding.details["split"] == "pseudo_audit" for finding in result.findings)


def test_audit_uses_cleaned_manifest_hashes_and_split_membership_end_to_end(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows.append(_row(image_id="unlabeled-0", class_id=0, split="train_pool", source="fruit_360"))
    rows.append(_row(image_id="unlabeled-0", class_id=1, split="train_pool", source="fruit_360"))
    for row in rows:
        row["split"] = "train_pool"  # cleaned input predates the fixed Task 8 protocol.
    fingerprint_rows = []
    for index, row in enumerate(rows):
        image_id = str(row["source_image_id"])
        fingerprint_rows.append({"source_image_id": image_id, "file_path": row["file_path"], "split": "train_pool", "sha256": f"hash-{image_id}", "perceptual_hash": "0"})
    # The cleaned manifest itself has no per-record hash.  Two protocol members
    # are deliberately assigned the same Task 7 dedup SHA across splits.
    fingerprint_rows[0]["sha256"] = "cross-split-hash"
    fingerprint_rows[5]["sha256"] = "cross-split-hash"
    cleaned = tmp_path / "cleaned.json"
    cleaned.write_text(json.dumps({"records": rows, "deduplication": {"fingerprints": fingerprint_rows}}), encoding="utf-8")
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(json.dumps({
        "split_image_ids": {
            "validation": [f"validation-{index}" for index in range(5)],
            "test": [f"test-{index}" for index in range(5)],
            "pseudo_audit": [f"pseudo_audit-{index}" for index in range(5)],
            "external_test": [],
        },
        "train_pool_image_ids": [f"train_pool-{index}" for index in range(5)],
        "budget_image_ids": {"20": ["train_pool-0"], "100": [f"train_pool-{index}" for index in range(5)]},
        "unlabeled_image_ids": ["unlabeled-0"],
    }), encoding="utf-8")
    output = tmp_path / "audit"

    assert main(["--annotations", str(cleaned), "--split-manifest", str(split_manifest), "--output-root", str(output)]) == 1

    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    assert any(item["code"] == "DUPLICATE_HASH_CROSS_SPLIT" for item in audit["findings"])
    assert not any(item["code"] == "SPLIT_MEMBERSHIP_MISSING" for item in audit["findings"])
    assert "unlabeled" not in audit["class_box_counts"]
    unlabeled_license = next(item for item in audit["source_license_summary"] if item["source"] == "fruit_360")
    assert unlabeled_license["image_count"] == 1
    assert unlabeled_license["box_count"] == 0
    with (output / "data_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    assert {"source", "source_image_id", "original_source_image_id", "image_hash", "effective_split", "classes", "class_count", "box_count", "license_name", "license_url", "attribution", "cleaning_status"} <= set(manifest_rows[0])
    assert next(item for item in manifest_rows if item["source_image_id"] == "validation-0")["effective_split"] == "validation"
    assert next(item for item in manifest_rows if item["source_image_id"] == "train_pool-0")["image_hash"] == "cross-split-hash"
    unlabeled = next(item for item in manifest_rows if item["source_image_id"] == "unlabeled-0")
    assert unlabeled["effective_split"] == "unlabeled"
    assert unlabeled["classes"] == ""
    assert unlabeled["class_count"] == "0"
    assert unlabeled["box_count"] == "0"


def test_audit_rejects_mixed_budget_keys_with_actionable_error() -> None:
    with pytest.raises(DatasetAuditError, match="budget"):
        audit_annotations(_complete_rows(), split_manifest={"budget_image_ids": {"20": [], "custom": []}})


def test_montage_keeps_same_image_id_from_different_sources_separate(tmp_path: Path) -> None:
    first = _row(image_id="shared-id", class_id=0, split="train_pool", source="first_source")
    second = _row(image_id="shared-id", class_id=1, split="train_pool", source="second_source")
    first["file_path"] = "images/first.png"
    second["file_path"] = "images/second.png"
    (tmp_path / "images").mkdir()
    Image.new("RGB", (32, 24), "red").save(tmp_path / "images" / "first.png")
    Image.new("RGB", (32, 24), "blue").save(tmp_path / "images" / "second.png")

    output = tmp_path / "audit"
    result = audit_annotations([first, second])
    from fruit_ssod.data.audit import write_audit_outputs
    write_audit_outputs(result, output, image_root=tmp_path)

    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    assert audit["sample_annotation_montage_image_count"] == 2


def test_audit_computes_missing_hashes_and_fails_cross_split_same_file(tmp_path: Path) -> None:
    image = tmp_path / "shared.png"
    Image.new("RGB", (32, 24), "white").save(image)
    train = _row(image_id="train-image", class_id=0, split="train_pool")
    validation = _row(image_id="validation-image", class_id=0, split="validation")
    train["file_path"] = validation["file_path"] = "shared.png"

    result = audit_annotations([train, validation], expected_class_ids=(), required_splits=("train_pool", "validation"), image_root=tmp_path)

    assert result.critical_finding_count == 1
    assert [finding.code for finding in result.findings] == ["DUPLICATE_HASH_CROSS_SPLIT"]


def test_audit_cli_verifies_task8_sealed_label_fingerprints(tmp_path: Path) -> None:
    rows = _complete_rows()
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(rows), encoding="utf-8")
    split_root = tmp_path / "splits"
    protected = split_root / "protected_splits"
    protected.mkdir(parents=True)
    payloads = {
        "test": {"records": [{"source_image_id": "test-0", "labels": [{"class_id": 0}]}]},
        "pseudo_audit": {"records": [{"source_image_id": "pseudo_audit-0", "labels": [{"class_id": 0}]}]},
    }
    import hashlib
    for name, payload in payloads.items():
        (protected / f"{name}_labels.json").write_text(json.dumps(payload), encoding="utf-8")
    fingerprints = {f"protected/{name}": hashlib.sha256(json.dumps(payload["records"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest() for name, payload in payloads.items()}
    (split_root / "split_manifest.json").write_text(json.dumps({"fingerprints": fingerprints}), encoding="utf-8")

    good = tmp_path / "good"
    assert main(["--annotations", str(annotations), "--split-output-root", str(split_root), "--output-root", str(good)]) == 1  # source images intentionally have no hashes
    good_audit = json.loads((good / "dataset_audit.json").read_text(encoding="utf-8"))
    assert not any(finding["code"] == "SEALED_LABEL_FINGERPRINT_MISMATCH" for finding in good_audit["findings"])
    payloads["test"]["records"][0]["labels"][0]["class_id"] = 4
    (protected / "test_labels.json").write_text(json.dumps(payloads["test"]), encoding="utf-8")
    bad = tmp_path / "bad"
    assert main(["--annotations", str(annotations), "--split-output-root", str(split_root), "--output-root", str(bad)]) == 1
    audit = json.loads((bad / "dataset_audit.json").read_text(encoding="utf-8"))
    assert any(finding["code"] == "SEALED_LABEL_FINGERPRINT_MISMATCH" for finding in audit["findings"])
    payloads["test"]["records"][0]["labels"][0]["class_id"] = float("nan")
    (protected / "test_labels.json").write_text(json.dumps(payloads["test"]), encoding="utf-8")
    nonfinite = tmp_path / "nonfinite"
    assert main(["--annotations", str(annotations), "--split-output-root", str(split_root), "--output-root", str(nonfinite)]) == 1
    audit = json.loads((nonfinite / "dataset_audit.json").read_text(encoding="utf-8"))
    assert any(finding["code"] == "SEALED_LABEL_ARTIFACT_MALFORMED" for finding in audit["findings"])


def test_audit_cli_missing_local_paths_are_actionable_parser_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing_annotations:
        main(["--annotations", str(tmp_path / "missing.json"), "--output-root", str(tmp_path / "out")])
    assert missing_annotations.value.code == 2
    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as missing_split:
        main(["--annotations", str(annotations), "--split-manifest", str(tmp_path / "missing-split.json"), "--output-root", str(tmp_path / "out2")])
    assert missing_split.value.code == 2


def test_audit_result_is_deeply_immutable() -> None:
    rows = _complete_rows()
    result = audit_annotations(rows, image_hashes={(str(row["source_image_id"]), str(row["file_path"])): "hash-" + str(index) for index, row in enumerate(rows)})
    rows[0]["source"] = "mutated-after-audit"

    with pytest.raises(TypeError):
        result.manifest_rows[0]["source"] = "mutated"
    with pytest.raises(TypeError):
        result.class_box_counts["train_pool"]["open_images_v7"]["0"] = 99
    assert result.manifest_rows[0]["source"] == "open_images_v7"
