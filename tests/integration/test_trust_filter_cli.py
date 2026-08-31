"""CLI contract tests for sealed Trust Filter distribution bounds."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from fruit_ssod.cli.filter_pseudo_labels import main
from fruit_ssod.data.schema import LicenseMetadata, UnlabeledImageRecord
from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.pseudo.generator import canonical_unlabeled_fingerprint
from fruit_ssod.pseudo.transforms import horizontal_flip_xyxy


def _prepare_task8_unlabeled_manifests(root: Path) -> tuple[Path, Path]:
    record = UnlabeledImageRecord(
        source="fruit_360", source_image_id="unlabeled-1", file_path="unlabeled-1.jpg",
        width=640, height=640, split="train_pool", label_status="unlabeled",
        license_metadata=LicenseMetadata(name="CC BY"),
    )
    unlabeled = root / "unlabeled.json"
    unlabeled.write_text(json.dumps({"records": [{
        "source": record.source, "source_image_id": record.source_image_id,
        "file_path": record.file_path, "width": record.width, "height": record.height,
        "split": record.split, "label_status": record.label_status,
        "license_metadata": {"name": record.license_metadata.name},
    }]}), encoding="utf-8")
    split = root / "split_manifest.json"
    split.write_text(json.dumps({
        "unlabeled_image_ids": [record.source_image_id],
        "split_image_ids": {"validation": [], "test": [], "pseudo_audit": [], "external_test": []},
        "fingerprints": {"unlabeled": canonical_unlabeled_fingerprint((record,))},
    }), encoding="utf-8")
    return unlabeled, split


def _candidate_envelope(path: Path) -> None:
    candidates = []
    for view in ("original", "horizontal_flip"):
        xyxy = (100, 100, 200, 200)
        raw_xyxy = xyxy if view == "original" else horizontal_flip_xyxy(xyxy, width=640)
        candidates.append(PseudoCandidate("teacher-20", "unlabeled-1", "unlabeled-1.jpg", view, 0, "Apple", .96, raw_xyxy, xyxy, "teacher.pt").mapping())
    path.write_text(json.dumps({"manifest_version": "1.0", "teacher_run_id": "teacher-20", "candidate_count": len(candidates), "candidates": candidates}), encoding="utf-8")


def test_trust_cli_requires_sealed_aspect_ratio_bounds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    unlabeled, split = _prepare_task8_unlabeled_manifests(tmp_path)
    candidates = tmp_path / "candidates.json"; _candidate_envelope(candidates)
    validation_pr = tmp_path / "validation-pr.json"; validation_pr.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([
            "--candidates", str(candidates), "--unlabeled-manifest", str(unlabeled), "--split-manifest", str(split),
            "--validation-pr", str(validation_pr), "--output", str(tmp_path / "filtered"), "--mode", "trust",
        ])

    assert error.value.code == 2
    assert "--aspect-ratio-bounds is required in trust mode" in capsys.readouterr().err


def test_global_cli_allows_explicit_threshold_above_trust_per_class_clamp(tmp_path: Path) -> None:
    unlabeled, split = _prepare_task8_unlabeled_manifests(tmp_path)
    candidates = tmp_path / "candidates.json"; _candidate_envelope(candidates)
    assert main([
        "--candidates", str(candidates), "--unlabeled-manifest", str(unlabeled), "--split-manifest", str(split),
        "--output", str(tmp_path / "filtered"), "--mode", "global", "--global-confidence", "0.95",
    ]) == 0
    manifest = json.loads((tmp_path / "filtered" / "decision_manifest.json").read_text(encoding="utf-8"))
    audit = (tmp_path / "filtered" / "audit.jsonl").read_bytes()
    assert manifest["candidate_artifact_sha256"] == hashlib.sha256(candidates.read_bytes()).hexdigest()
    assert manifest["decision_records_sha256"] == hashlib.sha256(audit).hexdigest()
    assert manifest["decision_record_count"] == len(audit.splitlines())
