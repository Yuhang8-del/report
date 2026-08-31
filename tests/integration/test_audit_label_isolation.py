from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fruit_ssod.cli.audit_pseudo_labels import main as audit_main
from fruit_ssod.cli.audit_pseudo_labels import _example_rows
from fruit_ssod.cli.filter_pseudo_labels import _parser as filter_parser
from fruit_ssod.cli.generate_pseudo_labels import _parser as generation_parser
from fruit_ssod.cli.train_supervised import _parser as training_parser
from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.evaluation.pseudo_metrics import AuditBox, AuditImage


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _record() -> dict[str, object]:
    return {
        "source": "open_images_v7", "source_image_id": "audit-1", "file_path": "audit/audit-1.jpg",
        "width": 640, "height": 640, "class_presence": [0],
        "labels": [{"class_id": 0, "xyxy": [100, 100, 200, 200]}],
        "duplicate_group_id": "audit-group-1", "protected_split": None,
        "license_metadata": {"name": "CC BY", "url": None, "attribution": None},
    }


def _write_sealed_audit(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    records = [_record()]
    labels = root / "pseudo_audit_labels.json"; labels.write_text(json.dumps({"records": records}), encoding="utf-8")
    manifest = root / "split_manifest.json"; manifest.write_text(json.dumps({
        "split_image_ids": {"validation": [], "test": [], "pseudo_audit": ["audit-1"], "external_test": []},
        "fingerprints": {"protected/pseudo_audit": _fingerprint(records)},
    }), encoding="utf-8")
    return labels, manifest


def _candidate(*, class_id: int, confidence: float) -> PseudoCandidate:
    return PseudoCandidate("audit-teacher", "audit-1", "audit/audit-1.jpg", "original", class_id, ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")[class_id], confidence, (100., 100., 200., 200.), (100., 100., 200., 200.), "teacher.pt")


def _write_decision_manifest(candidates: Path, filtered: Path, *, provenance: object = None) -> Path:
    manifest = filtered.with_name("decision_manifest.json")
    manifest.write_text(json.dumps({
        "schema_version": "1.0",
        "artifact_type": "sealed_task14_filter_decisions",
        "teacher_run_id": "audit-teacher",
        "candidate_artifact_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "decision_record_count": len(filtered.read_text(encoding="utf-8").splitlines()),
        "decision_records_sha256": hashlib.sha256(filtered.read_bytes()).hexdigest(),
        "filter_provenance": provenance,
        "filter_provenance_sha256": _fingerprint(provenance),
    }, sort_keys=True), encoding="utf-8")
    return manifest


def _write_bound_candidate_and_filter(root: Path) -> tuple[Path, Path]:
    candidates = (_candidate(class_id=0, confidence=.96), _candidate(class_id=1, confidence=.91))
    candidate_path = root / "candidates.json"; candidate_path.write_text(json.dumps({"manifest_version": "1.0", "teacher_run_id": "audit-teacher", "candidate_count": 2, "candidates": [item.mapping() for item in candidates]}), encoding="utf-8")
    rows = []
    for item, decision in zip(candidates, ("accepted", "rejected")):
        row = item.mapping(); row.update({"decision": decision, "reason_code": None if decision == "accepted" else "below_class_threshold", "paired_with_view": None, "paired_with_confidence": None, "filter_provenance": None}); rows.append(row)
    filter_path = root / "filter.jsonl"; filter_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_decision_manifest(candidate_path, filter_path)
    return candidate_path, filter_path


def test_audit_cli_is_the_only_command_accepting_sealed_audit_labels(tmp_path: Path) -> None:
    labels, manifest = _write_sealed_audit(tmp_path)
    candidates, filtered = _write_bound_candidate_and_filter(tmp_path)
    output = tmp_path / "audit-output"
    assert audit_main(["--audit-labels", str(labels), "--split-manifest", str(manifest), "--candidates", str(candidates), "--filter-audit", str(filtered), "--output", str(output)]) == 0
    payload = json.loads((output / "pseudo_audit.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["before_filter"]["overall"]["precision"] == .5
    assert payload["metrics"]["after_filter"]["overall"]["precision"] == 1.0
    assert payload["pseudo_refresh"] == {"allowed": True, "reason": "precision_at_or_above_threshold"}
    assert set(path.stem.split("_")[0] for path in (output / "examples").glob("*.png")) >= {"kept", "rejected", "false", "missed"}
    for parser in (training_parser(), generation_parser(), filter_parser()):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(["--audit-labels", str(labels)])
        assert error.value.code == 2


def test_audit_rejects_substituted_membership_and_unbound_filter_rows(tmp_path: Path) -> None:
    labels, manifest = _write_sealed_audit(tmp_path)
    candidates, filtered = _write_bound_candidate_and_filter(tmp_path)
    altered = json.loads(labels.read_text(encoding="utf-8")); altered["records"][0]["labels"][0]["xyxy"] = [1, 1, 2, 2]
    labels.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        audit_main(["--audit-labels", str(labels), "--split-manifest", str(manifest), "--candidates", str(candidates), "--filter-audit", str(filtered), "--output", str(tmp_path / "bad")])
    assert error.value.code == 2


@pytest.mark.parametrize("field,value", [
    ("decision", "rejected"),
    ("reason_code", "tampered_reason"),
    ("filter_provenance", {"policy": "tampered"}),
])
def test_audit_rejects_tampered_sealed_decision_evidence(tmp_path: Path, field: str, value: object) -> None:
    labels, manifest = _write_sealed_audit(tmp_path)
    candidates, filtered = _write_bound_candidate_and_filter(tmp_path)
    rows = [json.loads(line) for line in filtered.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    filtered.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        audit_main([
            "--audit-labels", str(labels), "--split-manifest", str(manifest),
            "--candidates", str(candidates), "--filter-audit", str(filtered),
            "--output", str(tmp_path / f"tampered-{field}"),
        ])
    assert error.value.code == 2

    labels, manifest = _write_sealed_audit(tmp_path / "fresh")
    candidates, filtered = _write_bound_candidate_and_filter(tmp_path / "fresh")
    rows = filtered.read_text(encoding="utf-8").splitlines(); tampered = json.loads(rows[0]); tampered["confidence"] = .12
    filtered.write_text(json.dumps(tampered) + "\n" + rows[1] + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        audit_main(["--audit-labels", str(labels), "--split-manifest", str(manifest), "--candidates", str(candidates), "--filter-audit", str(filtered), "--output", str(tmp_path / "unbound")])
    assert error.value.code == 2


def test_audit_rejects_candidate_path_escape_and_stops_refresh_below_precision(tmp_path: Path) -> None:
    labels, manifest = _write_sealed_audit(tmp_path)
    candidates, filtered = _write_bound_candidate_and_filter(tmp_path)
    envelope = json.loads(candidates.read_text(encoding="utf-8")); envelope["candidates"][0]["source_file_path"] = "../test.jpg"
    candidates.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(SystemExit):
        audit_main(["--audit-labels", str(labels), "--split-manifest", str(manifest), "--candidates", str(candidates), "--filter-audit", str(filtered), "--output", str(tmp_path / "escape")])


def test_audit_output_stops_pseudo_refresh_when_post_filter_precision_is_low(tmp_path: Path) -> None:
    labels, manifest = _write_sealed_audit(tmp_path)
    candidates, filtered = _write_bound_candidate_and_filter(tmp_path)
    rows = [json.loads(line) for line in filtered.read_text(encoding="utf-8").splitlines()]
    rows[0]["decision"] = "rejected"; rows[0]["reason_code"] = "below_class_threshold"
    rows[1]["decision"] = "accepted"; rows[1]["reason_code"] = None
    filtered.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_decision_manifest(candidates, filtered)
    output = tmp_path / "stopped"
    assert audit_main(["--audit-labels", str(labels), "--split-manifest", str(manifest), "--candidates", str(candidates), "--filter-audit", str(filtered), "--output", str(output)]) == 0
    payload = json.loads((output / "pseudo_audit.json").read_text(encoding="utf-8"))
    assert payload["pseudo_refresh"] == {"allowed": False, "reason": "stopped_precision_below_threshold"}


def test_example_rows_preserve_byte_identical_rejected_occurrences() -> None:
    image = AuditImage("audit-1", "audit/audit-1.jpg", 640, 640)
    duplicate = AuditBox("audit-1", 0, (100.0, 100.0, 200.0, 200.0), .96)
    rows = _example_rows((duplicate, duplicate), (duplicate,), (), iou=.5, images={"audit-1": image})
    assert len(rows["kept"]) == 1
    assert len(rows["rejected"]) == 1
