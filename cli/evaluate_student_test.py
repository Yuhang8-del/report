"""Evaluate one completed exploratory Student on the sealed fixed test split.

Student training snapshots intentionally contain only train/validation data, so
the normal supervised evaluator cannot infer a test YAML from the Student run
record. This CLI requires the caller to provide the already sealed v2 dataset
YAML, verifies its membership against the Student split fingerprint, and then
reuses the canonical evaluator implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from fruit_ssod.cli.evaluate_model import _canonical_dataset_evidence, _evaluate
from fruit_ssod.training.run_record import read_run_record
from fruit_ssod.training.supervised import SupervisedTrainingError, file_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a completed Student on a sealed fixed test YAML.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed Student run directory.")
    parser.add_argument("--data", type=Path, required=True, help="Sealed five-class dataset YAML containing the fixed test list.")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Paired split_manifest.json used by the Student run.")
    parser.add_argument("--device", default="cuda:0", help="Ultralytics device selector.")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_evidence(run_dir: Path) -> Mapping[str, Any]:
    payload = json.loads((run_dir / "checkpoint_evidence.json").read_text(encoding="utf-8"))
    expected = payload.get("best.pt") if isinstance(payload, Mapping) else None
    if not isinstance(expected, Mapping):
        raise SupervisedTrainingError("Problem: Student checkpoint evidence lacks best.pt. Likely cause: training did not publish a complete checkpoint record. Remediation: use a completed Student run.")
    actual = file_evidence(run_dir / "weights" / "best.pt", description="Student fixed-test checkpoint")
    expected_bytes, expected_sha = expected.get("bytes"), expected.get("sha256")
    if actual.get("bytes") != expected_bytes or actual.get("sha256") != expected_sha:
        raise SupervisedTrainingError("Problem: Student best.pt differs from its completion evidence. Likely cause: the checkpoint was replaced after training. Remediation: restore the immutable completed run.")
    return {"relative_path": "weights/best.pt", **actual}


def _verify_test_membership(data: Path, split_manifest: Path, effective: Mapping[str, Any], expected_split_fingerprint: str) -> dict[str, Any]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    fingerprints = manifest.get("fingerprints") if isinstance(manifest, Mapping) else None
    split_ids = manifest.get("split_image_ids") if isinstance(manifest, Mapping) else None
    if not isinstance(fingerprints, Mapping) or fingerprints.get("split_protocol") != expected_split_fingerprint or not isinstance(split_ids, Mapping) or not isinstance(split_ids.get("test"), list):
        raise SupervisedTrainingError("Problem: fixed-test split manifest does not match the Student run. Likely cause: an unrelated split was supplied. Remediation: pass the exact split_manifest.json bound to the completed Student.")
    root = Path(str(effective["path"]))
    test_value = effective.get("test")
    if not isinstance(test_value, str) or not test_value:
        raise SupervisedTrainingError("Problem: fixed-test YAML has no test list. Likely cause: the supplied YAML is a Student train/validation snapshot. Remediation: pass the sealed v2 supervised dataset YAML.")
    test_list = Path(test_value)
    test_list = test_list if test_list.is_absolute() else root / test_list
    lines = [line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed = sorted(Path(line).stem for line in lines)
    expected = sorted(str(value) for value in split_ids["test"])
    if observed != expected:
        raise SupervisedTrainingError("Problem: fixed-test image membership differs from the sealed split. Likely cause: the dataset YAML test list was altered. Remediation: restore the exact v2 dataset YAML and test list.")
    return {"split_fingerprint": expected_split_fingerprint, "test_list_sha256": _sha256(test_list), "test_image_count": len(observed)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        run_dir = args.run_dir.resolve(strict=True) if (args := parser.parse_args(argv)) else None
        record = read_run_record(run_dir / "run_record.json")
        if record.status != "complete":
            raise SupervisedTrainingError("Problem: Student run is not complete. Likely cause: training evidence is still running or failed. Remediation: evaluate only a completed Student run.")
        data, effective, data_digest = _canonical_dataset_evidence(args.data.resolve(strict=True), protocol="fixed test")
        checkpoint = _checkpoint_evidence(run_dir)
        split_manifest = args.split_manifest.resolve(strict=True)
        membership = _verify_test_membership(data, split_manifest, effective, record.split_fingerprint)
        snapshot = dict(record.config_snapshot)
        snapshot["dataset_yaml"] = str(data)
        snapshot["dataset_yaml_sha256"] = data_digest
        proxy = SimpleNamespace(config_snapshot=snapshot)
        execution = _evaluate(run_dir, proxy, split="test", data=data, device=args.device)
        result = {
            "schema_version": "1.0",
            "artifact_type": "student_fixed_test_evaluation",
            "run_id": record.run_id,
            "split": "test",
            "split_fingerprint": record.split_fingerprint,
            "checkpoint": checkpoint,
            "dataset_yaml": str(data),
            "dataset_yaml_sha256": data_digest,
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": _sha256(split_manifest),
            "membership": membership,
            "metrics": dict(execution.metrics),
            "raw_evaluator_outputs": dict(execution.raw_evaluator_outputs),
            "exploratory": True,
        }
        output = run_dir / "evaluations" / "test.json"
        if output.exists():
            raise SupervisedTrainingError(f"Problem: fixed-test evaluation already exists. Likely cause: {output} would be overwritten. Remediation: preserve immutable evidence and choose a new run for another evaluation.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (SupervisedTrainingError, OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps({"run_id": record.run_id, "split": "test", "output": str(output), "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
