from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fruit_ssod.cli.aggregate_results import publish_result_package, verify_result_package
from fruit_ssod.evaluation.acceptance import evaluate_acceptance
from fruit_ssod.evaluation.aggregate import ResultAggregationError, aggregate_results, canonical_json
from fruit_ssod.reporting.result_tables import fruitdet_rows, primary_result_rows
from fruit_ssod.reporting.result_figures import evaluator_figure_sources
from fruit_ssod.training.run_record import RunRecord, write_run_record
from fruit_ssod.training.supervised import _dataset_evidence


FINGERPRINT = "a" * 64
METRICS = {"map50": 0.82, "map50_95": 0.61, "precision": 0.84, "recall": 0.80, "f1": 0.82, "per_class_ap50": {str(index): 0.80 + index / 100 for index in range(5)}}
EXTERNAL_METRICS = {"map50": 0.82, "map50_95": 0.61, "precision": 0.84, "recall": 0.80, "f1": 0.82, "reported_class_ids": [0, 1, 2, 3], "per_class_ap50": {str(index): 0.80 + index / 100 for index in range(4)}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fruitdet_manifest(directory: Path) -> Path:
    path = directory / "fruitdet_manifest.json"
    members: list[Path] = []
    for index in range(4):
        member = directory / f"fruitdet-{index}.jpg"
        member.write_bytes(f"fruitdet-{index}".encode())
        members.append(member)
    payload = {
        "manifest_version": "1.0",
        "source": {"name": "fruitdet", "version": "fixture", "page": "https://example.invalid", "license": {"name": "fixture"}},
        "category_mapping_source": "limited_external_set",
        "mapped_class_ids": [0, 1, 2, 3],
        "mapped_class_names": ["Apple", "Banana", "Orange", "Strawberry"],
        "split": "external_test",
        "label_status": "labeled",
        "records": [
            {"source_dataset": "fruitdet", "source": "limited_external_set", "split": "external_test", "label_status": "labeled", "class_id": index, "file_path": str(members[index])}
            for index in range(4)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(tmp_path: Path, name: str, *, method: str, seed: int, status: str = "complete", test: bool = True, external: bool = False, raw_plots: bool = False) -> Path:
    directory = tmp_path / name
    (directory / "weights").mkdir(parents=True)
    weight = directory / "weights" / "best.pt"
    weight.write_bytes(b"checkpoint-" + name.encode())
    digest = _sha(weight)
    config = {"experiment_name": name, "matrix_role": method, "seed": seed, "label_budget_percent": 20, "dataset_yaml_sha256": "b" * 64}
    record = RunRecord(run_id=name, status=status, config_snapshot=config, split_fingerprint=FINGERPRINT, command=("python", "train"), environment={"python": "3.10"}, result=METRICS if status == "complete" else None, failure={"problem": "test", "cause": "test", "remediation": "test"} if status == "failed" else None)
    write_run_record(record, directory / "run_record.json")
    (directory / "checkpoint_evidence.json").write_text(json.dumps({"best.pt": {"relative_path": "weights/best.pt", "bytes": weight.stat().st_size, "sha256": digest}}), encoding="utf-8")
    if test:
        metrics = dict(METRICS)
        protocol = {"schema": "fruit_ssod_evaluation_evidence_v1", "run_id": name, "split": "test", "checkpoint_sha256": digest, "dataset_yaml_sha256": "b" * 64, "metrics_sha256": hashlib.sha256(json.dumps(metrics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        (directory / "evaluations").mkdir()
        if raw_plots:
            raw_root = directory / "evaluations" / "raw" / "test"
            raw_root.mkdir(parents=True)
            raw = {}
            for key, filename in (("precision_recall", "PR_curve.png"), ("confusion_matrix", "confusion_matrix.png")):
                artifact = raw_root / filename
                artifact.write_bytes(f"{name}-{key}".encode())
                raw[key] = {"path": str(artifact.resolve()), "relative_path": f"evaluations/raw/test/{filename}", "bytes": artifact.stat().st_size, "sha256": _sha(artifact), "split": "test"}
            protocol["raw_evaluator_outputs"] = raw
        (directory / "evaluations" / "test.json").write_text(json.dumps({"metrics": metrics, "protocol": protocol}), encoding="utf-8")
    if external:
        manifest = _fruitdet_manifest(directory)
        members = sorted(directory.glob("fruitdet-*.jpg"))
        member_list = directory / "fruitdet_images.txt"
        member_list.write_text("\n".join(str(member) for member in members) + "\n", encoding="utf-8")
        data = directory / "fruitdet.yaml"
        data.write_text("path: .\ntrain: fruitdet_images.txt\nval: fruitdet_images.txt\ntest: fruitdet_images.txt\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
        _, data_digest = _dataset_evidence(data)
        membership = hashlib.sha256(json.dumps(sorted(str(member.resolve()) for member in members), separators=(",", ":")).encode()).hexdigest()
        manifest_digest = _sha(manifest)
        fingerprint = hashlib.sha256(json.dumps({"dataset_yaml_sha256": data_digest, "fruitdet_manifest_sha256": manifest_digest, "membership_sha256": membership}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        protocol = {"schema": "fruit_ssod_evaluation_evidence_v1", "run_id": name, "split": "external_test", "checkpoint_sha256": digest, "metrics_sha256": hashlib.sha256(json.dumps(EXTERNAL_METRICS, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "dataset_yaml": str(data.resolve()), "dataset_yaml_sha256": data_digest, "mapped_class_ids": [0, 1, 2, 3], "mapped_class_names": ["Apple", "Banana", "Orange", "Strawberry"], "observed_class_ids": [0, 1, 2, 3], "observed_class_names": ["Apple", "Banana", "Orange", "Strawberry"], "mapping_source": "limited_external_set", "external_protocol": {"protocol_id": "fruitdet_external_mapped_v1", "mapping_source": "limited_external_set", "mapped_class_ids": [0, 1, 2, 3], "mapped_class_names": ["Apple", "Banana", "Orange", "Strawberry"]}, "fruitdet_manifest": {"path": str(manifest.resolve()), "bytes": manifest.stat().st_size, "sha256": manifest_digest, "membership_sha256": membership, "member_count": 4}, "fruitdet_dataset_fingerprint": fingerprint}
        (directory / "evaluations" / "external_test.json").write_text(json.dumps({"metrics": EXTERNAL_METRICS, "protocol": protocol}), encoding="utf-8")
    return directory


def test_aggregation_keeps_failed_and_missing_runs_visible(tmp_path: Path) -> None:
    complete = _run(tmp_path, "trust42", method="trust_main", seed=42)
    failed = _run(tmp_path, "trust3407", method="trust_main", seed=3407, status="failed", test=False)
    missing = tmp_path / "missing"
    result = aggregate_results((complete, failed, missing))
    assert {row["run_id"]: row["status"] for row in result["rows"]} == {"trust42": "complete", "trust3407": "failed", None: "unreadable"}
    assert result["summary"]["main_groups"]["trust_main"]["complete"] is False
    assert next(row for row in result["rows"] if row["run_id"] == "trust3407")["failure"]["problem"] == "test"


def test_aggregation_three_seed_summary_is_immutable_and_fruitdet_is_mapped_only(tmp_path: Path) -> None:
    runs = [_run(tmp_path, f"supervised{seed}", method="supervised_20", seed=seed, external=seed == 42) for seed in (42, 3407, 2026)]
    runs += [_run(tmp_path, f"trust{seed}", method="trust_main", seed=seed) for seed in (42, 3407, 2026)]
    result = aggregate_results(runs)
    assert result["summary"]["main_groups"]["supervised_20"]["metrics"]["map50"] == {"mean": 0.82, "std": 0.0, "n": 3}
    with pytest.raises(TypeError):
        result["summary"]["main_groups"]["trust_main"] = {}  # type: ignore[index]
    external = fruitdet_rows(result)
    assert set(external[0]["mapped_classes"].split(",")) == {"Apple", "Banana", "Orange", "Strawberry"}
    assert "ap50_class_4" not in external[0]


def test_corrupt_fixed_test_downgrades_evaluation_but_preserves_training_row(tmp_path: Path) -> None:
    run = _run(tmp_path, "trust42", method="trust_main", seed=42)
    path = run / "evaluations" / "test.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"]["metrics_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = aggregate_results((run,))["rows"][0]
    assert row["status"] == "complete"
    assert row["evaluation_status"] == "invalid_evaluation"
    assert row["primary_test"] is None
    assert any("metrics digest" in issue for issue in row["issues"])


def test_fruitdet_external_evidence_requires_exact_protocol_and_labeled_record_semantics(tmp_path: Path) -> None:
    run = _run(tmp_path, "fruitdet", method="supervised_20", seed=42, external=True)
    evaluation_path = run / "evaluations" / "external_test.json"
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    payload["protocol"]["mapped_class_ids"] = [0]
    payload["protocol"]["external_protocol"]["mapped_class_ids"] = [0]
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    row = aggregate_results((run,))["rows"][0]
    assert row["fruitdet"] is None
    assert any("exactly match" in issue for issue in row["issues"])

    run = _run(tmp_path, "fruitdet-record", method="supervised_20", seed=42, external=True)
    evaluation_path = run / "evaluations" / "external_test.json"
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    manifest_path = Path(payload["protocol"]["fruitdet_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["label_status"] = "pseudo"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_evidence = payload["protocol"]["fruitdet_manifest"]
    manifest_evidence["bytes"] = manifest_path.stat().st_size
    manifest_evidence["sha256"] = _sha(manifest_path)
    payload["protocol"]["fruitdet_dataset_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "dataset_yaml_sha256": payload["protocol"]["dataset_yaml_sha256"],
                "fruitdet_manifest_sha256": manifest_evidence["sha256"],
                "membership_sha256": manifest_evidence["membership_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    row = aggregate_results((run,))["rows"][0]
    assert row["fruitdet"] is None
    assert any("approved labeled external-test semantics" in issue for issue in row["issues"])


def test_main_group_requires_shared_split_and_fixed_test_dataset(tmp_path: Path) -> None:
    runs = [_run(tmp_path, f"trust{seed}", method="trust_main", seed=seed) for seed in (42, 3407, 2026)]
    record_path = runs[-1] / "run_record.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    record_path.unlink()
    write_run_record(RunRecord(run_id=payload["run_id"], status=payload["status"], config_snapshot=payload["config_snapshot"], split_fingerprint="c" * 64, command=tuple(payload["command"]), environment=payload["environment"], result=payload["result"]), record_path)
    summary = aggregate_results(runs)["summary"]["main_groups"]["trust_main"]
    assert summary["comparability"]["compatible"] is False
    assert summary["complete"] is False


def test_plot_sources_are_deterministically_taken_from_designated_final_trust_run(tmp_path: Path) -> None:
    final = _run(tmp_path, "ssod_trust_seed42", method="trust_main", seed=42, raw_plots=True)
    _run(tmp_path, "ssod_trust_seed3407", method="trust_main", seed=3407, raw_plots=True)
    _run(tmp_path, "ssod_trust_seed2026", method="trust_main", seed=2026, raw_plots=True)
    aggregate = aggregate_results((final, tmp_path / "ssod_trust_seed3407", tmp_path / "ssod_trust_seed2026"))
    sources, missing = evaluator_figure_sources(aggregate)
    assert sources["precision_recall"].read_bytes() == b"ssod_trust_seed42-precision_recall"
    assert sources["confusion"].read_bytes() == b"ssod_trust_seed42-confusion_matrix"
    assert not missing


def test_publish_package_is_atomic_and_non_overwriting(tmp_path: Path) -> None:
    runs = [_run(tmp_path, f"supervised{seed}", method="supervised_20", seed=seed) for seed in (42, 3407, 2026)]
    runs += [_run(tmp_path, f"trust{seed}", method="trust_main", seed=seed) for seed in (42, 3407, 2026)]
    aggregate = aggregate_results(runs)
    output = publish_result_package(aggregate, evaluate_acceptance(aggregate), tmp_path / "package")
    assert (output / "aggregate.json").is_file()
    assert (output / "tables" / "results.xlsx").is_file()
    assert len(list((output / "figures").glob("*.svg"))) == 4
    assert len(list((output / "figures").glob("*.missing.json"))) == 2
    assert verify_result_package(output)["protocol"] == "task18_result_package_v1"
    (output / "aggregate.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ResultAggregationError, match="digest differs"):
        verify_result_package(output)
    with pytest.raises(ResultAggregationError, match="already exists"):
        publish_result_package(aggregate, evaluate_acceptance(aggregate), output)


def test_publish_recomputes_acceptance_and_verifier_rejects_unlisted_or_linked_files(tmp_path: Path) -> None:
    runs = [_run(tmp_path, f"supervised{seed}", method="supervised_20", seed=seed) for seed in (42, 3407, 2026)]
    runs += [_run(tmp_path, f"trust{seed}", method="trust_main", seed=seed) for seed in (42, 3407, 2026)]
    aggregate = aggregate_results(runs)
    acceptance = json.loads(canonical_json(evaluate_acceptance(aggregate)))
    acceptance["status"] = "pass"
    with pytest.raises(ResultAggregationError, match="differs from the aggregate"):
        publish_result_package(aggregate, acceptance, tmp_path / "forged")

    output = publish_result_package(aggregate, evaluate_acceptance(aggregate), tmp_path / "package-extra")
    (output / "unlisted.txt").write_text("not sealed", encoding="utf-8")
    with pytest.raises(ResultAggregationError, match="file set differs"):
        verify_result_package(output)

    linked = publish_result_package(aggregate, evaluate_acceptance(aggregate), tmp_path / "package-link")
    try:
        (linked / "tables" / "linked.csv").symlink_to(linked / "tables" / "primary_results.csv")
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable in this Windows test environment: {error}")
    with pytest.raises(ResultAggregationError, match="symbolic link"):
        verify_result_package(linked)
