"""CLI fixture tests which never instantiate a real Ultralytics model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fruit_ssod.cli.evaluate_model as evaluate_module
from fruit_ssod.cli.evaluate_model import _capture_evaluator_artifacts, _external_metric_mapping, main as evaluate_main
from fruit_ssod.cli.train_supervised import main as train_main
from fruit_ssod.evaluation.detection_metrics import DetectionMetrics
from fruit_ssod.training.run_record import complete_run_record, create_run_record, write_run_record
from fruit_ssod.training.supervised import load_supervised_experiment


def _experiment(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\ntrain: train\nval: val\ntest: test\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"fingerprints": {"split_protocol": "c" * 64}}), encoding="utf-8")
    model = tmp_path / "model.yaml"
    model.write_text("model: yolov8s.yaml\nimage_size: 640\namp: true\nbatch: 4\n", encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment_name: supervised_fixture",
                f"model_config: '{model.as_posix()}'",
                f"dataset_yaml: '{dataset.as_posix()}'",
                f"split_manifest: '{split.as_posix()}'",
                f"artifact_root: '{(tmp_path / 'artifacts').as_posix()}'",
                "seed: 42",
                "patience: 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_cpu_fixture_dry_run_writes_reproducible_record_without_weights(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _experiment(tmp_path)

    assert train_main(["--config", str(config), "--dry-run", "--epochs", "1", "--batch", "2", "--device", "cpu"]) == 0

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "dry_run"
    assert record["config_snapshot"]["image_size"] == 640
    assert record["config_snapshot"]["amp"] is True
    assert record["config_snapshot"]["batch"] == 2
    assert record["config_snapshot"]["patience"] == 7
    assert record["split_fingerprint"] == "c" * 64
    assert set(record["config_snapshot"]["dataset_paths"]) == {"train", "val", "test"}
    assert len(record["config_snapshot"]["dataset_yaml_sha256"]) == 64
    assert not (run_dir / "weights" / "best.pt").exists()


def test_test_evaluation_refuses_a_run_that_is_not_complete(tmp_path: Path) -> None:
    config = _experiment(tmp_path)
    assert train_main(["--config", str(config), "--dry-run", "--device", "cpu"]) == 0
    run_root = tmp_path / "artifacts" / "runs"
    run_dir = next(run_root.iterdir())

    with pytest.raises(SystemExit) as status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--dry-run"])

    assert status.value.code == 2


def _completed_run(tmp_path: Path) -> tuple[Path, Path]:
    config = _experiment(tmp_path)
    experiment = load_supervised_experiment(config)
    run_dir = experiment.artifact_root / "runs" / "completed-fixture"
    run_dir.mkdir(parents=True)
    running = create_run_record(
        config_snapshot=experiment.snapshot(), split_fingerprint="c" * 64,
        command=("python", "-m", "fruit_ssod.cli.train_supervised"), environment={"python": "3.10"}, run_id="completed-fixture",
    )
    result = DetectionMetrics(0.81, 0.61, 0.8, 0.7, 0.7467, {index: 0.8 for index in range(5)}).mapping()
    write_run_record(complete_run_record(running, result), run_dir / "run_record.json")
    weights = run_dir / "weights"
    weights.mkdir()
    best = weights / "best.pt"
    best.write_bytes(b"trusted checkpoint")
    (run_dir / "checkpoint_evidence.json").write_text(
        json.dumps(
            {
                "best.pt": {
                    "relative_path": "weights/best.pt",
                    "bytes": best.stat().st_size,
                    "sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return run_dir, experiment.dataset_yaml


def _fruitdet_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "fruitdet_manifest.json"
    images = tmp_path / "fruitdet-images"
    images.mkdir()
    members: list[Path] = []
    for index in range(4):
        image = images / f"fruit-{index}.jpg"
        image.write_bytes(f"fixture-{index}".encode())
        members.append(image)
    (tmp_path / "fruitdet_images.txt").write_text("\n".join(str(member) for member in members) + "\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    return path


def test_heldout_test_refuses_data_override_and_snapshot_yaml_change(tmp_path: Path) -> None:
    run_dir, dataset = _completed_run(tmp_path)
    override = tmp_path / "override.yaml"
    override.write_text(dataset.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(SystemExit) as override_status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--data", str(override), "--dry-run"])
    assert override_status.value.code == 2

    dataset.write_text("path: changed\ntrain: train\nval: val\ntest: test\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as changed_status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--dry-run"])
    assert changed_status.value.code == 2


def test_external_test_requires_canonical_yaml_and_records_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, _ = _completed_run(tmp_path)
    invalid = tmp_path / "external-invalid.yaml"
    invalid.write_text("path: .\ntrain: train\nval: val\ntest: test\nnames: [Apple, Banana, Orange, Strawberry, pear]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as invalid_status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(invalid), "--dry-run"])
    assert invalid_status.value.code == 2

    manifest = _fruitdet_manifest(tmp_path)
    external = tmp_path / "external.yaml"
    external.write_text("path: .\ntrain: fruitdet_images.txt\nval: fruitdet_images.txt\ntest: fruitdet_images.txt\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    metric_mapping = {
        "map50": 0.82, "map50_95": 0.62, "precision": 0.81, "recall": 0.71, "f1": 0.7567,
        "reported_class_ids": [0, 1, 2, 3],
        "per_class_ap50": {str(index): 0.8 for index in range(4)},
    }
    monkeypatch.setattr(evaluate_module, "_evaluate", lambda *_args, **_kwargs: metric_mapping)

    assert evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(external), "--fruitdet-manifest", str(manifest)]) == 0
    output = json.loads((run_dir / "evaluations" / "external_test.json").read_text(encoding="utf-8"))
    assert output["metrics"]["map50"] == 0.82
    assert output["protocol"]["split"] == "external_test"
    assert output["protocol"]["run_id"] == "completed-fixture"
    assert len(output["protocol"]["checkpoint_sha256"]) == 64
    assert len(output["protocol"]["metrics_sha256"]) == 64
    assert output["protocol"]["dataset_yaml"] == str(external.resolve())
    assert len(output["protocol"]["dataset_yaml_sha256"]) == 64
    assert output["protocol"]["external_protocol"] == {
        "protocol_id": "fruitdet_external_mapped_v1",
        "mapping_source": "limited_external_set",
        "mapped_class_ids": [0, 1, 2, 3],
        "mapped_class_names": ["Apple", "Banana", "Orange", "Strawberry"],
    }
    assert output["protocol"]["mapped_class_ids"] == [0, 1, 2, 3]
    assert output["protocol"]["observed_class_ids"] == [0, 1, 2, 3]
    assert output["protocol"]["fruitdet_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(output["protocol"]["fruitdet_manifest"]["membership_sha256"]) == 64
    assert len(output["protocol"]["fruitdet_dataset_fingerprint"]) == 64


def test_external_metric_serializer_preserves_only_actual_observed_classes() -> None:
    metrics = SimpleNamespace(
        box=SimpleNamespace(
            all_ap=[[0.41, 0.30], [0.52, 0.39]], ap_class_index=[0, 1],
            map50=0.465, map=0.345, mp=0.50, mr=0.40,
        )
    )

    serialized = _external_metric_mapping(metrics, expected_class_ids=[0, 1])

    assert serialized["reported_class_ids"] == [0, 1]
    assert serialized["per_class_ap50"] == {"0": 0.41, "1": 0.52}
    assert "2" not in serialized["per_class_ap50"]
    with pytest.raises(Exception, match="reported class IDs"):
        _external_metric_mapping(metrics, expected_class_ids=[0, 1, 2])


def test_external_test_rejects_arbitrary_canonical_yaml_without_fruitdet_provenance(tmp_path: Path) -> None:
    run_dir, _ = _completed_run(tmp_path)
    external = tmp_path / "external.yaml"
    external.write_text("path: .\ntrain: train\nval: val\ntest: test\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")

    with pytest.raises(SystemExit) as status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(external), "--dry-run"])

    assert status.value.code == 2


def test_external_test_rejects_valid_fruitdet_manifest_paired_with_other_canonical_membership(tmp_path: Path) -> None:
    run_dir, _ = _completed_run(tmp_path)
    manifest = _fruitdet_manifest(tmp_path)
    foreign = tmp_path / "foreign.jpg"
    foreign.write_bytes(b"not-fruitdet")
    (tmp_path / "foreign.txt").write_text(str(foreign) + "\n", encoding="utf-8")
    external = tmp_path / "other-canonical.yaml"
    external.write_text("path: .\ntrain: foreign.txt\nval: foreign.txt\ntest: foreign.txt\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(external), "--fruitdet-manifest", str(manifest), "--dry-run"])
    assert status.value.code == 2


def test_external_test_rejects_duplicate_resolved_fruitdet_image_list_members(tmp_path: Path) -> None:
    run_dir, _ = _completed_run(tmp_path)
    manifest = _fruitdet_manifest(tmp_path)
    image_list = tmp_path / "fruitdet_images.txt"
    first = image_list.read_text(encoding="utf-8").splitlines()[0]
    image_list.write_text(image_list.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
    external = tmp_path / "external.yaml"
    external.write_text("path: .\ntrain: fruitdet_images.txt\nval: fruitdet_images.txt\ntest: fruitdet_images.txt\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(external), "--fruitdet-manifest", str(manifest), "--dry-run"])
    assert status.value.code == 2


def test_external_test_rejects_duplicate_absolute_and_relative_inline_members(tmp_path: Path) -> None:
    run_dir, _ = _completed_run(tmp_path)
    manifest = _fruitdet_manifest(tmp_path)
    members = sorted((tmp_path / "fruitdet-images").glob("*.jpg"))
    # Both spellings resolve to the same first image after the YAML's path
    # root is applied.  A set-only membership comparison would hide this.
    inline_members = [members[0].name, str(members[0].resolve())] + [member.name for member in members[1:]]
    external = tmp_path / "inline-duplicate.yaml"
    external.write_text(
        "path: fruitdet-images\n"
        "train: ../fruitdet_images.txt\n"
        "val: ../fruitdet_images.txt\n"
        f"test: {json.dumps(inline_members)}\n"
        "names: [Apple, Banana, Orange, Strawberry, Pineapple]\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(external), "--fruitdet-manifest", str(manifest), "--dry-run"])
    assert status.value.code == 2


def test_ultralytics_raw_plots_are_copied_and_hashed_into_run_evidence(tmp_path: Path) -> None:
    source = tmp_path / "framework-output"
    source.mkdir()
    (source / "PR_curve.png").write_bytes(b"real-framework-pr-pixels")
    (source / "confusion_matrix.png").write_bytes(b"real-framework-confusion-pixels")
    destination = tmp_path / "run" / "evaluations" / "raw" / "test"

    evidence = _capture_evaluator_artifacts(source, destination, split="test")

    assert set(evidence) == {"precision_recall", "confusion_matrix"}
    for item in evidence.values():
        copied = Path(item["path"])
        assert copied.is_file()
        assert item["relative_path"].startswith("evaluations/raw/test/")
        assert item["bytes"] == copied.stat().st_size
        assert item["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()


def test_heldout_evaluation_rejects_missing_or_modified_checkpoint_evidence(tmp_path: Path) -> None:
    run_dir, dataset = _completed_run(tmp_path)
    evidence = run_dir / "checkpoint_evidence.json"
    evidence.unlink()
    with pytest.raises(SystemExit) as missing_status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--dry-run"])
    assert missing_status.value.code == 2

    modified_root = tmp_path / "modified"
    modified_root.mkdir()
    run_dir, _ = _completed_run(modified_root)
    (run_dir / "weights" / "best.pt").write_bytes(b"post-complete modification")
    with pytest.raises(SystemExit) as modified_status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--dry-run"])
    assert modified_status.value.code == 2

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    run_dir, _ = _completed_run(empty_root)
    (run_dir / "weights" / "best.pt").write_bytes(b"")
    external = tmp_path / "external.yaml"
    external.write_text(dataset.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit) as empty_status:
        evaluate_main(["--run-dir", str(run_dir), "--split", "external_test", "--data", str(external), "--dry-run"])
    assert empty_status.value.code == 2
