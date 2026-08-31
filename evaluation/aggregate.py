"""Immutable aggregation of training and held-out evaluation evidence.

The collector is intentionally conservative: a directory that cannot prove a
metric is retained as a visible row, but never contributes to a mean.  This
keeps the final report honest when a queue was interrupted or a run failed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.evaluation.detection_metrics import DetectionMetricsError, metrics_from_mapping
from fruit_ssod.training.run_record import RunRecordError, read_run_record
from fruit_ssod.training.supervised import file_evidence


class ResultAggregationError(ValueError):
    """Raised for invalid aggregation requests or immutable publication errors."""


METRIC_NAMES = ("map50", "map50_95", "precision", "recall", "f1")
MAIN_GROUPS = ("supervised_20", "trust_main")
FINAL_TRUST_FIGURE_SOURCE = {"method": "trust_main", "seed": 42, "run_id": "ssod_trust_seed42"}


def _problem(problem: str, cause: str, remediation: str) -> ResultAggregationError:
    return ResultAggregationError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _problem("aggregation contains a non-string object key", "JSON evidence keys would be coerced", "use canonical string keys only")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise _problem("aggregation contains a non-finite number", "NaN/infinity cannot be sealed in JSON", "record finite metric values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise _problem("aggregation contains a non-JSON value", type(value).__name__, "use JSON-safe evidence values")


def thaw(value: Any) -> Any:
    """Return a JSON-serializable deep copy of an immutable aggregate."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _checkpoint_digest(run_dir: Path) -> str:
    try:
        payload = json.loads((run_dir / "checkpoint_evidence.json").read_text(encoding="utf-8"))
        evidence = payload["best.pt"]
        digest = evidence["sha256"]
        relative = evidence["relative_path"]
        if relative != "weights/best.pt" or not _is_sha256(digest):
            raise ValueError("best.pt evidence is malformed")
        current = file_evidence(run_dir / "weights" / "best.pt", description="completed best checkpoint")
        if current["sha256"] != digest.lower() or current["bytes"] != evidence.get("bytes"):
            raise ValueError("best.pt differs from checkpoint evidence")
        return digest.lower()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint evidence cannot prove the completed best.pt: {error}") from error


def _verified_raw_evaluator_outputs(protocol: Mapping[str, Any], *, run_dir: Path, split: str) -> None:
    """Verify copied evaluator plots before a report can present them.

    The protocol stores absolute paths for the publisher, but each one must
    still resolve under this immutable run's ``evaluations/raw/<split>`` tree.
    This prevents a hand-edited envelope from pointing report generation at an
    arbitrary PNG with a convenient hash.
    """
    raw = protocol.get("raw_evaluator_outputs")
    if raw is None:
        return  # Backwards-compatible no-plot evidence; aggregation marks no plot.
    if not isinstance(raw, Mapping):
        raise ValueError("raw evaluator outputs is not an object")
    allowed = {"precision_recall", "confusion_matrix"}
    if any(key not in allowed for key in raw):
        raise ValueError("raw evaluator outputs contains an unsupported artifact kind")
    root = (run_dir / "evaluations" / "raw" / split).resolve(strict=False)
    for kind, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"raw evaluator {kind} evidence is not an object")
        path_value, relative, size, digest = entry.get("path"), entry.get("relative_path"), entry.get("bytes"), entry.get("sha256")
        expected_relative = f"evaluations/raw/{split}/"
        if not isinstance(path_value, str) or not isinstance(relative, str) or not relative.startswith(expected_relative) or Path(relative).name != Path(path_value).name:
            raise ValueError(f"raw evaluator {kind} path evidence is malformed")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not _is_sha256(digest):
            raise ValueError(f"raw evaluator {kind} byte/SHA-256 evidence is malformed")
        try:
            actual_path = Path(path_value).resolve(strict=True)
            actual_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise ValueError(f"raw evaluator {kind} path is outside the sealed run output") from error
        actual = file_evidence(actual_path, description=f"sealed raw evaluator {kind}")
        if actual["bytes"] != size or actual["sha256"] != digest.lower():
            raise ValueError(f"raw evaluator {kind} differs from sealed SHA-256/byte evidence")


def _verified_fruitdet_manifest(protocol: Mapping[str, Any]) -> list[int]:
    """Verify source-specific external-test provenance held beside the result."""
    expected_ids = [0, 1, 2, 3]
    expected_names = ["Apple", "Banana", "Orange", "Strawberry"]
    expected_external_protocol = {
        "protocol_id": "fruitdet_external_mapped_v1",
        "mapping_source": "limited_external_set",
        "mapped_class_ids": expected_ids,
        "mapped_class_names": expected_names,
    }
    # The envelope itself is part of the external-test claim.  Checking only
    # that a list is a subset (or only that a manifest names FruitDet) would
    # let a hand-edited one-class evaluation be presented as the approved
    # four-class FruitDet protocol.
    if (
        protocol.get("external_protocol") != expected_external_protocol
        or protocol.get("mapped_class_ids") != expected_ids
        or protocol.get("mapped_class_names") != expected_names
        or protocol.get("mapping_source") != "limited_external_set"
    ):
        raise ValueError("external test protocol does not exactly match the approved FruitDet four-class contract")
    evidence = protocol.get("fruitdet_manifest")
    if not isinstance(evidence, Mapping):
        raise ValueError("external test lacks sealed FruitDet importer manifest evidence")
    path_value, size, digest = evidence.get("path"), evidence.get("bytes"), evidence.get("sha256")
    if not isinstance(path_value, str) or isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not _is_sha256(digest):
        raise ValueError("FruitDet importer manifest SHA-256/byte evidence is malformed")
    path = Path(path_value)
    actual = file_evidence(path, description="sealed FruitDet importer manifest")
    if actual["bytes"] != size or actual["sha256"] != digest.lower():
        raise ValueError("FruitDet importer manifest differs from sealed SHA-256/byte evidence")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"FruitDet importer manifest cannot be read: {error}") from error
    source = manifest.get("source") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("name") != "fruitdet"
        or manifest.get("category_mapping_source") != "limited_external_set"
        or manifest.get("split") != "external_test"
        or manifest.get("mapped_class_ids") != [0, 1, 2, 3]
        or manifest.get("mapped_class_names") != ["Apple", "Banana", "Orange", "Strawberry"]
    ):
        raise ValueError("FruitDet importer manifest does not prove the reviewed source/mapping/categories")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("FruitDet importer manifest has no accepted labeled records")
    observed_class_ids: set[int] = set()
    for row in records:
        if (
            not isinstance(row, Mapping)
            or row.get("source_dataset") != "fruitdet"
            or row.get("source") != "limited_external_set"
            or row.get("split") != "external_test"
            or row.get("label_status") != "labeled"
            or isinstance(row.get("class_id"), bool)
            or not isinstance(row.get("class_id"), int)
            or row.get("class_id") not in expected_ids
            or not isinstance(row.get("file_path"), str)
            or not row.get("file_path")
        ):
            raise ValueError("FruitDet importer manifest has a record outside the approved labeled external-test semantics")
        observed_class_ids.add(int(row["class_id"]))
    observed = sorted(observed_class_ids)
    names = list(DEFAULT_CLASS_REGISTRY.class_names)
    if protocol.get("observed_class_ids") != observed or protocol.get("observed_class_names") != [names[class_id] for class_id in observed]:
        raise ValueError("external test observed class scope differs from the sealed FruitDet records")
    # Recompute the dataset-membership binding from the YAML currently named
    # by the evaluation envelope.  A valid FruitDet manifest paired with an
    # arbitrary canonical YAML is not valid external-test evidence.
    data_value = protocol.get("dataset_yaml")
    dataset_digest = protocol.get("dataset_yaml_sha256")
    expected_fingerprint = protocol.get("fruitdet_dataset_fingerprint")
    expected_membership = evidence.get("membership_sha256")
    expected_count = evidence.get("member_count")
    if not isinstance(data_value, str) or not _is_sha256(dataset_digest) or not _is_sha256(expected_fingerprint) or not _is_sha256(expected_membership) or isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("FruitDet YAML membership/dataset fingerprint evidence is malformed")
    try:
        # Import locally to avoid making the core aggregator depend on CLI
        # startup at module import time; this helper is pure evidence parsing.
        from fruit_ssod.cli.evaluate_model import _canonical_dataset_evidence, fruitdet_dataset_binding
        data, effective, current_digest = _canonical_dataset_evidence(Path(data_value), protocol="external test")
        if current_digest != dataset_digest.lower():
            raise ValueError("external YAML digest differs from the sealed evaluation envelope")
        binding = fruitdet_dataset_binding(data, effective, current_digest, evidence)
    except (OSError, ValueError) as error:
        raise ValueError(f"FruitDet evaluated YAML cannot be rebound to the manifest: {error}") from error
    if binding["membership_sha256"] != expected_membership.lower() or binding["member_count"] != expected_count or binding["dataset_fingerprint"] != expected_fingerprint.lower():
        raise ValueError("FruitDet manifest membership/dataset fingerprint differs from the sealed evaluation evidence")
    return observed


def _external_metrics_from_mapping(value: Mapping[str, Any], *, observed_class_ids: list[int]) -> dict[str, Any]:
    """Decode partial-coverage FruitDet metrics without adding absent classes."""
    try:
        required = {"map50", "map50_95", "precision", "recall", "f1", "reported_class_ids", "per_class_ap50"}
        if set(value) != required:
            raise ValueError("external metric keys are incomplete or unexpected")
        reported = value["reported_class_ids"]
        per_class = value["per_class_ap50"]
        if reported != observed_class_ids:
            raise ValueError("reported class IDs differ from classes actually present in sealed FruitDet records")
        if not isinstance(per_class, Mapping) or set(per_class) != {str(class_id) for class_id in observed_class_ids}:
            raise ValueError("external per_class_ap50 keys differ from the reported class IDs")
        normalized: dict[str, Any] = {"reported_class_ids": list(reported), "per_class_ap50": {str(class_id): float(per_class[str(class_id)]) for class_id in observed_class_ids}}
        for name in METRIC_NAMES:
            metric = value[name]
            if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)) or not 0 <= float(metric) <= 1:
                raise ValueError(f"external {name} is not a finite normalized metric")
            normalized[name] = float(metric)
        if any(not math.isfinite(metric) or not 0 <= metric <= 1 for metric in normalized["per_class_ap50"].values()):
            raise ValueError("external per_class_ap50 contains a non-finite or unnormalized value")
        return {name: normalized[name] for name in ("map50", "map50_95", "precision", "recall", "f1", "reported_class_ids", "per_class_ap50")}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"external metrics do not match sealed FruitDet observed coverage: {error}") from error


def _read_evaluation(path: Path, *, record: Any, run_dir: Path, split: str) -> tuple[dict[str, Any] | None, list[str], Mapping[str, Any] | None]:
    """Validate one held-out evaluation envelope against its run evidence."""
    if not path.is_file():
        return None, [f"{split} evaluation is missing"], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("metrics"), Mapping) or not isinstance(payload.get("protocol"), Mapping):
            raise ValueError("evaluation has no metrics/protocol objects")
        protocol = payload["protocol"]
        if protocol.get("schema") != "fruit_ssod_evaluation_evidence_v1":
            raise ValueError("protocol schema is not fruit_ssod_evaluation_evidence_v1")
        if protocol.get("run_id") != record.run_id or protocol.get("split") != split:
            raise ValueError("protocol does not bind the requested run and split")
        if protocol.get("checkpoint_sha256") != _checkpoint_digest(run_dir):
            raise ValueError("protocol checkpoint digest differs from completion evidence")
        _verified_raw_evaluator_outputs(protocol, run_dir=run_dir, split=split)
        if split == "test":
            expected = record.config_snapshot.get("dataset_yaml_sha256")
            if not _is_sha256(expected) or protocol.get("dataset_yaml_sha256") != expected.lower():
                raise ValueError("test dataset digest differs from the frozen training dataset")
        if split == "external_test":
            observed = _verified_fruitdet_manifest(protocol)
            metrics = _external_metrics_from_mapping(payload["metrics"], observed_class_ids=observed)
        else:
            metrics = metrics_from_mapping(payload["metrics"]).mapping()
        if protocol.get("metrics_sha256") != _sha256_json(metrics):
            raise ValueError("protocol metrics digest differs from metric content")
        return metrics, [], protocol
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DetectionMetricsError, ValueError, KeyError, TypeError) as error:
        return None, [f"{path.name}: {error}"], None


def _method(snapshot: Mapping[str, Any]) -> str:
    role = snapshot.get("matrix_role")
    if isinstance(role, str) and role:
        return role
    budget = snapshot.get("label_budget_percent")
    if isinstance(budget, int) and not isinstance(budget, bool):
        return f"supervised_{budget}"
    return "unclassified"


def _external_mapped_metrics(metrics: Mapping[str, Any], protocol: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    """Return FruitDet-only mapped-class metrics; never silently use five classes."""
    if protocol is None:
        return None, ["external test protocol is unavailable"]
    external_protocol = protocol.get("external_protocol")
    if not isinstance(external_protocol, Mapping) or external_protocol.get("protocol_id") != "fruitdet_external_mapped_v1" or external_protocol.get("mapping_source") != "limited_external_set":
        return None, ["external test lacks the sealed FruitDet mapped protocol"]
    raw_ids = protocol.get("mapped_class_ids")
    if not isinstance(raw_ids, list) or not raw_ids or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_ids):
        return None, ["external test lacks explicit mapped_class_ids; it cannot be reported as FruitDet evidence"]
    expected_ids = set(DEFAULT_CLASS_REGISTRY.class_ids)
    if raw_ids != external_protocol.get("mapped_class_ids") or len(set(raw_ids)) != len(raw_ids) or not set(raw_ids).issubset(expected_ids):
        return None, ["external mapped_class_ids are duplicate or outside the canonical fruit classes"]
    observed_ids = protocol.get("observed_class_ids")
    if not isinstance(observed_ids, list) or not observed_ids or any(isinstance(item, bool) or not isinstance(item, int) for item in observed_ids) or observed_ids != sorted(set(observed_ids)) or not set(observed_ids).issubset(set(raw_ids)):
        return None, ["external metrics lacks a valid observed FruitDet class scope"]
    ap = metrics.get("per_class_ap50")
    if metrics.get("reported_class_ids") != observed_ids or not isinstance(ap, Mapping) or set(ap) != {str(class_id) for class_id in observed_ids}:
        return None, ["external metrics lacks per_class_ap50"]
    mapped_ap = {str(class_id): ap[str(class_id)] for class_id in observed_ids}
    names = list(DEFAULT_CLASS_REGISTRY.class_names)
    return {
        "mapped_class_ids": list(observed_ids),
        "mapped_class_names": [names[class_id] for class_id in observed_ids],
        "declared_mapped_class_ids": list(raw_ids),
        "declared_mapped_class_names": [names[class_id] for class_id in raw_ids],
        "per_class_ap50": mapped_ap,
        "mapped_mean_ap50": sum(float(value) for value in mapped_ap.values()) / len(mapped_ap),
    }, []


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for group in MAIN_GROUPS:
        eligible = [row for row in rows if row["method"] == group and row["evaluation_status"] == "complete" and isinstance(row["primary_test"], Mapping)]
        values = {metric: [float(row["primary_test"][metric]) for row in eligible] for metric in METRIC_NAMES}
        per_class = {
            str(class_id): [float(row["primary_test"]["per_class_ap50"][str(class_id)]) for row in eligible]
            for class_id in DEFAULT_CLASS_REGISTRY.class_ids
        }
        def stats(items: list[float]) -> dict[str, Any]:
            if not items:
                return {"mean": None, "std": None, "n": 0}
            mean = sum(items) / len(items)
            # Sample SD makes the three-seed uncertainty explicit; a singleton
            # has no estimate instead of a misleading 0.0.
            std = None if len(items) < 2 else math.sqrt(sum((item - mean) ** 2 for item in items) / (len(items) - 1))
            return {"mean": mean, "std": std, "n": len(items)}
        observed_seeds = sorted(row["seed"] for row in eligible if isinstance(row["seed"], int) and not isinstance(row["seed"], bool))
        split_fingerprints = sorted({str(row["split_fingerprint"]) for row in eligible})
        dataset_digests = sorted({str(row["primary_test_protocol"]["dataset_yaml_sha256"]) for row in eligible if isinstance(row.get("primary_test_protocol"), Mapping)})
        checkpoint_bound = all(isinstance(row.get("primary_test_protocol"), Mapping) and _is_sha256(row["primary_test_protocol"].get("checkpoint_sha256")) for row in eligible)
        compatible = len(split_fingerprints) == 1 and len(dataset_digests) == 1 and checkpoint_bound
        groups[group] = {
            "required_seeds": [42, 3407, 2026],
            "observed_seeds": observed_seeds,
            "complete": len(eligible) == 3 and observed_seeds == [42, 2026, 3407] and compatible,
            "comparability": {"split_fingerprints": split_fingerprints, "fixed_test_dataset_sha256": dataset_digests, "checkpoint_bound_to_each_run": checkpoint_bound, "compatible": compatible},
            "metrics": {metric: stats(items) for metric, items in values.items()},
            "per_class_ap50": {class_id: stats(items) for class_id, items in per_class.items()},
        }
    fruitdet_groups: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        evidence = [row["fruitdet"] for row in rows if row["method"] == method and isinstance(row["fruitdet"], Mapping)]
        mapped_ids = sorted({class_id for item in evidence for class_id in item["mapped_class_ids"]})
        values = [float(item["mapped_mean_ap50"]) for item in evidence]
        mean = sum(values) / len(values) if values else None
        std = None if len(values) < 2 else math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        fruitdet_groups[method] = {
            "n": len(evidence), "mapped_class_ids": mapped_ids,
            "mapped_mean_ap50": {"mean": mean, "std": std, "n": len(values)},
            "per_class_ap50": {str(class_id): {"mean": (sum(float(item["per_class_ap50"][str(class_id)]) for item in evidence if str(class_id) in item["per_class_ap50"]) / sum(str(class_id) in item["per_class_ap50"] for item in evidence)) if any(str(class_id) in item["per_class_ap50"] for item in evidence) else None, "n": sum(str(class_id) in item["per_class_ap50"] for item in evidence)} for class_id in mapped_ids},
        }
    return {
        "submitted_runs": len(rows),
        "complete_training_runs": sum(row["status"] == "complete" for row in rows),
        "complete_evaluations": sum(row["evaluation_status"] == "complete" for row in rows),
        "failed_runs": sum(row["status"] == "failed" for row in rows),
        "incomplete_or_unreadable_runs": sum(row["evaluation_status"] != "complete" for row in rows),
        "main_groups": groups,
        "fruitdet_mapped_groups": fruitdet_groups,
        # This is a protocol designation, not a score-based selection.  It
        # prevents a report refresh from choosing whichever replication happened
        # to provide a plot or the largest validation number.
        "final_trust_figure_source": dict(FINAL_TRUST_FIGURE_SOURCE),
    }


def aggregate_results(run_directories: Iterable[Path | str]) -> Mapping[str, Any]:
    """Aggregate all supplied runs without suppressing invalid/incomplete evidence."""
    supplied = list(run_directories)
    if not supplied:
        raise _problem("no run directories were supplied", "aggregation has no evidence to inspect", "pass every required run directory with --run-dir")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in supplied:
        directory = Path(raw)
        key = str(directory.resolve(strict=False))
        if key in seen:
            raise _problem("duplicate run directory supplied", key, "supply each run artifact directory once")
        seen.add(key)
        row: dict[str, Any] = {"run_dir": key, "run_id": None, "status": "unreadable", "evaluation_status": "unreadable", "method": "unclassified", "seed": None, "label_budget_percent": None, "split_fingerprint": None, "validation": None, "primary_test": None, "primary_test_protocol": None, "fruitdet": None, "failure": None, "issues": []}
        try:
            record = read_run_record(directory / "run_record.json")
            snapshot = record.config_snapshot
            row.update({"run_id": record.run_id, "status": record.status, "method": _method(snapshot), "seed": snapshot.get("seed"), "label_budget_percent": snapshot.get("label_budget_percent"), "split_fingerprint": record.split_fingerprint, "failure": None if record.failure is None else thaw(record.failure)})
            if record.status == "complete":
                row["validation"] = metrics_from_mapping(record.result or {}).mapping()
                primary, primary_issues, primary_protocol = _read_evaluation(directory / "evaluations" / "test.json", record=record, run_dir=directory, split="test")
                row["primary_test"] = primary
                row["primary_test_protocol"] = None if primary_protocol is None else thaw(primary_protocol)
                row["issues"].extend(primary_issues)
                ext, ext_issues, protocol = _read_evaluation(directory / "evaluations" / "external_test.json", record=record, run_dir=directory, split="external_test")
                row["issues"].extend(ext_issues)
                if ext is not None:
                    fruitdet, mapping_issues = _external_mapped_metrics(ext, protocol)
                    row["fruitdet"] = fruitdet
                    row["issues"].extend(mapping_issues)
                row["evaluation_status"] = "complete" if primary is not None else "invalid_evaluation"
            else:
                row["issues"].append("run is not complete; validation and held-out metrics are unavailable")
                row["evaluation_status"] = "failed" if record.status == "failed" else "incomplete"
        except (RunRecordError, DetectionMetricsError, OSError, ValueError) as error:
            row["issues"].append(str(error))
        rows.append(row)
    rows.sort(key=lambda row: (str(row["method"]), str(row["seed"]), str(row["run_id"]), str(row["run_dir"])))
    result = {"schema_version": "1.0", "protocol": "task18_result_aggregation_v1", "canonical_classes": list(DEFAULT_CLASS_REGISTRY.class_names), "rows": rows, "summary": _summary(rows)}
    return _freeze(result)
