"""Deterministic Task 12 supervised-reference matrix support.

This module deliberately owns only the small supervised reference matrix.  It
does not start training and it never converts a missing or failed run into a
successful result.  The fuller cross-method statistical aggregation belongs to
Task 18; this early collector is the evidence needed to decide whether the
100% reference is credible enough to proceed to pseudo-label experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.evaluation.detection_metrics import DetectionMetricsError, metrics_from_mapping
from fruit_ssod.training.run_record import RunRecord, RunRecordError, read_run_record


class SupervisedMatrixError(ValueError):
    """Raised when reference-matrix metadata would cease to be reproducible."""


def _problem(problem: str, cause: str, remediation: str) -> SupervisedMatrixError:
    return SupervisedMatrixError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _canonical_json(value: Any) -> str:
    """Serialize evidence exactly once for content-addressed comparisons."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metrics_sha256(metrics: Mapping[str, Any]) -> str:
    """Return the digest of the normalized framework-neutral metric mapping."""
    return hashlib.sha256(_canonical_json(metrics).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class SupervisedMatrixEntry:
    """One fixed label-budget/seed experiment required by the protocol."""

    label_budget_percent: int
    seed: int

    @property
    def experiment_name(self) -> str:
        return f"supervised_{self.label_budget_percent}_seed{self.seed}"

    @property
    def filename(self) -> str:
        return f"{self.experiment_name}.yaml"


# Keep this tuple in protocol order: the three repeated 20% baselines are
# adjacent, while the reference curve remains easy to read in the queue.
SUPERVISED_REFERENCE_MATRIX: tuple[SupervisedMatrixEntry, ...] = (
    SupervisedMatrixEntry(10, 42),
    SupervisedMatrixEntry(20, 42),
    SupervisedMatrixEntry(20, 3407),
    SupervisedMatrixEntry(20, 2026),
    SupervisedMatrixEntry(40, 42),
    SupervisedMatrixEntry(100, 42),
)

_TEMPLATE_REQUIRED_KEYS = frozenset(
    {
        "template_id",
        "experiment_name",
        "model_config",
        "dataset_yaml",
        "split_manifest",
        "artifact_root",
        "seed",
        "label_budget_percent",
        "epochs",
        "pretrained_weights",
        "initialization_policy",
    }
)
_TEMPLATE_ID = "supervised_reference_v1"
_SNAPSHOT_PROVENANCE_KEYS = frozenset(
    {
        "model_config_sha256",
        "dataset_yaml_sha256",
        "model_config_effective",
        "dataset_yaml_effective",
        "split_manifest",
        "canonical_classes",
    }
)


def matrix_entries() -> tuple[SupervisedMatrixEntry, ...]:
    """Return the immutable experimental queue in its published order."""
    return SUPERVISED_REFERENCE_MATRIX


def _read_yaml_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the UTF-8 YAML source and retry") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise _problem(f"{description} is not a string-keyed YAML object", "the YAML is empty or has an unsupported top-level shape", "use one key/value object")
    return payload


def _render_value(value: Any, entry: SupervisedMatrixEntry) -> Any:
    if isinstance(value, str):
        # `${FRUIT_SSOD_DATA_ROOT}` is intentionally preserved for the later
        # experiment loader.  Python's ``str.format`` would mistake it for a
        # template placeholder, so only protocol placeholders are replaced.
        unknown = set(re.findall(r"(?<!\$)\{([^{}]+)\}", value)).difference({"label_budget", "seed"})
        if unknown:
            raise _problem("supervised template has an invalid placeholder", repr(sorted(unknown)), "use only {label_budget} and {seed} placeholders")
        rendered = value.replace("{label_budget}", str(entry.label_budget_percent)).replace("{seed}", str(entry.seed))
        if value == "{label_budget}":
            return entry.label_budget_percent
        if value == "{seed}":
            return entry.seed
        return rendered
    if isinstance(value, list):
        return [_render_value(item, entry) for item in value]
    if isinstance(value, Mapping):
        return {key: _render_value(item, entry) for key, item in value.items()}
    return value


def load_reference_template(path: Path | str) -> dict[str, Any]:
    """Load and strictly validate the single canonical configuration source."""
    template = _read_yaml_mapping(Path(path), description="supervised reference template")
    missing = _TEMPLATE_REQUIRED_KEYS.difference(template)
    extra = set(template).difference(_TEMPLATE_REQUIRED_KEYS)
    if missing or extra:
        raise _problem(
            "supervised reference template keys do not match the protocol",
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}",
            "keep exactly the canonical Task 12 template keys",
        )
    if template["template_id"] != _TEMPLATE_ID:
        raise _problem("supervised reference template ID is unsupported", f"received {template['template_id']!r}", f"use {_TEMPLATE_ID!r}")
    if not isinstance(template["epochs"], int) or isinstance(template["epochs"], bool) or template["epochs"] <= 0:
        raise _problem("supervised reference template epochs are invalid", f"received {template['epochs']!r}", "set epochs to a positive integer")
    for key in _TEMPLATE_REQUIRED_KEYS.difference({"epochs", "initialization_policy"}):
        if not isinstance(template[key], str) or not template[key]:
            raise _problem("supervised reference template field is invalid", f"{key} is {template[key]!r}", "use a nonempty string for every template field")
    policy = template["initialization_policy"]
    if not isinstance(policy, Mapping) or set(policy) != {"policy_id", "model_initialization", "comparison_group"} or any(not isinstance(value, str) or not value for value in policy.values()):
        raise _problem("supervised reference template initialization policy is invalid", f"received {policy!r}", "declare the shared pretrained-weight policy with nonempty string fields")
    if policy["model_initialization"] != "shared_pretrained_weights":
        raise _problem("supervised reference template initialization policy is unsupported", f"received {policy['model_initialization']!r}", "use shared_pretrained_weights for comparable Teacher/Student experiments")
    return template


def render_reference_config(template: Mapping[str, Any], entry: SupervisedMatrixEntry) -> dict[str, Any]:
    """Render one config without relying on filesystem or mutable global state."""
    if set(template) != _TEMPLATE_REQUIRED_KEYS or template.get("template_id") != _TEMPLATE_ID:
        raise _problem("supervised reference template was not validated", "its schema or ID differs from the canonical source", "call load_reference_template before rendering")
    rendered = {key: _render_value(value, entry) for key, value in template.items()}
    if rendered["experiment_name"] != entry.experiment_name:
        raise _problem("template generated an unexpected experiment name", f"expected {entry.experiment_name!r}, got {rendered['experiment_name']!r}", "restore the canonical experiment_name placeholder")
    if rendered["seed"] != entry.seed or rendered["label_budget_percent"] != entry.label_budget_percent:
        raise _problem("template did not preserve matrix seed or budget", "seed or label_budget_percent is not an exact placeholder", "use {seed} and {label_budget} as their complete field values")
    return rendered


def render_reference_matrix(template_path: Path | str) -> dict[str, dict[str, Any]]:
    """Generate every committed config payload deterministically from one YAML."""
    template = load_reference_template(template_path)
    return {entry.filename: render_reference_config(template, entry) for entry in matrix_entries()}


def validate_reference_configs(template_path: Path | str, config_directory: Path | str) -> tuple[Path, ...]:
    """Prove checked-in matrix configs exactly match the canonical template."""
    expected = render_reference_matrix(template_path)
    directory = Path(config_directory)
    validated: list[Path] = []
    for filename, expected_payload in expected.items():
        path = directory / filename
        actual = _read_yaml_mapping(path, description=f"supervised reference config {filename}")
        if actual != expected_payload:
            raise _problem(
                "supervised reference config diverges from the canonical template",
                f"{path} does not equal the deterministic rendering",
                "regenerate the config from supervised_reference_template.yaml instead of editing it independently",
            )
        validated.append(path)
    # Exploratory v2/v3/v13 continuations are deliberately kept beside the fixed
    # reference matrix but are not comparable reference entries.
    exploratory_prefixes = ("supervised_v2_", "supervised_v3_")
    unexpected = {path.name for path in directory.glob("supervised_*_seed*.yaml") if not path.name.startswith(exploratory_prefixes)}.difference(expected)
    if unexpected:
        raise _problem("unexpected supervised reference config is present", f"found {sorted(unexpected)!r}", "add it to the fixed matrix deliberately or remove it")
    return tuple(validated)


def conda_train_command(
    config_path: Path | str,
    *,
    conda_executable: str = "conda",
    environment_name: str = "fruit-ssod",
    dry_run: bool = False,
    run_id: str | None = None,
) -> tuple[str, ...]:
    """Return an argument-vector safe for PowerShell's ``& executable @args``.

    No shell string is composed, so spaces in a Conda path or config path are
    preserved as a single argument and cannot be interpreted as shell syntax.
    """
    if not isinstance(conda_executable, str) or not conda_executable.strip() or "\x00" in conda_executable:
        raise _problem("Conda executable is invalid", f"received {conda_executable!r}", "provide a nonempty executable path or command name")
    if not isinstance(environment_name, str) or not environment_name.strip() or "\x00" in environment_name:
        raise _problem("Conda environment name is invalid", f"received {environment_name!r}", "provide the named Conda environment")
    config = str(Path(config_path))
    if not config or "\x00" in config:
        raise _problem("supervised config path is invalid", f"received {config_path!r}", "provide a nonempty YAML path")
    command = [conda_executable, "run", "--no-capture-output", "--name", environment_name, "python", "-m", "fruit_ssod.cli.train_supervised", "--config", config]
    if dry_run:
        command.append("--dry-run")
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id or "\x00" in run_id:
            raise _problem("run ID is invalid", f"received {run_id!r}", "use a nonempty safe run ID")
        command.extend(("--run-id", run_id))
    return tuple(command)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _checkpoint_sha256_or_error(directory: Path) -> str:
    """Read the completion-time identity of the exact test checkpoint."""
    path = directory / "checkpoint_evidence.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint evidence cannot be read: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("best.pt"), Mapping):
        raise ValueError("checkpoint evidence has no best.pt object")
    digest = payload["best.pt"].get("sha256")
    if not _is_sha256(digest):
        raise ValueError("checkpoint evidence best.pt.sha256 is malformed")
    return digest.lower()


def _fixed_test_or_issue(path: Path, record: RunRecord, directory: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a test result as immutable evidence, not merely plausible metrics.

    A result is only useful to the matrix gate when it binds its metrics to the
    specific completed run, its recorded best checkpoint, the frozen dataset
    YAML, and the requested held-out split.  This prevents a copied JSON file
    (or a different checkpoint's score) from becoming credibility evidence.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("top-level JSON value is not an object")
        metrics_payload = payload.get("metrics")
        protocol = payload.get("protocol")
        if not isinstance(metrics_payload, Mapping):
            raise TypeError("metrics is not an object")
        if not isinstance(protocol, Mapping):
            raise TypeError("protocol is not an object")
        metrics = metrics_from_mapping(metrics_payload).mapping()
        if protocol.get("run_id") != record.run_id:
            raise ValueError("protocol.run_id does not bind to run_record.json")
        if protocol.get("split") != "test":
            raise ValueError("protocol.split is not 'test'")
        expected_checkpoint = _checkpoint_sha256_or_error(directory)
        if protocol.get("checkpoint_sha256") != expected_checkpoint:
            raise ValueError("protocol.checkpoint_sha256 does not match completion checkpoint evidence")
        expected_dataset = record.config_snapshot.get("dataset_yaml_sha256")
        if not _is_sha256(expected_dataset):
            raise ValueError("run record dataset_yaml_sha256 is malformed")
        if protocol.get("dataset_yaml_sha256") != expected_dataset.lower():
            raise ValueError("protocol.dataset_yaml_sha256 does not match the frozen training dataset")
        expected_metrics = _metrics_sha256(metrics)
        if protocol.get("metrics_sha256") != expected_metrics:
            raise ValueError("protocol.metrics_sha256 does not match the metrics content")
        return metrics, None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, DetectionMetricsError) as error:
        return None, f"{path.name}: {error}"


def _entry_from_record(record: RunRecord) -> SupervisedMatrixEntry | None:
    snapshot = record.config_snapshot
    budget = snapshot.get("label_budget_percent")
    seed = snapshot.get("seed")
    if isinstance(budget, bool) or isinstance(seed, bool) or not isinstance(budget, int) or not isinstance(seed, int):
        return None
    entry = SupervisedMatrixEntry(budget, seed)
    return entry if entry in SUPERVISED_REFERENCE_MATRIX else None


def _canonical_protocol_issues(record: RunRecord, entry: SupervisedMatrixEntry | None) -> list[str]:
    """Return reasons why a row cannot influence the 100% credibility gate.

    Records remain visible regardless of these checks.  They merely cannot
    impersonate the named, template-derived actual reference run with a high
    score.  In particular, Task 12 dry-run UUID records are intentionally not
    gate evidence.
    """
    snapshot = record.config_snapshot
    issues: list[str] = []
    if entry is None:
        return ["run is not one of the fixed Task 12 supervised reference configurations"]
    if record.run_id != entry.experiment_name:
        issues.append("run_id does not equal the required fixed matrix experiment name")
    if snapshot.get("experiment_name") != entry.experiment_name:
        issues.append("config snapshot experiment_name does not match its label budget and seed")
    if snapshot.get("matrix_template_id") != _TEMPLATE_ID:
        issues.append("config snapshot is not proven to come from supervised_reference_v1")
    missing_provenance = sorted(key for key in _SNAPSHOT_PROVENANCE_KEYS if key not in snapshot)
    if missing_provenance:
        issues.append(f"config snapshot lacks required provenance fields: {missing_provenance!r}")
    for key in ("model_config_sha256", "dataset_yaml_sha256"):
        value = snapshot.get(key)
        if key in snapshot and (not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower())):
            issues.append(f"config snapshot {key} is not a SHA-256 digest")
    # RunRecord deep-freezes JSON arrays as tuples; compare to the frozen
    # canonical representation rather than accidentally rejecting the real
    # runner's own immutable snapshot.
    expected_names = tuple(DEFAULT_CLASS_REGISTRY.class_names)
    if snapshot.get("canonical_classes") != expected_names:
        issues.append("config snapshot canonical_classes does not exactly match the canonical five-class registry")
    for evidence_key in ("model_config_effective", "dataset_yaml_effective"):
        evidence = snapshot.get(evidence_key)
        if not isinstance(evidence, Mapping):
            # The missing-provenance diagnostic above remains useful; this
            # distinct message makes a present-but-malformed snapshot clear.
            if evidence_key in snapshot:
                issues.append(f"config snapshot {evidence_key} is not a YAML object")
            continue
        if evidence.get("names") != expected_names:
            issues.append(f"config snapshot {evidence_key}.names does not exactly match the canonical five-class registry")
    return issues


def aggregate_supervised_matrix(run_directories: Iterable[Path | str]) -> dict[str, Any]:
    """Collect validation and sealed fixed-test evidence without dropping rows.

    A bad record does not abort aggregation: it becomes a visible ``unreadable``
    row.  This prevents a failed run from disappearing merely because result
    presentation is being refreshed.
    """
    rows: list[dict[str, Any]] = []
    seen_directories: set[str] = set()
    for raw_directory in run_directories:
        directory = Path(raw_directory)
        key = str(directory.resolve(strict=False))
        if key in seen_directories:
            raise _problem("duplicate run directory supplied", key, "list each matrix run directory only once")
        seen_directories.add(key)
        row: dict[str, Any] = {"run_dir": key, "status": "unreadable", "run_id": None, "experiment_name": None, "label_budget_percent": None, "seed": None, "split_fingerprint": None, "validation": None, "fixed_test": None, "failure": None, "canonical_protocol": False, "issues": []}
        try:
            record = read_run_record(directory / "run_record.json")
        except RunRecordError as error:
            row["issues"].append(str(error))
            rows.append(row)
            continue
        entry = _entry_from_record(record)
        row.update({
            "run_id": record.run_id,
            "status": record.status,
            "experiment_name": record.config_snapshot.get("experiment_name"),
            "label_budget_percent": record.config_snapshot.get("label_budget_percent"),
            "seed": record.config_snapshot.get("seed"),
            "split_fingerprint": record.split_fingerprint,
            "failure": None if record.failure is None else dict(record.failure),
        })
        protocol_issues = _canonical_protocol_issues(record, entry)
        row["canonical_protocol"] = not protocol_issues
        row["issues"].extend(protocol_issues)
        if record.status == "complete":
            try:
                row["validation"] = metrics_from_mapping(record.result or {}).mapping()
            except DetectionMetricsError as error:  # Defensive: read_run_record already rejects this.
                row["issues"].append(f"validation result: {error}")
        elif record.status != "failed":
            row["issues"].append("run is not complete; validation and fixed-test metrics are unavailable")
        test_path = directory / "evaluations" / "test.json"
        if test_path.exists():
            test_metrics, issue = _fixed_test_or_issue(test_path, record, directory)
            row["fixed_test"] = test_metrics
            if issue is not None:
                row["issues"].append(issue)
        elif record.status == "complete":
            row["issues"].append("fixed-test evaluation is missing")
        rows.append(row)

    # One named matrix cell may have one and only one scientific run.  Two
    # directories asserting the same (budget, seed, run ID) indicate copied
    # or competing evidence, so neither is allowed to influence the gate.
    identities: dict[tuple[object, object, object], list[dict[str, Any]]] = {}
    for row in rows:
        identity = (row["label_budget_percent"], row["seed"], row["run_id"])
        if (
            isinstance(identity[0], int) and not isinstance(identity[0], bool)
            and isinstance(identity[1], int) and not isinstance(identity[1], bool)
            and isinstance(identity[2], str) and identity[2]
        ):
            identities.setdefault(identity, []).append(row)
    for identity, duplicates in identities.items():
        if len(duplicates) > 1:
            issue = f"duplicate canonical matrix identity supplied: budget={identity[0]!r}, seed={identity[1]!r}, run_id={identity[2]!r}"
            for row in duplicates:
                row["canonical_protocol"] = False
                row["issues"].append(issue)

    protocol_order = {(entry.label_budget_percent, entry.seed): index for index, entry in enumerate(SUPERVISED_REFERENCE_MATRIX)}
    rows.sort(
        key=lambda row: (
            protocol_order.get((row["label_budget_percent"], row["seed"]), len(protocol_order)),
            str(row["experiment_name"]), str(row["run_id"]), str(row["run_dir"]),
        )
    )
    hundred_percent = [
        row for row in rows
        if row["label_budget_percent"] == 100 and row["status"] == "complete" and row["canonical_protocol"]
    ]
    test_scores = [row["fixed_test"]["map50"] for row in hundred_percent if isinstance(row["fixed_test"], Mapping)]
    upper_bound = max(test_scores) if test_scores else None
    investigation_required = upper_bound is not None and upper_bound < 0.85
    if upper_bound is None:
        upper_bound_status = "missing_fixed_test_evidence"
        upper_bound_message = "The 100% reference has no valid fixed-test mAP@0.5; do not claim the upper bound is credible."
    elif investigation_required:
        upper_bound_status = "data_quality_investigation_required"
        upper_bound_message = "The 100% fixed-test mAP@0.5 is below 0.85; investigate data quality before continuing."
    else:
        upper_bound_status = "credible"
        upper_bound_message = "The available 100% fixed-test mAP@0.5 meets the 0.85 credibility screen."
    return {
        "schema_version": "1.0",
        "protocol": "task12_supervised_reference_matrix",
        "rows": rows,
        "summary": {
            "submitted_runs": len(rows),
            "complete_runs": sum(row["status"] == "complete" for row in rows),
            "failed_runs": sum(row["status"] == "failed" for row in rows),
            "noncomplete_runs": sum(row["status"] != "complete" for row in rows),
        },
        "upper_bound_gate": {
            "threshold_map50": 0.85,
            "fixed_test_map50": upper_bound,
            "status": upper_bound_status,
            "data_quality_investigation_required": investigation_required,
            "message": upper_bound_message,
        },
    }


def write_supervised_matrix_aggregate(result: Mapping[str, Any], output: Path | str) -> Path:
    """Atomically publish an aggregation once without replacing evidence.

    The completed JSON is synced to a temporary sibling first.  A hard-link
    publication is an exclusive operation on the destination name, unlike
    ``replace`` which would overwrite an aggregate created by another process.
    The temporary artifact is removed on every failure path.
    """
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise _problem("supervised matrix aggregate already exists", str(destination), "preserve the original output or choose a new output path")
    try:
        encoded = _canonical_json(result) + "\n"
    except (TypeError, ValueError) as error:
        raise _problem("supervised matrix aggregate cannot be serialized", str(error), "use a JSON-safe aggregation result") from error
    temporary: Path | None = None
    try:
        ancestor = destination.parent
        while ancestor != ancestor.parent:
            if ancestor.exists() and not ancestor.is_dir():
                raise OSError(f"aggregate parent ancestor is not a directory: {ancestor}")
            ancestor = ancestor.parent
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.parent.is_dir():
            raise OSError(f"aggregate parent is not a directory: {destination.parent}")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # ``link`` fails atomically if another worker already published this
        # target.  It also keeps the final file whole: the linked inode was
        # fully fsynced before publication.
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
    except FileExistsError as error:
        raise _problem("supervised matrix aggregate already exists", str(destination), "preserve the original output or choose a new output path") from error
    except OSError as error:
        raise _problem("supervised matrix aggregate cannot be written", str(error), "choose a writable path and JSON-safe aggregation result") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Do not hide the original publication error.  A future run
                # uses a distinct temporary name and still cannot overwrite
                # the requested evidence destination.
                pass
    return destination
