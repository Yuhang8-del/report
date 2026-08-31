"""Immutable, JSON-safe provenance records for every training invocation.

The record is deliberately small and dependency-free.  Model frameworks may
change their result object shape, but the configuration, split fingerprint,
exact invocation and terminal metrics remain stable project evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fruit_ssod.evaluation.detection_metrics import DetectionMetricsError, metrics_from_mapping


class RunRecordError(ValueError):
    """Raised when a run record would lose provenance or overwrite evidence."""


_STATUSES = frozenset({"running", "complete", "failed", "dry_run"})
_TERMINAL_STATUSES = frozenset({"complete", "failed", "dry_run"})


def _problem(problem: str, cause: str, remediation: str) -> RunRecordError:
    return RunRecordError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _freeze_json(value: Any, *, field_name: str) -> Any:
    """Deep-copy JSON values and reject lossy/non-canonical inputs."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _problem(f"{field_name} has a non-string object key", "JSON keys would be coerced and might collide", "use unique string keys in the run metadata")
        return MappingProxyType({key: _freeze_json(item, field_name=field_name) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name=field_name) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise _problem(f"{field_name} contains NaN or infinity", "non-finite values are not portable JSON evidence", "store finite metric and configuration values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise _problem(f"{field_name} is not JSON-compatible", f"unsupported value type {type(value).__name__}", "use objects, arrays, strings, finite numbers, booleans, and null only")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_thaw_json(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not run_id or len(run_id) > 160:
        raise _problem("run_id is missing or malformed", "a run ID must be a short nonempty string", "supply a stable name or let the runner generate one")
    if any(character in run_id for character in "\\/:*?\"<>|") or run_id in {".", ".."}:
        raise _problem("run_id is unsafe for an artifact path", "the ID contains a path separator or Windows-reserved character", "use letters, numbers, dashes, underscores, and periods only")
    return run_id


def _validate_fingerprint(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise _problem("split fingerprint is not a SHA-256 hex digest", "the Task 8 split manifest is missing or malformed", "regenerate deterministic split artifacts and pass their split_protocol fingerprint")
    return value.lower()


@dataclass(frozen=True)
class RunRecord:
    """A deeply immutable record whose ID/config/split never change in place."""

    run_id: str
    status: str
    config_snapshot: Mapping[str, Any]
    split_fingerprint: str
    command: tuple[str, ...]
    environment: Mapping[str, Any]
    result: Mapping[str, Any] | None = None
    failure: Mapping[str, Any] | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _validate_run_id(self.run_id))
        if self.status not in _STATUSES:
            raise _problem("run status is unsupported", f"{self.status!r} is not a known lifecycle state", "use running, complete, failed, or dry_run")
        object.__setattr__(self, "split_fingerprint", _validate_fingerprint(self.split_fingerprint))
        if not isinstance(self.command, Sequence) or isinstance(self.command, (str, bytes)) or not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise _problem("exact command is missing or malformed", "the invocation cannot be reproduced", "record the complete command as a nonempty array of strings")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise _problem("run record schema version is missing", "the record cannot be decoded safely", "use the current schema version")
        frozen_config = _freeze_json(self.config_snapshot, field_name="config_snapshot")
        frozen_environment = _freeze_json(self.environment, field_name="environment")
        frozen_result = None if self.result is None else _freeze_json(self.result, field_name="result")
        frozen_failure = None if self.failure is None else _freeze_json(self.failure, field_name="failure")
        if not isinstance(frozen_config, Mapping) or not isinstance(frozen_environment, Mapping):
            raise _problem("run config or environment is not an object", "top-level evidence has an unsupported shape", "store config_snapshot and environment as JSON objects")
        if self.status == "complete" and not isinstance(frozen_result, Mapping):
            raise _problem("complete run has no result", "a terminal successful run omitted its metrics", "save validated result metrics before marking a run complete")
        if self.status == "complete":
            try:
                # A successful run record is public scientific evidence.  Do
                # not merely require an arbitrary JSON object: round-trip it
                # through the canonical detector metric type so missing global
                # metrics, missing AP50 classes, invalid bounds, and remapped
                # labels cannot be persisted or loaded as a completion.
                canonical_metrics = metrics_from_mapping(_thaw_json(frozen_result))
            except DetectionMetricsError as error:
                raise _problem("complete run has invalid detection metrics", str(error), "store the full canonical DetectionMetrics mapping before marking complete") from error
            frozen_result = _freeze_json(canonical_metrics.mapping(), field_name="result")
        if self.status == "failed" and not isinstance(frozen_failure, Mapping):
            raise _problem("failed run has no failure record", "the terminal failure cannot be diagnosed", "save a problem, cause, and remediation before marking failed")
        object.__setattr__(self, "config_snapshot", frozen_config)
        object.__setattr__(self, "environment", frozen_environment)
        object.__setattr__(self, "result", frozen_result)
        object.__setattr__(self, "failure", frozen_failure)
        object.__setattr__(self, "command", tuple(self.command))

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "config_snapshot": _thaw_json(self.config_snapshot),
            "split_fingerprint": self.split_fingerprint,
            "command": list(self.command),
            "environment": _thaw_json(self.environment),
            "result": _thaw_json(self.result),
            "failure": _thaw_json(self.failure),
        }


def create_run_record(*, config_snapshot: Mapping[str, Any], split_fingerprint: str, command: Sequence[str], environment: Mapping[str, Any], run_id: str | None = None, status: str = "running") -> RunRecord:
    """Create a fresh record; UUID IDs prevent accidental artifact collisions."""
    return RunRecord(
        run_id=run_id or f"run-{uuid.uuid4().hex}", status=status, config_snapshot=config_snapshot,
        split_fingerprint=split_fingerprint, command=tuple(command), environment=environment,
    )


def complete_run_record(record: RunRecord, result: Mapping[str, Any]) -> RunRecord:
    """Return (never mutate) the sole valid successful terminal version."""
    if record.status != "running":
        raise _problem("run cannot be completed from its current state", f"run {record.run_id} is {record.status}", "complete only a running run and create a new run for retries")
    return replace(record, status="complete", result=result, failure=None)


def fail_run_record(record: RunRecord, *, problem: str, cause: str, remediation: str) -> RunRecord:
    """Return a diagnostic terminal failure record without erasing provenance."""
    if record.status != "running":
        raise _problem("run cannot be failed from its current state", f"run {record.run_id} is {record.status}", "record failure only while a run is active")
    return replace(record, status="failed", failure={"problem": problem, "cause": cause, "remediation": remediation}, result=None)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temporary_name = None
    try:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False)
        temporary_name = handle.name
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary_name, path)
    except OSError as error:
        raise _problem("run record could not be written", str(error), "ensure the run directory is writable and retry") from error
    finally:
        if handle is not None:
            handle.close()
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink(missing_ok=True)


def write_run_record(record: RunRecord, path: Path | str, *, allow_status_update: bool = False) -> Path:
    """Write a new record, or one monotonic terminal update to the same record."""
    destination = Path(path)
    if destination.exists():
        if not allow_status_update:
            raise _problem("run record already exists", f"{destination} would be overwritten", "choose a new run ID or resume the existing run explicitly")
        previous = read_run_record(destination)
        immutable_fields = ("run_id", "config_snapshot", "split_fingerprint", "command", "environment", "schema_version")
        if any(getattr(previous, key) != getattr(record, key) for key in immutable_fields):
            raise _problem("run record immutable provenance changed", "a status update changed the run ID, configuration, split, command, or environment", "create a new run instead of modifying an existing run")
        if previous.status != "running" or record.status not in {"complete", "failed", "dry_run"}:
            raise _problem("run record has an invalid terminal transition", f"cannot change {previous.status} to {record.status}", "only change a running record once to complete, failed, or dry_run")
    _atomic_write(destination, json.dumps(record.mapping(), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return destination


def read_run_record(path: Path | str) -> RunRecord:
    """Load and validate a run record rather than trusting manually edited JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("run record cannot be read", str(error), "provide the untouched run_record.json emitted by train_supervised") from error
    if not isinstance(payload, Mapping):
        raise _problem("run record is not a JSON object", "the file was truncated or manually replaced", "restore the run_record.json artifact")
    try:
        return RunRecord(
            run_id=payload["run_id"], status=payload["status"], config_snapshot=payload["config_snapshot"],
            split_fingerprint=payload["split_fingerprint"], command=tuple(payload["command"]),
            environment=payload["environment"], result=payload.get("result"), failure=payload.get("failure"),
            schema_version=payload.get("schema_version", ""),
        )
    except (KeyError, TypeError, RunRecordError) as error:
        if isinstance(error, RunRecordError):
            raise
        raise _problem("run record is incomplete", str(error), "restore all required run record fields from the original artifact") from error


def split_fingerprint_from_manifest(path: Path | str) -> str:
    """Read the authoritative Task 8 protocol fingerprint, not a guessed hash."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("split manifest cannot be read", str(error), "pass Task 8's readable split_manifest.json") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("fingerprints"), Mapping):
        raise _problem("split manifest has no fingerprints object", "the manifest is not an untouched Task 8 output", "regenerate deterministic splits before training")
    return _validate_fingerprint(payload["fingerprints"].get("split_protocol"))


def canonical_snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Expose a stable checksum for result aggregators and manual provenance checks."""
    frozen = _freeze_json(snapshot, field_name="config_snapshot")
    if not isinstance(frozen, Mapping):
        raise _problem("config snapshot is not an object", "a scalar or list was passed", "save experiment configuration as a JSON object")
    return hashlib.sha256(_canonical_json(frozen).encode("utf-8")).hexdigest()
