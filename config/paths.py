"""Path validation that avoids unsafe data-directory side effects."""

from __future__ import annotations

from pathlib import Path
from typing import Union


PathInput = Union[str, Path]


class PathValidationError(ValueError):
    """Raised when a configured filesystem location is unsafe or unusable."""


def _error(problem: str, likely_cause: str, remediation: str) -> PathValidationError:
    """Build consistently actionable configuration errors."""
    return PathValidationError(
        f"{problem} Likely cause: {likely_cause} Remediation: {remediation}"
    )


def _coerce_root(value: PathInput, field_name: str) -> Path:
    """Convert a non-empty configured root into a native path without rewriting it."""
    if isinstance(value, str):
        if not value.strip():
            raise _error(
                f"{field_name} is empty.",
                "The environment variable or YAML value was not set.",
                f"Set {field_name} to the intended directory and rerun the command.",
            )
        return Path(value)
    if isinstance(value, Path):
        return value
    raise _error(
        f"{field_name} must be a path string.",
        f"Its configured value has unsupported type {type(value).__name__}.",
        f"Set {field_name} to a quoted directory path in project.yaml.",
    )


def _same_location(left: Path, right: Path) -> bool:
    """Compare local paths after resolving equivalent spellings when possible."""
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return False


def validate_data_root(
    value: PathInput,
    *,
    repository_root: Path,
    require_exists: bool = True,
) -> Path:
    """Validate a dataset root without ever creating it.

    UNC paths remain ordinary ``Path`` values, preserving their backslashes.  Callers
    may set ``require_exists=False`` when validating a configuration intended for a
    machine where the share is not currently mounted.
    """
    data_root = _coerce_root(value, "data_root")
    if _same_location(data_root, repository_root):
        raise _error(
            "data_root must not be the repository root.",
            "The source checkout (or an equivalent path containing '..') was configured as data storage.",
            "Choose the approved dataset directory outside the repository and update FRUIT_SSOD_DATA_ROOT.",
        )
    if require_exists and not data_root.is_dir():
        raise _error(
            f"data_root does not exist or is not a directory: {data_root}",
            "The dataset share is unavailable, the path is misspelled, or it has not been mounted.",
            "Mount or create the approved dataset location outside this repository, then set FRUIT_SSOD_DATA_ROOT.",
        )
    return data_root


def validate_artifact_root(value: PathInput, *, create: bool = False) -> Path:
    """Validate an artifact directory, creating it only after explicit opt-in."""
    artifact_root = _coerce_root(value, "artifact_root")
    if artifact_root.exists():
        if not artifact_root.is_dir():
            raise _error(
                f"artifact_root is not a directory: {artifact_root}",
                "A file occupies the configured artifact path.",
                "Choose an empty directory path for FRUIT_SSOD_ARTIFACT_ROOT.",
            )
        return artifact_root
    if not create:
        raise _error(
            f"artifact_root does not exist: {artifact_root}",
            "The output directory has not been provisioned yet.",
            "Create it yourself or call validation with create=True when directory creation is intended.",
        )
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _error(
            f"artifact_root could not be created: {artifact_root}",
            f"The location is inaccessible ({error}).",
            "Choose a writable output directory or create it with the required permissions.",
        ) from error
    return artifact_root
