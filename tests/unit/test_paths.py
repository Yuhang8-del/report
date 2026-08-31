"""Tests for data and artifact path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from fruit_ssod.config.paths import PathValidationError, validate_artifact_root, validate_data_root


def test_data_root_accepts_unc_string_without_mangling_backslashes(tmp_path: Path) -> None:
    """UNC syntax remains intact even when the share is not mounted locally."""
    unc_root = r"\\fileserver\fruit-data\images"

    validated = validate_data_root(unc_root, repository_root=tmp_path, require_exists=False)

    assert str(validated) == unc_root


def test_data_root_rejects_missing_directory(tmp_path: Path) -> None:
    """Required data roots must already exist and are never implicitly created."""
    with pytest.raises(PathValidationError, match="does not exist") as error:
        validate_data_root(tmp_path / "missing", repository_root=tmp_path)

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_data_root_rejects_repository_root_and_equivalent_path(tmp_path: Path) -> None:
    """The source checkout cannot be accidentally used to store a dataset."""
    equivalent_root = tmp_path / "nested" / ".."
    (tmp_path / "nested").mkdir()

    with pytest.raises(PathValidationError, match="repository root") as error:
        validate_data_root(equivalent_root, repository_root=tmp_path)

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_artifact_root_is_created_only_when_explicitly_requested(tmp_path: Path) -> None:
    """Artifact creation requires an explicit opt-in rather than loader side effects."""
    artifact_root = tmp_path / "new-artifacts"

    with pytest.raises(PathValidationError, match="does not exist"):
        validate_artifact_root(artifact_root)
    assert not artifact_root.exists()

    assert validate_artifact_root(artifact_root, create=True) == artifact_root
    assert artifact_root.is_dir()
