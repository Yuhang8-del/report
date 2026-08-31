"""Tests for project configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from fruit_ssod.config.loader import ConfigValidationError, load_project_config


def write_config(path: Path, data_root: str, artifact_root: str) -> None:
    """Write the smallest valid project configuration fixture."""
    path.write_text(
        "\n".join(
            [
                f"data_root: {data_root}",
                f"artifact_root: {artifact_root}",
                "classes:",
                "  - {id: 0, name: Apple}",
                "  - {id: 1, name: Banana}",
                "  - {id: 2, name: Orange}",
                "  - {id: 3, name: Strawberry}",
                "  - {id: 4, name: Pineapple}",
                "image_size: 640",
                "device: cuda:0",
                "workers: 2",
                "seed: 42",
                "experiment_name: unit-test",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_loader_expands_environment_variables_before_applying_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML values can use environment variables without baking in local paths."""
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifact_root.mkdir()
    config_path = tmp_path / "project.yaml"
    write_config(config_path, "${TEST_FRUIT_DATA_ROOT}", "${TEST_FRUIT_ARTIFACT_ROOT}")
    monkeypatch.setenv("TEST_FRUIT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("TEST_FRUIT_ARTIFACT_ROOT", str(artifact_root))

    config = load_project_config(config_path)

    assert config.data_root == data_root
    assert config.artifact_root == artifact_root
    assert tuple((item.id, item.name) for item in config.classes) == (
        (0, "Apple"),
        (1, "Banana"),
        (2, "Orange"),
        (3, "Strawberry"),
        (4, "Pineapple"),
    )


def test_loader_uses_fruit_environment_overrides_after_yaml_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit Fruit overrides take precedence over values expanded from YAML."""
    yaml_data_root = tmp_path / "yaml-data"
    override_data_root = tmp_path / "override-data"
    yaml_data_root.mkdir()
    override_data_root.mkdir()
    (tmp_path / "artifacts").mkdir()
    config_path = tmp_path / "project.yaml"
    write_config(config_path, str(yaml_data_root), str(tmp_path / "artifacts"))
    monkeypatch.setenv("FRUIT_SSOD_DATA_ROOT", str(override_data_root))

    config = load_project_config(config_path)

    assert config.data_root == override_data_root


def test_loader_reports_missing_required_config_key(tmp_path: Path) -> None:
    """Missing settings fail with a problem, cause, and remediation."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    config_path = tmp_path / "project.yaml"
    config_path.write_text(f"data_root: {data_root}\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="artifact_root") as error:
        load_project_config(config_path)

    message = str(error.value)
    assert "Likely cause:" in message
    assert "Remediation:" in message


def test_loader_reports_invalid_utf8_as_actionable_config_error(tmp_path: Path) -> None:
    """Unreadable configuration bytes do not escape as a raw decoding exception."""
    config_path = tmp_path / "project.yaml"
    config_path.write_bytes(b"data_root: \xff\xfe")

    with pytest.raises(ConfigValidationError, match="could not be read") as error:
        load_project_config(config_path)

    message = str(error.value)
    assert "Likely cause:" in message
    assert "Remediation:" in message


def test_loader_rejects_nonexistent_data_root(tmp_path: Path) -> None:
    """A typo in the dataset root never falls back to an unrelated directory."""
    config_path = tmp_path / "project.yaml"
    write_config(config_path, str(tmp_path / "does-not-exist"), str(tmp_path / "artifacts"))

    with pytest.raises(ConfigValidationError, match="does not exist") as error:
        load_project_config(config_path)

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_loader_rejects_legacy_or_reordered_class_names_with_canonical_remediation(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifact_root.mkdir()
    config_path = tmp_path / "project.yaml"
    write_config(config_path, str(data_root), str(artifact_root))
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "{id: 3, name: Strawberry}", "{id: 3, name: pear}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as error:
        load_project_config(config_path)

    assert str(error.value) == (
        "Configuration key 'classes' does not match the approved five-class registry. "
        "Likely cause: Class ids or names were changed, reordered, duplicated, or omitted. "
        "Remediation: Use IDs 0-4 in order for Apple, Banana, Orange, Strawberry, and Pineapple."
    )
