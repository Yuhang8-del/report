"""YAML project configuration loader with explicit environment handling."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import APPROVED_CLASS_NAMES, ClassDefinition, ProjectConfig
from .paths import PathValidationError, validate_artifact_root, validate_data_root


_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ConfigValidationError(ValueError):
    """Raised when project YAML cannot be safely converted to ``ProjectConfig``."""


def _error(problem: str, likely_cause: str, remediation: str) -> ConfigValidationError:
    """Build errors that tell operators how to correct the configuration."""
    return ConfigValidationError(
        f"{problem} Likely cause: {likely_cause} Remediation: {remediation}"
    )


def _expand_environment_value(value: Any) -> Any:
    """Recursively expand ``${NAME}`` values while rejecting unset variables."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            environment_value = os.environ.get(name)
            if environment_value is None:
                raise _error(
                    f"Environment variable {name} is required by project.yaml but is not set.",
                    "The local environment was not configured for this machine.",
                    f"Set {name} to the intended path, then rerun the command.",
                )
            return environment_value

        return _ENVIRONMENT_VARIABLE.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment_value(item) for key, item in value.items()}
    return value


def _required_string(config: Mapping[str, Any], key: str) -> str:
    """Read a non-empty string required by the project schema."""
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _error(
            f"Required configuration key '{key}' is missing or empty.",
            "The YAML file is incomplete or its environment variable expanded to an empty value.",
            f"Add a non-empty '{key}' entry to project.yaml or set its referenced environment variable.",
        )
    return value


def _optional_int(config: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    """Read a bounded integer and reject booleans, which YAML treats as integers."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(
            f"Configuration key '{key}' must be an integer greater than or equal to {minimum}.",
            f"The YAML value {value!r} has an invalid type or range.",
            f"Set '{key}' to a whole number greater than or equal to {minimum}.",
        )
    return value


def _optional_string(config: Mapping[str, Any], key: str, default: str) -> str:
    """Read a non-empty string setting or use its safe default."""
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise _error(
            f"Configuration key '{key}' must be a non-empty string.",
            "The YAML value is missing, empty, or has the wrong type.",
            f"Set '{key}' to a non-empty string in project.yaml.",
        )
    return value


def _classes(config: Mapping[str, Any]) -> tuple[ClassDefinition, ...]:
    """Validate the fixed five-class detector registry."""
    raw_classes = config.get("classes")
    if raw_classes is None:
        return ProjectConfig.__dataclass_fields__["classes"].default_factory()
    if not isinstance(raw_classes, list) or len(raw_classes) != len(APPROVED_CLASS_NAMES):
        raise _error(
            "Configuration key 'classes' must list exactly five approved classes.",
            "The dataset label registry is incomplete or uses an unsupported schema.",
            "Provide IDs 0 through 4 in order for Apple, Banana, Orange, Strawberry, and Pineapple.",
        )
    classes: list[ClassDefinition] = []
    for index, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise _error(
                f"classes[{index}] must be a mapping with id and name.",
                "The class registry entry has the wrong YAML shape.",
                "Use '- {id: 0, name: Apple}' and repeat for each approved class.",
            )
        class_id = item.get("id")
        class_name = item.get("name")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or not isinstance(class_name, str):
            raise _error(
                f"classes[{index}] must contain integer id and string name.",
                "A class entry has an invalid field type.",
                "Provide an integer id and one of the approved fruit names.",
            )
        classes.append(ClassDefinition(id=class_id, name=class_name))
    expected = tuple((index, name) for index, name in enumerate(APPROVED_CLASS_NAMES))
    received = tuple((item.id, item.name) for item in classes)
    if received != expected:
        raise _error(
            "Configuration key 'classes' does not match the approved five-class registry.",
            "Class ids or names were changed, reordered, duplicated, or omitted.",
            "Use IDs 0-4 in order for Apple, Banana, Orange, Strawberry, and Pineapple.",
        )
    return tuple(classes)


def load_project_config(
    config_path: Path | str,
    *,
    create_artifact_root: bool = False,
    repository_root: Path | None = None,
) -> ProjectConfig:
    """Load and validate a project YAML file without hidden local-path fallbacks.

    ``FRUIT_SSOD_DATA_ROOT`` and ``FRUIT_SSOD_ARTIFACT_ROOT`` take precedence over
    the YAML values after all ``${NAME}`` references have been expanded.
    """
    path = Path(config_path)
    if not path.is_file():
        raise _error(
            f"Project configuration file does not exist: {path}",
            "The command was run from the wrong directory or the config path is misspelled.",
            "Pass the path to configs/project.yaml or another existing YAML configuration file.",
        )
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, UnicodeError) as error:
        raise _error(
            f"Project configuration could not be read: {path}",
            "The file is inaccessible, changed while being read, or is not valid UTF-8 text.",
            "Check file permissions and save project.yaml as UTF-8, then rerun the command.",
        ) from error
    except yaml.YAMLError as error:
        raise _error(
            f"Project configuration is not valid YAML: {path}",
            f"YAML parsing failed ({error}).",
            "Fix the YAML indentation or quoting and rerun the command.",
        ) from error
    if not isinstance(loaded, dict):
        raise _error(
            "Project configuration must contain a YAML mapping at its top level.",
            "The file is empty or uses a list where key/value settings are required.",
            "Use key/value entries such as data_root and artifact_root in project.yaml.",
        )

    expanded = _expand_environment_value(loaded)
    data_root_value = os.environ.get("FRUIT_SSOD_DATA_ROOT", _required_string(expanded, "data_root"))
    artifact_root_value = os.environ.get(
        "FRUIT_SSOD_ARTIFACT_ROOT", _required_string(expanded, "artifact_root")
    )
    if not data_root_value:
        raise _error(
            "FRUIT_SSOD_DATA_ROOT is empty.",
            "An environment override was set without a directory value.",
            "Unset the override to use YAML or set it to the approved existing dataset directory.",
        )
    if not artifact_root_value:
        raise _error(
            "FRUIT_SSOD_ARTIFACT_ROOT is empty.",
            "An environment override was set without a directory value.",
            "Unset the override to use YAML or set it to an artifact directory.",
        )

    defaults = ProjectConfig(data_root=Path("."), artifact_root=Path("."))
    try:
        data_root = validate_data_root(
            data_root_value,
            repository_root=repository_root or _REPOSITORY_ROOT,
        )
        artifact_root = validate_artifact_root(artifact_root_value, create=create_artifact_root)
    except PathValidationError as error:
        raise ConfigValidationError(str(error)) from error
    return ProjectConfig(
        data_root=data_root,
        artifact_root=artifact_root,
        classes=_classes(expanded),
        image_size=_optional_int(expanded, "image_size", defaults.image_size, minimum=1),
        device=_optional_string(expanded, "device", defaults.device),
        workers=_optional_int(expanded, "workers", defaults.workers, minimum=0),
        seed=_optional_int(expanded, "seed", defaults.seed, minimum=0),
        experiment_name=_optional_string(expanded, "experiment_name", defaults.experiment_name),
    )
