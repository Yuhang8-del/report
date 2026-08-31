"""Immutable canonical fruit-class registry and source-aware label mapping.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class ClassMappingError(ValueError):
    """Raised when a source/category cannot be safely mapped to a class."""


@dataclass(frozen=True)
class CanonicalClass:
    """A detector class with a permanent identifier and approved display name."""

    id: int
    name: str


@dataclass(frozen=True)
class ClassRegistry:
    """Read-only class registry loaded from the versioned JSON source of truth."""

    version: str
    classes: tuple[CanonicalClass, ...]
    source_aliases: Mapping[str, Mapping[str, str]]

    @property
    def class_ids(self) -> frozenset[int]:
        """Return the approved IDs without exposing mutable registry state."""
        return frozenset(item.id for item in self.classes)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return names in their fixed detector order."""
        return tuple(item.name for item in self.classes)


_EXPECTED_CLASSES = (
    CanonicalClass(0, "Apple"),
    CanonicalClass(1, "Banana"),
    CanonicalClass(2, "Orange"),
    CanonicalClass(3, "Strawberry"),
    CanonicalClass(4, "Pineapple"),
)
_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "class_registry.json"


def _problem(problem: str, cause: str, remediation: str) -> str:
    """Keep domain failures useful to data-curation users."""
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def load_class_registry(path: Path = _REGISTRY_PATH) -> ClassRegistry:
    """Load and validate the fixed registry without allowing runtime mutation."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClassMappingError(
            _problem(
                f"Class registry at {path} could not be read",
                str(error),
                "restore a valid UTF-8 configs/class_registry.json file",
            )
        ) from error

    try:
        classes = tuple(CanonicalClass(id=item["id"], name=item["name"]) for item in raw["classes"])
        version = raw["registry_version"]
        aliases = raw["source_aliases"]
    except (KeyError, TypeError) as error:
        raise ClassMappingError(
            _problem(
                "Class registry has a missing or malformed required field",
                str(error),
                "use the committed registry schema and its fixed five classes",
            )
        ) from error

    if any(
        isinstance(item.id, bool)
        or not isinstance(item.id, int)
        or not isinstance(item.name, str)
        or not item.name.strip()
        for item in classes
    ):
        raise ClassMappingError(
            _problem(
                "Class registry class definition is malformed",
                "a class ID is not a non-bool integer or a class name is not nonempty text",
                "use integer IDs and nonempty canonical class names in every class definition",
            )
        )
    if not isinstance(version, str) or not version.strip() or classes != _EXPECTED_CLASSES:
        raise ClassMappingError(
            _problem(
                "Class registry does not define the approved fixed class IDs and order",
                "the registry was edited or has incompatible class values",
                "restore IDs 0-4 for Apple, Banana, Orange, Strawberry, and Pineapple",
            )
        )
    if not isinstance(aliases, dict) or not all(
        isinstance(source, str)
        and source.strip()
        and isinstance(values, dict)
        and all(
            isinstance(label, str)
            and label.strip()
            and isinstance(name, str)
            and name.strip()
            for label, name in values.items()
        )
        for source, values in aliases.items()
    ):
        raise ClassMappingError(
            _problem(
                "Class registry source aliases are malformed",
                "a source alias map is not an object of string labels",
                "use explicit string-to-canonical-name mappings",
            )
        )

    class_names = {item.name for item in classes}
    if any(name not in class_names for values in aliases.values() for name in values.values()):
        raise ClassMappingError(
            _problem(
                "Class registry alias targets an unknown canonical class",
                "an alias is misspelled or maps to an unapproved class",
                "point every alias at one of the five canonical class names",
            )
        )
    frozen_aliases = MappingProxyType(
        {source: MappingProxyType(dict(values)) for source, values in aliases.items()}
    )
    return ClassRegistry(version=version, classes=classes, source_aliases=frozen_aliases)


DEFAULT_CLASS_REGISTRY = load_class_registry()
"""The application registry; this is the only canonical class definition."""


def resolve_class_id(
    source: str, source_category: str, registry: ClassRegistry = DEFAULT_CLASS_REGISTRY
) -> int:
    """Resolve an explicitly approved source label to its stable canonical ID."""
    aliases = registry.source_aliases.get(source)
    if aliases is None:
        raise ClassMappingError(
            _problem(
                f"Unknown source {source!r}",
                "no reviewed alias map exists for this source",
                "add and review an explicit source alias map before importing its annotations",
            )
        )
    canonical_name = aliases.get(source_category)
    if canonical_name is None:
        raise ClassMappingError(
            _problem(
                f"Unknown category {source_category!r} for source {source!r}",
                "the label is not an explicitly approved alias (it may be a distinct variety)",
                "review the category and add an explicit safe alias only if it is the same canonical fruit",
            )
        )
    return next(item.id for item in registry.classes if item.name == canonical_name)
