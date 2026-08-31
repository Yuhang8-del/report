"""Typed project configuration models with safe defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APPROVED_CLASS_NAMES: tuple[str, ...] = (
    "Apple",
    "Banana",
    "Orange",
    "Strawberry",
    "Pineapple",
)
"""The fixed five fruit labels approved for the initial detector."""


@dataclass(frozen=True)
class ClassDefinition:
    """One detector class and its stable numeric identifier."""

    id: int
    name: str


def _default_classes() -> tuple[ClassDefinition, ...]:
    """Create the fixed class registry without sharing mutable state."""
    return tuple(ClassDefinition(id=index, name=name) for index, name in enumerate(APPROVED_CLASS_NAMES))


@dataclass(frozen=True)
class ProjectConfig:
    """Validated runtime settings for a Fruit SSOD experiment."""

    data_root: Path
    artifact_root: Path
    classes: tuple[ClassDefinition, ...] = field(default_factory=_default_classes)
    image_size: int = 640
    device: str = "cuda:0"
    workers: int = 4
    seed: int = 42
    experiment_name: str = "fruit-ssod"
