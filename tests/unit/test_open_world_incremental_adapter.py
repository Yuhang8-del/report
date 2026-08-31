from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.open_world.incremental_adapter import ReviewedUltralyticsDetectorAdapter


class Boxes:
    xyxy = [[10.0, 20.0, 30.0, 40.0]]
    conf = [0.9]
    cls = [5.0]


class Result:
    def __init__(self, names: dict[int, str]) -> None:
        self.names = names
        self.boxes = Boxes()


class Model:
    def __init__(self, names: dict[int, str]) -> None:
        self.names = names

    def __call__(self, _image: object, **_kwargs: object):
        return [Result(self.names)]


def registry(path: Path) -> dict[int, str]:
    names = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple", "Avocado"]
    path.write_text(
        json.dumps({"classes": [{"id": index, "name": name} for index, name in enumerate(names)]}),
        encoding="utf-8",
    )
    return dict(enumerate(names))


def test_reviewed_adapter_accepts_append_only_class_mapping(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    names = registry(path)
    adapter = ReviewedUltralyticsDetectorAdapter(registry_path=path, model=Model(names))
    detection = adapter.predict("image.jpg", confidence=0.5)[0]
    assert detection.class_id == 5
    assert detection.class_name == "Avocado"
    assert detection.xyxy == (10.0, 20.0, 30.0, 40.0)


def test_reviewed_adapter_rejects_checkpoint_registry_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    names = registry(path)
    names[5] = "Wrong"
    with pytest.raises(ValueError, match="mapping mismatch"):
        ReviewedUltralyticsDetectorAdapter(registry_path=path, model=Model(names))
