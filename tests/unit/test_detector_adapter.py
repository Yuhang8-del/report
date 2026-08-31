"""Tests for the dependency-injected Ultralytics detector adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from fruit_ssod.data.class_mapping import CanonicalClass, ClassRegistry, DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapterError
from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter


class FakeBoxes:
    """Minimal in-memory stand-in for an Ultralytics Boxes result."""

    def __init__(self) -> None:
        self.xyxy = [[1.0, 2.0, 30.0, 40.0], [3.0, 4.0, 20.0, 25.0]]
        self.conf = [0.91, 0.75]
        self.cls = [0, 4]


class FakeResult:
    """Minimal result object returned by the injected model."""

    names = {0: "Apple", 1: "Banana", 2: "Orange", 3: "Strawberry", 4: "Pineapple"}
    boxes = FakeBoxes()


class FakeModel:
    """A fake callable model that proves no weights or GPU are needed in tests."""

    names = FakeResult.names

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, image: object, **kwargs: object) -> list[FakeResult]:
        self.calls.append((image, kwargs))
        return [FakeResult()]


def test_adapter_converts_fake_ultralytics_results_to_validated_records() -> None:
    """The first implementation returns known classes and preserves model provenance."""
    model = FakeModel()
    adapter = UltralyticsDetectorAdapter(model=model, source_model="fixture.pt")

    detections = adapter.predict(Path("example.jpg"), confidence=0.4)

    assert [(item.class_id, item.class_name) for item in detections] == [(0, "Apple"), (4, "Pineapple")]
    assert all(item.is_unknown is False for item in detections)
    assert all(item.source_model == "fixture.pt" for item in detections)
    assert model.calls == [(Path("example.jpg"), {"conf": 0.4, "verbose": False})]


def test_adapter_passes_a_validated_nms_iou_to_the_backend() -> None:
    """The GUI NMS control remains part of the audited backend invocation."""
    model = FakeModel()
    adapter = UltralyticsDetectorAdapter(model=model, source_model="fixture.pt")

    adapter.predict(Path("example.jpg"), confidence=0.4, nms_iou=0.6)

    assert model.calls == [(Path("example.jpg"), {"conf": 0.4, "iou": 0.6, "verbose": False})]


def test_adapter_accepts_array_like_inputs_without_touching_the_filesystem() -> None:
    """PySide6 can pass an already loaded image array to the model adapter."""
    image_array = [[0, 1], [2, 3]]
    model = FakeModel()

    detections = UltralyticsDetectorAdapter(model=model, source_model="fixture.pt").predict(image_array)

    assert len(detections) == 2
    assert model.calls[0] == (image_array, {"verbose": False})


def test_adapter_rejects_a_model_class_mapping_that_differs_from_the_registry() -> None:
    """A mismatched checkpoint cannot silently relabel detector output."""
    model = FakeModel()
    model.names = {0: "Apple", 1: "Banana", 2: "Orange", 3: "Strawberry", 4: "Mango"}

    with pytest.raises(DetectorAdapterError, match="class mapping") as error:
        UltralyticsDetectorAdapter(model=model, source_model="wrong.pt")

    assert "Problem:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_adapter_reports_malformed_model_output_actionably() -> None:
    """Unexpected box tensor shapes fail with remediation rather than an index error."""
    model = FakeModel()
    FakeResult.boxes.xyxy = [[1.0, 2.0, 3.0]]
    try:
        adapter = UltralyticsDetectorAdapter(model=model, source_model="fixture.pt")
        with pytest.raises(DetectorAdapterError, match="malformed box") as error:
            adapter.predict("image.jpg")
    finally:
        FakeResult.boxes = FakeBoxes()

    assert "Problem:" in str(error.value)
    assert "Likely cause:" in str(error.value)


def test_adapter_discards_finite_degenerate_border_boxes() -> None:
    """A zero-area framework artifact must not abort an otherwise valid image."""
    model = FakeModel()
    boxes = FakeBoxes()
    boxes.xyxy = [boxes.xyxy[0], [10.0, 20.0, 10.0, 30.0]]
    boxes.conf = [boxes.conf[0], .01]
    boxes.cls = [boxes.cls[0], 0]
    FakeResult.boxes = boxes
    try:
        detections = UltralyticsDetectorAdapter(model=model, source_model="fixture.pt").predict("image.jpg")
    finally:
        FakeResult.boxes = FakeBoxes()

    assert [(item.class_id, item.confidence) for item in detections] == [(0, .91)]


def test_adapter_delays_ultralytics_import_until_a_model_is_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing project code stays usable when optional Ultralytics is absent."""
    adapter = UltralyticsDetectorAdapter(weights_path="missing.pt")

    def fail_import() -> object:
        raise ModuleNotFoundError("No module named 'ultralytics'")

    monkeypatch.setattr(adapter, "_build_model", fail_import)
    with pytest.raises(DetectorAdapterError, match="could not be initialized"):
        adapter.predict("image.jpg")


def test_adapter_rejects_a_noncanonical_custom_registry() -> None:
    """A caller cannot use registry injection to make Mango an accepted model class."""
    mango_registry = ClassRegistry(
        version="test",
        classes=(
            CanonicalClass(0, "Apple"),
            CanonicalClass(1, "Banana"),
            CanonicalClass(2, "Orange"),
            CanonicalClass(3, "Strawberry"),
            CanonicalClass(4, "Mango"),
        ),
        source_aliases={},
    )

    with pytest.raises(DetectorAdapterError, match="configured registry does not match") as error:
        UltralyticsDetectorAdapter(model=FakeModel(), source_model="fixture.pt", registry=mango_registry)

    assert "Remediation:" in str(error.value)


@pytest.mark.parametrize("invalid_class_id", [float("nan"), float("inf"), float("-inf")])
def test_adapter_turns_nonfinite_model_class_ids_into_actionable_errors(invalid_class_id: float) -> None:
    """NaN and infinity class IDs do not escape as implementation-level conversion errors."""
    model = FakeModel()
    boxes = FakeBoxes()
    boxes.xyxy = [boxes.xyxy[0]]
    boxes.conf = [boxes.conf[0]]
    boxes.cls = [invalid_class_id]
    FakeResult.boxes = boxes
    try:
        adapter = UltralyticsDetectorAdapter(model=model, source_model="fixture.pt")
        with pytest.raises(DetectorAdapterError, match="malformed box") as error:
            adapter.predict("image.jpg")
    finally:
        FakeResult.boxes = FakeBoxes()

    assert "Problem:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_adapter_rejects_an_ambiguous_injected_model_and_weights_path() -> None:
    """Tests and production callers must choose one unambiguous model source."""
    with pytest.raises(DetectorAdapterError, match="configuration is ambiguous") as error:
        UltralyticsDetectorAdapter(model=FakeModel(), weights_path="best.pt")

    assert "Remediation:" in str(error.value)


def test_lazy_model_mapping_failure_is_not_cached_for_a_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected candidate model cannot become the adapter's cached model instance."""
    adapter = UltralyticsDetectorAdapter(weights_path="best.pt")
    invalid_model = FakeModel()
    invalid_model.names = {0: "Apple"}
    build_calls = 0

    def build_invalid_model() -> FakeModel:
        nonlocal build_calls
        build_calls += 1
        return invalid_model

    monkeypatch.setattr(adapter, "_build_model", build_invalid_model)
    for _ in range(2):
        with pytest.raises(DetectorAdapterError, match="class mapping"):
            adapter.predict("image.jpg")

    assert build_calls == 2
    assert adapter._model is None


def test_adapter_canonicalizes_an_equivalent_mutable_registry() -> None:
    """Later caller-side registry changes cannot change inference class interpretation."""
    external_classes = list(DEFAULT_CLASS_REGISTRY.classes)
    equivalent_registry = ClassRegistry(
        version="test",
        classes=external_classes,  # type: ignore[arg-type]
        source_aliases={},
    )
    adapter = UltralyticsDetectorAdapter(
        model=FakeModel(), source_model="fixture.pt", registry=equivalent_registry
    )
    external_classes[-1] = CanonicalClass(4, "Mango")

    assert adapter._registry is DEFAULT_CLASS_REGISTRY
    assert adapter.predict("image.jpg")[-1].class_name == "Pineapple"
