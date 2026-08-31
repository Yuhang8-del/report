"""pytest-qt coverage for asynchronous single-model ownership."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from fruit_ssod.detection import DetectorAdapter
from fruit_ssod.gui.model_manager import ModelLoadWorker, ModelManager


class FakeAdapter(DetectorAdapter):
    """Adapter stand-in that records where eager validation actually runs."""

    def __init__(self, *, weights_path: Path) -> None:
        self.weights_path = weights_path
        self.initialized = False
        self.initialize_thread_id: int | None = None

    def initialize(self) -> None:
        self.initialize_thread_id = threading.get_ident()
        self.initialized = True

    def predict(self, image: object, *, confidence: float | None = None) -> tuple[object, ...]:
        return ()


class BlockingAdapter(DetectorAdapter):
    """Test backend that pauses after the worker thread has started loading."""

    def __init__(self, *, weights_path: Path, started: threading.Event, continue_loading: threading.Event) -> None:
        self.started = started
        self.continue_loading = continue_loading

    def initialize(self) -> None:
        self.started.set()
        assert self.continue_loading.wait(3)

    def predict(self, image: object, *, confidence: float | None = None) -> tuple[object, ...]:
        return ()


def _load(manager: ModelManager, weights: Path, qtbot: object) -> FakeAdapter:
    with qtbot.waitSignal(manager.model_loaded, timeout=3_000):  # type: ignore[attr-defined]
        assert manager.start_loading(weights) is True
    qtbot.waitUntil(lambda: not manager.is_loading, timeout=3_000)  # type: ignore[attr-defined]
    assert isinstance(manager.active_model, FakeAdapter)
    return manager.active_model


def test_invalid_weights_error_is_actionable_and_keeps_no_model(tmp_path: Path) -> None:
    """Missing checkpoints fail before a worker or expensive backend import is started."""
    manager = ModelManager()
    errors: list[str] = []
    manager.load_failed.connect(errors.append)

    assert manager.start_loading(tmp_path / "missing.pt") is False

    assert len(errors) == 1
    assert "Problem: model weights file was not found" in errors[0]
    assert "Remediation:" in errors[0]
    assert manager.has_active_model is False
    assert manager.is_loading is False


def test_worker_loads_off_the_gui_thread_and_replaces_the_previous_model(
    tmp_path: Path, qtbot: object
) -> None:
    """Validation is asynchronous and a replacement leaves exactly one active adapter."""
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.touch()
    second.touch()
    built: list[FakeAdapter] = []
    gui_thread_id = threading.get_ident()

    def factory(*, weights_path: Path) -> FakeAdapter:
        adapter = FakeAdapter(weights_path=weights_path)
        built.append(adapter)
        return adapter

    manager = ModelManager(adapter_factory=factory)
    first_adapter = _load(manager, first, qtbot)
    second_adapter = _load(manager, second, qtbot)

    assert first_adapter is built[0]
    assert second_adapter is built[1]
    assert first_adapter is not manager.active_model
    assert manager.active_model is second_adapter
    assert manager.active_weights_path == second.resolve()
    assert all(adapter.initialized for adapter in built)
    assert all(adapter.initialize_thread_id != gui_thread_id for adapter in built)


def test_release_clears_the_active_model(tmp_path: Path, qtbot: object) -> None:
    """Explicit release removes the model reference used by later inference workers."""
    weights = tmp_path / "best.pt"
    weights.touch()
    manager = ModelManager(adapter_factory=FakeAdapter)
    _load(manager, weights, qtbot)

    manager.release_model()

    assert manager.active_model is None
    assert manager.active_weights_path is None
    assert manager.has_active_model is False


def test_shutdown_discards_a_queued_successful_load(tmp_path: Path, qtbot: object) -> None:
    """A load queued before ``shutdown(wait)`` must never reactivate the model."""
    weights = tmp_path / "best.pt"
    weights.touch()
    started = threading.Event()
    continue_loading = threading.Event()
    manager = ModelManager(
        adapter_factory=lambda *, weights_path: BlockingAdapter(
            weights_path=weights_path,
            started=started,
            continue_loading=continue_loading,
        )
    )

    assert manager.start_loading(weights) is True
    qtbot.waitUntil(started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    worker = manager._load_worker
    assert isinstance(worker, ModelLoadWorker)
    worker_emitted_success = threading.Event()
    # This direct observer runs in the worker thread.  It proves that the worker
    # has queued ``loaded`` to ModelManager before the main test thread calls
    # shutdown, without pumping the GUI event queue that would deliver that slot.
    worker.loaded.connect(
        lambda *_args: worker_emitted_success.set(),
        Qt.ConnectionType.DirectConnection,
    )
    continue_loading.set()
    assert worker_emitted_success.wait(3)
    # The worker emits ``loaded`` while the main thread is blocked in wait(), so
    # delivery occurs only after shutdown has completed its cancellation boundary.
    assert manager.shutdown(wait_ms=3_000) is True
    assert manager._load_thread is None
    assert manager._load_worker is None

    qtbot.wait(100)  # type: ignore[attr-defined]
    assert manager.has_active_model is False
    assert manager.active_weights_path is None
    assert manager._load_thread is None
    assert manager._load_worker is None
