"""Cancellation coverage for the cooperative image batch worker."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtCore import QThread

from fruit_ssod.detection import DetectorAdapter
from fruit_ssod.gui.workers.image_worker import ImageInferenceSettings, ImageInferenceWorker


class BlockingAdapter(DetectorAdapter):
    """Pause the first call so cancellation can be requested from the GUI thread."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.continue_prediction = threading.Event()
        self.calls = 0

    def predict(
        self, image: object, *, confidence: float | None = None, nms_iou: float | None = None
    ) -> tuple[object, ...]:
        self.calls += 1
        self.started.set()
        assert self.continue_prediction.wait(3)
        return ()


def _image(path: Path) -> Path:
    Image.new("RGB", (20, 20), (255, 255, 255)).save(path)
    return path


def test_cancel_stops_before_next_image_and_preserves_completed_result(tmp_path: Path, qtbot: object) -> None:
    """Cancellation is cooperative: it never tears down a model call mid-image."""
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png")
    adapter = BlockingAdapter()
    worker = ImageInferenceWorker(
        adapter=adapter,
        image_paths=[first, second],
        settings=ImageInferenceSettings(),
    )
    cancellation: list[tuple[int, int]] = []
    worker.cancelled.connect(lambda completed, total: cancellation.append((completed, total)))
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.start()
    qtbot.waitUntil(adapter.started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    worker.request_cancel()
    adapter.continue_prediction.set()
    qtbot.waitUntil(lambda: bool(cancellation), timeout=3_000)  # type: ignore[attr-defined]
    assert thread.wait(3_000)
    assert adapter.calls == 1
    assert cancellation == [(1, 2)]
