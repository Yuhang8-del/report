"""pytest-qt coverage for the no-model and asynchronous-shell states."""

from __future__ import annotations

import threading
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QComboBox, QGroupBox, QLabel, QListWidget, QPushButton, QTableWidget, QWidget

from fruit_ssod.detection import DetectorAdapter
from fruit_ssod.gui.app import create_main_window
from fruit_ssod.gui.main_window import NAVIGATION_PAGES, MainWindow
from fruit_ssod.gui.model_manager import ModelManager
from fruit_ssod.gui.open_world_window import OpenWorldWindow
from fruit_ssod.gui.widgets.image_view import ImageInferencePage
from fruit_ssod.gui.widgets.status_panel import StatusPanel


class _BlockingAdapter(DetectorAdapter):
    def __init__(self, *, weights_path: Path, started: threading.Event, continue_loading: threading.Event) -> None:
        self.started = started
        self.continue_loading = continue_loading

    def initialize(self) -> None:
        self.started.set()
        assert self.continue_loading.wait(3)

    def predict(self, image: object, *, confidence: float | None = None) -> tuple[object, ...]:
        return ()


class _BlockingPredictAdapter(DetectorAdapter):
    """Hold an image worker in predict() until a close attempt has timed out."""

    def __init__(self) -> None:
        self.predict_started = threading.Event()
        self.release_predict = threading.Event()

    def predict(
        self,
        image: object,
        *,
        confidence: float | None = None,
        nms_iou: float | None = None,
    ) -> tuple[object, ...]:
        self.predict_started.set()
        assert self.release_predict.wait(3)
        return ()


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (230, 190, 120)).save(path)
    return path


def test_application_starts_without_weights(qtbot: object) -> None:
    """The demo opens safely even before a trained checkpoint has been produced."""
    window = create_main_window()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()

    status_panel = window.findChild(StatusPanel, "statusPanel")
    assert status_panel is not None
    assert status_panel.state_text == "No model loaded"
    assert window.model_manager.has_active_model is False
    assert window.current_page_name == "Camera"


def test_navigation_exposes_camera_and_file_workflows(qtbot: object) -> None:
    """The delivered shell exposes real-time camera and existing file workflows."""
    window = create_main_window()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    navigation = window.findChild(QListWidget, "navigationList")

    assert navigation is not None
    assert [navigation.item(index).text() for index in range(navigation.count())] == list(NAVIGATION_PAGES)
    assert "Live Camera" in NAVIGATION_PAGES
    assert "Open-world" not in " ".join(NAVIGATION_PAGES)


def test_customer_gui_visible_text_is_english_only(qtbot: object) -> None:
    """Prevent Chinese labels from returning to either delivered desktop window."""
    chinese = re.compile(r"[\u3400-\u9fff]")
    windows = (create_main_window(), OpenWorldWindow(None))  # type: ignore[arg-type]
    visible_text: list[str] = []
    for window in windows:
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        visible_text.append(window.windowTitle())
        for kind in (QLabel, QPushButton, QGroupBox):
            visible_text.extend(widget.text() if hasattr(widget, "text") else widget.title() for widget in window.findChildren(kind))
        for navigation in window.findChildren(QListWidget):
            visible_text.extend(navigation.item(index).text() for index in range(navigation.count()))
        for combo in window.findChildren(QComboBox):
            visible_text.extend(combo.itemText(index) for index in range(combo.count()))
        for table in window.findChildren(QTableWidget):
            visible_text.extend(
                table.horizontalHeaderItem(index).text()
                for index in range(table.columnCount())
                if table.horizontalHeaderItem(index) is not None
            )

    assert not [text for text in visible_text if chinese.search(text)]


def test_loading_is_asynchronous_and_shell_does_not_claim_detection(tmp_path: Path, qtbot: object) -> None:
    """A slow adapter leaves navigation responsive and documents the shell's boundary."""
    weights = tmp_path / "best.pt"
    weights.touch()
    started = threading.Event()
    continue_loading = threading.Event()

    def factory(*, weights_path: Path) -> _BlockingAdapter:
        return _BlockingAdapter(
            weights_path=weights_path,
            started=started,
            continue_loading=continue_loading,
        )

    manager = ModelManager(adapter_factory=factory)
    window = MainWindow(model_manager=manager)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()

    assert window.load_model(weights) is True
    qtbot.waitUntil(started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    load_button = window.findChild(QPushButton, "loadModelButton")
    navigation = window.findChild(QListWidget, "navigationList")
    assert load_button is not None and not load_button.isEnabled()
    assert navigation is not None
    navigation.setCurrentRow(2)
    assert window.current_page_name == "Batch images"

    page = window.findChild(QWidget, "single_imagePage")
    assert page is not None
    assert page.findChild(QPushButton, "singleRunInferenceButton") is not None

    continue_loading.set()
    qtbot.waitUntil(lambda: not manager.is_loading, timeout=3_000)  # type: ignore[attr-defined]
    assert manager.has_active_model is True
    status_panel = window.findChild(StatusPanel, "statusPanel")
    release_button = window.findChild(QPushButton, "releaseModelButton")
    assert status_panel is not None and "Select an image" in status_panel.message_text
    assert release_button is not None and release_button.isEnabled()


def test_window_close_releases_the_model_and_cuda_cleanup_path(tmp_path: Path, qtbot: object) -> None:
    """Closing a loaded shell drops the adapter before Qt can destroy the window."""
    weights = tmp_path / "best.pt"
    weights.touch()
    started = threading.Event()
    continue_loading = threading.Event()
    continue_loading.set()
    manager = ModelManager(
        adapter_factory=lambda *, weights_path: _BlockingAdapter(
            weights_path=weights_path,
            started=started,
            continue_loading=continue_loading,
        )
    )
    window = MainWindow(model_manager=manager)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()

    assert window.load_model(weights) is True
    qtbot.waitUntil(lambda: manager.has_active_model and not manager.is_loading, timeout=3_000)  # type: ignore[attr-defined]
    assert window.close() is True
    assert manager.has_active_model is False


def test_rejected_close_keeps_idle_single_page_available_after_batch_timeout(
    tmp_path: Path, qtbot: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later batch shutdown refusal must not permanently close the idle single page."""
    image = _image(tmp_path / "apple.png")
    adapter = _BlockingPredictAdapter()
    manager = ModelManager()
    # This test exercises window/page shutdown sequencing, not asynchronous
    # checkpoint construction.  Give both pages one already-owned detector.
    manager._active_adapter = adapter
    window = MainWindow(model_manager=manager)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()

    single = window.findChild(ImageInferencePage, "single_imagePage")
    batch = window.findChild(ImageInferencePage, "batch_imagePage")
    assert single is not None and batch is not None
    assert batch.set_images([image]) is True
    assert batch.start_inference() is True
    qtbot.waitUntil(adapter.predict_started.is_set, timeout=3_000)  # type: ignore[attr-defined]

    # ``closeEvent`` calls pages in single/batch order.  Make the actively
    # blocked batch worker refuse this close promptly, as a real worker would
    # after the normal shutdown timeout.
    batch_shutdown = batch.shutdown
    monkeypatch.setattr(batch, "shutdown", lambda: batch_shutdown(wait_ms=1))
    assert window.close() is False
    assert window.isVisible() is True
    assert single._shutdown_requested is False

    adapter.release_predict.set()
    qtbot.waitUntil(lambda: not batch.is_running, timeout=3_000)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not batch._bridges, timeout=3_000)  # type: ignore[attr-defined]
    assert batch._shutdown_requested is False

    # The rejected close did not consume the idle single-image page; it can
    # start and complete a new run after the batch worker has stopped.
    assert single.set_images([image]) is True
    assert single.start_inference() is True
    qtbot.waitUntil(lambda: not single.is_running, timeout=3_000)  # type: ignore[attr-defined]
    assert len(single.results) == 1
    assert window.close() is True
