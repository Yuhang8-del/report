"""Camera-device, real-time worker and GUI page coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, QThread
from PySide6.QtWidgets import QComboBox, QPushButton

from fruit_ssod.gui.widgets.camera_view import CameraFrameView, CameraInferencePage, _place_label_rect
from fruit_ssod.gui.workers.camera_worker import (
    FIVE_CLASS_NAMES,
    CameraFrameResult,
    CameraInferenceSettings,
    CameraInferenceWorker,
    CameraModelProfile,
    normalize_camera_result,
    probe_camera_devices,
)


class _Array:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "_Array":
        return self

    def cpu(self) -> "_Array":
        return self

    def __array__(self, dtype: object = None) -> np.ndarray:
        return np.asarray(self.value, dtype=dtype)


class _Boxes:
    xyxy = _Array([[3.0, 4.0, 18.0, 20.0]])
    conf = _Array([0.88])
    cls = _Array([0])


class _Result:
    names = dict(enumerate(FIVE_CLASS_NAMES))
    boxes = _Boxes()


class _Model:
    names = dict(enumerate(FIVE_CLASS_NAMES))

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[_Result]:
        assert frame.shape == (24, 32, 3)
        assert kwargs["conf"] == 0.35
        assert kwargs["iou"] == 0.55
        assert kwargs["imgsz"] == 640
        return [_Result()]


class _Capture:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.frame = np.full((24, 32, 3), 120, dtype=np.uint8)
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API
        return self.opened

    def set(self, key: int, value: float) -> bool:
        self.properties[key] = value
        return True

    def get(self, key: int) -> float:
        return self.properties.get(key, 30.0)

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


def _profile(weights: Path) -> CameraModelProfile:
    weights.touch()
    return CameraModelProfile("Test 5-Class Model", weights, FIVE_CLASS_NAMES)


def test_normalize_camera_result_accepts_packaged_five_class_taxonomy(tmp_path: Path) -> None:
    detections = normalize_camera_result(_Result(), _profile(tmp_path / "best.pt"))

    assert len(detections) == 1
    assert detections[0].class_name == "Apple"
    assert detections[0].confidence == pytest.approx(0.88)
    assert detections[0].xyxy == (3.0, 4.0, 18.0, 20.0)


def test_probe_camera_devices_releases_every_temporary_handle() -> None:
    captures: list[_Capture] = []

    def factory(index: int, _backend: int) -> _Capture:
        capture = _Capture(opened=index == 1)
        captures.append(capture)
        return capture

    assert probe_camera_devices(max_devices=3, capture_factory=factory) == (1,)
    assert captures
    assert all(capture.released for capture in captures)


def test_worker_runs_model_off_gui_thread_and_releases_camera(tmp_path: Path, qtbot: object) -> None:
    capture = _Capture()
    worker = CameraInferenceWorker(
        profile=_profile(tmp_path / "best.pt"),
        device_index=0,
        settings=CameraInferenceSettings(confidence=0.35, nms_iou=0.55, width=32, height=24),
        capture_factory=lambda _index, _backend: capture,
        model_factory=lambda _weights: _Model(),
    )
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    frames: list[CameraFrameResult] = []

    def stop_after_first(result: object) -> None:
        assert isinstance(result, CameraFrameResult)
        frames.append(result)
        worker.request_stop()

    worker.frame_ready.connect(stop_after_first)
    worker.finished.connect(thread.quit)
    thread.start()
    qtbot.waitUntil(lambda: not thread.isRunning(), timeout=3_000)  # type: ignore[attr-defined]

    assert frames
    assert frames[0].detections[0].class_name == "Apple"
    assert frames[0].rgb_frame.shape == (24, 32, 3)
    assert capture.released is True


def test_camera_page_exposes_professional_controls_and_two_model_profiles(
    tmp_path: Path, qtbot: object
) -> None:
    five = _profile(tmp_path / "student.pt")
    eleven_path = tmp_path / "incremental.pt"
    eleven_path.touch()
    eleven = CameraModelProfile(
        "Extended Detector (11 Classes)",
        eleven_path,
        FIVE_CLASS_NAMES + ("Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon"),
    )
    page = CameraInferencePage(model_profiles=(five, eleven))
    qtbot.addWidget(page)  # type: ignore[attr-defined]

    model_combo = page.findChild(QComboBox, "cameraModelCombo")
    device_combo = page.findChild(QComboBox, "cameraDeviceCombo")
    start_button = page.findChild(QPushButton, "cameraStartButton")
    snapshot_button = page.findChild(QPushButton, "cameraSnapshotButton")
    preview = page.findChild(CameraFrameView, "cameraFramePreview")

    assert model_combo is not None and model_combo.count() == 2
    assert device_combo is not None and device_combo.currentData() == 0
    assert start_button is not None and start_button.text() == "Start Detection"
    assert snapshot_button is not None and not snapshot_button.isEnabled()
    assert preview is not None


def test_overlapping_camera_boxes_receive_non_overlapping_labels() -> None:
    bounds = QRectF(0.0, 0.0, 640.0, 480.0)
    box = QRectF(200.0, 80.0, 90.0, 140.0)
    first = _place_label_rect(box, width=100.0, height=28.0, bounds=bounds, occupied=())
    second = _place_label_rect(box, width=100.0, height=28.0, bounds=bounds, occupied=(first,))
    third = _place_label_rect(box, width=100.0, height=28.0, bounds=bounds, occupied=(first, second))

    assert not first.intersects(second)
    assert not first.intersects(third)
    assert not second.intersects(third)
    assert bounds.contains(first) and bounds.contains(second) and bounds.contains(third)
