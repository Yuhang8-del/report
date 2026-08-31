"""Professional real-time camera workspace for five/eleven-class fruit detection."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fruit_ssod.gui.workers.camera_worker import (
    ELEVEN_CLASS_NAMES,
    FIVE_CLASS_NAMES,
    CameraDetection,
    CameraFrameResult,
    CameraInferenceError,
    CameraInferenceSettings,
    CameraInferenceWorker,
    CameraModelProfile,
    CameraSourceInfo,
    probe_camera_devices,
)


_FRUIT_DISPLAY_NAMES = {name: name for name in ELEVEN_CLASS_NAMES}

_BOX_COLORS = (
    QColor("#EF4444"),
    QColor("#F59E0B"),
    QColor("#F97316"),
    QColor("#EC4899"),
    QColor("#84CC16"),
    QColor("#14B8A6"),
    QColor("#38BDF8"),
    QColor("#FB7185"),
    QColor("#22C55E"),
    QColor("#FACC15"),
    QColor("#A78BFA"),
)


def _place_label_rect(
    box: QRectF,
    *,
    width: float,
    height: float,
    bounds: QRectF,
    occupied: Iterable[QRectF],
) -> QRectF:
    """Place one label inside the preview without covering earlier labels."""
    width = min(width, bounds.width())
    x = max(bounds.left(), min(box.left(), bounds.right() - width))
    occupied_rects = tuple(occupied)
    candidates = [box.top() - height, box.top()]
    candidates.extend(box.top() + offset * (height + 3.0) for offset in range(1, 8))
    candidates.extend(bounds.top() + offset * (height + 3.0) for offset in range(12))
    for candidate_y in candidates:
        y = max(bounds.top(), min(candidate_y, bounds.bottom() - height))
        candidate = QRectF(x, y, width, height)
        if not any(candidate.intersects(existing) for existing in occupied_rects):
            return candidate
    return QRectF(x, max(bounds.top(), min(box.top(), bounds.bottom() - height)), width, height)


def default_camera_profiles() -> tuple[CameraModelProfile, ...]:
    """Resolve packaged camera models without depending on the current CWD."""
    configured_root = os.environ.get("FRUIT_SSOD_DELIVERY_ROOT")
    delivery_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path(__file__).resolve().parents[5]
    )
    return (
        CameraModelProfile(
            label="Semi-Supervised Student (5 Classes)",
            weights_path=delivery_root / "models" / "student_best.pt",
            class_names=FIVE_CLASS_NAMES,
        ),
        CameraModelProfile(
            label="Extended Detector (11 Classes)",
            weights_path=delivery_root / "models" / "incremental_11class_best.pt",
            class_names=ELEVEN_CLASS_NAMES,
        ),
    )


class CameraFrameView(QWidget):
    """Draw the latest camera frame and scale all model boxes consistently."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraFramePreview")
        self.setMinimumSize(600, 430)
        self._image = QImage()
        self._detections: tuple[CameraDetection, ...] = ()

    @property
    def has_frame(self) -> bool:
        return not self._image.isNull()

    def show_frame(self, result: CameraFrameResult) -> None:
        frame = result.rgb_frame
        height, width, channels = frame.shape
        if channels != 3:
            raise ValueError("camera RGB frame must have exactly three channels")
        self._image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._detections = result.detections
        self.update()

    def clear(self) -> None:
        self._image = QImage()
        self._detections = ()
        self.update()

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07131F"))
        if self._image.isNull():
            painter.setPen(QColor("#8FA8BA"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Connect a camera to display live detections")
            return
        target = self.rect().adjusted(10, 10, -10, -10)
        scaled = self._image.size().scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = target.x() + (target.width() - scaled.width()) / 2
        y = target.y() + (target.height() - scaled.height()) / 2
        destination = QRectF(x, y, scaled.width(), scaled.height())
        painter.drawImage(destination, self._image, QRectF(self._image.rect()))
        scale_x = destination.width() / self._image.width()
        scale_y = destination.height() / self._image.height()
        occupied_labels: list[QRectF] = []
        for detection in self._detections:
            x1, y1, x2, y2 = detection.xyxy
            box = QRectF(
                destination.x() + x1 * scale_x,
                destination.y() + y1 * scale_y,
                (x2 - x1) * scale_x,
                (y2 - y1) * scale_y,
            )
            color = _BOX_COLORS[detection.class_id % len(_BOX_COLORS)]
            painter.setPen(QPen(color, 3))
            painter.drawRoundedRect(box, 3, 3)
            label = f"{_FRUIT_DISPLAY_NAMES.get(detection.class_name, detection.class_name)} {detection.confidence:.2f}"
            metrics = painter.fontMetrics()
            text_rect = _place_label_rect(
                box,
                width=float(metrics.horizontalAdvance(label) + 14),
                height=float(metrics.height() + 8),
                bounds=destination,
                occupied=occupied_labels,
            )
            occupied_labels.append(text_rect)
            painter.fillRect(text_rect, color)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)


class CameraInferencePage(QWidget):
    """Device selection, model switching and real-time inference controls."""

    busy_changed = Signal(bool)

    def __init__(
        self,
        *,
        model_profiles: Iterable[CameraModelProfile] | None = None,
        start_allowed: Callable[[], bool] = lambda: True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cameraPage")
        self._profiles = tuple(model_profiles or default_camera_profiles())
        self._start_allowed = start_allowed
        self._thread: QThread | None = None
        self._worker: CameraInferenceWorker | None = None
        self._last_result: CameraFrameResult | None = None
        self._failed = False
        self._build_ui()

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @property
    def selected_profile(self) -> CameraModelProfile | None:
        index = self._model_combo.currentIndex()
        return self._profiles[index] if 0 <= index < len(self._profiles) else None

    @property
    def last_result(self) -> CameraFrameResult | None:
        return self._last_result

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)

        heading = QHBoxLayout()
        title_group = QVBoxLayout()
        title = QLabel("Live Camera Detection", self)
        title.setObjectName("pageTitle")
        intro = QLabel("Connect a USB camera and run real-time fruit detection with the 5-class or 11-class model.", self)
        intro.setObjectName("pageIntro")
        title_group.addWidget(title)
        title_group.addWidget(intro)
        heading.addLayout(title_group, 1)
        self._live_badge = QLabel("● DISCONNECTED", self)
        self._live_badge.setObjectName("cameraStateBadge")
        heading.addWidget(self._live_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(heading)

        body = QHBoxLayout()
        body.setSpacing(14)
        preview_card = QFrame(self)
        preview_card.setObjectName("cameraPreviewCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 12)
        preview_layout.setSpacing(10)
        self._preview = CameraFrameView(preview_card)
        preview_layout.addWidget(self._preview, 1)
        stats = QHBoxLayout()
        self._fps_value = self._metric("--", "LIVE FPS", preview_card)
        self._latency_value = self._metric("--", "INFERENCE TIME", preview_card)
        self._count_value = self._metric("0", "OBJECTS", preview_card)
        self._classes_value = self._metric("--", "DETECTED CLASSES", preview_card)
        for card in (self._fps_value[0], self._latency_value[0], self._count_value[0], self._classes_value[0]):
            stats.addWidget(card, 1)
        preview_layout.addLayout(stats)
        body.addWidget(preview_card, 1)

        controls = QFrame(self)
        controls.setObjectName("cameraControlPanel")
        controls.setMinimumWidth(330)
        controls.setMaximumWidth(380)
        panel = QVBoxLayout(controls)
        panel.setContentsMargins(18, 18, 18, 18)
        panel.setSpacing(12)

        model_group = QGroupBox("Detection Model", controls)
        model_form = QFormLayout(model_group)
        self._model_combo = QComboBox(model_group)
        self._model_combo.setObjectName("cameraModelCombo")
        for profile in self._profiles:
            suffix = "" if profile.weights_path.is_file() else " (File Missing)"
            self._model_combo.addItem(profile.label + suffix)
        model_form.addRow("Model", self._model_combo)
        panel.addWidget(model_group)

        device_group = QGroupBox("Camera Device", controls)
        device_form = QFormLayout(device_group)
        self._device_combo = QComboBox(device_group)
        self._device_combo.setObjectName("cameraDeviceCombo")
        self._device_combo.addItem("Camera 0 (Default)", 0)
        self._refresh_button = QPushButton("Refresh", device_group)
        self._refresh_button.setObjectName("cameraRefreshButton")
        device_row = QHBoxLayout()
        device_row.addWidget(self._device_combo, 1)
        device_row.addWidget(self._refresh_button)
        device_form.addRow("Input", device_row)
        self._resolution_combo = QComboBox(device_group)
        self._resolution_combo.setObjectName("cameraResolutionCombo")
        for label, size in (
            ("640 × 480", (640, 480)),
            ("1280 × 720", (1280, 720)),
            ("1920 × 1080", (1920, 1080)),
        ):
            self._resolution_combo.addItem(label, size)
        self._resolution_combo.setCurrentIndex(1)
        device_form.addRow("Resolution", self._resolution_combo)
        panel.addWidget(device_group)

        settings_group = QGroupBox("Live Parameters", controls)
        settings_form = QFormLayout(settings_group)
        self._confidence = QDoubleSpinBox(settings_group)
        self._confidence.setObjectName("cameraConfidenceSpinBox")
        self._confidence.setRange(0.05, 0.95)
        self._confidence.setSingleStep(0.05)
        self._confidence.setDecimals(2)
        self._confidence.setValue(0.25)
        self._nms_iou = QDoubleSpinBox(settings_group)
        self._nms_iou.setObjectName("cameraNmsIouSpinBox")
        self._nms_iou.setRange(0.05, 0.95)
        self._nms_iou.setSingleStep(0.05)
        self._nms_iou.setDecimals(2)
        self._nms_iou.setValue(0.50)
        settings_form.addRow("Confidence", self._confidence)
        settings_form.addRow("NMS IoU", self._nms_iou)
        panel.addWidget(settings_group)

        action_row = QHBoxLayout()
        self._start_button = QPushButton("Start Detection", controls)
        self._start_button.setObjectName("cameraStartButton")
        self._start_button.setProperty("primary", True)
        self._stop_button = QPushButton("Stop", controls)
        self._stop_button.setObjectName("cameraStopButton")
        self._stop_button.setEnabled(False)
        action_row.addWidget(self._start_button, 1)
        action_row.addWidget(self._stop_button)
        panel.addLayout(action_row)
        self._snapshot_button = QPushButton("Save Current Frame", controls)
        self._snapshot_button.setObjectName("cameraSnapshotButton")
        self._snapshot_button.setEnabled(False)
        panel.addWidget(self._snapshot_button)
        self._summary = QLabel("Select a model and camera, then start live detection.", controls)
        self._summary.setObjectName("cameraSummary")
        self._summary.setWordWrap(True)
        panel.addWidget(self._summary)
        panel.addStretch(1)
        note = QLabel("Note: live mode uses trained fixed-class detectors. Full open-world analysis does not run per frame.", controls)
        note.setObjectName("cameraBoundaryNote")
        note.setWordWrap(True)
        panel.addWidget(note)
        body.addWidget(controls)
        outer.addLayout(body, 1)

        self._refresh_button.clicked.connect(self.refresh_devices)
        self._start_button.clicked.connect(self.start_camera)
        self._stop_button.clicked.connect(self.stop_camera)
        self._snapshot_button.clicked.connect(self._choose_snapshot_path)

    @staticmethod
    def _metric(value: str, caption: str, parent: QWidget) -> tuple[QFrame, QLabel]:
        card = QFrame(parent)
        card.setObjectName("liveMetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(13, 9, 13, 9)
        value_label = QLabel(value, card)
        value_label.setObjectName("liveMetricValue")
        caption_label = QLabel(caption, card)
        caption_label.setObjectName("liveMetricCaption")
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return card, value_label

    @Slot()
    def refresh_devices(self) -> None:
        if self.is_running:
            return
        self._refresh_button.setEnabled(False)
        self._summary.setText("Scanning camera devices. Please wait...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            devices = probe_camera_devices(max_devices=6)
            self._device_combo.clear()
            if devices:
                for index in devices:
                    self._device_combo.addItem(f"Camera {index}", index)
                self._summary.setText(f"Found {len(devices)} available camera(s).")
            else:
                self._device_combo.addItem("Camera 0 (Not Detected)", 0)
                self._summary.setText("No camera detected. Connect a device and check Windows camera permissions.")
        finally:
            QApplication.restoreOverrideCursor()
            self._refresh_button.setEnabled(True)

    @Slot()
    def start_camera(self) -> bool:
        if self.is_running or not self._start_allowed():
            self._show_error("Another inference task is running. Stop it before starting the camera.")
            return False
        profile = self.selected_profile
        if profile is None or not profile.weights_path.is_file():
            self._show_error("The selected model file is missing. Restore the models folder in the delivery package.")
            return False
        device_index = self._device_combo.currentData()
        if not isinstance(device_index, int):
            self._show_error("Select a valid camera device.")
            return False
        size = self._resolution_combo.currentData()
        if not isinstance(size, tuple) or len(size) != 2:
            self._show_error("The camera resolution is invalid.")
            return False
        try:
            settings = CameraInferenceSettings(
                confidence=self._confidence.value(),
                nms_iou=self._nms_iou.value(),
                width=int(size[0]),
                height=int(size[1]),
                image_size=640,
                target_fps=30,
                device=0,
            )
            worker = CameraInferenceWorker(
                profile=profile,
                device_index=device_index,
                settings=settings,
            )
        except CameraInferenceError as error:
            self._show_error(str(error))
            return False
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.opened.connect(self._on_opened)
        worker.frame_ready.connect(self._on_frame_ready)
        worker.stopped.connect(self._on_stopped)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._last_result = None
        self._failed = False
        self._set_busy(True)
        self._live_badge.setText("● CONNECTING")
        self._summary.setText(f"Loading {profile.label} and connecting to camera {device_index}...")
        thread.start()
        return True

    @Slot()
    def stop_camera(self) -> None:
        if self._worker is not None:
            self._summary.setText("Stopping inference and releasing the camera...")
            self._worker.request_stop()

    def shutdown(self, *, wait_ms: int = 6_000) -> bool:
        thread = self._thread
        if thread is None:
            return True
        if self._worker is not None:
            self._worker.request_stop()
        thread.quit()
        if not thread.wait(wait_ms):
            self._show_error("The camera thread is still stopping. Please wait before closing the application.")
            return False
        self._thread = None
        self._worker = None
        return True

    @Slot(object)
    def _on_opened(self, info: object) -> None:
        if not isinstance(info, CameraSourceInfo):
            return
        self._live_badge.setText("● LIVE")
        source = f", source {info.source_fps:.1f} FPS" if info.source_fps > 0 else ""
        self._summary.setText(
            f"Camera {info.device_index} connected at {info.width}×{info.height}{source}. "
            f"Active model: {info.model_label}."
        )

    @Slot(object)
    def _on_frame_ready(self, result: object) -> None:
        if not isinstance(result, CameraFrameResult):
            return
        self._last_result = result
        self._preview.show_frame(result)
        self._fps_value[1].setText(f"{result.display_fps:.1f}")
        self._latency_value[1].setText(f"{result.inference_ms:.1f} ms")
        self._count_value[1].setText(str(len(result.detections)))
        names = ", ".join(_FRUIT_DISPLAY_NAMES.get(name, name) for name in result.class_counts)
        self._classes_value[1].setText(names or "None")
        self._snapshot_button.setEnabled(True)

    @Slot(int)
    def _on_stopped(self, frames: int) -> None:
        self._summary.setText(f"Live detection stopped after {frames} frame(s). Camera released.")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._failed = True
        self._live_badge.setText("● CONNECTION FAILED")
        self._show_error(message)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)
        if not self._failed:
            self._live_badge.setText("● STOPPED")

    def _set_busy(self, busy: bool) -> None:
        self._model_combo.setEnabled(not busy)
        self._device_combo.setEnabled(not busy)
        self._resolution_combo.setEnabled(not busy)
        self._confidence.setEnabled(not busy)
        self._nms_iou.setEnabled(not busy)
        self._refresh_button.setEnabled(not busy)
        self._start_button.setEnabled(not busy)
        self._stop_button.setEnabled(busy)
        if not busy:
            self._snapshot_button.setEnabled(self._last_result is not None)
        self.busy_changed.emit(busy)

    @Slot()
    def _choose_snapshot_path(self) -> None:
        default = str(Path.home() / "fruit_camera_snapshot.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save Live Detection Frame", default, "PNG image (*.png)")
        if path:
            self.save_snapshot(path)

    def save_snapshot(self, path: str | Path) -> bool:
        if self._last_result is None or not self._preview.has_frame:
            self._show_error("There is no camera frame to save yet.")
            return False
        destination = Path(path).expanduser()
        if destination.suffix.lower() != ".png":
            destination = destination.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self._preview.grab().save(str(destination), "PNG"):
            self._show_error("The snapshot could not be saved. Select a writable location.")
            return False
        self._summary.setText(f"Saved live detection snapshot: {destination}")
        return True

    def _show_error(self, message: str) -> None:
        self._summary.setText(message)
        QMessageBox.warning(self, "Live Camera Detection", message)
