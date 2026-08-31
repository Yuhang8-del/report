"""Video-file processing page; deliberately no camera-device functionality."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fruit_ssod.detection import DetectorAdapter
from fruit_ssod.gui.workers.video_worker import (
    VideoFrameResult,
    VideoInferenceError,
    VideoInferenceSettings,
    VideoInferenceWorker,
    VideoSourceInfo,
    resolve_video_path,
)


class VideoFrameView(QWidget):
    """Display a detached RGB frame emitted by the worker thread."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoFramePreview")
        self.setMinimumSize(420, 320)
        self._pixmap = QPixmap()
        self.frame_index: int | None = None

    def show_frame(self, rgb_frame: np.ndarray, frame_index: int) -> bool:
        """Copy data into Qt ownership before the worker advances to the next frame."""
        if not isinstance(rgb_frame, np.ndarray) or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            self._pixmap = QPixmap()
            self.frame_index = None
            self.update()
            return False
        height, width, _channels = rgb_frame.shape
        image = QImage(rgb_frame.data, width, height, rgb_frame.strides[0], QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.frame_index = frame_index
        self.update()
        return True

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap.isNull():
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignCenter, "Select a video and output file")
            return
        target = self.rect().adjusted(8, 8, -8, -8)
        painter.drawPixmap(target, self._pixmap.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class _VideoRunBridge(QObject):
    """Associate all queued worker notifications with one immutable page run."""

    opened = Signal(int, object)
    frame_completed = Signal(int, object, int, int)
    progress_changed = Signal(int, int, int)
    pause_changed = Signal(int, bool)
    stopped = Signal(int, int, int)
    completed = Signal(int, str, int, float)
    failed = Signal(int, str)
    finished = Signal(int, bool)
    thread_finished = Signal(int, object)

    def __init__(self, run_id: int, parent: QObject) -> None:
        super().__init__(parent)
        self._run_id = run_id

    @Slot(object)
    def relay_opened(self, info: object) -> None:
        self.opened.emit(self._run_id, info)

    @Slot(object, int, int)
    def relay_frame_completed(self, result: object, current: int, total: int) -> None:
        self.frame_completed.emit(self._run_id, result, current, total)

    @Slot(int, int)
    def relay_progress_changed(self, current: int, total: int) -> None:
        self.progress_changed.emit(self._run_id, current, total)

    @Slot(bool)
    def relay_pause_changed(self, paused: bool) -> None:
        self.pause_changed.emit(self._run_id, paused)

    @Slot(int, int)
    def relay_stopped(self, current: int, total: int) -> None:
        self.stopped.emit(self._run_id, current, total)

    @Slot(str, int, float)
    def relay_completed(self, path: str, frames: int, elapsed: float) -> None:
        self.completed.emit(self._run_id, path, frames, elapsed)

    @Slot(str)
    def relay_failed(self, message: str) -> None:
        self.failed.emit(self._run_id, message)

    @Slot(bool)
    def relay_finished(self, stopped: bool) -> None:
        self.finished.emit(self._run_id, stopped)

    @Slot()
    def relay_thread_finished(self) -> None:
        self.thread_finished.emit(self._run_id, self)


class VideoInferencePage(QWidget):
    """Open/process/pause/stop/save one local video using a stable detector."""

    busy_changed = Signal(bool)

    def __init__(
        self,
        *,
        model_provider: Callable[[], DetectorAdapter | None],
        start_allowed: Callable[[], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_provider = model_provider
        self._start_allowed = start_allowed
        self._video_path: Path | None = None
        self._output_path: Path | None = None
        self._source_info: VideoSourceInfo | None = None
        self._thread: QThread | None = None
        self._worker: VideoInferenceWorker | None = None
        self._bridge: _VideoRunBridge | None = None
        self._bridges: dict[int, _VideoRunBridge] = {}
        self._run_id = 0
        self._failed = False
        self._shutdown_requested = False
        self._shutdown_timed_out = False
        self._finished_successfully = False
        self._build_ui()

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @property
    def selected_video_path(self) -> Path | None:
        return self._video_path

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def _build_ui(self) -> None:
        self.setObjectName("videoPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title = QLabel("Video Fruit Detection", self)
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        intro = QLabel("Process a local video frame by frame and export an annotated MP4. Use Live Camera for real-time input.", self)
        intro.setObjectName("pageIntro")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        controls = QHBoxLayout()
        self._open_button = QPushButton("Open Video...", self)
        self._open_button.setObjectName("videoOpenButton")
        self._save_button = QPushButton("Set Output File...", self)
        self._save_button.setObjectName("videoSaveButton")
        self._play_button = QPushButton("Start Processing", self)
        self._play_button.setObjectName("videoPlayButton")
        self._pause_button = QPushButton("Pause", self)
        self._pause_button.setObjectName("videoPauseButton")
        self._stop_button = QPushButton("Stop", self)
        self._stop_button.setObjectName("videoStopButton")
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        controls.addWidget(self._open_button)
        controls.addWidget(self._save_button)
        controls.addWidget(self._play_button)
        controls.addWidget(self._pause_button)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        settings_box = QGroupBox("Detection Parameters", self)
        settings = QFormLayout(settings_box)
        self._confidence = QDoubleSpinBox(self)
        self._confidence.setObjectName("videoConfidenceSpinBox")
        self._confidence.setRange(0.0, 1.0)
        self._confidence.setDecimals(2)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(0.25)
        self._nms_iou = QDoubleSpinBox(self)
        self._nms_iou.setObjectName("videoNmsIouSpinBox")
        self._nms_iou.setRange(0.0, 1.0)
        self._nms_iou.setDecimals(2)
        self._nms_iou.setSingleStep(0.05)
        self._nms_iou.setValue(0.50)
        settings.addRow("Confidence Threshold", self._confidence)
        settings.addRow("NMS IoU Threshold", self._nms_iou)
        layout.addWidget(settings_box)

        self._frame_view = VideoFrameView(self)
        layout.addWidget(self._frame_view, 1)
        self._file_label = QLabel("No video selected", self)
        self._file_label.setObjectName("videoFileLabel")
        self._fps_label = QLabel("Inference FPS: --", self)
        self._fps_label.setObjectName("videoFpsLabel")
        layout.addWidget(self._file_label)
        layout.addWidget(self._fps_label)
        self._progress = QProgressBar(self)
        self._progress.setObjectName("videoProgressBar")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._summary = QLabel("Open a local video, set a new .mp4 output file, then start processing.", self)
        self._summary.setObjectName("videoInferenceSummary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._open_button.clicked.connect(self._choose_video)
        self._save_button.clicked.connect(self._choose_output)
        self._play_button.clicked.connect(self.start_processing)
        self._pause_button.clicked.connect(self.toggle_pause)
        self._stop_button.clicked.connect(self.stop_processing)

    @Slot()
    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a Local Fruit Video", "", "Video files (*.mp4 *.mov *.avi *.mkv *.m4v)"
        )
        if path:
            self.set_video(path)

    def set_video(self, path: str | Path) -> bool:
        if self._shutdown_requested or self.is_running:
            self._show_error("Processing is running or the page is closing. The video cannot be changed.")
            return False
        try:
            selected = resolve_video_path(path)
        except VideoInferenceError as error:
            self._show_error(str(error))
            return False
        self._video_path = selected
        self._source_info = None
        self._finished_successfully = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._file_label.setText(f"Input video: {selected.name}")
        self._summary.setText("Set a new .mp4 output file, then click Start Processing.")
        return True

    @Slot()
    def _choose_output(self) -> None:
        default = "processed_fruit_video.mp4" if self._video_path is None else f"{self._video_path.stem}_processed.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed Video", default, "MP4 video (*.mp4)")
        if path:
            self.set_output_path(path)

    def set_output_path(self, path: str | Path) -> bool:
        if self._shutdown_requested or self.is_running:
            self._show_error("Processing is running or the page is closing. The output file cannot be changed.")
            return False
        output = Path(path).expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            self._show_error("Select a .mp4 filename for safe export.")
            return False
        if output.exists():
            self._show_error("The output file already exists. Existing videos will not be overwritten.")
            return False
        self._output_path = output
        self._finished_successfully = False
        self._summary.setText(f"Output file set to {output.name}. Click Start Processing when ready.")
        return True

    @Slot()
    def start_processing(self) -> bool:
        if self._shutdown_requested or self.is_running:
            return False
        if not self._start_allowed():
            self._show_error("Another inference task is running. Wait for it to finish or stop it first.")
            return False
        adapter = self._model_provider()
        if adapter is None:
            self._show_error("Load compatible Student model weights first.")
            return False
        if self._video_path is None:
            self._show_error("Open a local video first.")
            return False
        if self._output_path is None:
            self._show_error("Set a new .mp4 output file first.")
            return False
        try:
            settings = VideoInferenceSettings(self._confidence.value(), self._nms_iou.value())
            worker = VideoInferenceWorker(
                adapter=adapter,
                video_path=self._video_path,
                output_path=self._output_path,
                settings=settings,
            )
        except VideoInferenceError as error:
            self._show_error(str(error))
            return False
        self._run_id += 1
        run_id = self._run_id
        self._failed = False
        self._finished_successfully = False
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        bridge = _VideoRunBridge(run_id, self)
        worker.opened.connect(bridge.relay_opened)
        worker.frame_completed.connect(bridge.relay_frame_completed)
        worker.progress_changed.connect(bridge.relay_progress_changed)
        worker.pause_changed.connect(bridge.relay_pause_changed)
        worker.stopped.connect(bridge.relay_stopped)
        worker.completed.connect(bridge.relay_completed)
        worker.failed.connect(bridge.relay_failed)
        worker.finished.connect(bridge.relay_finished)
        bridge.opened.connect(self._on_opened)
        bridge.frame_completed.connect(self._on_frame_completed)
        bridge.progress_changed.connect(self._on_progress_changed)
        bridge.pause_changed.connect(self._on_pause_changed)
        bridge.stopped.connect(self._on_stopped)
        bridge.completed.connect(self._on_completed)
        bridge.failed.connect(self._on_failed)
        bridge.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(bridge.relay_thread_finished)
        bridge.thread_finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._bridge = bridge
        self._bridges[run_id] = bridge
        self._set_busy(True)
        thread.start()
        return True

    @Slot()
    def toggle_pause(self) -> None:
        worker = self._worker
        if worker is None:
            return
        paused = self._pause_button.text() == "Pause"
        worker.request_pause(paused)

    @Slot()
    def stop_processing(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            self._summary.setText("Stop requested. Processing will stop after the current frame without publishing an incomplete video.")

    def shutdown(self, *, wait_ms: int = 5_000) -> bool:
        worker, thread = self._worker, self._thread
        if worker is None and thread is None:
            return True
        self._shutdown_requested = True
        self._run_id += 1
        self._finished_successfully = False
        if worker is not None:
            worker.request_stop()
        if thread is None:
            self._worker = None
            self._set_busy(False)
            return True
        thread.quit()
        if not thread.wait(wait_ms):
            # A failed close is not a permanent close.  Keep the run id
            # invalidated so stale queued worker output stays harmless, then
            # let this exact bridge restore a usable page after the worker has
            # actually exited.
            self._shutdown_timed_out = True
            self._show_error("The video thread is still stopping. Wait for the current frame before closing the application.")
            return False
        self._thread = None
        self._worker = None
        self._shutdown_timed_out = False
        self._set_busy(False)
        return True

    @Slot(int, object)
    def _on_opened(self, run_id: int, info: object) -> None:
        if run_id != self._run_id or not isinstance(info, VideoSourceInfo):
            return
        self._source_info = info
        self._progress.setRange(0, max(1, info.frame_count))
        self._summary.setText(f"Processing {info.path.name}: {info.frame_count} frames at {info.fps:.2f} source FPS.")

    @Slot(int, object, int, int)
    def _on_frame_completed(self, run_id: int, result: object, current: int, total: int) -> None:
        if run_id != self._run_id or not isinstance(result, VideoFrameResult):
            return
        self._frame_view.show_frame(result.rgb_frame, result.frame_index)
        self._fps_label.setText(f"Inference FPS: {result.inference_fps:.2f}; source FPS: {result.source_fps:.2f}")
        self._summary.setText(f"Processing frame {current}/{total if total > 0 else '?'}.")

    @Slot(int, int, int)
    def _on_progress_changed(self, run_id: int, current: int, total: int) -> None:
        if run_id == self._run_id:
            self._progress.setRange(0, max(1, total))
            self._progress.setValue(current)

    @Slot(int, bool)
    def _on_pause_changed(self, run_id: int, paused: bool) -> None:
        if run_id != self._run_id:
            return
        self._pause_button.setText("Resume" if paused else "Pause")
        self._summary.setText("Video processing paused." if paused else "Video processing resumed.")

    @Slot(int, int, int)
    def _on_stopped(self, run_id: int, current: int, total: int) -> None:
        if run_id == self._run_id:
            self._finished_successfully = False
            self._summary.setText(f"Stopped after {current}/{total if total > 0 else '?'} frames. No output file was published.")

    @Slot(int, str, int, float)
    def _on_completed(self, run_id: int, path: str, frames: int, elapsed: float) -> None:
        if run_id == self._run_id:
            self._finished_successfully = True
            self._summary.setText(f"Completed {frames} frames and wrote {Path(path).name} in {elapsed:.1f} seconds.")

    @Slot(int, str)
    def _on_failed(self, run_id: int, message: str) -> None:
        if run_id == self._run_id:
            self._failed = True
            self._finished_successfully = False
            self._show_error(message)

    @Slot(int, bool)
    def _on_finished(self, run_id: int, stopped: bool) -> None:
        if run_id == self._run_id and (stopped or self._failed):
            self._finished_successfully = False

    @Slot(int, object)
    def _on_thread_finished(self, run_id: int, candidate: object) -> None:
        bridge = self._bridges.get(run_id)
        if bridge is not candidate:
            return
        self._bridges.pop(run_id, None)
        active = self._bridge is bridge
        if active:
            self._bridge = None
        bridge.deleteLater()
        if not active:
            return
        self._thread = None
        self._worker = None
        if self._shutdown_timed_out:
            # ``shutdown() == False`` refused application close because a
            # worker was still in flight.  Once that worker exits, the page is
            # again usable; a successful shutdown deliberately leaves this
            # flag true and never reaches this recovery branch.
            self._shutdown_requested = False
            self._shutdown_timed_out = False
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._open_button.setEnabled(not busy)
        self._save_button.setEnabled(not busy)
        self._play_button.setEnabled(not busy)
        self._confidence.setEnabled(not busy)
        self._nms_iou.setEnabled(not busy)
        self._pause_button.setEnabled(busy)
        self._stop_button.setEnabled(busy)
        if not busy:
            self._pause_button.setText("Pause")
        self.busy_changed.emit(busy)

    def _show_error(self, message: str) -> None:
        self._summary.setText(message)
