"""Image preview and file-inference controls for the desktop demonstrator."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from fruit_ssod.detection import DetectionRecord, DetectorAdapter
from fruit_ssod.gui.result_exporter import ExportManifest, ResultExportError, export_inference_results
from fruit_ssod.gui.workers.image_worker import (
    ImageInferenceError,
    ImageInferenceResult,
    ImageInferenceSettings,
    ImageInferenceWorker,
    SUPPORTED_IMAGE_SUFFIXES,
    resolve_image_paths,
)


_BOX_COLORS = {
    0: QColor(220, 45, 45), 1: QColor(240, 190, 25), 2: QColor(245, 130, 35),
    3: QColor(210, 55, 100), 4: QColor(120, 155, 35),
}

_FRUIT_DISPLAY_NAMES = {name: name for name in ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")}


def _display_fruit_name(name: str) -> str:
    """Show Chinese labels in the GUI while keeping English class IDs in exports."""
    return _FRUIT_DISPLAY_NAMES.get(name, name)


class ImageView(QWidget):
    """Render a selected local image and validated known-fruit detections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePreview")
        self.setMinimumSize(420, 320)
        self._pixmap = QPixmap()
        self._detections: tuple[DetectionRecord, ...] = ()
        self._path: Path | None = None

    @property
    def image_path(self) -> Path | None:
        """Expose the currently previewed source image for tests and navigation."""
        return self._path

    @property
    def detections(self) -> tuple[DetectionRecord, ...]:
        """Expose the current immutable detection tuple without model internals."""
        return self._detections

    def show_image(self, path: str | Path, detections: Iterable[DetectionRecord] = ()) -> bool:
        """Display an image path; return False when Qt cannot decode it."""
        resolved = Path(path).resolve()
        pixmap = QPixmap(str(resolved))
        if pixmap.isNull():
            self._pixmap = QPixmap()
            self._detections = ()
            self._path = None
            self.update()
            return False
        self._pixmap = pixmap
        self._detections = tuple(detections)
        self._path = resolved
        self.update()
        return True

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt API name
        """Scale the source once and transform model XYXY boxes consistently."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap.isNull():
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignCenter, "Select an image to view detection results")
            return
        target = self.rect().adjusted(8, 8, -8, -8)
        source_size = self._pixmap.size()
        scaled = source_size.scaled(target.size(), Qt.KeepAspectRatio)
        x_offset = target.x() + (target.width() - scaled.width()) / 2
        y_offset = target.y() + (target.height() - scaled.height()) / 2
        destination = QRectF(x_offset, y_offset, scaled.width(), scaled.height())
        painter.drawPixmap(destination, self._pixmap, QRectF(self._pixmap.rect()))
        scale_x = destination.width() / source_size.width()
        scale_y = destination.height() / source_size.height()
        for detection in self._detections:
            x1, y1, x2, y2 = detection.xyxy
            color = _BOX_COLORS[detection.class_id]
            painter.setPen(QPen(color, 2))
            painter.drawRect(
                QRectF(
                    destination.x() + x1 * scale_x,
                    destination.y() + y1 * scale_y,
                    (x2 - x1) * scale_x,
                    (y2 - y1) * scale_y,
                )
            )
            painter.drawText(
                destination.x() + x1 * scale_x + 2,
                max(destination.y() + 14, destination.y() + y1 * scale_y - 3),
                f"{_display_fruit_name(detection.class_name)} {detection.confidence:.2f}",
            )


class _RunSignalBridge(QObject):
    """Attach a worker's queued signals to one immutable page-run identifier."""

    image_completed = Signal(int, object, int, int)
    image_failed = Signal(int, str, str, int, int)
    progress_changed = Signal(int, int, int)
    cancelled = Signal(int, int, int)
    finished = Signal(int, int, int, bool)
    # Include this immutable bridge in the notification.  A run id alone is
    # deliberately insufficient: ``shutdown()`` invalidates the visible run
    # id before a blocked worker can finish, and a later queued notification
    # must still be able to retire *its own* active QThread/worker pair.
    thread_finished = Signal(int, object)

    def __init__(self, run_id: int, parent: QObject) -> None:
        super().__init__(parent)
        self._run_id = run_id

    @Slot(object, int, int)
    def relay_image_completed(self, result: object, completed: int, total: int) -> None:
        self.image_completed.emit(self._run_id, result, completed, total)

    @Slot(str, str, int, int)
    def relay_image_failed(self, path: str, message: str, completed: int, total: int) -> None:
        self.image_failed.emit(self._run_id, path, message, completed, total)

    @Slot(int, int)
    def relay_progress_changed(self, completed: int, total: int) -> None:
        self.progress_changed.emit(self._run_id, completed, total)

    @Slot(int, int)
    def relay_cancelled(self, completed: int, total: int) -> None:
        self.cancelled.emit(self._run_id, completed, total)

    @Slot(int, int, bool)
    def relay_finished(self, completed: int, total: int, was_cancelled: bool) -> None:
        self.finished.emit(self._run_id, completed, total, was_cancelled)

    @Slot()
    def relay_thread_finished(self) -> None:
        self.thread_finished.emit(self._run_id, self)


class ImageInferencePage(QWidget):
    """Reusable single-image or folder workflow with one cooperative worker at a time."""

    busy_changed = Signal(bool)

    def __init__(
        self,
        *,
        model_provider: Callable[[], DetectorAdapter | None],
        mode: str,
        start_allowed: Callable[[], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in {"single", "batch"}:
            raise ValueError("mode must be 'single' or 'batch'")
        self._mode = mode
        self._model_provider = model_provider
        self._start_allowed = start_allowed
        self._paths: tuple[Path, ...] = ()
        self._results: dict[Path, ImageInferenceResult] = {}
        self._preview_index = 0
        self._thread: QThread | None = None
        self._worker: ImageInferenceWorker | None = None
        self._signal_bridge: _RunSignalBridge | None = None
        # Keep every bridge alive until its thread-finished notification is
        # delivered.  ``shutdown()`` can join a thread while GUI events are not
        # being processed, so clearing the sole reference there would leak a
        # child bridge and make its deferred deletion unobservable.
        self._bridges: dict[int, _RunSignalBridge] = {}
        # Every worker signal is tied to a monotonically increasing UI run.  A
        # queued signal from a just-finished worker must never repopulate the
        # result set for a newer run.
        self._run_id = 0
        self._shutdown_requested = False
        # A false-returning shutdown leaves this page alive.  Track that case
        # separately so a later queued thread-finished notification restores
        # usability only after a close was actually refused.
        self._shutdown_timed_out = False
        self._round_finished_successfully = False
        self._build_ui()

    @property
    def is_running(self) -> bool:
        """Whether this page currently owns an inference worker."""
        return self._thread is not None

    @property
    def selected_paths(self) -> tuple[Path, ...]:
        """The validated ordered list selected by file or folder controls."""
        return self._paths

    @property
    def results(self) -> tuple[ImageInferenceResult, ...]:
        """Completed successful results in the original selection order."""
        return tuple(self._results[path] for path in self._paths if path in self._results)

    def _build_ui(self) -> None:
        self.setObjectName(f"{self._mode}_imagePage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title = QLabel("Single-Image Fruit Detection" if self._mode == "single" else "Batch Fruit Detection", self)
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        intro = QLabel(
            "Select a local image and run the Student detector to view classes, confidence scores, boxes and latency."
            if self._mode == "single"
            else "Select multiple images or a folder for batch processing, result review and export.",
            self,
        )
        intro.setObjectName("pageIntro")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        controls = QHBoxLayout()
        self._choose_images_button = QPushButton("Select Images...", self)
        self._choose_images_button.setObjectName(f"{self._mode}ChooseImagesButton")
        controls.addWidget(self._choose_images_button)
        self._choose_folder_button: QPushButton | None = None
        if self._mode == "batch":
            self._choose_folder_button = QPushButton("Select Folder...", self)
            self._choose_folder_button.setObjectName("batchChooseFolderButton")
            controls.addWidget(self._choose_folder_button)
        self._run_button = QPushButton("Run Detection", self)
        self._run_button.setObjectName(f"{self._mode}RunInferenceButton")
        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.setObjectName(f"{self._mode}CancelInferenceButton")
        self._cancel_button.setEnabled(False)
        self._export_button = QPushButton("Export Results...", self)
        self._export_button.setObjectName(f"{self._mode}ExportResultsButton")
        self._export_button.setEnabled(False)
        controls.addWidget(self._run_button)
        controls.addWidget(self._cancel_button)
        controls.addWidget(self._export_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        settings_box = QGroupBox("Detection Parameters", self)
        settings = QFormLayout(settings_box)
        self._confidence = QDoubleSpinBox(self)
        self._confidence.setObjectName(f"{self._mode}ConfidenceSpinBox")
        self._confidence.setRange(0.0, 1.0)
        self._confidence.setDecimals(2)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(0.25)
        self._nms_iou = QDoubleSpinBox(self)
        self._nms_iou.setObjectName(f"{self._mode}NmsIouSpinBox")
        self._nms_iou.setRange(0.0, 1.0)
        self._nms_iou.setDecimals(2)
        self._nms_iou.setSingleStep(0.05)
        self._nms_iou.setValue(0.50)
        settings.addRow("Confidence Threshold", self._confidence)
        settings.addRow("NMS IoU Threshold", self._nms_iou)
        layout.addWidget(settings_box)

        self._image_view = ImageView(self)
        layout.addWidget(self._image_view, 1)
        navigation = QHBoxLayout()
        self._previous_button = QPushButton("Previous", self)
        self._previous_button.setObjectName(f"{self._mode}PreviousButton")
        self._next_button = QPushButton("Next", self)
        self._next_button.setObjectName(f"{self._mode}NextButton")
        self._file_label = QLabel("No image selected", self)
        self._file_label.setObjectName(f"{self._mode}FileLabel")
        navigation.addWidget(self._previous_button)
        navigation.addWidget(self._next_button)
        navigation.addWidget(self._file_label, 1)
        layout.addLayout(navigation)
        self._progress = QProgressBar(self)
        self._progress.setObjectName(f"{self._mode}ProgressBar")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._summary = QLabel("Load a model, select an image, then click Run Detection.", self)
        self._summary.setObjectName(f"{self._mode}InferenceSummary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._choose_images_button.clicked.connect(self._choose_images)
        if self._choose_folder_button is not None:
            self._choose_folder_button.clicked.connect(self._choose_folder)
        self._run_button.clicked.connect(self.start_inference)
        self._cancel_button.clicked.connect(self.cancel_inference)
        self._previous_button.clicked.connect(lambda: self._navigate(-1))
        self._next_button.clicked.connect(lambda: self._navigate(1))
        self._export_button.clicked.connect(self._choose_export_folder)
        self._update_navigation()

    @Slot()
    def _choose_images(self) -> None:
        filter_text = "Image files (*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Fruit Images", "", filter_text)
        if paths:
            self.set_images(paths)

    @Slot()
    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Fruit Image Folder")
        if not folder:
            return
        candidates = sorted(
            (path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
            key=lambda path: (path.name.casefold(), str(path)),
        )
        self.set_images(candidates)

    def set_images(self, paths: Iterable[str | Path]) -> bool:
        """Set an ordered preview list from non-dialog callers and automated tests."""
        if self._shutdown_requested:
            self._show_error("This page is closing and cannot accept new images.")
            return False
        if self.is_running:
            self._show_error("An inference task is running. Wait for it to finish or cancel it.")
            return False
        try:
            resolved = resolve_image_paths(paths)
        except ImageInferenceError as error:
            self._show_error(str(error))
            return False
        if self._mode == "single" and len(resolved) != 1:
            self._show_error("Single-image mode accepts one image only. Use Batch Processing for multiple images.")
            return False
        self._paths = resolved
        self._results.clear()
        self._preview_index = 0
        self._round_finished_successfully = False
        self._progress.setRange(0, len(resolved))
        self._progress.setValue(0)
        self._export_button.setEnabled(False)
        self._show_current_preview()
        return True

    @Slot()
    def start_inference(self) -> bool:
        """Create a worker/QThread pair; no detector call occurs in the GUI thread."""
        if self._shutdown_requested:
            self._show_error("This page is closing and cannot start a new inference task.")
            return False
        if self.is_running:
            return False
        if not self._start_allowed():
            self._show_error("Another inference task is running. Wait for it to finish or cancel it first.")
            return False
        adapter = self._model_provider()
        if adapter is None:
            self._show_error("Load compatible Student model weights first.")
            return False
        if not self._paths:
            self._show_error("Select an image or image folder first.")
            return False
        try:
            settings = ImageInferenceSettings(confidence=self._confidence.value(), nms_iou=self._nms_iou.value())
            worker = ImageInferenceWorker(adapter=adapter, image_paths=self._paths, settings=settings)
        except ImageInferenceError as error:
            self._show_error(str(error))
            return False
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Clear first, before starting the thread, so that a new run cannot
        # display or export prior controls/results during its busy period.
        self._run_id += 1
        run_id = self._run_id
        self._results.clear()
        self._preview_index = 0
        self._round_finished_successfully = False
        self._progress.setRange(0, len(self._paths))
        self._progress.setValue(0)
        self._export_button.setEnabled(False)
        bridge = _RunSignalBridge(run_id, self)
        worker.image_completed.connect(bridge.relay_image_completed)
        worker.image_failed.connect(bridge.relay_image_failed)
        worker.progress_changed.connect(bridge.relay_progress_changed)
        worker.cancelled.connect(bridge.relay_cancelled)
        worker.finished.connect(bridge.relay_finished)
        bridge.image_completed.connect(self._on_image_completed)
        bridge.image_failed.connect(self._on_image_failed)
        bridge.progress_changed.connect(self._on_progress_changed)
        bridge.cancelled.connect(self._on_cancelled)
        bridge.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(bridge.relay_thread_finished)
        bridge.thread_finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._signal_bridge = bridge
        self._bridges[run_id] = bridge
        self._set_busy(True)
        self._show_current_preview()
        thread.start()
        return True

    @Slot()
    def cancel_inference(self) -> None:
        """Signal cancellation without terminating a model call mid-execution."""
        if self._worker is not None:
            self._worker.request_cancel()
            self._cancel_button.setEnabled(False)
            self._summary.setText("Cancellation requested. The task will stop after the current image.")

    def shutdown(self, *, wait_ms: int = 5_000) -> bool:
        """Cancel and join an inference worker before Qt destroys this page."""
        worker = self._worker
        thread = self._thread
        # MainWindow shuts pages down sequentially.  An idle page may be
        # visited before another page refuses close because its worker did not
        # stop in time.  Returning success for this no-op must leave the idle
        # page reusable after that rejected window close.
        if worker is None and thread is None:
            return True
        # Invalidate the run *before* requesting cancellation.  A worker may
        # have already queued image/finished signals, and ``wait`` below blocks
        # the GUI event loop that would otherwise deliver them.  Those queued
        # slots must be harmless after close has begun.
        self._shutdown_requested = True
        self._run_id += 1
        self._results.clear()
        self._round_finished_successfully = False
        self._export_button.setEnabled(False)
        if thread is None:
            # This is an internal-invariant fallback: no QThread can still be
            # alive, so there is nothing to join.  Keep behaviour safe even if
            # an unexpected detached worker reference is observed.
            self._worker = None
            self._set_busy(False)
            return True
        if worker is not None:
            worker.request_cancel()
        thread.quit()
        if not thread.wait(wait_ms):
            self._shutdown_timed_out = True
            self._show_error(
                "The inference thread is still stopping. Wait for the current image before closing the application."
            )
            return False
        # ``finished`` can be queued while the GUI thread is blocked in wait();
        # clear only the active worker/thread references now.  The bridge stays
        # in ``_bridges`` until its queued thread-finished slot performs a safe
        # deleteLater cleanup.
        self._thread = None
        self._worker = None
        self._shutdown_timed_out = False
        self._set_busy(False)
        return True

    @Slot(int, object, int, int)
    def _on_image_completed(self, run_id: int, result: object, completed: int, total: int) -> None:
        if run_id != self._run_id:
            return
        if not isinstance(result, ImageInferenceResult):
            self._show_error("The inference worker returned an invalid result. Please retry.")
            return
        self._results[result.image_path] = result
        try:
            self._preview_index = self._paths.index(result.image_path)
        except ValueError:
            return
        self._show_current_preview()
        self._summary.setText(self._summary_for(result, completed, total))

    @Slot(int, str, str, int, int)
    def _on_image_failed(self, run_id: int, path: str, message: str, completed: int, total: int) -> None:
        if run_id != self._run_id:
            return
            self._summary.setText(f"{completed}/{total}: {Path(path).name} failed. {message}")

    @Slot(int, int, int)
    def _on_progress_changed(self, run_id: int, completed: int, total: int) -> None:
        if run_id != self._run_id:
            return
        self._progress.setRange(0, total)
        self._progress.setValue(completed)

    @Slot(int, int, int)
    def _on_cancelled(self, run_id: int, completed: int, total: int) -> None:
        if run_id != self._run_id:
            return
            self._summary.setText(f"Cancelled after {completed}/{total} image(s). This run cannot be exported.")

    @Slot(int, int, int, bool)
    def _on_finished(self, run_id: int, completed: int, total: int, was_cancelled: bool) -> None:
        if run_id != self._run_id:
            return
        self._round_finished_successfully = not was_cancelled
        if not was_cancelled:
            self._summary.setText(f"Completed {completed}/{total} image(s); {len(self._results)} succeeded. Results are ready to export.")
        self._refresh_export_enabled()

    @Slot(int, object)
    def _on_thread_finished(self, run_id: int, finished_bridge: object) -> None:
        # Always retire this bridge, including stale runs invalidated by
        # shutdown.  The conditional below protects a newer UI run from an old
        # queued notification, while this cleanup prevents one QObject child
        # from accumulating per completed round.
        bridge = self._bridges.get(run_id)
        # The bridge is created once for a particular QThread.  Do not let a
        # malformed or stale queued signal for a reused integer run id retire a
        # different bridge (and therefore a newer worker).
        if bridge is not finished_bridge:
            return
        self._bridges.pop(run_id, None)
        if bridge is not None:
            was_active_bridge = self._signal_bridge is bridge
            if self._signal_bridge is bridge:
                self._signal_bridge = None
            bridge.deleteLater()
        else:
            was_active_bridge = False
        # Normally the visible run id matches.  A shutdown that timed out is
        # different: it intentionally advanced ``_run_id`` to discard queued
        # worker output while the same bridge remains the active owner.  In
        # that case this exact bridge is still authoritative for clearing the
        # active thread/worker references and restoring the page's usable
        # state.  An old bridge can never clear a newer run because it no
        # longer equals ``_signal_bridge``.
        if run_id != self._run_id and not was_active_bridge:
            return
        self._thread = None
        self._worker = None
        if self._shutdown_timed_out and was_active_bridge:
            # ``shutdown() == False`` means application close was refused, not
            # that this page must remain permanently disabled.  Once the
            # cooperatively cancelled worker has actually exited, allow the
            # user to select images and start a clean run again.
            self._shutdown_requested = False
            self._shutdown_timed_out = False
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._choose_images_button.setEnabled(not busy)
        if self._choose_folder_button is not None:
            self._choose_folder_button.setEnabled(not busy)
        self._run_button.setEnabled(not busy)
        self._cancel_button.setEnabled(busy)
        self._confidence.setEnabled(not busy)
        self._nms_iou.setEnabled(not busy)
        self._refresh_export_enabled()
        self._update_navigation()
        self.busy_changed.emit(busy)

    def _refresh_export_enabled(self) -> None:
        """Expose exports only for a successfully completed, non-busy UI run."""
        self._export_button.setEnabled(
            not self.is_running and self._round_finished_successfully and bool(self._results)
        )

    def _navigate(self, direction: int) -> None:
        if not self._paths:
            return
        self._preview_index = (self._preview_index + direction) % len(self._paths)
        self._show_current_preview()

    def _show_current_preview(self) -> None:
        if not self._paths:
            self._file_label.setText("No image selected")
            self._update_navigation()
            return
        path = self._paths[self._preview_index]
        result = self._results.get(path)
        if not self._image_view.show_image(path, () if result is None else result.detections):
            self._show_error(f"Cannot preview {path.name}. Confirm that the image is readable.")
            return
        prefix = f"{self._preview_index + 1}/{len(self._paths)}"
        self._file_label.setText(f"{prefix}: {path.name}")
        self._update_navigation()

    def _update_navigation(self) -> None:
        enabled = len(self._paths) > 1 and not self.is_running
        self._previous_button.setEnabled(enabled)
        self._next_button.setEnabled(enabled)

    @staticmethod
    def _summary_for(result: ImageInferenceResult, completed: int, total: int) -> str:
        counts = ", ".join(f"{_display_fruit_name(name)} × {count}" for name, count in result.class_counts.items()) or "No objects detected"
        confidences = ", ".join(f"{item.confidence:.2f}" for item in result.detections) or "None"
        return (
            f"{completed}/{total}: {result.image_path.name}; {counts}; confidence {confidences}; "
            f"latency {result.latency_ms:.1f} ms; settings: confidence {result.confidence:.2f}, NMS IoU {result.nms_iou:.2f}."
        )

    @Slot()
    def _choose_export_folder(self) -> None:
        parent = QFileDialog.getExistingDirectory(self, "Select Results Folder")
        if parent:
            # The exporter deliberately refuses to overwrite an existing
            # package.  Select a parent directory in the dialog and create a
            # fresh, human-readable child package automatically.
            destination = Path(parent) / f"fruit_detection_results_{uuid4().hex[:8]}"
            self.export_results(destination)

    def export_results(self, output_dir: str | Path) -> ExportManifest | None:
        """Export only completed image results and present a direct local path summary."""
        if self._shutdown_requested:
            self._show_error("This page is closing and cannot export results.")
            return None
        if self.is_running:
            self._show_error("Inference is still running. Export after it finishes.")
            return None
        if not self._round_finished_successfully:
            self._show_error("There are no successful detection results to export.")
            return None
        try:
            manifest = export_inference_results(self.results, output_dir)
        except ResultExportError as error:
            self._show_error(str(error))
            return None
        self._summary.setText(f"Exported {len(manifest.annotated_images)} annotated image(s), CSV and JSON to {manifest.output_dir}")
        return manifest

    def _show_error(self, message: str) -> None:
        self._summary.setText(message)
