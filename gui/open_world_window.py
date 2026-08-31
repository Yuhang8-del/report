"""English PySide6 demonstrator for box-level open-world fruit detection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fruit_ssod.open_world.pipeline import OpenWorldDetection, OpenWorldFruitPipeline, OpenWorldInferenceResult


CLASS_NAMES = {name: name for name in (
    "Apple", "Banana", "Orange", "Strawberry", "Pineapple",
    "Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon",
)}


def display_label(value: str) -> str:
    """Translate detector labels while preserving auditable cluster suffixes."""
    return CLASS_NAMES.get(value, value)


class OpenWorldCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(680, 500)
        self._pixmap = QPixmap()
        self._detections: tuple[OpenWorldDetection, ...] = ()

    def show_result(self, path: str | Path, detections: tuple[OpenWorldDetection, ...] = ()) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        self._pixmap = pixmap
        self._detections = detections
        self.update()
        return True

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        if self._pixmap.isNull():
            painter.setPen(QColor("#CBD5E1"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Select an image and run open-world detection")
            return
        target = self.rect().adjusted(12, 12, -12, -12)
        scaled = self._pixmap.size().scaled(target.size(), Qt.KeepAspectRatio)
        destination = QRectF(
            target.x() + (target.width() - scaled.width()) / 2,
            target.y() + (target.height() - scaled.height()) / 2,
            scaled.width(),
            scaled.height(),
        )
        painter.drawPixmap(destination, self._pixmap, QRectF(self._pixmap.rect()))
        sx = destination.width() / self._pixmap.width()
        sy = destination.height() / self._pixmap.height()
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        known_colors = [QColor("#EF4444"), QColor("#F59E0B"), QColor("#FB923C"), QColor("#EC4899"), QColor("#84CC16")]
        dense_scene = len(self._detections) > 40
        known_labels_drawn = 0
        for detection in self._detections:
            x1, y1, x2, y2 = detection.xyxy
            color = (
                QColor("#FACC15")
                if detection.kind == "unknown"
                else known_colors[(detection.class_id or 0) % len(known_colors)]
            )
            pen = QPen(color, 4 if detection.kind == "unknown" else (2 if dense_scene else 3))
            if detection.kind == "unknown":
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            rect = QRectF(
                destination.x() + x1 * sx,
                destination.y() + y1 * sy,
                (x2 - x1) * sx,
                (y2 - y1) * sy,
            )
            painter.drawRect(rect)
            show_label = detection.kind == "unknown" or not dense_scene or known_labels_drawn < 12
            if not show_label:
                continue
            if detection.kind == "known":
                known_labels_drawn += 1
            label = f"{display_label(detection.label)}  {detection.score:.2f}"
            text_rect = painter.fontMetrics().boundingRect(label).adjusted(-6, -4, 6, 4)
            # QRect requires an integer QPoint. Passing QRectF.topLeft()
            # raises inside Qt's paint callback after the first rectangle,
            # leaving every later detection invisible while the table remains
            # complete.
            text_rect.moveTopLeft(rect.topLeft().toPoint())
            painter.fillRect(text_rect, QColor(15, 23, 42, 220))
            painter.setPen(color)
            painter.drawText(text_rect, Qt.AlignCenter, label)


class _InferenceWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, pipeline: OpenWorldFruitPipeline, image_path: Path) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path

    @Slot()
    def run(self) -> None:
        try:
            self.completed.emit(self.pipeline.predict(self.image_path))
        except Exception as error:  # GUI boundary must surface backend context.
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            self.finished.emit()


class OpenWorldWindow(QMainWindow):
    def __init__(self, pipeline: OpenWorldFruitPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image_path: Path | None = None
        self.result: OpenWorldInferenceResult | None = None
        self.thread: QThread | None = None
        self.worker: _InferenceWorker | None = None
        self.setWindowTitle("Open-World Fruit Detection Studio")
        self.resize(1380, 850)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#F1F5F9;color:#0F172A;font-family:'Segoe UI';}"
            "QFrame#header{background:#0F172A;border-radius:12px;}"
            "QLabel#title{color:white;font-size:25px;font-weight:700;}"
            "QLabel#subtitle{color:#94A3B8;font-size:13px;}"
            "QPushButton{background:#2563EB;color:white;border:0;border-radius:7px;padding:10px 18px;font-weight:600;}"
            "QPushButton:disabled{background:#94A3B8;}"
            "QTableWidget{background:white;border:1px solid #CBD5E1;border-radius:8px;gridline-color:#E2E8F0;}"
        )
        self._build()

    def _build(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        header = QFrame(root)
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 22, 16)
        text = QVBoxLayout()
        title = QLabel("Open-World Fruit Detection", header)
        title.setObjectName("title")
        subtitle = QLabel("Five known classes · Unknown box proposals · Cluster assignment · Candidate naming", header)
        subtitle.setObjectName("subtitle")
        text.addWidget(title)
        text.addWidget(subtitle)
        header_layout.addLayout(text, 1)
        self.choose_button = QPushButton("Select Image", header)
        self.run_button = QPushButton("Run Open Detection", header)
        self.run_button.setEnabled(False)
        header_layout.addWidget(self.choose_button)
        header_layout.addWidget(self.run_button)
        layout.addWidget(header)

        content = QHBoxLayout()
        self.canvas = OpenWorldCanvas(root)
        content.addWidget(self.canvas, 3)
        side = QVBoxLayout()
        self.status = QLabel("Status: waiting for an image", root)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("background:white;border-radius:8px;padding:12px;font-size:14px;")
        side.addWidget(self.status)
        legend = QLabel("Solid colored box: known fruit\nYellow dashed box: unknown candidate\nA '?' suffix marks a cluster-derived name requiring review", root)
        legend.setWordWrap(True)
        legend.setStyleSheet("background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:12px;")
        side.addWidget(legend)
        self.table = QTableWidget(0, 4, root)
        self.table.setHorizontalHeaderLabels(("Type", "Result", "Score", "Position"))
        self.table.horizontalHeader().setStretchLastSection(True)
        side.addWidget(self.table, 1)
        content.addLayout(side, 2)
        layout.addLayout(content, 1)
        self.choose_button.clicked.connect(self.choose_image)
        self.run_button.clicked.connect(self.run_inference)

    @Slot()
    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a Fruit Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        self.image_path = Path(path)
        self.result = None
        self.canvas.show_result(self.image_path)
        self.table.setRowCount(0)
        self.status.setText(f"Selected: {self.image_path.name}")
        self.run_button.setEnabled(True)

    @Slot()
    def run_inference(self) -> None:
        if self.image_path is None or self.thread is not None:
            return
        self.choose_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status.setText("Running known detection, unknown box proposals and box-level clustering...")
        self.thread = QThread(self)
        self.worker = _InferenceWorker(self.pipeline, self.image_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._show_result)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(object)
    def _show_result(self, result: object) -> None:
        if not isinstance(result, OpenWorldInferenceResult):
            self._show_error("Inference returned an incompatible result")
            return
        self.result = result
        self.canvas.show_result(result.image_path, result.detections)
        self.table.setRowCount(len(result.detections))
        for row, detection in enumerate(result.detections):
            values = (
                "Unknown" if detection.kind == "unknown" else "Known",
                display_label(detection.label),
                f"{detection.score:.3f}",
                ", ".join(f"{value:.0f}" for value in detection.xyxy),
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.status.setText(f"Completed: {result.known_count} known object(s), {result.unknown_count} unknown candidate(s)")

    @Slot(str)
    def _show_error(self, message: str) -> None:
        self.status.setText("Run failed: " + message)
        QMessageBox.warning(self, "Open-World Detection Failed", message)

    @Slot()
    def _thread_finished(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.choose_button.setEnabled(True)
        self.run_button.setEnabled(self.image_path is not None)
