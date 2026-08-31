"""Chinese customer-facing desktop shell for file and live-camera inference."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QApplication,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fruit_ssod.gui.model_manager import ModelManager
from fruit_ssod.gui.theme import APP_STYLE
from fruit_ssod.gui.widgets.camera_view import CameraInferencePage
from fruit_ssod.gui.widgets.image_view import ImageInferencePage
from fruit_ssod.gui.widgets.status_panel import StatusPanel
from fruit_ssod.gui.widgets.video_view import VideoInferencePage


# The visible labels are intentionally Chinese.  ``current_page_name`` keeps
# the historical English identifiers as a small compatibility API for scripts
# and tests that used the first prototype.
NAVIGATION_PAGES = ("Live Camera", "Image Detection", "Batch Processing", "Video File", "Experiment Overview")
_CANONICAL_NAVIGATION_PAGES = ("Camera", "Single image", "Batch images", "Video", "Logs")
KNOWN_CLASSES = ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")
EXTENDED_CLASSES = ("Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon")


class MainWindow(QMainWindow):
    """Application shell with a polished Chinese presentation layer."""

    def __init__(self, *, model_manager: ModelManager | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Semi-Supervised Fruit Detection Studio")
        self.setMinimumSize(1180, 760)
        self.resize(1440, 900)
        self._model_manager = model_manager or ModelManager(self)
        self._inference_pages: tuple[CameraInferencePage | ImageInferencePage | VideoInferencePage, ...] = ()
        self._build_ui()
        self._connect_model_manager()

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    @property
    def current_page_name(self) -> str:
        """Return the stable English page key used by the original API."""
        row = self._navigation.currentRow()
        return _CANONICAL_NAVIGATION_PAGES[row] if 0 <= row < len(_CANONICAL_NAVIGATION_PAGES) else "Camera"

    def _build_ui(self) -> None:
        # Set the application font explicitly for consistent Windows rendering.
        application = QApplication.instance()
        if application is not None:
            font_path = Path("C:/Windows/Fonts/segoeui.ttf")
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))
            application.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(APP_STYLE)
        root = QWidget(self)
        root.setObjectName("mainWindowContent")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 14, 16, 12)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._build_header())

        content = QHBoxLayout()
        content.setSpacing(12)
        navigation_card = QFrame(root)
        navigation_card.setObjectName("navigationCard")
        navigation_layout = QVBoxLayout(navigation_card)
        navigation_layout.setContentsMargins(10, 14, 10, 14)
        navigation_layout.setSpacing(10)
        navigation_title = QLabel("Detection Workspace", navigation_card)
        navigation_title.setObjectName("navigationTitle")
        navigation_caption = QLabel("Select a workflow", navigation_card)
        navigation_caption.setObjectName("navigationCaption")
        navigation_layout.addWidget(navigation_title)
        navigation_layout.addWidget(navigation_caption)
        self._navigation = QListWidget(navigation_card)
        self._navigation.setObjectName("navigationList")
        navigation_card.setMaximumWidth(205)
        navigation_card.setMinimumWidth(185)
        self._navigation.setSelectionMode(QListWidget.SingleSelection)
        for page_name in NAVIGATION_PAGES:
            self._navigation.addItem(QListWidgetItem(page_name))
        navigation_layout.addWidget(self._navigation, 1)
        navigation_footer = QLabel("CUDA acceleration enabled\n5-class / 11-class models", navigation_card)
        navigation_footer.setObjectName("navigationFooter")
        navigation_layout.addWidget(navigation_footer)

        self._pages = QStackedWidget(root)
        self._pages.setObjectName("workflowPages")
        self._camera_page = CameraInferencePage(
            start_allowed=lambda: not self._has_active_inference(),
            parent=self._pages,
        )
        self._single_image_page = ImageInferencePage(
            model_provider=lambda: self._model_manager.active_model,
            mode="single",
            start_allowed=lambda: not self._has_active_inference(),
            parent=self._pages,
        )
        self._batch_image_page = ImageInferencePage(
            model_provider=lambda: self._model_manager.active_model,
            mode="batch",
            start_allowed=lambda: not self._has_active_inference(),
            parent=self._pages,
        )
        self._video_page = VideoInferencePage(
            model_provider=lambda: self._model_manager.active_model,
            start_allowed=lambda: not self._has_active_inference(),
            parent=self._pages,
        )
        self._inference_pages = (
            self._camera_page,
            self._single_image_page,
            self._batch_image_page,
            self._video_page,
        )
        self._pages.addWidget(self._camera_page)
        self._pages.addWidget(self._single_image_page)
        self._pages.addWidget(self._batch_image_page)
        self._pages.addWidget(self._video_page)
        self._pages.addWidget(self._build_experiment_page())
        content.addWidget(navigation_card)
        content.addWidget(self._pages, 1)
        root_layout.addLayout(content, 1)

        self._status_panel = StatusPanel(root)
        root_layout.addWidget(self._status_panel)
        self._navigation.setCurrentRow(0)

        self._navigation.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._load_model_button.clicked.connect(self._choose_model)
        self._release_model_button.clicked.connect(self._model_manager.release_model)
        for page in self._inference_pages:
            page.busy_changed.connect(self._on_inference_busy_changed)

    def _build_header(self) -> QFrame:
        header = QFrame(self)
        header.setObjectName("topHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 14, 20, 13)
        layout.setSpacing(7)

        first_row = QHBoxLayout()
        brand = QVBoxLayout()
        title = QLabel("Fruit Detection & Open-Category Research Studio", header)
        title.setObjectName("brandTitle")
        subtitle = QLabel("Fruit SSOD Studio  ·  Semi-supervised Learning  ·  Live Camera  ·  11-Class Extension", header)
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        first_row.addLayout(brand, 1)

        badge = QLabel("SYSTEM ONLINE  ·  RTX 3080  ·  CUDA", header)
        badge.setObjectName("headerBadge")
        badge.setAlignment(Qt.AlignCenter)
        first_row.addWidget(badge, 0, Qt.AlignVCenter)

        self._load_model_button = QPushButton("Load Model", header)
        self._load_model_button.setObjectName("loadModelButton")
        self._release_model_button = QPushButton("Release Model", header)
        self._release_model_button.setObjectName("releaseModelButton")
        self._release_model_button.setEnabled(False)
        first_row.addWidget(self._load_model_button, 0, Qt.AlignVCenter)
        first_row.addWidget(self._release_model_button, 0, Qt.AlignVCenter)
        layout.addLayout(first_row)

        self._model_hint = QLabel("File inference model not loaded · Camera models can be selected on the Live Camera page", header)
        self._model_hint.setObjectName("modelHint")
        layout.addWidget(self._model_hint)
        return header

    def _build_experiment_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("experimentInfoPage")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)

        title = QLabel("Experimental Results & System Capabilities", page)
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        intro = QLabel(
            "The semi-supervised experiment uses real public images: the Teacher generates filtered pseudo-labels, "
            "the Student detects five fruit classes, and the extension model adds six classes for live demonstration.",
            page,
        )
        intro.setObjectName("pageIntro")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        for index, (value, caption) in enumerate(
            (("0.6285", "Teacher fixed-test mAP@0.5"), ("0.5323", "Student fixed-test mAP@0.5"), ("11 classes", "Real-time extended detection"), ("≈15 ms", "RTX 3080 / 640-pixel input"))
        ):
            card = QFrame(page)
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(15, 12, 15, 12)
            value_label = QLabel(value, card)
            value_label.setObjectName("metricValue")
            caption_label = QLabel(caption, card)
            caption_label.setObjectName("metricCaption")
            card_layout.addWidget(value_label)
            card_layout.addWidget(caption_label)
            metrics.addWidget(card, index // 2, index % 2)
        outer.addLayout(metrics)

        pipeline = QFrame(page)
        pipeline.setObjectName("infoCard")
        pipeline_layout = QVBoxLayout(pipeline)
        pipeline_layout.setContentsMargins(17, 14, 17, 14)
        pipeline_title = QLabel("Experimental Pipeline", pipeline)
        pipeline_title.setObjectName("pageTitle")
        pipeline_title.setStyleSheet("font-size: 15px;")
        pipeline_layout.addWidget(pipeline_title)
        pipeline_text = QLabel("Real public data → Fixed split → Teacher → Trust Filter → Student → Incremental class training → GUI & camera demo", pipeline)
        pipeline_text.setObjectName("infoBody")
        pipeline_text.setWordWrap(True)
        pipeline_layout.addWidget(pipeline_text)
        classes_layout = QHBoxLayout()
        classes_layout.setSpacing(7)
        for class_name in KNOWN_CLASSES:
            chip = QLabel(class_name, pipeline)
            chip.setObjectName("chip")
            classes_layout.addWidget(chip)
        classes_layout.addStretch(1)
        pipeline_layout.addLayout(classes_layout)
        extension_layout = QHBoxLayout()
        extension_layout.setSpacing(7)
        extension_label = QLabel("Extended", pipeline)
        extension_label.setObjectName("chipLabel")
        extension_layout.addWidget(extension_label)
        for class_name in EXTENDED_CLASSES:
            chip = QLabel(class_name, pipeline)
            chip.setObjectName("extendedChip")
            extension_layout.addWidget(chip)
        extension_layout.addStretch(1)
        pipeline_layout.addLayout(extension_layout)
        outer.addWidget(pipeline)

        note = QFrame(page)
        note.setObjectName("infoCard")
        note_layout = QVBoxLayout(note)
        note_layout.setContentsMargins(17, 14, 17, 14)
        note_title = QLabel("Live Demonstration Notes", note)
        note_title.setObjectName("pageTitle")
        note_title.setStyleSheet("font-size: 15px;")
        note_layout.addWidget(note_title)
        note_body = QLabel(
            "Supports single images, image batches, local video and external cameras. Live camera inference uses the "
            "5-class Student or 11-class incremental detector; full open-world Unknown analysis remains offline.",
            note,
        )
        note_body.setObjectName("infoBody")
        note_body.setWordWrap(True)
        note_layout.addWidget(note_body)
        outer.addWidget(note)
        outer.addStretch(1)
        return page

    def _connect_model_manager(self) -> None:
        self._model_manager.model_loading.connect(self._on_model_loading)
        self._model_manager.model_loaded.connect(self._on_model_loaded)
        self._model_manager.model_released.connect(self._on_model_released)
        self._model_manager.loading_finished.connect(self._on_loading_finished)
        self._model_manager.status_changed.connect(self._on_status_changed)
        self._model_manager.load_failed.connect(self._on_model_load_failed)

    def _has_active_inference(self) -> bool:
        return any(page.is_running for page in self._inference_pages)

    @Slot()
    def _choose_model(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Select fruit detection weights",
            "",
            "PyTorch weights (*.pt)",
        )
        if path_text:
            self.load_model(Path(path_text))

    def load_model(self, path: str | Path) -> bool:
        if self._has_active_inference():
            self._status_panel.set_error("An inference task is running. Wait for it to finish before changing the model.")
            return False
        return self._model_manager.start_loading(path)

    @Slot(str)
    def _on_model_loading(self, model_name: str) -> None:
        self._load_model_button.setEnabled(False)
        self._release_model_button.setEnabled(False)
        self._model_hint.setText(f"Checking model compatibility: {model_name}")
        self._status_panel.set_loading(model_name)

    @Slot(str)
    def _on_model_loaded(self, model_name: str) -> None:
        self._model_hint.setText(f"File inference model: {model_name}  ·  5 classes  ·  CUDA · Camera models switch independently")
        self._status_panel.set_model_loaded(model_name)

    @Slot()
    def _on_model_released(self) -> None:
        self._model_hint.setText("File inference model not loaded · Camera models can be selected on the Live Camera page")
        self._status_panel.set_model_released()

    @Slot(bool)
    def _on_loading_finished(self, model_is_active: bool) -> None:
        self._load_model_button.setEnabled(not self._has_active_inference())
        self._release_model_button.setEnabled(model_is_active and not self._has_active_inference())

    @Slot(bool)
    def _on_inference_busy_changed(self, _busy: bool) -> None:
        inference_busy = self._has_active_inference()
        self._load_model_button.setEnabled(not inference_busy and not self._model_manager.is_loading)
        self._release_model_button.setEnabled(
            not inference_busy and self._model_manager.has_active_model and not self._model_manager.is_loading
        )

    @Slot(str)
    def _on_status_changed(self, message: str) -> None:
        if not self._model_manager.has_active_model and not self._model_manager.is_loading:
            self._status_panel.set_error(message)

    @Slot(str)
    def _on_model_load_failed(self, message: str) -> None:
        self._status_panel.set_error(message)
        QMessageBox.warning(self, "Model Load Failed", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if any(not page.shutdown() for page in self._inference_pages):
            event.ignore()
            return
        if not self._model_manager.shutdown():
            event.ignore()
            return
        event.accept()
