"""Chinese status surface with a stable compatibility text API."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class StatusPanel(QWidget):
    """Present model state and an actionable Chinese status message."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusPanel")
        self._compat_state = "No model loaded"
        self._compat_message = "Choose a compatible .pt weights file to begin."

        self._state_label = QLabel("No Model Loaded", self)
        self._state_label.setObjectName("modelStateLabel")
        self._state_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._message_label = QLabel("Select compatible best.pt weights to begin the demonstration.", self)
        self._message_label.setObjectName("statusMessageLabel")
        self._message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(13, 8, 13, 8)
        layout.addWidget(self._state_label)
        layout.addStretch(1)
        layout.addWidget(self._message_label)

    @property
    def state_text(self) -> str:
        """Return the stable English state key used by legacy automation."""
        return self._compat_state

    @property
    def message_text(self) -> str:
        """Return the stable English message key used by legacy automation."""
        return self._compat_message

    def set_model_loaded(self, model_name: str) -> None:
        self._compat_state = f"Active model: {model_name}"
        self._compat_message = "Model loaded. Select an image, folder, or video file to start inference."
        self._state_label.setText(f"Model Loaded: {model_name}")
        self._message_label.setText("The model is ready. Select an image, folder or video to start inference.")

    def set_loading(self, model_name: str) -> None:
        self._compat_state = f"Loading model: {model_name}"
        self._compat_message = "Checking checkpoint compatibility in a background worker..."
        self._state_label.setText(f"Loading: {model_name}")
        self._message_label.setText("Checking weight compatibility in the background. The interface remains responsive.")

    def set_model_released(self) -> None:
        self._compat_state = "No model loaded"
        self._compat_message = "Choose a compatible .pt weights file to begin."
        self._state_label.setText("No Model Loaded")
        self._message_label.setText("Select compatible best.pt weights to begin the demonstration.")

    def set_error(self, message: str) -> None:
        self._compat_message = message
        self._message_label.setText(f"Notice: {message}")
