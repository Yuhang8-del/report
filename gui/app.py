"""Module entry point for the Fruit SSOD desktop demonstrator."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from fruit_ssod.gui.main_window import MainWindow


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the shared QApplication, creating it only when the process needs one."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(list(argv) if argv is not None else sys.argv)
    return application


def create_main_window() -> MainWindow:
    """Build a shell that is useful before weights are available."""
    create_application()
    return MainWindow()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch file, video and external-camera inference workflows."""
    application = create_application(argv)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
