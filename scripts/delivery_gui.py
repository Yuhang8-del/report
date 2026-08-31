"""Start the delivery GUI and preload the packaged Student checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer

from fruit_ssod.gui.app import create_application
from fruit_ssod.gui.main_window import MainWindow


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    delivery_root = project_dir.parent
    weights = Path(
        os.environ.get(
            "FRUIT_SSOD_DEFAULT_MODEL",
            delivery_root / "models" / "student_best.pt",
        )
    ).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"The default model was not found: {weights}")

    app = create_application(sys.argv)
    window = MainWindow()
    window.show()
    QTimer.singleShot(250, lambda: window.load_model(weights))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
