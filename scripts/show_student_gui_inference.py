"""Show one completed Student inference in the real PySide6 delivery GUI.

The process stays alive after printing its Win32 handle so an external capture
helper can take a screenshot of the actual visible application window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve(strict=True)
    image_path = args.image.resolve(strict=True)
    os.environ.pop("QT_QPA_PLATFORM", None)

    from PySide6.QtCore import QTimer
    from ultralytics.utils import SETTINGS

    SETTINGS["sync"] = False

    from fruit_ssod.detection import UltralyticsDetectorAdapter
    from fruit_ssod.gui.app import create_application
    from fruit_ssod.gui.main_window import MainWindow
    from fruit_ssod.gui.model_manager import ModelManager

    adapter = UltralyticsDetectorAdapter(weights_path=model_path)
    adapter.initialize()
    manager = ModelManager()
    # This is a deterministic screenshot utility.  The normal application uses
    # ModelManager's asynchronous loader; here the validated adapter is installed
    # before the window appears so each process performs exactly one inference.
    manager._active_adapter = adapter
    manager._weights_path = model_path

    app = create_application(sys.argv[:1])
    window = MainWindow(model_manager=manager)
    window.resize(1280, 820)
    window.show()
    window._on_model_loaded(model_path.name)
    page = window._single_image_page

    announced = False

    def announce_when_ready(_busy: bool = False) -> None:
        nonlocal announced
        if announced or page.is_running or not page.results:
            return
        announced = True
        result = page.results[0]
        counts = ",".join(f"{name}:{count}" for name, count in result.class_counts.items()) or "none"
        window.setWindowTitle(f"Semi-Supervised Fruit Detection System - Student Inference - {args.label}")
        app.processEvents()
        print(
            f"GUI_HANDLE={int(window.winId())};LABEL={args.label};"
            f"IMAGE_ID={image_path.stem};DETECTIONS={len(result.detections)};COUNTS={counts}",
            flush=True,
        )

    page.busy_changed.connect(announce_when_ready)

    def start_inference() -> None:
        if not page.set_images((image_path,)):
            raise RuntimeError(f"GUI rejected image: {image_path}")
        if not page.start_inference():
            raise RuntimeError(f"GUI failed to start inference: {image_path}")

    QTimer.singleShot(300, start_inference)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
