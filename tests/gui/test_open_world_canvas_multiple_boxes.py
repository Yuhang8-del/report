from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image
from fruit_ssod.gui.open_world_window import OpenWorldCanvas
from fruit_ssod.open_world.pipeline import OpenWorldDetection


def test_canvas_paints_every_detection_and_not_only_the_first(tmp_path: Path, qtbot: object) -> None:
    source = tmp_path / "fruit.png"
    Image.new("RGB", (100, 100), "white").save(source)
    canvas = OpenWorldCanvas()
    qtbot.addWidget(canvas)  # type: ignore[attr-defined]
    canvas.resize(680, 500)
    detections = (
        OpenWorldDetection("known", (10.0, 10.0, 30.0, 30.0), "Blueberry", 0.9, class_id=6),
        OpenWorldDetection("known", (60.0, 60.0, 80.0, 80.0), "Blueberry", 0.8, class_id=6),
    )
    assert canvas.show_result(source, detections)
    errors: list[BaseException] = []
    previous_hook = sys.excepthook
    sys.excepthook = lambda _type, value, _traceback: errors.append(value)
    try:
        canvas.show()
        canvas.repaint()
        qtbot.wait(50)  # type: ignore[attr-defined]
    finally:
        sys.excepthook = previous_hook
    assert not errors, f"paint callback stopped before completing all boxes: {errors}"
