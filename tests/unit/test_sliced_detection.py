from __future__ import annotations

from PIL import Image

from fruit_ssod.data.sliced_detection import (
    SliceWindow,
    generate_slice_windows,
    merge_sliced_detections,
    predict_sliced,
)
from fruit_ssod.detection.types import DetectionRecord


def _detection(class_id: int, name: str, confidence: float, xyxy: tuple[float, float, float, float]) -> DetectionRecord:
    return DetectionRecord(class_id, name, confidence, xyxy, False, "fixture.pt")


def test_slice_windows_cover_right_and_bottom_edges_deterministically() -> None:
    windows = generate_slice_windows(1000, 700, slice_size=400, overlap=0.25)

    assert windows[0] == SliceWindow(0, 0, 400, 400)
    assert windows[-1] == SliceWindow(600, 300, 1000, 700)
    assert {window.right for window in windows} >= {400, 700, 1000}
    assert {window.bottom for window in windows} >= {400, 700}
    assert len(windows) == len(set(windows))


def test_merge_sliced_detections_is_class_aware() -> None:
    detections = (
        _detection(0, "Apple", 0.9, (10, 10, 50, 50)),
        _detection(0, "Apple", 0.8, (12, 12, 52, 52)),
        _detection(2, "Orange", 0.7, (12, 12, 52, 52)),
    )

    merged = merge_sliced_detections(detections, iou_threshold=0.5)

    assert [(item.class_id, item.confidence) for item in merged] == [(0, 0.9), (2, 0.7)]


def test_predict_sliced_projects_local_boxes_and_merges_overlap() -> None:
    image = Image.new("RGB", (700, 400), "black")

    def predictor(_crop: Image.Image, window: SliceWindow) -> tuple[DetectionRecord, ...]:
        if window.left == 0:
            return (_detection(4, "Pineapple", 0.8, (320, 100, 390, 180)),)
        return (_detection(4, "Pineapple", 0.9, (20, 100, 90, 180)),)

    detections = predict_sliced(image, predictor, slice_size=400, overlap=0.25, nms_iou=0.5)

    assert len(detections) == 1
    assert detections[0].confidence == 0.9
    assert detections[0].xyxy == (320.0, 100.0, 390.0, 180.0)
