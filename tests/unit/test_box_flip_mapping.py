"""Properties of the dual-view horizontal-box geometry."""

from __future__ import annotations

import pytest

from fruit_ssod.pseudo.transforms import TransformError, horizontal_flip_xyxy


@pytest.mark.parametrize(
    ("width", "box"),
    [
        (10, (0.0, 0.0, 1.0, 2.0)),
        (10, (1.25, 2.5, 8.75, 9.5)),
        (640, (17.0, 31.0, 523.5, 428.0)),
        (4096, (100.125, 4.0, 4095.5, 512.75)),
    ],
)
def test_horizontal_flip_is_an_involution(width: int, box: tuple[float, float, float, float]) -> None:
    """Flip-view predictions map back to original coordinates exactly."""
    assert horizontal_flip_xyxy(horizontal_flip_xyxy(box, width=width), width=width) == box


def test_horizontal_flip_reverses_x_coordinates_but_preserves_y_coordinates() -> None:
    assert horizontal_flip_xyxy((10.0, 3.0, 40.0, 50.0), width=100) == (60.0, 3.0, 90.0, 50.0)


@pytest.mark.parametrize("box", [(-1.0, 1.0, 2.0, 3.0), (1.0, 1.0, 101.0, 3.0), (1.0, 1.0, 1.0, 3.0)])
def test_horizontal_flip_rejects_out_of_bounds_or_empty_boxes(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(TransformError, match="Problem:"):
        horizontal_flip_xyxy(box, width=100)
