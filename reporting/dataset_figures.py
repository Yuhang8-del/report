"""Small, dependency-light dataset figures used by the audit command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw


def render_annotation_montage(
    rows: Iterable[Mapping[str, Any]], output_path: Path, *, image_root: Path | None = None, maximum_images: int = 9
) -> int:
    """Render a deterministic contact sheet of readable images and their boxes.

    Unreadable or absent source files are skipped.  A valid placeholder is still
    written when no samples are locally available so a report build has a stable
    artifact and can explain why it contains no examples.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        image_id = row.get("source_image_id")
        source = row.get("source")
        if isinstance(image_id, str) and image_id and isinstance(source, str) and source:
            grouped.setdefault((source, image_id), []).append(row)
    tiles: list[tuple[Image.Image, list[Mapping[str, Any]], str]] = []
    for source, image_id in sorted(grouped)[:maximum_images]:
        image_rows = grouped[(source, image_id)]
        file_path = image_rows[0].get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        path = Path(file_path)
        if not path.is_absolute() and image_root is not None:
            path = image_root / path
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except OSError:
            continue
        tiles.append((image, image_rows, f"{source}:{image_id}"))
    if not tiles:
        canvas = Image.new("RGB", (640, 120), "white")
        ImageDraw.Draw(canvas).text((16, 50), "No readable local sample images were available for the dataset audit.", fill="black")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return 0
    tile_width, tile_height = 240, 180
    columns = min(3, len(tiles))
    rows_count = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    for index, (image, annotations, image_id) in enumerate(tiles):
        image.thumbnail((tile_width - 8, tile_height - 28))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(image, (4, 20))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 4), image_id, fill="black")
        scale_x = image.width / max(1, int(annotations[0].get("width", image.width)))
        scale_y = image.height / max(1, int(annotations[0].get("height", image.height)))
        for annotation in annotations:
            box = annotation.get("xyxy")
            if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
                draw.rectangle(tuple(float(value) * (scale_x if axis % 2 == 0 else scale_y) + (4 if axis % 2 == 0 else 20) for axis, value in enumerate(box)), outline="red", width=2)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return len(tiles)
