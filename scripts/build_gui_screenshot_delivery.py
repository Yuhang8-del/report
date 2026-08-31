"""Build a contact sheet and integrity manifest for captured GUI windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*_GUI.png"))
    if len(paths) != 12:
        raise RuntimeError(f"expected 12 GUI screenshots, found {len(paths)}")
    tile = (690, 425)
    columns = 2
    sheet = Image.new("RGB", (columns * tile[0], math.ceil(len(paths) / columns) * tile[1]), "#CBD5E1")
    rows = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            original_size = source.size
            preview = ImageOps.fit(image, tile, Image.Resampling.LANCZOS)
        x = (index % columns) * tile[0]
        y = (index // columns) * tile[1]
        sheet.paste(preview, (x, y))
        rows.append(
            {
                "file": path.name,
                "width": original_size[0],
                "height": original_size[1],
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    contact = args.input / "gui_inference_contact_sheet.jpg"
    sheet.save(contact, quality=94, subsampling=0)
    payload = {
        "artifact_type": "pyside6_gui_inference_screenshot_delivery",
        "model": str(args.model.resolve(strict=True)),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "capture_method": "real PySide6 window inference plus Win32 window-handle screenshot",
        "screenshot_count": len(rows),
        "screenshots": rows,
        "contact_sheet": contact.name,
    }
    (args.input / "gui_screenshot_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"screenshots": len(rows), "contact_sheet": str(contact)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
