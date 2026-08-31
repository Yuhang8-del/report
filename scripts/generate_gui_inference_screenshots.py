"""Run real PySide6 GUI inference and capture the rendered application window."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--objectness", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--protected-truth", type=Path, required=True)
    parser.add_argument("--example-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show-index", type=int)
    return parser.parse_args()


def selected_records(truth_path: Path, manifest_path: Path) -> list[dict]:
    records = {record["image_id"]: record for record in json.loads(truth_path.read_text(encoding="utf-8"))["records"]}
    rendered = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    selected = []
    for path in rendered:
        image_id = Path(path).stem.rsplit("_", 1)[-1]
        if image_id not in records:
            raise KeyError(f"example image id is absent from protected truth: {image_id}")
        selected.append(records[image_id])
    return selected


def contact_sheet(paths: list[Path], output: Path) -> None:
    tile = (690, 425)
    columns = 2
    sheet = Image.new("RGB", (columns * tile[0], math.ceil(len(paths) / columns) * tile[1]), "#CBD5E1")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            preview = ImageOps.fit(source.convert("RGB"), tile, Image.Resampling.LANCZOS)
        x = (index % columns) * tile[0]
        y = (index // columns) * tile[1]
        sheet.paste(preview, (x, y))
    sheet.save(output, quality=94, subsampling=0)


def main() -> int:
    args = arguments()
    # Offscreen still runs the real Qt paint pipeline and QWidget.grab(), while
    # avoiding focus changes on the user's Windows desktop during batch capture.
    if args.show_index is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from ultralytics.utils import SETTINGS

    # Disable optional telemetry threads during deterministic GUI capture. On
    # this Windows build an SSL telemetry callback can race Qt's offscreen
    # event processing and trigger a native access violation after inference.
    SETTINGS["sync"] = False

    from fruit_ssod.gui.open_world_window import OpenWorldWindow
    from fruit_ssod.open_world.box_clustering import BoxClusterer
    from fruit_ssod.open_world.box_proposals import UltralyticsObjectnessProposalProvider
    from fruit_ssod.open_world.pipeline import OpenWorldFruitPipeline
    from fruit_ssod.open_world.incremental_adapter import ReviewedUltralyticsDetectorAdapter

    required = (args.detector, args.objectness, args.encoder, args.clusters, args.names, args.registry)
    for path in required:
        path.resolve(strict=True)
    candidate_names = {
        int(key): value for key, value in json.loads(args.names.read_text(encoding="utf-8")).items()
    }
    clusterer = BoxClusterer(
        encoder_checkpoint=args.encoder,
        cluster_model=args.clusters,
        candidate_names=candidate_names,
        device=f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device),
    )
    pipeline = OpenWorldFruitPipeline(
        known_detector=ReviewedUltralyticsDetectorAdapter(
            weights_path=args.detector,
            registry_path=args.registry,
        ),
        proposal_provider=UltralyticsObjectnessProposalProvider(
            weights_path=args.objectness,
            objectness_threshold=0.10,
            known_iou_threshold=0.35,
            image_size=768,
            device=args.device,
        ),
        clusterer=clusterer,
    )
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = OpenWorldWindow(pipeline)
    window.resize(1380, 850)
    window.show()
    app.processEvents()
    args.output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    manifest_rows = []
    selected = selected_records(args.protected_truth, args.example_manifest)
    if args.show_index is not None:
        if not 0 <= args.show_index < len(selected):
            raise IndexError(f"show-index must be between 0 and {len(selected) - 1}")
        record = selected[args.show_index]
        image_path = Path(record["image_path"])
        window.image_path = image_path
        window.canvas.show_result(image_path)
        window.status.setText(f"正在通过GUI推理：{image_path.name}")
        app.processEvents()
        result = pipeline.predict(image_path)
        window._show_result(result)
        window.setWindowTitle(f"水果开放世界目标检测系统 - {record['category']}")
        app.processEvents()
        print(
            "GUI_HANDLE="
            + str(int(window.winId()))
            + ";CATEGORY="
            + str(record["category"])
            + ";IMAGE_ID="
            + str(record["image_id"]),
            flush=True,
        )
        return app.exec()
    if args.limit is not None:
        selected = selected[: args.limit]
    for index, record in enumerate(selected, start=1):
        image_path = Path(record["image_path"])
        window.image_path = image_path
        window.canvas.show_result(image_path)
        window.status.setText(f"正在通过GUI推理：{image_path.name}")
        app.processEvents()
        result = pipeline.predict(image_path)
        window._show_result(result)
        window.setWindowTitle(f"水果开放世界目标检测系统 - {record['category']}")
        destination = args.output / f"{index:02d}_{record['category']}_{record['image_id']}_GUI.png"
        capture = QImage(window.size(), QImage.Format_ARGB32)
        capture.fill(0)
        painter = QPainter(capture)
        try:
            window.render(painter, QPoint(0, 0))
        finally:
            painter.end()
        if not capture.save(str(destination), "PNG"):
            raise RuntimeError(f"failed to save GUI screenshot: {destination}")
        paths.append(destination)
        manifest_rows.append(
            {
                "category": record["category"],
                "image_id": record["image_id"],
                "source_image": str(image_path),
                "gui_screenshot": str(destination),
                "known_detections": result.known_count,
                "unknown_detections": result.unknown_count,
            }
        )
    window.close()
    contact_sheet(paths, args.output / "gui_inference_contact_sheet.jpg")
    (args.output / "gui_screenshot_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "pyside6_gui_inference_screenshots",
                "detector": str(args.detector.resolve()),
                "screenshot_method": "QWidget.grab after OpenWorldFruitPipeline.predict",
                "screenshots": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"screenshots": len(paths), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
