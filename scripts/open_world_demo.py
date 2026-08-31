"""Launch the standalone box-level open-world fruit demonstrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter
from fruit_ssod.gui.open_world_window import OpenWorldWindow
from fruit_ssod.open_world.box_clustering import BoxClusterer
from fruit_ssod.open_world.box_proposals import UltralyticsObjectnessProposalProvider
from fruit_ssod.open_world.pipeline import OpenWorldFruitPipeline
from fruit_ssod.open_world.incremental_adapter import ReviewedUltralyticsDetectorAdapter


def arguments() -> argparse.Namespace:
    delivery = Path(__file__).resolve().parents[2]
    incremental = delivery / "models" / "incremental_11class_best.pt"
    default_detector = incremental if incremental.is_file() else delivery / "models" / "student_best.pt"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student",
        type=Path,
        default=default_detector,
        help="Known-class detector; the reviewed 11-class model is preferred when available.",
    )
    parser.add_argument("--objectness", type=Path, default=delivery / "models" / "open_world_objectness.pt")
    parser.add_argument("--encoder", type=Path, default=delivery / "models" / "open_world_encoder.pt")
    parser.add_argument("--clusters", type=Path, default=delivery / "models" / "open_world_box_clusters.npz")
    parser.add_argument("--names", type=Path, default=delivery / "models" / "open_world_cluster_names.json")
    parser.add_argument("--registry", type=Path, default=delivery / "models" / "class_registry_v2.json")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    required = (args.student, args.objectness, args.encoder, args.clusters, args.names)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("开放世界运行文件尚未齐备：" + "; ".join(missing))
    names = {int(key): value for key, value in json.loads(args.names.read_text(encoding="utf-8")).items()}
    clusterer = BoxClusterer(
        encoder_checkpoint=args.encoder,
        cluster_model=args.clusters,
        candidate_names=names,
        device=f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device),
    )
    incremental_detector = args.student.name == "incremental_11class_best.pt" and args.registry.is_file()
    detector = (
        ReviewedUltralyticsDetectorAdapter(weights_path=args.student, registry_path=args.registry)
        if incremental_detector
        else UltralyticsDetectorAdapter(weights_path=args.student)
    )
    pipeline = OpenWorldFruitPipeline(
        known_detector=detector,
        proposal_provider=UltralyticsObjectnessProposalProvider(
            weights_path=args.objectness,
            objectness_threshold=0.10,
            known_iou_threshold=0.35,
            image_size=768,
            device=args.device,
        ),
        clusterer=clusterer,
    )
    application = QApplication(sys.argv)
    window = OpenWorldWindow(pipeline)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
