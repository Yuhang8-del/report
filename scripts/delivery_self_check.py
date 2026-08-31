"""Offline integrity and inference check for the D-drive delivery."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    root = project_dir.parent
    required = {
        "project": project_dir / "src" / "fruit_ssod",
        "dataset": root / "data" / "fruit_ssod",
        "student": root / "models" / "student_best.pt",
        "teacher": root / "models" / "teacher_best.pt",
        "incremental": root / "models" / "incremental_11class_best.pt",
        "open_world_objectness": root / "models" / "open_world_objectness.pt",
        "open_world_encoder": root / "models" / "open_world_encoder.pt",
        "open_world_clusters": root / "models" / "open_world_box_clusters.npz",
        "open_world_names": root / "models" / "open_world_cluster_names.json",
        "camera_view": project_dir / "src" / "fruit_ssod" / "gui" / "widgets" / "camera_view.py",
        "camera_worker": project_dir / "src" / "fruit_ssod" / "gui" / "workers" / "camera_worker.py",
        "report_pdf": root / "reports" / "final_report" / "final_report.pdf",
        "sample": root / "samples" / "images",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        print("[FAILED] The delivery package is incomplete:")
        print("\n".join(f"  - {item}" for item in missing))
        return 2

    import cv2
    import torch
    from ultralytics import YOLO

    from fruit_ssod.gui.widgets.camera_view import default_camera_profiles
    from fruit_ssod.gui.workers.camera_worker import normalize_camera_result

    images = sorted(
        path
        for path in required["sample"].iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        print("[FAILED] No demonstration image was found in samples/images.")
        return 3

    model = YOLO(str(required["student"]))
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=str(images[0]),
        imgsz=640,
        conf=0.25,
        device=device,
        verbose=False,
    )
    detections = 0 if results[0].boxes is None else len(results[0].boxes)
    incremental_model = YOLO(str(required["incremental"]))
    incremental_results = incremental_model.predict(
        source=str(images[0]),
        imgsz=640,
        conf=0.25,
        device=device,
        verbose=False,
    )
    profiles = default_camera_profiles()
    if len(profiles) != 2 or not all(profile.weights_path.is_file() for profile in profiles):
        print("[FAILED] The camera page does not have complete 5-class and 11-class model profiles.")
        return 4
    normalize_camera_result(incremental_results[0], profiles[1])
    incremental_detections = (
        0 if incremental_results[0].boxes is None else len(incremental_results[0].boxes)
    )
    summary = {
        "status": "passed",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "student_sha256": sha256(required["student"]),
        "student_classes": model.names,
        "incremental_sha256": sha256(required["incremental"]),
        "incremental_classes": incremental_model.names,
        "open_world_files": {
            name: str(required[name])
            for name in (
                "open_world_objectness",
                "open_world_encoder",
                "open_world_clusters",
                "open_world_names",
            )
        },
        "sample_image": str(images[0]),
        "sample_detections": detections,
        "incremental_sample_detections": incremental_detections,
        "camera_profiles": [profile.label for profile in profiles],
        "opencv": cv2.__version__,
        "camera_device_note": "Device scanning and access occur only after the user clicks Refresh Devices or Start Detection in the GUI.",
        "data_root": os.environ.get("FRUIT_SSOD_DATA_ROOT", str(root / "data")),
    }
    output = root / "outputs" / "self_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[PASSED] The self-check result was written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
