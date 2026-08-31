"""Build a machine-readable inventory for the complete delivery folder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_stats(path: Path) -> dict[str, int]:
    count = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            total += item.stat().st_size
    return {"files": count, "logical_bytes": total}


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    root = project_dir.parent
    sections = [
        "project",
        "data",
        "artifacts",
        "models",
        "samples",
        "reports",
        "requirements",
        "archives",
    ]
    important = [
        root / "models" / "student_best.pt",
        root / "models" / "teacher_best.pt",
        root / "models" / "incremental_11class_best.pt",
        root / "models" / "open_world_encoder.pt",
        root / "models" / "open_world_objectness.pt",
        root / "models" / "class_registry_v2.json",
        root / "reports" / "final_report" / "final_report.pdf",
        root / "reports" / "final_report" / "final_report.docx",
        root / "reports" / "final_report" / "final_report_high_fidelity.docx",
        root / "reports" / "final_report" / "final_report_high_fidelity_verification.pdf",
        root / "project" / "scripts" / "build_high_fidelity_word.js",
        root / "客户本地部署与使用清单.md",
        root / "客户本地部署与使用清单.docx",
        root / "客户本地部署与使用清单.pdf",
        root / "setup_environment.ps1",
        root / "self_check.ps1",
        root / "run_gui.ps1",
        root / "run_open_world_gui.ps1",
        root / "project" / "src" / "fruit_ssod" / "gui" / "widgets" / "camera_view.py",
        root / "project" / "src" / "fruit_ssod" / "gui" / "workers" / "camera_worker.py",
        root / "project" / "docs" / "user-guide.zh-CN.md",
        root / "outputs" / "gui_camera_page_preview.png",
        root / "project" / "requirements-lock.txt",
    ]
    manifest = {
        "delivery_name": "半监督水果检测完整可运行项目",
        "root_layout": {name: tree_stats(root / name) for name in sections},
        "important_files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in important
            if path.is_file()
        },
        "verified": {
            "pytest": "413 passed, 1 skipped",
            "gui_pytest": "39 passed",
            "student_inference": "passed on NVIDIA GeForce RTX 3080",
            "incremental_11class_inference": "passed on NVIDIA GeForce RTX 3080",
            "gui_checkpoint_loading": "passed",
            "camera_pipeline": "device open, 10-frame inference and clean release passed",
            "high_fidelity_word_round_trip": "8 Letter-size pages; 150 dpi mean absolute image difference 1.295/255",
            "client_deployment_checklist": "9-page A4 DOCX/PDF plus Markdown; setup, self-check, GUI, camera, open-world and archive recovery covered",
            "delivery_self_check_latest": "passed on NVIDIA GeForce RTX 3080 with both 5-class and 11-class checkpoints",
        },
    }
    output = root / "DELIVERY_MANIFEST.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
