"""Finalize a generated professional report after Word has exported the PDF."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("fruit_report_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(strict=True)
    builder = load_builder(args.builder.resolve(strict=True))
    builder.create_data_index(out, args.split_manifest, args.dataset_yaml, args.unlabeled_manifest)
    docx = (out / "final_report.docx").resolve(strict=True)
    pdf = (out / "final_report.pdf").resolve(strict=True)
    builder.build_inventory(
        out,
        docx,
        pdf,
        Path(r"E:\bishe\fruit\.worktrees\fruit-ssod-implementation"),
        Path(r"E:\fruit_ssod_runtime"),
    )
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report_docx"] = {"path": str(docx), "sha256": builder.sha256(docx)}
    manifest["report_pdf"] = {"path": str(pdf), "sha256": builder.sha256(pdf)}
    manifest["data_index"] = {
        "unlabeled_train": str(out / "data" / "splits" / "unlabeled_train.txt"),
        "unlabeled_count": sum(1 for line in (out / "data" / "splits" / "unlabeled_train.txt").read_text(encoding="utf-8").splitlines() if line.strip()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "pdf_sha256": manifest["report_pdf"]["sha256"], "unlabeled_count": manifest["data_index"]["unlabeled_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
