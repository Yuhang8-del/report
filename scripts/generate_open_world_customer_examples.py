"""Build customer-facing cards from the protected open-world holdout results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_ZH = {
    "Avocado": "牛油果",
    "Blueberry": "蓝莓",
    "Cherry": "樱桃",
    "Kiwi": "猕猴桃",
    "Mango": "芒果",
    "Rockmelon": "网纹瓜",
}

CLASS_COLORS = {
    "Avocado": "#4D7C0F",
    "Blueberry": "#4338CA",
    "Cherry": "#BE123C",
    "Kiwi": "#65A30D",
    "Mango": "#EA580C",
    "Rockmelon": "#0F766E",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def make_card(record: dict, truth: str, mapped: str | None, destination: Path) -> None:
    width, height = 1200, 900
    header_height, footer_height = 88, 210
    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    color = CLASS_COLORS[truth]
    draw.rectangle((0, 0, width, header_height), fill=color)
    draw.text((34, 20), "开放类别发现（图像级）", fill="white", font=font(36, bold=True))
    draw.text((830, 28), "独立留出测试样本", fill="white", font=font(25))

    image_area = (width, height - header_height - footer_height)
    image = fit_image(Path(record["path"]), (image_area[0] - 40, image_area[1] - 30))
    image_x = (width - image.width) // 2
    image_y = header_height + (image_area[1] - image.height) // 2
    canvas.paste(image, (image_x, image_y))

    footer_y = height - footer_height
    draw.rectangle((0, footer_y, width, height), fill="white")
    draw.line((0, footer_y, width, footer_y), fill=color, width=5)
    novelty = float(record.get("novelty_score", 0.0))
    is_semantic_match = mapped == truth
    semantic_text = f"{CLASS_ZH[truth]}（{truth}）" if is_semantic_match else "待人工确认命名"
    draw.text((35, footer_y + 25), "判定：新类别候选", fill="#0F172A", font=font(31, bold=True))
    draw.text((420, footer_y + 25), f"发现簇：Cluster {record['cluster_id']}", fill="#334155", font=font(29))
    draw.text((35, footer_y + 78), f"聚类后验映射：{semantic_text}", fill=color, font=font(29, bold=True))
    draw.text((680, footer_y + 78), f"新颖度：{novelty:.2f}", fill="#0F172A", font=font(29, bold=True))
    draw.text(
        (35, footer_y + 128),
        f"测试参考类别：{CLASS_ZH[truth]}（{truth}）   ·   新颖度越高，表示越不像已知五类水果",
        fill="#475569",
        font=font(23),
    )
    draw.text(
        (35, footer_y + 170),
        "说明：该结果展示图像级未知类别发现与聚类，不代表框级开放世界目标检测。",
        fill="#64748B",
        font=font(20),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95, subsampling=0)


def make_sheet(paths: list[Path], output: Path) -> None:
    tile_size = (500, 375)
    margin, caption_height = 18, 42
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_size[0] + (columns + 1) * margin, rows * (tile_size[1] + caption_height) + (rows + 1) * margin),
        "#E2E8F0",
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        card = fit_image(path, tile_size)
        x = margin + (index % columns) * (tile_size[0] + margin)
        y = margin + (index // columns) * (tile_size[1] + caption_height + margin)
        sheet.paste(card, (x + (tile_size[0] - card.width) // 2, y))
        parts = path.stem.split("_")
        caption = f"{parts[0]}  {parts[1]}" if len(parts) > 1 else path.stem
        draw.text((x + 8, y + tile_size[1] + 6), caption, fill="#0F172A", font=font(21, bold=True))
    sheet.save(output, quality=94, subsampling=0)


def main() -> None:
    args = arguments()
    results = json.loads((args.artifact_dir / "discovery_results.json").read_text(encoding="utf-8"))
    protected = json.loads((args.artifact_dir / "protected_evaluation_labels.json").read_text(encoding="utf-8"))
    truth_by_id = {record["image_id"]: record["category"] for record in protected["records"]}
    mapping = {
        int(cluster): category
        for cluster, category in results["metrics"]["discovery"]["posthoc_cluster_to_category"].items()
    }
    assignments = [
        json.loads(line)
        for line in (args.artifact_dir / "cluster_assignments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in assignments:
        if record.get("split") != "holdout" or not Path(record["path"]).exists():
            continue
        truth = truth_by_id[record["image_id"]]
        record["mapped_category"] = mapping.get(int(record["cluster_id"]))
        grouped[truth].append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for truth in CLASS_ZH:
        candidates = sorted(
            grouped[truth],
            key=lambda item: (
                item["mapped_category"] == truth,
                item.get("novelty_score", 0.0) >= 0.5,
                item.get("novelty_score", 0.0),
                -item.get("cluster_distance", 99.0),
            ),
            reverse=True,
        )
        for candidate in candidates[: args.per_class]:
            index = len(written) + 1
            destination = args.output_dir / f"{index:02d}_{truth}_{candidate['image_id']}.jpg"
            make_card(candidate, truth, candidate["mapped_category"], destination)
            written.append(destination)

    sheet = args.output_dir / "open_world_holdout_contact_sheet.jpg"
    make_sheet(written, sheet)
    summary = {
        "source": "protected holdout split",
        "image_count": len(written),
        "classes": list(CLASS_ZH),
        "individual_images": [path.name for path in written],
        "contact_sheet": sheet.name,
        "limitation": "image-level open-world discovery; not box-level open-world detection",
    }
    (args.output_dir / "README_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
