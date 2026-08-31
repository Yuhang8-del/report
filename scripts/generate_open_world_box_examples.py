"""Render representative held-out box-level open-world predictions for customers."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ZH = {"Avocado": "牛油果", "Blueberry": "蓝莓", "Cherry": "樱桃", "Kiwi": "猕猴桃", "Mango": "芒果", "Rockmelon": "网纹瓜"}
KNOWN_ZH = {"Apple": "苹果", "Banana": "香蕉", "Orange": "橙子", "Strawberry": "草莓", "Pineapple": "菠萝"}


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def truth_xyxy(record: dict, size: tuple[int, int]):
    width, height = size
    result = []
    for box in record["boxes"]:
        xc, yc, bw, bh = box["x_center"], box["y_center"], box["width"], box["height"]
        result.append(((xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height))
    return result


def dashed_rectangle(draw: ImageDraw.ImageDraw, box, color: str, width: int = 6, dash: int = 18) -> None:
    x1, y1, x2, y2 = [int(value) for value in box]
    for start in range(x1, x2, dash * 2):
        draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=width)
    for start in range(y1, y2, dash * 2):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=width)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> None:
    face = font(24, bold=True)
    bounds = draw.textbbox(xy, text, font=face, stroke_width=0)
    draw.rectangle((bounds[0] - 5, bounds[1] - 3, bounds[2] + 5, bounds[3] + 3), fill="#0F172AE6")
    draw.text(xy, text, fill=color, font=face)


def render(row: dict, assignments: dict[str, dict], truth: dict, output: Path) -> None:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    known_colors = ["#EF4444", "#F59E0B", "#FB923C", "#EC4899", "#84CC16"]
    for detection in row["known_detections"]:
        box = detection["xyxy"]
        color = known_colors[detection["class_id"]]
        draw.rectangle(box, outline=color, width=5)
        text = f"{KNOWN_ZH.get(detection['class_name'], detection['class_name'])} {detection['confidence']:.2f}"
        label(draw, (int(box[0]) + 3, max(3, int(box[1]) - 34)), text, color)
    for proposal in row["unknown_proposals"]:
        box = proposal["xyxy"]
        assignment = assignments.get(proposal["proposal_id"], {})
        cluster = assignment.get("cluster_id")
        candidate = assignment.get("candidate_name")
        text = f"未知水果 C{cluster}" if cluster is not None else "未知水果"
        if candidate:
            text += f" / {ZH.get(candidate, candidate)}?"
        text += f" {proposal['novelty_score']:.2f}"
        dashed_rectangle(draw, box, "#FACC15")
        label(draw, (int(box[0]) + 3, max(3, int(box[1]) - 34)), text, "#FACC15")
    footer = 92
    canvas = Image.new("RGB", (image.width, image.height + footer), "white")
    canvas.paste(image, (0, 0))
    footer_draw = ImageDraw.Draw(canvas)
    category = truth["category"]
    footer_draw.text((22, image.height + 12), "框级开放世界检测 · 黄色虚线框为Unknown候选", fill="#0F172A", font=font(25, True))
    footer_draw.text((22, image.height + 51), f"独立测试参考类别：{ZH.get(category, category)}（{category}）", fill="#475569", font=font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--protected-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    assignments = {item["proposal_id"]: item for item in (json.loads(line) for line in args.assignments.read_text(encoding="utf-8").splitlines() if line.strip())}
    truth = {item["image_id"]: item for item in json.loads(args.protected_truth.read_text(encoding="utf-8"))["records"]}
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "holdout":
            continue
        with Image.open(row["image_path"]) as image:
            boxes = truth_xyxy(truth[row["image_id"]], image.size)
        proposals = row["unknown_proposals"]
        matched = sum(max((iou(proposal["xyxy"], box) for proposal in proposals), default=0.0) >= 0.5 for box in boxes)
        fp = sum(max((iou(proposal["xyxy"], box) for box in boxes), default=0.0) < 0.5 for proposal in proposals)
        score = matched * 10 - fp * 2 + sum(item["novelty_score"] for item in proposals)
        grouped[truth[row["image_id"]]["category"]].append((score, matched, row))
    selected = []
    for category in ZH:
        ranked = sorted(grouped[category], key=lambda item: (item[1] > 0, item[0]), reverse=True)
        selected.extend(item[2] for item in ranked[: args.per_class])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, row in enumerate(selected, start=1):
        category = truth[row["image_id"]]["category"]
        path = args.output / f"{index:02d}_{category}_{row['image_id']}.jpg"
        render(row, assignments, truth[row["image_id"]], path)
        paths.append(path)
    # Contact sheet uses the complete annotated cards without altering their labels.
    tile = (520, 390)
    columns = 3
    rows_count = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * tile[0], rows_count * tile[1]), "#E2E8F0")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            preview = ImageOps.contain(source.convert("RGB"), tile, Image.Resampling.LANCZOS)
        x = (index % columns) * tile[0] + (tile[0] - preview.width) // 2
        y = (index // columns) * tile[1] + (tile[1] - preview.height) // 2
        sheet.paste(preview, (x, y))
    sheet.save(args.output / "open_world_box_contact_sheet.jpg", quality=94, subsampling=0)
    print(json.dumps({"images": len(paths), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
