"""Render representative protected-holdout predictions from the 11-class detector."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ZH = {
    "Apple": "苹果",
    "Banana": "香蕉",
    "Orange": "橙子",
    "Strawberry": "草莓",
    "Pineapple": "菠萝",
    "Avocado": "牛油果",
    "Blueberry": "蓝莓",
    "Cherry": "樱桃",
    "Kiwi": "猕猴桃",
    "Mango": "芒果",
    "Rockmelon": "网纹瓜",
}
COLORS = ["#EF4444", "#F59E0B", "#FB923C", "#EC4899", "#84CC16", "#22C55E", "#3B82F6", "#E11D48", "#14B8A6", "#A855F7", "#06B6D4"]


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def truth_xyxy(record: dict, width: int, height: int) -> list[tuple[float, float, float, float]]:
    result = []
    for box in record["boxes"]:
        xc, yc, bw, bh = box["x_center"], box["y_center"], box["width"], box["height"]
        result.append(((xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height))
    return result


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> None:
    face = font(23, True)
    bounds = draw.textbbox(xy, text, font=face)
    draw.rectangle((bounds[0] - 4, bounds[1] - 2, bounds[2] + 4, bounds[3] + 2), fill="#0F172AE8")
    draw.text(xy, text, fill=color, font=face)


def render(item: dict, output: Path) -> None:
    with Image.open(item["record"]["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for detection in item["detections"]:
        box = detection["xyxy"]
        color = COLORS[detection["class_id"] % len(COLORS)]
        draw.rectangle(box, outline=color, width=5)
        text = f"{ZH.get(detection['name'], detection['name'])} {detection['confidence']:.2f}"
        label(draw, (int(box[0]) + 3, max(3, int(box[1]) - 32)), text, color)
    footer = 90
    canvas = Image.new("RGB", (image.width, image.height + footer), "white")
    canvas.paste(image, (0, 0))
    foot = ImageDraw.Draw(canvas)
    category = item["record"]["category"]
    foot.text((20, image.height + 10), "开放类别增量检测 · 彩色实线框为模型预测", fill="#0F172A", font=font(25, True))
    foot.text((20, image.height + 49), f"独立保护测试集参考类别：{ZH.get(category, category)}（{category}）", fill="#475569", font=font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--protected-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    from ultralytics import YOLO

    records = [
        record
        for record in json.loads(args.protected_truth.read_text(encoding="utf-8"))["records"]
        if record["split"] == "holdout"
    ]
    model = YOLO(str(args.weights.resolve(strict=True)))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        with Image.open(record["image_path"]) as source:
            width, height = source.size
        truths = truth_xyxy(record, width, height)
        result = model.predict(record["image_path"], conf=args.confidence, imgsz=args.image_size, device=args.device, verbose=False)[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "class_id": class_id,
                    "name": str(model.names[class_id]),
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                }
            )
        correct_class = [detection for detection in detections if detection["name"] == record["category"]]
        matched = sum(max((iou(detection["xyxy"], truth) for detection in correct_class), default=0.0) >= 0.5 for truth in truths)
        score = matched * 20 + sum(detection["confidence"] for detection in correct_class) - max(0, len(detections) - matched)
        grouped[record["category"]].append({"score": score, "matched": matched, "record": record, "detections": detections})

    selected = []
    for category in ("Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon"):
        selected.extend(sorted(grouped[category], key=lambda item: (item["matched"] > 0, item["score"]), reverse=True)[: args.per_class])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, item in enumerate(selected, start=1):
        record = item["record"]
        path = args.output / f"{index:02d}_{record['category']}_{record['image_id']}.jpg"
        render(item, path)
        paths.append(path)
    tile = (520, 390)
    columns = 3
    sheet = Image.new("RGB", (columns * tile[0], math.ceil(len(paths) / columns) * tile[1]), "#E2E8F0")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            preview = ImageOps.contain(source.convert("RGB"), tile, Image.Resampling.LANCZOS)
        x = (index % columns) * tile[0] + (tile[0] - preview.width) // 2
        y = (index // columns) * tile[1] + (tile[1] - preview.height) // 2
        sheet.paste(preview, (x, y))
    sheet.save(args.output / "incremental_11class_contact_sheet.jpg", quality=94, subsampling=0)
    (args.output / "example_manifest.json").write_text(
        json.dumps({"weights": str(args.weights.resolve()), "images": [str(path) for path in paths]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"images": len(paths), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
