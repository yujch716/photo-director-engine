from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLASS_NAMES = ["youtube_logo"]


def _font(size: int):
    # Use a common system font if available; fall back to PIL default.
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def random_background(w: int, h: int) -> Image.Image:
    """Create a simple synthetic SNS/drone-like background."""
    rng = np.random.default_rng(random.randint(0, 2**32 - 1))

    # Smooth gradient base
    c1 = rng.integers(30, 230, size=3)
    c2 = rng.integers(30, 230, size=3)
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    blend = 0.6 * x + 0.4 * y
    arr = (c1 * (1 - blend) + c2 * blend).astype(np.uint8)
    arr = np.repeat(arr, h, axis=0) if arr.shape[0] == 1 else arr

    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Landscape-like horizon/sea/sky bands sometimes
    if random.random() < 0.45:
        horizon = random.randint(int(h * 0.35), int(h * 0.7))
        sky = tuple(rng.integers(110, 235, size=3).tolist()) + (180,)
        ground = tuple(rng.integers(40, 180, size=3).tolist()) + (200,)
        draw.rectangle([0, 0, w, horizon], fill=sky)
        draw.rectangle([0, horizon, w, h], fill=ground)
        for _ in range(random.randint(2, 8)):
            y0 = random.randint(horizon, h)
            draw.line([0, y0, w, y0 + random.randint(-10, 10)], fill=(255, 255, 255, random.randint(15, 50)), width=random.randint(1, 3))

    # Random visual elements: boxes, circles, text-like strips
    for _ in range(random.randint(15, 55)):
        color = tuple(rng.integers(0, 255, size=3).tolist()) + (random.randint(25, 120),)
        if random.random() < 0.5:
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2 = min(w, x1 + random.randint(20, max(25, w // 4)))
            y2 = min(h, y1 + random.randint(10, max(15, h // 5)))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            r = random.randint(5, max(8, min(w, h) // 10))
            cx, cy = random.randint(0, w), random.randint(0, h)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

    # Mild blur/noise/compression-like variance
    if random.random() < 0.55:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
    if random.random() < 0.7:
        arr = np.array(img).astype(np.int16)
        noise = rng.normal(0, random.uniform(1.5, 8.0), size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")

    return img


def make_youtube_like_logo(width: int, variant: str) -> Image.Image:
    """Make a simple YouTube-like logo graphic with alpha."""
    width = max(32, width)
    if variant == "play_only":
        h = int(width * 0.70)
        logo = Image.new("RGBA", (width, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(logo)
        radius = max(4, int(h * 0.18))
        red = (230 + random.randint(-25, 20), random.randint(0, 35), random.randint(0, 35), random.randint(150, 255))
        d.rounded_rectangle([0, 0, width - 1, h - 1], radius=radius, fill=red)
        tri_w = int(width * 0.32)
        tri_h = int(h * 0.42)
        cx, cy = width // 2 + int(width * 0.03), h // 2
        pts = [(cx - tri_w//3, cy - tri_h//2), (cx - tri_w//3, cy + tri_h//2), (cx + tri_w//2, cy)]
        d.polygon(pts, fill=(255, 255, 255, random.randint(210, 255)))
        return logo

    if variant == "wordmark":
        h = int(width * 0.28)
        logo = Image.new("RGBA", (width, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(logo)
        icon_w = int(width * 0.28)
        radius = max(3, int(h * 0.18))
        d.rounded_rectangle([0, 0, icon_w, h - 1], radius=radius, fill=(230, 0, 0, random.randint(170, 255)))
        tri_w = int(icon_w * 0.36)
        tri_h = int(h * 0.42)
        cx, cy = icon_w // 2 + int(icon_w * 0.03), h // 2
        d.polygon([(cx - tri_w//3, cy - tri_h//2), (cx - tri_w//3, cy + tri_h//2), (cx + tri_w//2, cy)], fill=(255, 255, 255, 240))
        font = _font(max(12, int(h * 0.63)))
        color = random.choice([(20, 20, 20, 240), (255, 255, 255, 240)])
        d.text((icon_w + int(width * 0.03), int(h * 0.04)), "YouTube", font=font, fill=color)
        return logo

    # watermark: a translucent red play icon plus faint background rectangle
    h = int(width * 0.58)
    logo = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(logo)
    alpha = random.randint(70, 155)
    d.rounded_rectangle([0, 0, width - 1, h - 1], radius=max(3, h // 5), fill=(230, 0, 0, alpha))
    tri_w = int(width * 0.32)
    tri_h = int(h * 0.42)
    cx, cy = width // 2 + int(width * 0.03), h // 2
    d.polygon([(cx - tri_w//3, cy - tri_h//2), (cx - tri_w//3, cy + tri_h//2), (cx + tri_w//2, cy)], fill=(255, 255, 255, min(255, alpha + 80)))
    return logo


def paste_logo(bg: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int], Image.Image]:
    w, h = bg.size
    variant = random.choices(["play_only", "wordmark", "watermark"], weights=[0.62, 0.18, 0.20], k=1)[0]
    logo_w = int(w * random.uniform(0.06, 0.26))
    logo = make_youtube_like_logo(logo_w, variant)

    if random.random() < 0.18:
        angle = random.uniform(-8, 8)
        logo = logo.rotate(angle, expand=True, resample=Image.BICUBIC)

    lw, lh = logo.size
    margin_x = max(2, int(w * 0.015))
    margin_y = max(2, int(h * 0.015))
    positions = [
        (margin_x, margin_y),
        (w - lw - margin_x, margin_y),
        (margin_x, h - lh - margin_y),
        (w - lw - margin_x, h - lh - margin_y),
        (random.randint(margin_x, max(margin_x, w - lw - margin_x)), random.randint(margin_y, max(margin_y, h - lh - margin_y))),
    ]
    x, y = random.choices(positions, weights=[0.17, 0.17, 0.23, 0.23, 0.20], k=1)[0]
    x = max(0, min(w - lw, x))
    y = max(0, min(h - lh, y))

    out = bg.convert("RGBA")
    out.alpha_composite(logo, (x, y))

    mask = Image.new("L", (w, h), 0)
    logo_alpha = logo.split()[-1]
    mask.paste(logo_alpha, (x, y))
    # binarize mask
    mask_np = np.array(mask)
    mask_np = (mask_np > 10).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np, "L")

    return out.convert("RGB"), (x, y, x + lw, y + lh), mask


def add_negative_distractors(bg: Image.Image) -> Image.Image:
    # Add red shapes that are NOT labeled as logo for false-positive resistance.
    img = bg.convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for _ in range(random.randint(1, 4)):
        rw = random.randint(max(12, w // 30), max(18, w // 8))
        rh = random.randint(max(8, h // 40), max(12, h // 10))
        x = random.randint(0, max(0, w - rw))
        y = random.randint(0, max(0, h - rh))
        d.rounded_rectangle([x, y, x + rw, y + rh], radius=random.randint(1, max(2, min(rw, rh)//4)), fill=(random.randint(180, 255), random.randint(0, 70), random.randint(0, 70), random.randint(80, 170)))
    return img.convert("RGB")


def yolo_label_from_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def make_one(path_img: Path, path_lbl: Path, path_mask: Path, positive: bool = True) -> None:
    sizes = [(640, 640), (800, 600), (960, 540), (720, 960), (1024, 768), (1280, 720)]
    w, h = random.choice(sizes)
    bg = random_background(w, h)

    labels: List[str] = []
    if positive:
        img, bbox, mask = paste_logo(bg)
        labels.append(yolo_label_from_bbox(bbox, w, h))
        # Sometimes add a second small logo/watermark.
        if random.random() < 0.08:
            img, bbox2, mask2 = paste_logo(img)
            labels.append(yolo_label_from_bbox(bbox2, w, h))
            mask_np = np.maximum(np.array(mask), np.array(mask2))
            mask = Image.fromarray(mask_np.astype(np.uint8), "L")
    else:
        img = add_negative_distractors(bg)
        mask = Image.new("L", (w, h), 0)

    path_img.parent.mkdir(parents=True, exist_ok=True)
    path_lbl.parent.mkdir(parents=True, exist_ok=True)
    path_mask.parent.mkdir(parents=True, exist_ok=True)

    img.save(path_img, quality=random.randint(82, 96))
    path_lbl.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    mask.save(path_mask)


def build_dataset(root: Path, train_count: int = 320, val_count: int = 80, negative_ratio: float = 0.22, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if root.exists():
        shutil.rmtree(root)
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)

    for split, count in [("train", train_count), ("val", val_count)]:
        for i in range(count):
            positive = random.random() > negative_ratio
            stem = f"{split}_{i:05d}"
            make_one(
                root / "images" / split / f"{stem}.jpg",
                root / "labels" / split / f"{stem}.txt",
                root / "masks" / split / f"{stem}.png",
                positive=positive,
            )

    yaml_text = """# Synthetic YOLO dataset for detecting YouTube-like logo overlays.\n# Put this folder at data/logo_yolo so logo_yolo_train.py can find it.\npath: data/logo_yolo\ntrain: images/train\nval: images/val\nnames:\n  0: youtube_logo\n"""
    (root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    readme = f"""# Synthetic YouTube-logo YOLO dataset\n\nThis dataset is generated for a small logo detector used before mask-based inpainting.\n\n- Class: `youtube_logo`\n- Train images: {train_count}\n- Val images: {val_count}\n- YOLO labels: `labels/train`, `labels/val`\n- Optional binary masks for inpainting experiments: `masks/train`, `masks/val`\n\nThe graphics are synthetic YouTube-like play-button overlays on generated backgrounds.\nUse real collected screenshots later to improve robustness.\n\nTraining command with the provided `logo_yolo_train.py`:\n\n```bash\npip install ultralytics pyyaml\npython logo_yolo_train.py --data data/logo_yolo/dataset.yaml --epochs 80 --imgsz 640 --batch 8\n```\n"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/logo_yolo")
    parser.add_argument("--train", type=int, default=320)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--negative-ratio", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(Path(args.root), args.train, args.val, args.negative_ratio, args.seed)
    print(f"[OK] Dataset generated at {args.root}")


if __name__ == "__main__":
    main()
