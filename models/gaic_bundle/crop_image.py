"""Auto-crop your own image with GAIC.

Generates fixed-size 4:3 candidate windows, scores them with GAIC, and saves
the best crop (plus an annotated preview).

Examples:
    python crop_image.py /path/to/photo.jpg
    python crop_image.py photo.jpg --backbone vgg16 --ratio 0.8 --out out_dir
    python crop_image.py photo.jpg --topk 3          # also save top-3 crops
"""

import argparse
import os
import sys

import numpy as np
import cv2

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from gaic_scorer import GaicScorer, generate_fixed_crops

_REDDIM = {"vgg16": 32, "shufflenetv2": 32, "mobilenetv2": 16}


def parse_args():
    p = argparse.ArgumentParser(description="Auto-crop an image with GAIC")
    p.add_argument("image", help="path to your image")
    p.add_argument("--backbone", default="mobilenetv2", choices=list(_REDDIM))
    p.add_argument("--weight", default=None,
                   help="weight .pth (default: pretrained_models/GAIC-<backbone>-reddim<d>.pth)")
    p.add_argument("--ratio", type=float, default=0.8, help="crop size ratio (default 0.8)")
    p.add_argument("--n", type=int, default=48, help="number of candidate windows")
    p.add_argument("--object", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                   default=None, help="object bbox that every crop must fully contain")
    p.add_argument("--topk", type=int, default=1, help="how many top crops to save")
    p.add_argument("--out", default=os.path.join(_REPO, "crop_out"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    assert os.path.exists(a.image), a.image
    weight = a.weight or os.path.join(
        _REPO, "pretrained_models",
        "GAIC-{}-reddim{}.pth".format(a.backbone, _REDDIM[a.backbone]))
    assert os.path.exists(weight), "weight not found: {}".format(weight)

    img = cv2.imread(a.image)
    assert img is not None, "cannot read image: {}".format(a.image)
    H, W = img.shape[:2]
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]

    scorer = GaicScorer(a.backbone, weight, device=a.device)
    boxes = generate_fixed_crops(img.shape, ratio=a.ratio, n=a.n, contain=a.object)
    scores = np.asarray(scorer.score_crops(img, boxes))
    order = np.argsort(-scores)                       # best first

    print("image: {} ({}x{})  candidates: {}  ratio: {}".format(
        a.image, W, H, len(boxes), a.ratio))

    if a.object is not None:
        ox1, oy1, ox2, oy2 = a.object
        def contains(b):
            return b[0] <= ox1 and b[1] <= oy1 and b[2] >= ox2 and b[3] >= oy2
        n_ok = sum(contains(b) for b in boxes)
        cw, ch = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
        print("  object bbox: {}  -> {}/{} crops fully contain it (window {}x{})".format(
            a.object, n_ok, len(boxes), cw, ch))
        if n_ok < len(boxes):
            print("  WARNING: object larger than / off the image edge — "
                  "some crops can't fully contain it (best-effort centered).")

    # save top-k crops
    for rank, i in enumerate(order[:a.topk], start=1):
        x1, y1, x2, y2 = boxes[i]
        fn = os.path.join(a.out, "{}_crop{}_s{:.3f}.jpg".format(stem, rank, scores[i]))
        cv2.imwrite(fn, img[y1:y2, x1:x2])
        print("  #{} score={:.4f} bbox=({},{},{},{}) -> {}".format(
            rank, scores[i], x1, y1, x2, y2, fn))

    # annotated preview: best box (green) + object box (blue) on the original
    prev = img.copy()
    thick = max(2, W // 400)
    bx = boxes[int(order[0])]
    cv2.rectangle(prev, bx[:2], bx[2:], (0, 220, 0), thick)          # best crop
    if a.object is not None:
        ox1, oy1, ox2, oy2 = a.object
        cv2.rectangle(prev, (ox1, oy1), (ox2, oy2), (235, 0, 0), thick)  # object
    prev_fn = os.path.join(a.out, "{}_preview.jpg".format(stem))
    cv2.imwrite(prev_fn, prev)
    print("preview (best box on original): {}".format(prev_fn))


if __name__ == "__main__":
    main()
