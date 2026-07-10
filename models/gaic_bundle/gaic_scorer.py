"""GAIC as a pluggable crop scorer.

Unlike evaluate/demo.py (which lets GAIC generate its own grid anchors), this
wrapper takes an externally supplied list of candidate crop windows (in the
*original* image coordinate system) and returns one GAIC score per window in a
single batched forward pass.

The preprocessing and the box coordinate handling are replicated *exactly* from
the training/eval pipeline so the pretrained weights stay valid:
  - image: cv2 BGR -> RGB, aspect-preserving resize (short side 256, both sides
    rounded to a multiple of 32), ToTensor + ImageNet normalization
    (see evaluate/demo.py:image_preprocessing, dataset/cropping_dataset.py).
  - boxes: fed to the model in the RESIZED image coordinate system. Original
    coords are mapped with the same rule as dataset/cropping_dataset.py:
    rescale_crops -> x1,y1 = floor(coord * ratio), x2,y2 = ceil(coord * ratio).
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from networks.GAIC_model import build_crop_model  # noqa: E402

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# backbone -> reddim, mirroring evaluate/demo.py:build_network
_REDDIM = {"vgg16": 32, "shufflenetv2": 32, "mobilenetv2": 16}

_SHORT_SIDE = 256   # cfg.image_size[0]
_STRIDE = 32        # both sides rounded to a multiple of 32


def _inscribed_4x3(W, H):
    """Largest w:h = 4:3 box that fits inside a W x H image."""
    if W * 3 >= H * 4:               # image at least as wide as 4:3 -> height-limited
        return H * 4.0 / 3.0, float(H)
    return float(W), W * 3.0 / 4.0   # width-limited


def _grid_dims(n):
    """Factor n into (nx, ny), nx >= ny, as close to square as possible."""
    nx, ny = n, 1
    for k in range(1, int(n ** 0.5) + 1):
        if n % k == 0:
            ny, nx = k, n // k       # largest such k -> closest to square
    return nx, ny


def generate_fixed_crops(image_shape, ratio=0.8, n=48, contain=None):
    """Fixed-size 4:3 crop windows slid over a grid, in ORIGINAL image coords.

    Every window has the SAME size (``ratio`` * the largest 4:3 box inscribed
    in the image); only the position changes. This matches the production
    pipeline (constant crop size, varying location) and avoids the score-spread
    inflation that GAIC's content-preservation term causes for mixed sizes.

    Args:
        image_shape: cv2-style shape, (H, W) or (H, W, C).
        ratio: crop size as a fraction of the inscribed 4:3 box (default 0.8).
        n: number of grid positions (default 48). Factored into an nx*ny grid.
        contain: optional (x1, y1, x2, y2) object box that every crop must fully
            enclose (so the object is never cut off). The window is enlarged
            (kept 4:3) if needed to fit the object, and the sliding grid is
            restricted to positions that keep the object inside.

    Returns:
        list of ``n`` (x1, y1, x2, y2) int tuples, all in bounds.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    bw, bh = _inscribed_4x3(W, H)
    cw = min(int(round(ratio * bw)), W)
    ch = min(int(round(ratio * bh)), H)

    if contain is None:
        x_lo, x_hi = 0, W - cw
        y_lo, y_hi = 0, H - ch
    else:
        ox1, oy1 = max(0, int(contain[0])), max(0, int(contain[1]))
        ox2, oy2 = min(W, int(contain[2])), min(H, int(contain[3]))
        ow, oh = ox2 - ox1, oy2 - oy1
        # enlarge the 4:3 window until it can hold the object box
        if cw < ow or ch < oh:
            need_w = max(float(ow), oh * 4.0 / 3.0)
            cw = min(int(np.ceil(need_w)), W)
            ch = min(int(np.ceil(need_w * 3.0 / 4.0)), H)
        # top-left positions that keep [ox1,oy1,ox2,oy2] fully inside the window
        x_lo, x_hi = max(0, ox2 - cw), min(ox1, W - cw)
        y_lo, y_hi = max(0, oy2 - ch), min(oy1, H - ch)
        if x_lo > x_hi or y_lo > y_hi:
            # object cannot be fully contained by any in-image 4:3 window of this
            # size -> best effort: center the window on the object and clamp
            cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            x_lo = x_hi = min(max(0, cx - cw // 2), W - cw)
            y_lo = y_hi = min(max(0, cy - ch // 2), H - ch)

    nx, ny = _grid_dims(n)
    xs = np.linspace(x_lo, x_hi, nx).round().astype(int)
    ys = np.linspace(y_lo, y_hi, ny).round().astype(int)
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((int(x), int(y), int(x + cw), int(y + ch)))
    return boxes


class GaicScorer:
    """Load a GAIC model once and score arbitrary candidate crops with it."""

    def __init__(self, backbone: str, weight_path: str, device: str = "cuda"):
        assert backbone in _REDDIM, \
            "backbone must be one of {}, got {!r}".format(list(_REDDIM), backbone)
        assert os.path.exists(weight_path), weight_path

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[GaicScorer] CUDA unavailable, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.backbone = backbone
        self.reddim = _REDDIM[backbone]

        net = build_crop_model(scale="multi", alignsize=9, reddim=self.reddim,
                               loadweight=False, model=backbone)
        state = torch.load(weight_path, map_location="cpu")
        # strict=True: a backbone/reddim mismatch (wrong weight file) fails loudly
        net.load_state_dict(state)
        self.net = net.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])

    # -- internals ---------------------------------------------------------
    def _preprocess(self, image_bgr: np.ndarray):
        """cv2 BGR image -> (tensor[1,3,H,W], ratio_w, ratio_h) in resized coords."""
        assert isinstance(image_bgr, np.ndarray) and image_bgr.ndim == 3, \
            "image_bgr must be an HxWx3 array (cv2.imread result)"
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])  # BGR -> RGB
        im = Image.fromarray(rgb)
        im_w, im_h = im.size
        scale = float(_SHORT_SIDE) / min(im_h, im_w)
        h = int(round(im_h * scale / _STRIDE) * _STRIDE)
        w = int(round(im_w * scale / _STRIDE) * _STRIDE)
        resized = im.resize((w, h), Image.Resampling.LANCZOS)
        tensor = self.transform(resized).unsqueeze(0)
        ratio_w = float(w) / im_w
        ratio_h = float(h) / im_h
        return tensor, ratio_w, ratio_h

    @staticmethod
    def _rescale_boxes(bboxes, ratio_w, ratio_h) -> np.ndarray:
        """Original-coord [x1,y1,x2,y2] -> resized coords (dataset.rescale_crops)."""
        b = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
        out = np.empty((b.shape[0], 4), dtype=np.float32)
        out[:, 0] = np.floor(b[:, 0] * ratio_w)
        out[:, 1] = np.floor(b[:, 1] * ratio_h)
        out[:, 2] = np.ceil(b[:, 2] * ratio_w)
        out[:, 3] = np.ceil(b[:, 3] * ratio_h)
        return out

    # -- public API --------------------------------------------------------
    @torch.no_grad()
    def score_crops(self, image_bgr: np.ndarray, bboxes) -> list:
        """Return one GAIC score per bbox (input order preserved), one forward pass."""
        assert len(bboxes) > 0, "bboxes is empty"
        tensor, ratio_w, ratio_h = self._preprocess(image_bgr)
        tensor = tensor.to(self.device)

        boxes = self._rescale_boxes(bboxes, ratio_w, ratio_h)         # (N, 4) resized
        rois = torch.from_numpy(boxes).unsqueeze(0).to(self.device)   # (1, N, 4)

        scores = self.net(tensor, rois)                              # (N, 1, 1, 1)
        scores = scores.detach().float().cpu().numpy().reshape(-1)
        assert scores.shape[0] == len(bboxes), (scores.shape, len(bboxes))
        return scores.tolist()

    def best_crop(self, image_bgr: np.ndarray, bboxes):
        """Return (best_index, best_score)."""
        scores = self.score_crops(image_bgr, bboxes)
        idx = int(np.argmax(scores))
        return idx, float(scores[idx])
