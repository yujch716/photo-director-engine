"""GAIC(Grid Anchor based Image Cropping) 구도 점수 — lazy singleton 로더.

models/gaic_bundle 의 GaicScorer를 감싸, 원본 이미지 + 후보 박스들을 받아
각 박스(크롭)의 구도 점수를 반환한다. 랜드마크가 없는 캡처에서 best 크롭을
DINO 유사도 대신 GAIC 구도 점수로 고를 때 쓴다.

환경변수:
  GAIC_BACKBONE : vgg16 | mobilenetv2 | shufflenetv2 (기본 mobilenetv2)
  GAIC_WEIGHT   : .pth 경로 (기본 gaic_bundle/pretrained_models/GAIC-<backbone>-reddim<d>.pth)

가중치가 없으면 안내 메시지를 담아 RuntimeError.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_BUNDLE = Path(__file__).resolve().parent / "gaic_bundle"
_REDDIM = {"vgg16": 32, "mobilenetv2": 16, "shufflenetv2": 32}

GAIC_BACKBONE = os.environ.get("GAIC_BACKBONE", "mobilenetv2")
GAIC_WEIGHT = os.environ.get(
    "GAIC_WEIGHT",
    str(_BUNDLE / "pretrained_models" / f"GAIC-{GAIC_BACKBONE}-reddim{_REDDIM.get(GAIC_BACKBONE, 32)}.pth"),
)

_scorer: Any | None = None


def _get_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("GAIC는 torch가 필요합니다.") from exc

    if str(_BUNDLE) not in sys.path:
        sys.path.insert(0, str(_BUNDLE))  # gaic_scorer / networks import 위해 번들을 루트로

    if not os.path.exists(GAIC_WEIGHT):
        raise RuntimeError(
            f"GAIC 가중치 없음: {GAIC_WEIGHT}\n"
            "       models/gaic_bundle/pretrained_models/ 에 GAIC-<backbone>-reddim<d>.pth 를 넣으세요."
        )

    from gaic_scorer import GaicScorer

    # GAIC 커스텀 op(roi/rod align)은 순수 PyTorch라 CPU에서 동작. cuda 있으면 cuda.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gaic] loading {GAIC_BACKBONE} on {device} ({GAIC_WEIGHT})")
    _scorer = GaicScorer(GAIC_BACKBONE, GAIC_WEIGHT, device=device)
    return _scorer


def score_boxes(image, boxes) -> list[float]:
    """원본 이미지 + 후보 박스들의 GAIC 구도 점수 리스트를 반환한다.

    Args:
        image: PIL.Image(RGB) 또는 np.ndarray(RGB).
        boxes: [[x1, y1, x2, y2], ...] 원본 픽셀 좌표.
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    img_bgr = np.ascontiguousarray(arr[:, :, ::-1])  # RGB → BGR (GAIC는 cv2 관례)
    scorer = _get_scorer()
    return [float(s) for s in scorer.score_crops(img_bgr, [list(b) for b in boxes])]
