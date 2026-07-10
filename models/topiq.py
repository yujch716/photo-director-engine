"""TOPIQ (pyiqa) 무참조 이미지 품질 점수 — lazy singleton 로더.

pyiqa의 TOPIQ 계열 지표로 이미지 한 장의 품질 점수를 낸다. NIMA(nima.py)와 같은
pyiqa 기반이라 구조가 동일하다.

환경변수:
  TOPIQ_METRIC : pyiqa 지표명 (기본 "topiq_nr" — 무참조 품질).
                 다른 예: topiq_iaa(미학, AVA), topiq_nr-spaq 등.

첫 호출 시 가중치를 자동 다운로드(~수백 MB)하고 캐시한다.
점수는 대부분 높을수록 좋음(higher_better). 지표에 따라 범위가 다르다.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

TOPIQ_METRIC = os.environ.get("TOPIQ_METRIC", "topiq_nr")

_metric: Any | None = None
_device: Any | None = None


def _get_metric() -> tuple[Any, Any]:
    global _metric, _device

    if _metric is not None:
        return _metric, _device

    try:
        import pyiqa
        import torch
    except ImportError as exc:
        raise RuntimeError("TOPIQ requires pyiqa and torch. Install with: pip install pyiqa") from exc

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _metric = pyiqa.create_metric(TOPIQ_METRIC, device=_device)
    return _metric, _device


def run_topiq_score(image_bytes: bytes, filename: str = "image.jpg") -> dict[str, Any]:
    metric, device = _get_metric()
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        score = metric(temp_path)
        return {
            "score": float(score.detach().cpu().item()),
            "metric": TOPIQ_METRIC,                 # 어떤 topiq 변형을 썼는지
            "lower_better": bool(getattr(metric, "lower_better", False)),
            "device": str(device),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
