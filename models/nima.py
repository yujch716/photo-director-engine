from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


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
        raise RuntimeError("NIMA requires pyiqa and torch. Install with: pip install pyiqa") from exc

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _metric = pyiqa.create_metric("nima", device=_device)
    return _metric, _device


def run_nima_score(image_bytes: bytes, filename: str = "image.jpg") -> dict[str, Any]:
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
            "device": str(device),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
