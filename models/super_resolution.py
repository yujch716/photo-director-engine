from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import types

import numpy as np


MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_NAME = "RealESRGAN_x4plus"

_upsampler: Any | None = None
_device: Any | None = None
_model_path: Path | None = None


def _default_weights_path() -> Path:
    return Path(__file__).resolve().parents[1] / "weights" / f"{MODEL_NAME}.pth"


def _resolve_weights_path() -> Path:
    import os

    configured_path = os.environ.get("REAL_ESRGAN_MODEL_PATH")
    if configured_path:
        return Path(configured_path)
    return _default_weights_path()


def _ensure_weights(model_path: Path) -> None:
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        torch.hub.download_url_to_file(MODEL_URL, str(model_path), progress=True)
    except Exception as exc:
        raise RuntimeError(
            "Real-ESRGAN weights are missing and automatic download failed. "
            f"Download {MODEL_URL} to {model_path}, or set REAL_ESRGAN_MODEL_PATH."
        ) from exc


def _install_torchvision_functional_tensor_shim() -> None:
    """BasicSR expects an older torchvision module path that newer torchvision removed."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return

    try:
        from torchvision.transforms.functional import rgb_to_grayscale
    except Exception:
        return

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _get_realesrgan_upsampler() -> tuple[Any, Any, Path]:
    global _upsampler, _device, _model_path

    if _upsampler is not None and _device is not None and _model_path is not None:
        return _upsampler, _device, _model_path

    try:
        import torch

        _install_torchvision_functional_tensor_shim()

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:
        raise RuntimeError(
            "Real-ESRGAN SR requires torch, basicsr, and realesrgan. "
            f"Install with: pip install realesrgan. Import error: {exc}"
        ) from exc

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model_path = _resolve_weights_path()
    _ensure_weights(_model_path)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    _upsampler = RealESRGANer(
        scale=4,
        model_path=str(_model_path),
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=_device.type == "cuda",
        device=_device,
    )

    return _upsampler, _device, _model_path


def apply_pretrained_cnn_sr(img_bgr: np.ndarray, outscale: float = 4.0) -> tuple[np.ndarray, dict[str, Any]]:
    upsampler, device, model_path = _get_realesrgan_upsampler()
    output_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale)

    return output_bgr, {
        "backend": "real_esrgan",
        "model": MODEL_NAME,
        "model_path": str(model_path),
        "device": str(device),
        "scale": outscale,
        "applied": True,
    }
