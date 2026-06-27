from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


DEFAULT_LOGO_MODEL_PATH = Path(__file__).resolve().parent / "logo_yolo.pt"
_logo_model: Any | None = None


def _image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read image bytes with PIL.") from exc
    return np.array(pil_img)


def _rgb_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()


def _resolve_logo_model_path() -> Path:
    configured_path = os.environ.get("LOGO_YOLO_MODEL_PATH")
    if configured_path:
        return Path(configured_path)
    return DEFAULT_LOGO_MODEL_PATH


def _get_logo_yolo_model() -> Any:
    global _logo_model

    if _logo_model is not None:
        return _logo_model

    model_path = _resolve_logo_model_path()
    if not model_path.exists():
        raise RuntimeError(
            "Logo/arrow YOLO weights not found. "
            f"Put a custom logo detector at {model_path} or set LOGO_YOLO_MODEL_PATH. "
            "You can train it with: python models/logo_yolo_train.py "
            "YOLO is required for automatic logo/arrow detection; OpenCV inpaint only removes masked pixels."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Logo detection requires ultralytics. Install with: pip install ultralytics") from exc

    _logo_model = YOLO(str(model_path))
    return _logo_model


def _detections_to_mask(
    detections: list[dict[str, Any]],
    width: int,
    height: int,
    dilation_px: int = 12,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    if dilation_px > 0 and mask.any():
        kernel_size = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _run_logo_detection(
    img_rgb: np.ndarray,
    confidence: float = 0.25,
    dilation_px: int = 12,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    model = _get_logo_yolo_model()
    results = model(Image.fromarray(img_rgb), conf=confidence)
    result = results[0]
    height, width = img_rgb.shape[:2]

    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append({
            "class": model.names.get(class_id, str(class_id)),
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
        })

    mask = _detections_to_mask(detections, width, height, dilation_px=dilation_px)
    report = {
        "detector": "yolo",
        "model_path": str(_resolve_logo_model_path()),
        "confidence_threshold": confidence,
        "dilation_px": dilation_px,
        "detections": len(detections),
        "mask_pixels": int(np.count_nonzero(mask)),
    }
    return mask, detections, report


def _inpaint_with_opencv(img_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, {
        "backend": "opencv_telea",
        "applied": True,
        "error": None,
    }


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    return _rgb_to_png_bytes(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))


def remove_logo_arrows(
    image_bytes: bytes,
    *,
    confidence: float = 0.25,
    dilation_px: int = 12,
) -> tuple[bytes, bytes, dict[str, Any]]:
    img_rgb = _image_bytes_to_rgb(image_bytes)
    mask, detections, detection_report = _run_logo_detection(
        img_rgb,
        confidence=confidence,
        dilation_px=dilation_px,
    )

    if not detections:
        report = {
            "detection": detection_report,
            "inpaint": {
                "backend": "opencv_telea",
                "applied": False,
                "error": "No logo/arrow detections.",
            },
            "detections": detections,
        }
        return _rgb_to_png_bytes(img_rgb), _mask_to_png_bytes(mask), report

    output_rgb, inpaint_report = _inpaint_with_opencv(img_rgb, mask)
    report = {
        "detection": detection_report,
        "inpaint": inpaint_report,
        "detections": detections,
    }
    return _rgb_to_png_bytes(output_rgb), _mask_to_png_bytes(mask), report
