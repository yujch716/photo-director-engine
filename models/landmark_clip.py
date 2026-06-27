from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from models.yolo import _model as yolo_model


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

LANDMARK_CANDIDATES: list[dict[str, Any]] = [
    {
        "label": "상생의 손(바다)",
        "prompts": [
            "상생의 손 바다",
            "Hand of Harmony in the sea",
            "Homigot hand sculpture in the ocean",
            "Pohang Homigot sea hand sunrise sculpture",
        ],
    },
    {
        "label": "상생의 손(육지)",
        "prompts": [
            "상생의 손 육지",
            "Hand of Harmony on land",
            "Homigot hand sculpture on land",
            "Pohang Homigot land hand sculpture",
        ],
    },
    {
        "label": "호미곶 등대",
        "prompts": [
            "호미곶 등대",
            "Homigot lighthouse",
            "Pohang Homigot lighthouse",
            "white lighthouse at Homigot",
        ],
    },
    {
        "label": "호미곶 광장",
        "prompts": [
            "호미곶 광장",
            "Homigot square",
            "Homigot plaza",
            "Pohang Homigot public square",
        ],
    },
    {
        "label": "새천년기념관",
        "prompts": [
            "새천년기념관",
            "New Millennium Memorial Hall",
            "Homigot New Millennium Memorial Hall",
            "Pohang Saecheonnyeon Memorial Hall building",
        ],
    },
]

_clip_model: Any | None = None
_clip_processor: Any | None = None
_clip_device: Any | None = None


def _image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read image bytes with PIL.") from exc


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _get_clip() -> tuple[Any, Any, Any, str]:
    global _clip_model, _clip_processor, _clip_device

    model_name = os.environ.get("LANDMARK_CLIP_MODEL", DEFAULT_CLIP_MODEL)

    if _clip_model is not None and _clip_processor is not None and _clip_device is not None:
        return _clip_model, _clip_processor, _clip_device, model_name

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("Landmark CLIP requires transformers and torch. Install with: pip install transformers") from exc

    _clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model = CLIPModel.from_pretrained(model_name).to(_clip_device)
    _clip_model.eval()
    return _clip_model, _clip_processor, _clip_device, model_name


def _run_yolo_crops(img: Image.Image, confidence: float = 0.20) -> tuple[list[dict[str, Any]], Image.Image]:
    results = yolo_model(img, conf=confidence)
    result = results[0]
    annotated_arr = result.plot()
    annotated_img = Image.fromarray(annotated_arr[..., ::-1])

    width, height = img.size
    crops: list[dict[str, Any]] = []
    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        pad_x = max(4, int((x2 - x1) * 0.08))
        pad_y = max(4, int((y2 - y1) * 0.08))
        crop_box = (
            max(0, int(round(x1)) - pad_x),
            max(0, int(round(y1)) - pad_y),
            min(width, int(round(x2)) + pad_x),
            min(height, int(round(y2)) + pad_y),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue

        crops.append({
            "id": f"yolo_{index}",
            "source": "yolo_box",
            "image": img.crop(crop_box),
            "yolo_class": yolo_model.names.get(class_id, str(class_id)),
            "yolo_confidence": conf,
            "bbox_xyxy": [round(v, 2) for v in [x1, y1, x2, y2]],
        })

    crops.insert(0, {
        "id": "full_image",
        "source": "full_image",
        "image": img,
        "yolo_class": None,
        "yolo_confidence": 1.0,
        "bbox_xyxy": [0, 0, width, height],
    })
    return crops, annotated_img


def _flatten_prompts() -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    labels: list[str] = []
    for candidate in LANDMARK_CANDIDATES:
        for prompt in candidate["prompts"]:
            prompts.append(prompt)
            labels.append(candidate["label"])
    return prompts, labels


def _score_crops(crops: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    import torch

    model, processor, device, model_name = _get_clip()
    prompts, prompt_labels = _flatten_prompts()
    images = [crop["image"] for crop in crops]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()

    crop_reports: list[dict[str, Any]] = []
    label_scores = {candidate["label"]: 0.0 for candidate in LANDMARK_CANDIDATES}

    for crop, prompt_probs in zip(crops, probs):
        per_label: dict[str, float] = {label: 0.0 for label in label_scores}
        for prompt_label, prob in zip(prompt_labels, prompt_probs):
            per_label[prompt_label] = max(per_label[prompt_label], float(prob))

        weight = 1.0 if crop["source"] == "full_image" else max(0.2, float(crop["yolo_confidence"]))
        for label, score in per_label.items():
            label_scores[label] = max(label_scores[label], score * weight)

        best_label = max(per_label, key=per_label.get)
        crop_reports.append({
            "id": crop["id"],
            "source": crop["source"],
            "yolo_class": crop["yolo_class"],
            "yolo_confidence": round(float(crop["yolo_confidence"]), 4),
            "bbox_xyxy": crop["bbox_xyxy"],
            "best_label": best_label,
            "best_score": round(per_label[best_label], 6),
            "scores": {label: round(score, 6) for label, score in per_label.items()},
        })

    clip_report = {
        "model": model_name,
        "device": str(device),
        "candidate_labels": [candidate["label"] for candidate in LANDMARK_CANDIDATES],
        "prompt_count": len(prompts),
    }
    return crop_reports, label_scores, clip_report


def classify_landmark(image_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    img = _image_bytes_to_pil(image_bytes)
    crops, annotated_img = _run_yolo_crops(img)
    crop_reports, label_scores, clip_report = _score_crops(crops)

    best_label = max(label_scores, key=label_scores.get)
    report = {
        "best_landmark": best_label,
        "best_score": round(label_scores[best_label], 6),
        "scores": {label: round(score, 6) for label, score in label_scores.items()},
        "clip": clip_report,
        "yolo": {
            "model": "models.yolo._model",
            "crop_count": len(crops),
            "detected_boxes": max(0, len(crops) - 1),
        },
        "crops": crop_reports,
    }

    return _pil_to_png_bytes(annotated_img), report
