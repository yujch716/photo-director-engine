"""Qwen2.5-VL 로딩 + 멀티이미지 호출.

여러 장의 이미지 + 텍스트 지시를 넣고 텍스트 응답을 받는다(사진 심사 등).
모델은 lazy singleton으로 로드(yolo/nima/dinov2와 동일 패턴).

환경변수:
  QWEN_VL_MODEL       : HF 모델 id (기본 Qwen/Qwen2.5-VL-3B-Instruct; GPU서버는 7B로 지정)
  VLM_MAX_NEW_TOKENS  : 생성 최대 토큰 (기본 512)
  VLM_IMAGE_MAX_SIDE  : 입력 이미지 긴 변 최대 픽셀 (기본 512; 속도/메모리용 축소)

가중치가 없거나 deps가 없으면 안내 메시지를 담아 RuntimeError를 던진다.
(서비스단은 이를 잡아 NIMA 최고점 fallback으로 degrade)
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

DEFAULT_MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "512"))
IMAGE_MAX_SIDE = int(os.environ.get("VLM_IMAGE_MAX_SIDE", "512"))

_model: Any | None = None
_processor: Any | None = None
_device: Any | None = None


def _get_model() -> tuple[Any, Any, Any]:
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:  # deps 미설치
        raise RuntimeError(
            "Qwen2.5-VL 로드 실패: transformers/torch 필요. "
            "pip install 'transformers>=4.49' accelerate qwen-vl-utils"
        ) from exc

    if torch.cuda.is_available():
        _device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    print(f"[qwen-vl] loading {DEFAULT_MODEL} on {_device} ...")
    try:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            DEFAULT_MODEL,
            dtype="auto",
            device_map=_device,
        )
        _processor = AutoProcessor.from_pretrained(DEFAULT_MODEL)
    except Exception as exc:  # 가중치 미다운로드 등
        raise RuntimeError(
            f"Qwen2.5-VL 모델 로드 실패({DEFAULT_MODEL}): {exc}. "
            "최초 1회 가중치 다운로드 필요(HF). 다른 크기는 QWEN_VL_MODEL 환경변수로 지정."
        ) from exc

    _model.eval()
    return _model, _processor, _device


def _resize_for_vlm(img: Image.Image, max_side: int = IMAGE_MAX_SIDE) -> Image.Image:
    """긴 변이 max_side를 넘으면 비율 유지 축소(속도/메모리)."""
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    return img


def run_vlm(images: list[Image.Image], prompt: str) -> str:
    """이미지 여러 장 + 지시(prompt)를 넣고 모델의 텍스트 응답을 반환한다."""
    import torch

    model, processor, device = _get_model()

    imgs = [_resize_for_vlm(im) for im in images]
    content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in imgs]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception:
        image_inputs, video_inputs = imgs, None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out_text
