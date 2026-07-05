"""CLIP zero-shot 태깅.

config/schema.yaml(태그 단일 소스)을 기준으로 두 가지를 수행한다.
- identify_object(image_bytes, bbox): YOLO bbox 안의 객체가 어떤 랜드마크인지 식별
- tag_image(image_bytes): 전체 이미지를 보고 schema.yaml 형식대로 태깅

백엔드는 transformers CLIP(openai/clip-vit-base-patch32).
"""

from __future__ import annotations

import io
import os
import pathlib
from functools import lru_cache
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "schema.yaml"

# 임계값 (raw 코사인 유사도 기준, 튜닝용). ViT-B/32에서 매칭 개념은 대략
# 0.25~0.33, 비매칭은 0.15~0.22 정도로 나온다. 실데이터로 보정 권장.
LANDMARK_THRESHOLD = 0.24   # bbox 객체 식별 시 랜드마크 인정 최소 코사인
TAG_LANDMARK_THRESHOLD = 0.24   # 전체 이미지 태깅 시 랜드마크 인정 최소 코사인
FACING_THRESHOLD = 0.22     # facing 인정 최소 코사인
NEG_MARGIN = 0.01           # nullable 판정 시 negative보다 이만큼은 높아야 인정
# 플래그는 present/absent 최소쌍(minimal pair)의 코사인 차이로 판정한다.
# 프롬프트별 baseline이 상쇄되어 플래그 간 비교가 가능해진다. (신호가 약하므로 실데이터 보정 권장)
FLAG_MARGIN = 0.003         # cos(present) - cos(absent) 가 이 값 이상이면 present

# "랜드마크 없음"을 흡수하기 위한 negative 프롬프트
_NEGATIVE_LANDMARK_PROMPTS = [
    "a random scene with no notable landmark",
    "ordinary background, sky, sea or ground with no landmark",
    "그냥 배경, 특별한 랜드마크 없음",
]

# 플래그 minimal-pair 판정용 간결 명사구 (present/absent 프롬프트에 삽입)
_FLAG_NOUNS: dict[str, str] = {
    "sea": "the sea or ocean",
    "ground": "bare ground or soil",
    "mountain": "a mountain",
    "forest": "trees or a forest",
    "river": "a river or stream",
    "cloud": "clouds in the sky",
    "sun": "the sun",
    "bystander": "a crowd of bystanders",
    "vehicle": "a vehicle such as a boat, car or airplane",
    "structure": "a building, bridge or man-made structure",
}
_FLAG_PRESENT = "a photo that clearly shows {noun}"
_FLAG_ABSENT = "a photo that does not show {noun}"


# ---------------------------------------------------------------------------
# 스키마 로딩 + 프롬프트 맵
# ---------------------------------------------------------------------------

# 스키마 값(영문 id) -> CLIP 텍스트 프롬프트 후보들.
# 여러 표현을 주고 그 중 최대 유사도를 그 id의 점수로 사용한다.
PROMPTS: dict[str, list[str]] = {
    # fields.time
    "day": ["a photo taken during the day", "bright daytime", "낮에 찍은 사진"],
    "evening": [
        "a photo at sunset or golden hour",
        "evening sky with sunset glow",
        "노을 지는 저녁, 골든아워",
    ],
    "night": ["a photo taken at night", "dark night scene", "밤에 찍은 사진"],
    # fields.weather
    "clear": ["clear sunny sky", "clear weather with blue sky", "맑은 날씨"],
    "cloudy": ["cloudy overcast sky", "gray cloudy weather", "흐린 날씨, 구름 낀 하늘"],
    "rain": ["rainy weather", "raining outside", "비 오는 날씨"],
    "snow": ["snowy weather", "snow falling, snowy scene", "눈 오는 날씨"],
    # fields.location
    "indoor": ["an indoor scene, inside a building", "실내"],
    "outdoor": ["an outdoor scene, outside", "실외, 야외"],
    # enums.facing
    "front": ["people facing the camera from the front", "정면을 보고 있는 사람들"],
    "side": ["people seen from the side, profile view", "측면으로 서 있는 사람들"],
    "back": ["people seen from behind, showing their backs", "등을 보이는 사람들"],
    "mixed": ["people facing different directions", "여러 방향을 향한 사람들"],
    # enums.landmark
    "hands_of_harmony": [
        "the Hand of Harmony sculpture in Pohang",
        "상생의 손 조각상",
        "a giant hand sculpture rising from the sea",
    ],
    "cheomseongdae": ["Cheomseongdae observatory in Gyeongju", "첨성대"],
    "dabotap": ["Dabotap stone pagoda", "다보탑"],
    "seokgatap": ["Seokgatap stone pagoda", "석가탑"],
    "daereungwon": ["Daereungwon ancient royal tombs in Gyeongju", "대릉원 고분"],
    "yeongildae": ["Yeongildae observatory tower in Pohang", "영일대 전망대"],
    "yeonorang_seonyeo": ["the Yeonorang Seonyeo statue", "연오랑 세오녀 상"],
    # flags (present 쪽 프롬프트)
    "sea": ["the sea, ocean water", "바다"],
    "ground": ["the ground, bare land or soil", "땅, 지면"],
    "mountain": ["a mountain", "산"],
    "forest": ["a forest or trees", "숲, 나무"],
    "river": ["a river or stream", "강, 하천"],
    "cloud": ["clouds in the sky", "구름"],
    "sun": ["the sun visible in the sky", "태양"],
    "bystander": ["bystanders, a crowd of unrelated people", "관중, 지나가는 사람들"],
    "vehicle": ["a vehicle such as a boat, car or airplane", "탈것(배/차/비행기)"],
    "structure": [
        "a man-made structure such as a building, bridge or tower",
        "인공 구조물(건물/다리/타워)",
    ],
}

@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prompts_for(value_id: str) -> list[str]:
    return PROMPTS.get(value_id, [value_id])


# ---------------------------------------------------------------------------
# CLIP 로더 (lazy singleton)
# ---------------------------------------------------------------------------

_clip_model: Any | None = None
_clip_processor: Any | None = None
_clip_device: Any | None = None


def _get_clip() -> tuple[Any, Any, Any]:
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_processor, _clip_device

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "CLIP requires transformers and torch. pip install transformers torch"
        ) from exc

    model_name = os.environ.get("LANDMARK_CLIP_MODEL", DEFAULT_CLIP_MODEL)
    _clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model = CLIPModel.from_pretrained(model_name).to(_clip_device)
    _clip_model.eval()
    return _clip_model, _clip_processor, _clip_device


# ---------------------------------------------------------------------------
# 코어: raw 코사인 유사도
# ---------------------------------------------------------------------------
# CLIP의 logits_per_image는 logit_scale(~100)이 곱해져 있어 softmax가 사실상
# hard argmax가 된다(임계값 판정 불가). 그래서 멀티라벨/nullable 판정에는
# image/text 임베딩의 raw 코사인 유사도(대략 0.15~0.35 범위)를 쓰고 임계로 자른다.

def _cosine_similarities(img: Image.Image, prompts: list[str]) -> np.ndarray:
    """img와 각 프롬프트의 raw 코사인 유사도(보통 0.15~0.35)를 반환."""
    import torch

    model, processor, device = _get_clip()
    inputs = processor(text=prompts, images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        img_feat = outputs.image_embeds  # (1, dim), 투영됨(정규화 전)
        txt_feat = outputs.text_embeds   # (num_prompts, dim)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        cos = (img_feat @ txt_feat.T)[0]  # (num_prompts,)
    return cos.detach().cpu().numpy()


def _score_group(
    img: Image.Image,
    value_ids: list[str],
    extra_prompts: list[str] | None = None,
) -> tuple[dict[str, float], float]:
    """value_ids(+extra_prompts)의 프롬프트를 한 번에 넣고 코사인 유사도 계산.

    각 value_id 점수 = 그 id에 속한 프롬프트들의 코사인 최댓값.
    반환: (id별 코사인 dict, extra 프롬프트들의 코사인 최댓값)
    """
    flat_prompts: list[str] = []
    owners: list[str] = []
    for vid in value_ids:
        for p in _prompts_for(vid):
            flat_prompts.append(p)
            owners.append(vid)
    extra_prompts = extra_prompts or []
    for p in extra_prompts:
        flat_prompts.append(p)
        owners.append("__extra__")

    cos = _cosine_similarities(img, flat_prompts)

    scores = {vid: -1.0 for vid in value_ids}
    extra_score = -1.0
    for owner, c in zip(owners, cos):
        c = float(c)
        if owner == "__extra__":
            extra_score = max(extra_score, c)
        else:
            scores[owner] = max(scores[owner], c)
    return scores, extra_score


# ---------------------------------------------------------------------------
# 이미지 유틸
# ---------------------------------------------------------------------------

def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read image bytes with PIL.") from exc


def _crop_bbox(img: Image.Image, bbox: list[float]) -> Image.Image:
    """정규화 xywhn [cx, cy, w, h] -> 픽셀 crop."""
    cx, cy, w, h = bbox
    W, H = img.size
    left = max(0, int((cx - w / 2) * W))
    top = max(0, int((cy - h / 2) * H))
    right = min(W, int((cx + w / 2) * W))
    bottom = min(H, int((cy + h / 2) * H))
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_object(image_bytes: bytes, bbox: list[float]) -> dict[str, Any]:
    """이미지 바이트 + YOLO bbox(정규화 xywhn) -> 박스 내 객체의 랜드마크 식별.

    후보는 schema.yaml enums.landmark.values. landmark는 nullable이므로
    negative 프롬프트를 함께 두고, top1이 negative거나 확률이 임계 미만이면 None.
    """
    schema = _load_schema()
    landmark_ids = schema["enums"]["landmark"]["values"]

    img = _bytes_to_pil(image_bytes)
    crop = _crop_bbox(img, bbox)

    scores, negative_score = _score_group(
        crop, landmark_ids, extra_prompts=_NEGATIVE_LANDMARK_PROMPTS
    )

    best_id = max(scores, key=scores.__getitem__)
    best_score = scores[best_id]
    landmark = (
        best_id
        if (best_score >= LANDMARK_THRESHOLD and best_score >= negative_score + NEG_MARGIN)
        else None
    )

    return {
        "landmark": landmark,
        "best_score": round(best_score, 6),
        "scores": {k: round(v, 6) for k, v in scores.items()},
    }


def tag_image(image_bytes: bytes) -> dict[str, Any]:
    """전체 이미지를 보고 schema.yaml 형식으로 태깅.

    person_count는 CLIP으로 세지 않으므로 None. 나머지 전부 채운다.
    """
    schema = _load_schema()
    img = _bytes_to_pil(image_bytes)

    result: dict[str, Any] = {}

    # fields: required 단일 선택 (argmax)
    for field, spec in schema.get("fields", {}).items():
        scores, _ = _score_group(img, spec["values"])
        result[field] = max(scores, key=scores.__getitem__)

    # counts: CLIP 미지원
    for count in schema.get("counts", {}):
        result[count] = None

    # flags: 멀티라벨. present/absent 최소쌍의 코사인 차이가 FLAG_MARGIN 이상이면 붙인다.
    # (프롬프트 baseline이 상쇄되어 플래그 간 비교 가능. 신호는 약해 실데이터 보정 권장.)
    flag_ids = list(schema.get("flags", {}).keys())
    if flag_ids:
        prompts: list[str] = []
        for f in flag_ids:
            noun = _FLAG_NOUNS.get(f, f)
            prompts.append(_FLAG_PRESENT.format(noun=noun))
            prompts.append(_FLAG_ABSENT.format(noun=noun))
        cos = _cosine_similarities(img, prompts)  # 한 번에 계산
        result["flags"] = [
            f for i, f in enumerate(flag_ids)
            if float(cos[2 * i] - cos[2 * i + 1]) >= FLAG_MARGIN
        ]
    else:
        result["flags"] = []

    # enums: nullable 단일 선택 (argmax + 코사인 임계)
    for enum_name, spec in schema.get("enums", {}).items():
        value_ids = spec["values"]
        if enum_name == "landmark":
            scores, negative_score = _score_group(
                img, value_ids, extra_prompts=_NEGATIVE_LANDMARK_PROMPTS
            )
            threshold = TAG_LANDMARK_THRESHOLD
        else:
            scores, negative_score = _score_group(img, value_ids)
            threshold = FACING_THRESHOLD if enum_name == "facing" else 0.0

        best_id = max(scores, key=scores.__getitem__)
        best_score = scores[best_id]
        nullable = spec.get("nullable", False)
        if nullable and (best_score < threshold or best_score < negative_score + NEG_MARGIN):
            result[enum_name] = None
        else:
            result[enum_name] = best_id

    return result
