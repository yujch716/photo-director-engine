"""
homigot_hand_yolo8n_hyun_compatible.py

상생의손 전용 YOLO 추론 래퍼.

기존 코드 호환 포인트:
- 기존 함수명 run_homigot_hand_yolo(image_bytes) 유지
- 기존 출력 키 bbox 유지
  bbox = [x_center, y_center, width, height]  # 0~1 normalized
- 기존 출력 키 class, is_homigot_hand, is_person, color, confidence 유지

추가 개선 포인트:
- run_yolo(), run_yolo_batch() 지원
- bytes / bytearray / PIL.Image / 이미지 경로 입력 지원
- bbox_xywhn, bbox_xyxyn 상세 좌표도 함께 반환
- 환경변수 HOMIGOT_YOLO_MODEL_PATH로 모델 경로 지정 가능
"""

import os
from pathlib import Path
from io import BytesIO
from typing import Any, Iterable

from PIL import Image
from ultralytics import YOLO
import torch


# 학습된 상생의손 전용 YOLO 모델 파일명
# 기본적으로 이 py 파일과 같은 폴더의 homigot_hand_yolo8n.pt를 사용한다.
# 다른 경로를 쓰고 싶으면 환경변수 HOMIGOT_YOLO_MODEL_PATH를 지정하면 된다.
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _THIS_DIR / "homigot_hand_yolo8n.pt"
MODEL_PATH = Path(os.getenv("HOMIGOT_YOLO_MODEL_PATH", str(_DEFAULT_MODEL_PATH))).expanduser()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

HOMIGOT_CLASSES = {"homigot_hand"}

_DEVICE = 0 if torch.cuda.is_available() else "cpu"
_model = YOLO(str(MODEL_PATH))


def _to_pil_image(x: Any) -> Image.Image:
    """bytes, PIL.Image, path 형태의 입력을 RGB PIL 이미지로 변환한다."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, (bytes, bytearray)):
        return Image.open(BytesIO(x)).convert("RGB")

    # str, pathlib.Path, file-like object 등을 PIL이 처리하도록 넘긴다.
    return Image.open(x).convert("RGB")


def _get_class_name(names: Any, cls_id: int) -> str:
    """Ultralytics names가 dict/list 어느 형태여도 class name을 안전하게 가져온다."""
    if isinstance(names, dict):
        return str(names.get(cls_id, "homigot_hand"))

    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])

    return "homigot_hand"


def _result_to_detections(result) -> list[dict]:
    """
    Ultralytics Result 1개를 기존 코드와 호환되는 detection list로 변환한다.

    기존 호환 출력:
    - bbox: [x_center, y_center, width, height], normalized, round(4)
    - confidence: round(4)

    추가 출력:
    - class_id
    - bbox_xywhn: dict
    - bbox_xyxyn: dict
    """
    detections = []

    if result.boxes is None:
        return detections

    names = getattr(result, "names", getattr(_model, "names", {}))

    for box in result.boxes:
        cls_id = int(box.cls[0].detach().cpu().item())
        cls_name = _get_class_name(names, cls_id)
        conf = float(box.conf[0].detach().cpu().item())

        xywhn = [float(v) for v in box.xywhn[0].detach().cpu().tolist()]
        xyxyn = [float(v) for v in box.xyxyn[0].detach().cpu().tolist()]

        detections.append({
            # 기존 코드 호환 필드
            "class": cls_name,
            "is_homigot_hand": cls_name in HOMIGOT_CLASSES or cls_name == "homigot_hand",
            "is_person": False,
            "color": "#FF9500",
            "confidence": round(conf, 4),
            "bbox": [round(v, 4) for v in xywhn],

            # 추가 필드: 새 코드에서 더 자세히 쓰고 싶을 때 사용
            "class_id": cls_id,
            "bbox_xywhn": {
                "x": xywhn[0],
                "y": xywhn[1],
                "w": xywhn[2],
                "h": xywhn[3],
            },
            "bbox_xyxyn": {
                "x1": xyxyn[0],
                "y1": xyxyn[1],
                "x2": xyxyn[2],
                "y2": xyxyn[3],
            },
        })

    return detections


def run_yolo_batch(
    image_items: Iterable[Any],
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 1,
) -> list[list[dict]]:
    """
    여러 이미지를 한 번에 추론한다.

    Args:
        image_items: list/tuple of bytes | bytearray | PIL.Image | image path
        conf: confidence threshold
        iou: IoU threshold
        max_det: 이미지당 최대 detection 개수

    Returns:
        list[list[dict]]: 입력 이미지 개수만큼 detection list 반환
    """
    if image_items is None:
        return []

    # bytes 하나가 들어오면 list(bytes)가 되어버리는 문제를 막는다.
    if isinstance(image_items, (bytes, bytearray, Image.Image, str, Path)):
        image_items = [image_items]
    else:
        image_items = list(image_items)

    if len(image_items) == 0:
        return []

    images = [_to_pil_image(x) for x in image_items]

    results = _model.predict(
        source=images,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=_DEVICE,
        verbose=False,
    )

    return [_result_to_detections(r) for r in results]


def run_yolo(
    image_item: Any,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 1,
) -> list[dict]:
    """
    단일 이미지 추론 함수.

    Args:
        image_item: bytes | bytearray | PIL.Image | image path

    Returns:
        list[dict]: detection list
    """
    out = run_yolo_batch(
        [image_item],
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    return out[0] if out else []


def run_homigot_hand_yolo(
    image_bytes: bytes,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 1,
) -> list[dict]:
    """
    기존 코드 호환용 함수.

    기존 코드가 run_homigot_hand_yolo(image_bytes)를 호출하고,
    결과에서 det["bbox"]를 읽는 구조라면 그대로 사용할 수 있다.

    bbox 형식:
    [x_center, y_center, width, height]
    전부 0~1 normalized 좌표.
    """
    return run_yolo(
        image_bytes,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )


def run_homigot_hand_yolo_batch(
    image_items: Iterable[Any],
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 1,
) -> list[list[dict]]:
    """기존 이름 스타일에 맞춘 배치 추론 alias."""
    return run_yolo_batch(
        image_items,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
