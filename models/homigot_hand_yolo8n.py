import io
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# 학습된 상생의손 전용 YOLO 모델 파일명
# 이 py 파일과 같은 폴더에 homigot_hand_yolo8n.pt를 두면 됨.
# 경로를 따로 지정하고 싶으면 환경변수 HOMIGOT_YOLO_MODEL_PATH 사용.

_DEFAULT_MODEL_PATH = Path(__file__).parent / "homigot_hand_yolo8n.pt"
MODEL_PATH = os.getenv("HOMIGOT_YOLO_MODEL_PATH", str(_DEFAULT_MODEL_PATH))

HOMIGOT_CLASSES = {"homigot_hand"}

_model = YOLO(MODEL_PATH)


def run_homigot_hand_yolo(image_bytes: bytes) -> list[dict]:
    """
    상생의손 전용 YOLO 추론 함수.

    bbox 형식은 기존 코드와 동일:
    [x_center, y_center, width, height]
    전부 0~1 normalized 좌표.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = _model.predict(
        source=img,
        conf=0.25,
        iou=0.45,
        max_det=1,
        verbose=False,
    )

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = _model.names.get(cls_id, "homigot_hand")
        conf = float(box.conf[0])

        detections.append({
            "class": cls_name,
            "is_homigot_hand": cls_name in HOMIGOT_CLASSES,
            "is_person": False,
            "color": "#FF9500",
            "confidence": round(conf, 4),
            "bbox": [round(v, 4) for v in box.xywhn[0].tolist()],
        })

    return detections