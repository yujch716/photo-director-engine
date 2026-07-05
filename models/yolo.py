import io
from PIL import Image
from ultralytics import YOLO

PERSON_CLASSES = {"person"}

_model = YOLO("yolov8n.pt")


def run_yolo(image_bytes: bytes) -> list[dict]:
    img = Image.open(io.BytesIO(image_bytes))
    results = _model(img)

    detections = []
    for box in results[0].boxes:
        cls_name = _model.names[int(box.cls[0])]
        if cls_name not in PERSON_CLASSES:
            continue
        detections.append({
            "class": cls_name,
            "is_person": cls_name in PERSON_CLASSES,
            "color": "#FF3B30" if cls_name in PERSON_CLASSES else "#34C759",
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 4) for v in box.xywhn[0].tolist()],
        })

    return detections
