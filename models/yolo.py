import io
from PIL import Image
from ultralytics import YOLO

_model = YOLO("yolov8n.pt")


def run_yolo(image_bytes: bytes) -> list[dict]:
    img = Image.open(io.BytesIO(image_bytes))
    results = _model(img)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": _model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 4) for v in box.xywhn[0].tolist()],
        })

    return detections