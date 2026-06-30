import io
from PIL import Image
from ultralytics import YOLOWorld

PERSON_CLASSES = {"person", "people"}

_model = YOLOWorld("yolov8x-worldv2.pt")
_model.set_classes([
    "giant hand sculpture rising from the sea",
    "stone hand monument", "statue", "sculpture", "landmark", "lighthouse",
    "person", "people",
    "tower", "building", "bridge", "boat", "car"
])

def run_yolo_world(image_bytes: bytes) -> list[dict]:
    img = Image.open(io.BytesIO(image_bytes))
    results = _model.predict(img, conf=0.1, iou=0.7, agnostic_nms=False)
    detections = []
    for box in results[0].boxes:
        cls_name = _model.names[int(box.cls[0])]
        detections.append({
            "class": cls_name,
            "is_person": cls_name in PERSON_CLASSES,
            "color": "#FF3B30" if cls_name in PERSON_CLASSES else "#34C759",
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 4) for v in box.xywhn[0].tolist()],
        })
    return detections