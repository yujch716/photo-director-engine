import io
import json
import pathlib
from PIL import Image
from ultralytics import YOLOWorld

_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "yolo_world_classes.json"


def _load_classes() -> list[str]:
    data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return data["default_classes"]


_model = YOLOWorld("yolov8x-worldv2.pt")
_model.set_classes(_load_classes())


def run_yolo_world(image_bytes: bytes) -> list[dict]:
    img = Image.open(io.BytesIO(image_bytes))
    results = _model.predict(img, conf=0.1, iou=0.7, agnostic_nms=False)
    detections = []
    for box in results[0].boxes:
        cls_name = _model.names[int(box.cls[0])]
        detections.append({
            "class": cls_name,
            "is_person": False,
            "color": "#34C759",
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 4) for v in box.xywhn[0].tolist()],
        })
    return detections
