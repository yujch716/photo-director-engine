import io
from PIL import Image
from ultralytics import YOLO

_model = YOLO("yolov8n.pt")


def run_detection(img: Image.Image) -> tuple[bytes, list[dict]]:
    results = _model(img)

    annotated = results[0].plot()
    annotated_img = Image.fromarray(annotated[..., ::-1])
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": _model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 4) for v in box.xywhn[0].tolist()],
        })

    return png_bytes, detections
