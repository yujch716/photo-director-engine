import io
import json
import pathlib
from datetime import datetime
from typing import Any

from PIL import Image

from models.landmark_clip import classify_pil
from services.kakao_places import get_landmarks_by_keyword

TEST_DATA_DIR = pathlib.Path("drone-info")


def process_capture(
    image_bytes: bytes,
    target_list: list[dict],
    lat: float | None,
    lng: float | None,
) -> dict[str, Any]:
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    TEST_DATA_DIR.mkdir(exist_ok=True)

    (TEST_DATA_DIR / f"{name}.jpg").write_bytes(image_bytes)

    nearby_places = get_landmarks_by_keyword(lat, lng) if lat is not None and lng is not None else []
    nearby_names = [p["name"] for p in nearby_places]

    results = []
    if target_list:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size
        for i, t in enumerate(target_list):
            cx, cy, w, h = t["bbox"]
            box = (
                max(0, int((cx - w / 2) * W)),
                max(0, int((cy - h / 2) * H)),
                min(W, int((cx + w / 2) * W)),
                min(H, int((cy + h / 2) * H)),
            )
            crop = img.crop(box)

            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            (TEST_DATA_DIR / f"{name}_crop{i}.jpg").write_bytes(buf.getvalue())

            landmark = None
            confidence = None
            scores = None
            try:
                clip_result = classify_pil(crop, nearby_names)
                landmark = clip_result["landmark"]
                confidence = clip_result["best_score"]
                scores = clip_result["scores"]
            except Exception:
                pass

            results.append({
                "class": t["class"],
                "bbox": t["bbox"],
                "landmark": landmark,
                "confidence": confidence,
                "scores": scores,
            })

    payload = {
        "location": {"lat": lat, "lng": lng},
        "nearby_landmarks": nearby_places,
        "targets": target_list,
        "results": results,
    }
    (TEST_DATA_DIR / f"{name}.json").write_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2).encode()
    )

    return {
        "saved": name,
        "location": {"lat": lat, "lng": lng},
        "nearby_landmarks": nearby_places,
        "results": results,
    }
