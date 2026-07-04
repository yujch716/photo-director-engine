import io
import json
import pathlib
from datetime import datetime
from typing import Any

from PIL import Image

from models.landmark_clip import classify_pil
from services.kakao_places import get_landmarks_by_keyword

TEST_DATA_DIR = pathlib.Path("drone-data")


def process_capture(
    image_bytes: bytes,
    target_list: list[dict],
    lat: float | None,
    lng: float | None,
) -> dict[str, Any]:
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    capture_dir = TEST_DATA_DIR / name
    capture_dir.mkdir(parents=True, exist_ok=True)

    (capture_dir / "original_1x.jpg").write_bytes(image_bytes)

    # 정가운데를 2배율로 줌 땡긴 사진 저장
    base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = base_img.size
    zoom_box = (W // 4, H // 4, W - W // 4, H - H // 4)  # 중앙 절반 영역
    zoomed = base_img.crop(zoom_box).resize((W, H), Image.LANCZOS)
    zoom_buf = io.BytesIO()
    zoomed.save(zoom_buf, format="JPEG")
    (capture_dir / "original_2x.jpg").write_bytes(zoom_buf.getvalue())

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
            (capture_dir / f"crop{i}.jpg").write_bytes(buf.getvalue())

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
                "bbox_pixel": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
                "landmark": landmark,
                "confidence": confidence,
                "scores": scores,
            })

    payload = {
        "location": {"lat": lat, "lng": lng},
        "nearby_landmarks": nearby_places,
        "candidate_labels": nearby_names,
        "targets": target_list,
        "results": results,
    }
    (capture_dir / "result.json").write_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2).encode()
    )

    return {
        "saved": name,
        "location": {"lat": lat, "lng": lng},
        "nearby_landmarks": nearby_places,
        "results": results,
    }
