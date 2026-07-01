import io
import json
import pathlib
from datetime import datetime
from typing import Any

from PIL import Image

from models.landmark_clip import classify_pil

TEST_DATA_DIR = pathlib.Path("test-data")


def process_capture(image_bytes: bytes, target_list: list[dict]) -> dict[str, Any]:
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    TEST_DATA_DIR.mkdir(exist_ok=True)

    (TEST_DATA_DIR / f"{name}.jpg").write_bytes(image_bytes)

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

            clip_result = None
            try:
                clip_result = classify_pil(crop)
            except Exception:
                pass

            results.append({"class": t["class"], "clip": clip_result, "bbox": t["bbox"]})

    payload = {"targets": target_list, "results": results}
    (TEST_DATA_DIR / f"{name}.json").write_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2).encode()
    )

    return {"saved": name, "results": results}
