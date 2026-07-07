"""정점 도착 후 촬영 결과 저장.

스캔으로 정점에 도착한 뒤 촬영한 사진(arrived.jpg)과 그 중앙 2배율 크롭(arrived_2x.jpg),
선택적으로 최종구도(target.jpg)와 메타데이터(meta.json)를
drone-data/result/<타임스탬프>/ 아래에 저장한다.

지금은 저장만 한다. 나중에 NIMA 등 품질 점수를 여기(save_scan_result)에서 계산해
meta에 추가/반환하도록 확장하기 쉽게 함수로 분리해 둔다.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

# 스캔 결과 저장 루트. 기존 /capture 폴더와 섞이지 않게 result/ 아래로 분리.
DRONE_DATA_DIR = Path("drone-data/result")


def save_scan_result(
    image_bytes: bytes,
    target_bytes: bytes | None = None,
    meta_json: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """정점 도착 사진(+선택 target/meta)을 새 캡처 폴더에 저장한다.

    저장 파일:
        arrived.jpg     : 정점 도착 후 촬영 원본
        arrived_2x.jpg  : arrived의 중앙 절반을 2배율로 확대한 이미지(capture의 original_2x와 동일 방식)
        target.jpg      : (선택) 최종구도 이미지
        meta.json       : (선택) 메타데이터

    Args:
        image_bytes: 정점 도착 후 촬영 사진(JPEG) — arrived.jpg / arrived_2x.jpg 로 저장.
        target_bytes: (선택) 최종구도 이미지(JPEG) — target.jpg 로 저장(비교 검증용).
        meta_json: (선택) JSON 문자열(theta, peak_index, 되돌아간 거리 등) — meta.json 으로 저장.
        data_dir: 저장 루트(기본 drone-data/result).

    Returns:
        {"saved": "<폴더명>", "message": "저장 완료"}
    """
    base = data_dir or DRONE_DATA_DIR
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder = base / name
    # 같은 밀리초에 두 번 호출되어도 덮어쓰지 않도록 유일한 폴더명 보장.
    suffix = 1
    while folder.exists():
        folder = base / f"{name}_{suffix}"
        suffix += 1
    name = folder.name
    folder.mkdir(parents=True)

    (folder / "arrived.jpg").write_bytes(image_bytes)

    # arrived의 중앙 절반을 2배율로 확대(capture.py의 original_2x와 동일 방식)해 저장.
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = base_img.size
        zoom_box = (W // 4, H // 4, W - W // 4, H - H // 4)  # 중앙 절반 영역
        zoomed = base_img.crop(zoom_box).resize((W, H), Image.LANCZOS)
        buf = io.BytesIO()
        zoomed.save(buf, format="JPEG")
        (folder / "arrived_2x.jpg").write_bytes(buf.getvalue())
    except Exception as e:
        print(f"[WARN] arrived_2x.jpg 생성 실패(무시): {e}")

    if target_bytes is not None:
        (folder / "target.jpg").write_bytes(target_bytes)

    if meta_json is not None:
        try:
            parsed = json.loads(meta_json)
        except Exception:
            parsed = {"raw": meta_json}  # JSON이 아니면 원문 그대로 보존
        (folder / "meta.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # TODO(확장): 여기서 NIMA 등 품질 점수를 계산해 meta.json에 추가하고 반환값에 실을 자리.

    return {"saved": name, "message": "저장 완료"}
