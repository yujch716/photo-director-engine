"""정점 도착 후 촬영 결과 저장.

스캔으로 정점에 도착한 뒤 촬영한 사진(arrived.jpg)과 그 중앙 1.4배율 크롭(arrived_1_5x.jpg),
선택적으로 최종구도(target.jpg)와 메타데이터(meta.json)를
drone-data/result/<타임스탬프>/ 아래에 저장한다.

지금은 저장만 한다. 나중에 NIMA 등 품질 점수를 여기(save_scan_result)에서 계산해
meta에 추가/반환하도록 확장하기 쉽게 함수로 분리해 둔다.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir


def save_scan_result(
    image_bytes: bytes,
    meta_json: str | None = None,
    session_id: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """정점 도착 사진(+선택 meta)을 저장한다. (target은 받지 않음 — 2_scan에 이미 best.jpg 존재)

    저장 파일(세션이면 2_scan 폴더 공유):
        arrived_1x.jpg    : 정점 도착 후 촬영 원본
        arrived_1_5x.jpg  : arrived의 중앙 5/7을 1.4배율로 확대한 이미지(capture의 original_1_5x와 동일 방식)
        meta.json         : (선택) 메타데이터

    Args:
        image_bytes: 정점 도착 후 촬영 사진(JPEG) — arrived_1x.jpg / arrived_1_5x.jpg 로 저장.
        meta_json: (선택) JSON 문자열(theta, peak_index, 되돌아간 거리 등) — meta.json 으로 저장.
        session_id: (선택) 있으면 drone-data/<session_id>/2_scan/ 에 저장(scan-peak와 공유).
        data_dir: 저장 루트(기본 drone-data).

    Returns:
        {"saved": "<drone-data 이하 폴더 경로>", "message": "저장 완료"}
    """
    base = data_dir or DRONE_DATA_DIR
    # 세션 없으면 기존 경로(drone-data/result/<ts>), 있으면 <session>/2_scan/ (scan-peak와 공유).
    folder = resolve_save_dir(session_id, "2_scan", legacy=base / "result" / make_ts(), data_dir=base)
    name = str(folder.relative_to(base))

    (folder / "arrived_1x.jpg").write_bytes(image_bytes)

    # arrived의 중앙 5/7을 1.4배율로 확대(capture.py의 original_2x와 동일 방식)해 저장.
    arrived_2x_bytes = None
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = base_img.size
        zoom_box = (W // 7, H // 7, W - W // 7, H - H // 7)  # 중앙 5/7 영역(1.4배)
        zoomed = base_img.crop(zoom_box).resize((W, H), Image.LANCZOS)
        buf = io.BytesIO()
        zoomed.save(buf, format="JPEG")
        arrived_2x_bytes = buf.getvalue()
        (folder / "arrived_1_5x.jpg").write_bytes(arrived_2x_bytes)
    except Exception as e:
        print(f"[WARN] arrived_1_5x.jpg 생성 실패(무시): {e}")


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
