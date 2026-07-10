"""최종 촬영 이미지 저장 (초점 조정 후 찍은 것).

drone-data/<session_id>/4_final/ 아래에 final.jpg (+선택 meta.json)를 저장한다.
지금은 저장만 한다(분석 없음).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.nima_log import append_session_nima
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir


def save_final_shot(
    image_bytes: bytes,
    session_id: str | None = None,
    meta_json: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """최종 촬영 이미지를 세션 폴더에 저장한다.
    세션 있으면 drone-data/<session_id>/4_final/, 없으면 drone-data/final_shot_<ts>/.
    Returns:
        {"saved": <session_id 또는 폴더경로>, "path": "/drone-data/.../final.jpg", "message": ...}
    """
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_save_dir(session_id, "4_final", legacy=base / f"final_shot_{make_ts()}", data_dir=base)
    rel = str(folder.relative_to(base))

    (folder / "final.jpg").write_bytes(image_bytes)

    if meta_json is not None:
        try:
            parsed = json.loads(meta_json)
        except Exception:
            parsed = {"raw": meta_json}
        (folder / "meta.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 단계별 NIMA 누적 로그(최종 촬영 사진).
    append_session_nima(session_id, "4_final", image_bytes=image_bytes, data_dir=base)

    saved = session_id if (session_id and session_id.strip()) else rel
    return {
        "saved": saved,
        "path": f"/drone-data/{rel}/final.jpg",
        "message": "최종 저장 완료",
    }
