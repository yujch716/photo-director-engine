"""세션 단계별 NIMA 점수 누적 로그.

각 API가 그 단계의 "선택된/최종 사진"에 NIMA 점수를 매겨
drone-data/<session_id>/nima-score.json 에 순서대로 append한다.
단계가 진행될수록 점수가 오르는지(구도 개선) 확인용.

이미 NIMA를 계산한 단계는 그 점수를 score= 로 넘겨 재계산을 피한다.
세션이 없으면(무세션 호출) 아무것도 하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.nima import run_nima_score
from services.session_paths import session_root


def append_session_nima(
    session_id: str | None,
    stage: str,
    image_bytes: bytes | None = None,
    score: float | None = None,
    data_dir: Path | None = None,
) -> float | None:
    """<session_id>/nima-score.json 에 {stage, nima} 를 append한다.

    Args:
        stage: 단계 라벨(예: "1_original", "2_scan/arrived", "3_nima-move/lateral/1").
        image_bytes: 점수를 매길 이미지(선택 사진). score가 없을 때만 사용.
        score: 이미 계산된 NIMA 점수가 있으면 그대로 사용(재계산 안 함).
    Returns:
        기록한 점수(무세션이거나 실패 시 None).
    """
    if not (session_id and session_id.strip()):
        return None

    if score is None:
        if not image_bytes:
            return None
        try:
            score = float(run_nima_score(image_bytes)["score"])
        except Exception as e:
            print(f"[nima-score] WARN NIMA 계산 실패({stage}): {e}")
            return None

    try:
        root = session_root(session_id, data_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "nima-score.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
        data.append({"stage": stage, "nima": round(float(score), 4)})
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[nima-score] {stage}: {float(score):.4f} → {session_id}/nima-score.json")
    except Exception as e:
        print(f"[nima-score] WARN 저장 실패({stage}): {e}")
        return None

    return float(score)
