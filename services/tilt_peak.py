"""틸트 sweep 최고 프레임 선택 (NIMA).

짐벌 틸트를 위→아래로 훑으며 찍은 프레임들을 받아 각 장에 NIMA 미학점수를 매기고,
최고 점수 프레임의 index와 실제 각도를 반환한다. (/scan-peak과 같은 "묶음→최고" 구조,
비교 기준이 SSIM이 아니라 무참조 NIMA 점수)

기존 models.nima.run_nima_score 재사용. 결과/프레임은 drone-data/<session_id>/tilt/에 저장.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from models.nima import run_nima_score
from services.nima_log import append_session_nima
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir

SAVE_DEBUG = True


def _save_tilt(
    frame_bytes_list: list[bytes],
    angles: list[float],
    result: dict[str, Any],
    session_id: str | None = None,
    data_dir: Path | None = None,
) -> str:
    """프레임들(각도 파일명) + best + scores.json을 drone-data/<session_id>/3_nima-move/tilt/에 저장."""
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_save_dir(session_id, "3_nima-move/tilt", legacy=base / "tilt" / make_ts(), data_dir=base)

    for i, (fb, ang) in enumerate(zip(frame_bytes_list, angles)):
        (folder / f"{i:02d}_{ang:+.1f}deg.jpg").write_bytes(fb)
    # 최고 점수 프레임
    peak = result["peak_index"]
    (folder / "best.jpg").write_bytes(frame_bytes_list[peak])
    (folder / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 단계별 NIMA 누적 로그(선택 프레임 = peak, 이미 계산된 peak_score 재사용).
    append_session_nima(session_id, "3_nima-move/tilt", score=result["peak_score"], data_dir=base)
    return str(folder.relative_to(base))


def find_tilt_peak(
    frame_bytes_list: list[bytes],
    angles: list[float],
    session_id: str | None = None,
    save: bool | None = None,
) -> dict[str, Any]:
    """프레임들에 NIMA를 매겨 최고 점수 프레임(각도)을 반환한다.

    Returns:
        {peak_index, peak_angle, peak_score, scores, angles, frame_count, timing_ms}
    Raises:
        ValueError: frames가 비었거나 angles 개수가 frames와 다를 때.
    """
    t_start = time.perf_counter()
    n = len(frame_bytes_list)
    if n == 0:
        raise ValueError("frames가 비어 있음")
    if angles is None or len(angles) != n:
        raise ValueError(f"angles 개수({0 if angles is None else len(angles)})가 frames({n})와 달라야 함")

    scores: list[float] = []
    t_n0 = time.perf_counter()
    for i, fb in enumerate(frame_bytes_list):
        ti = time.perf_counter()
        s = float(run_nima_score(fb)["score"])
        scores.append(round(s, 4))
        print(f"[tilt-peak] frame {i:>2} (angle={angles[i]:+.1f}): nima={s:.4f} ({(time.perf_counter() - ti) * 1000:.1f} ms)")
    nima_total_ms = (time.perf_counter() - t_n0) * 1000.0

    peak_index = int(max(range(n), key=lambda i: scores[i]))
    peak_angle = angles[peak_index]
    peak_score = scores[peak_index]
    total_ms = (time.perf_counter() - t_start) * 1000.0

    print(
        f"[tilt-peak] DONE n={n} → peak_index={peak_index}, peak_angle={peak_angle:+.1f}, "
        f"peak_score={peak_score} | nima_total={nima_total_ms:.1f}ms total={total_ms:.1f}ms"
    )

    result = {
        "peak_index": peak_index,
        "peak_angle": peak_angle,
        "peak_score": peak_score,
        "scores": scores,
        "angles": angles,
        "frame_count": n,
        "timing_ms": {
            "nima_total": round(nima_total_ms, 1),
            "per_frame_avg": round(nima_total_ms / n, 1),
            "total": round(total_ms, 1),
        },
    }

    if SAVE_DEBUG if save is None else save:
        try:
            saved = _save_tilt(frame_bytes_list, angles, result, session_id=session_id)
            print(f"[tilt-peak] saved: drone-data/{saved}")
        except Exception as e:
            print(f"[tilt-peak] WARN 저장 실패(무시): {e}")

    return result
