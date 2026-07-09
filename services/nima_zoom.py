"""NIMA 줌(전후진) 조정: 현재 프레임을 배율 크롭해 NIMA 점수로 전진/후진을 판단한다.

- 첫 회차(direction 없음): 1.0(현재) / 1.1(전진) / 0.9(후진) 세 크롭을 NIMA로 채점해 방향 확정.
- 이후 회차(direction 지정): 확정된 방향 배율 + 1.0(현재) 두 크롭만 채점해 개선 여부 판단.

9방향 /nima-directions(팬)와 별개, 전후진 전용. 기존 models.nima.run_nima_score 재사용.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

from PIL import Image

from models.nima import run_nima_score
from services.nima_log import append_session_nima
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_indexed_save_dir

# 중앙 2배율 크롭 기준 배율 스텝(±10%).
ZOOM_STEP = 0.1
ZOOM_IN_FACTOR = 1.0 + ZOOM_STEP    # 1.1 = 전진(더 좁게 크롭)
ZOOM_OUT_FACTOR = 1.0 - ZOOM_STEP   # 0.9 = 후진(더 넓게 크롭)

# 디버깅 저장 기본 on/off (실서비스에선 False). 함수 인자로도 덮어쓸 수 있음.
SAVE_DEBUG = True

# 저장 파일명(배율 표기).
_SAVE_NAMES = {
    "current": "center.jpg",         # 1.0
    "zoom_in": "center_1_1x.jpg",    # 1.1 (전진)
    "zoom_out": "center_0_9x.jpg",   # 0.9 (후진)
}


def _zoom_crop(img: Image.Image, factor: float) -> Image.Image:
    """중앙 2배율 크롭(창=중앙 절반)을 기준으로 factor 배율의 크롭을 만든다.

    factor>1 이면 더 좁게 크롭(전진 근사), <1 이면 더 넓게 크롭(후진 근사).
    결과는 항상 기준 창 크기(W/2 x H/2)로 리사이즈해 같은 크기로 맞춘다.
    """
    W, H = img.size
    base_w, base_h = W // 2, H // 2                 # 1.0배율(중앙 2배율 뷰) 창 크기
    win_w = max(1, min(round(base_w / factor), W))  # 배율↑ → 창↓, 배율↓ → 창↑
    win_h = max(1, min(round(base_h / factor), H))
    x = (W - win_w) // 2
    y = (H - win_h) // 2
    crop = img.crop((x, y, x + win_w, y + win_h))
    if (win_w, win_h) != (base_w, base_h):
        crop = crop.resize((base_w, base_h), Image.LANCZOS)
    return crop.convert("RGB")


def _nima_score(crop: Image.Image) -> float:
    buf = io.BytesIO()
    crop.save(buf, format="JPEG")
    return float(run_nima_score(buf.getvalue())["score"])


def _save_debug(
    image_bytes: bytes,
    crops: dict[str, Image.Image],
    result: dict[str, Any],
    session_id: str | None = None,
    data_dir=None,
) -> str:
    """원본 + 배율 크롭들 + 점수/판단을 저장.

    세션 없으면 drone-data/nima-move/<ts>/, 있으면 drone-data/<session_id>/3_nima-move/depth/<순번>/.
    (반복 호출 시 depth/1, depth/2 ... 순서대로 쌓임)
    """
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_indexed_save_dir(session_id, "3_nima-move/depth", legacy=base / "nima-move" / make_ts(), data_dir=base)

    (folder / "original.jpg").write_bytes(image_bytes)
    for name, crop in crops.items():
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        (folder / _SAVE_NAMES.get(name, f"{name}.jpg")).write_bytes(buf.getvalue())

    payload = {
        "type": "zoom",
        **{
            k: result[k]
            for k in ("scores", "direction", "best_zoom", "improved", "current_score", "best_score")
        },
    }
    (folder / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 단계별 NIMA 누적 로그(선택 = 최고 배율 크롭, 이미 계산된 best_score 재사용).
    append_session_nima(session_id, f"3_nima-move/depth/{folder.name}", score=result.get("best_score"), data_dir=base)
    return str(folder.relative_to(base))


def evaluate_zoom(
    image_bytes: bytes,
    direction: str | None = None,
    session_id: str | None = None,
    save: bool | None = None,
) -> dict[str, Any]:
    """배율 크롭 NIMA 점수로 전진/후진을 판단한다.

    direction 없음(첫 회차): 1.0/1.1/0.9 세 크롭 채점 → 방향 확정.
    direction 지정(이후 회차): 그 방향 배율 + 1.0 두 크롭 채점 → 개선 여부.

    Returns:
        {direction, best_zoom, improved, scores, current_score, best_score, timing_ms}
    Raises:
        ValueError: direction이 forward/backward가 아닐 때.
    """
    t_start = time.perf_counter()
    d = (direction or "").strip().lower()
    if d and d not in ("forward", "backward"):
        raise ValueError(f"direction은 forward/backward/없음 중 하나여야 함: {direction!r}")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    crops: dict[str, Image.Image] = {}
    scores: dict[str, float] = {}

    def score(name: str, factor: float) -> None:
        ti = time.perf_counter()
        crop = _zoom_crop(img, factor)
        crops[name] = crop
        s = _nima_score(crop)
        scores[name] = round(s, 4)
        print(f"[nima-zoom] {name:>8}(x{factor:.1f}): {s:.4f} ({(time.perf_counter() - ti) * 1000:.1f} ms)")

    t_n0 = time.perf_counter()
    score("current", 1.0)

    if not d:
        # 첫 회차: 양방향 비교
        score("zoom_in", ZOOM_IN_FACTOR)
        score("zoom_out", ZOOM_OUT_FACTOR)
        cur = scores["current"]
        best_name = max(scores, key=scores.__getitem__)  # current/zoom_in/zoom_out
        if best_name == "zoom_in" and scores["zoom_in"] > cur:
            best_zoom = "forward"
        elif best_name == "zoom_out" and scores["zoom_out"] > cur:
            best_zoom = "backward"
        else:
            best_zoom = "stay"
        direction_used = best_zoom
    else:
        # 이후 회차: 확정 방향만
        if d == "forward":
            score("zoom_in", ZOOM_IN_FACTOR)
            z = scores["zoom_in"]
        else:
            score("zoom_out", ZOOM_OUT_FACTOR)
            z = scores["zoom_out"]
        cur = scores["current"]
        improved_dir = z > cur
        best_zoom = d if improved_dir else "stay"
        direction_used = d

    nima_total_ms = (time.perf_counter() - t_n0) * 1000.0
    current_score = scores["current"]
    best_score = round(max(scores.values()), 4)
    improved = best_score > current_score
    total_ms = (time.perf_counter() - t_start) * 1000.0

    result = {
        "direction": direction_used,
        "best_zoom": best_zoom,
        "improved": improved,
        "scores": scores,
        "current_score": current_score,
        "best_score": best_score,
        "timing_ms": {
            "nima_total": round(nima_total_ms, 1),
            "per_call_avg": round(nima_total_ms / len(scores), 1),
            "total": round(total_ms, 1),
        },
    }

    print(
        f"[nima-zoom] DONE mode={'first' if not d else d} → direction={direction_used}, "
        f"best_zoom={best_zoom}, improved={improved}, current={current_score}, best={best_score} | "
        f"nima_total={nima_total_ms:.1f}ms total={total_ms:.1f}ms"
    )

    if SAVE_DEBUG if save is None else save:
        try:
            saved = _save_debug(image_bytes, crops, result, session_id=session_id)
            print(f"[nima-zoom] debug saved: drone-data/{saved}")
        except Exception as e:
            print(f"[WARN] nima-zoom 디버깅 저장 실패(무시): {e}")

    return result
