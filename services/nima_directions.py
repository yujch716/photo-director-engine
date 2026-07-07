"""NIMA 세부조정: 현재 프레임에서 9방향 2배율 크롭을 만들어 각각 미학 점수(NIMA)를 매긴다.

정점 근처에서 "어느 방향으로 살짝 옮기면 구도가 더 좋아지는지"를 고르는 용도.
중앙(center) + 상하좌우 + 대각선 4개 = 9방향의 2배율 크롭 창을 만들어 NIMA로 채점하고,
최고점 방향을 반환한다.

기존 models.nima.run_nima_score를 그대로 재사용(모델은 lazy singleton으로 1회 로드).
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from models.nima import run_nima_score

# 중앙 크롭 창을 각 방향으로 얼마나(창 크기 대비 비율) 옮길지.
# 너무 작으면 방향별 그림이 거의 같아 NIMA가 구분 못 함 → 20% 유지.
CROP_SHIFT_RATIO = 0.20

# 디버깅 저장 기본 on/off (실서비스에선 False로). 함수 인자로도 덮어쓸 수 있음.
SAVE_DEBUG = True
# 저장 루트. nima-directions/nima-zoom 결과를 drone-data/nima-move/<타임스탬프>/ 아래로 통일.
NIMA_MOVE_DIR = Path("drone-data/nima-move")

# 방향별 이동 단위 (dx, dy). +x=오른쪽, +y=아래. "up"=위(=y 감소).
DIRECTIONS: dict[str, tuple[int, int]] = {
    "center": (0, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "up-left": (-1, -1),
    "up-right": (1, -1),
    "down-left": (-1, 1),
    "down-right": (1, 1),
}


def _make_direction_crops(img: Image.Image) -> dict[str, Image.Image]:
    """9방향 2배율 크롭(중앙 절반 크기 창을 방향별로 이동)을 만든다. 밖으로 나가면 clamp."""
    W, H = img.size
    cw, ch = W // 2, H // 2                      # 2배율 뷰 = 중앙 절반 크기
    cx0, cy0 = (W - cw) // 2, (H - ch) // 2       # 중앙 창의 좌상단
    shift_x = round(cw * CROP_SHIFT_RATIO)
    shift_y = round(ch * CROP_SHIFT_RATIO)

    crops: dict[str, Image.Image] = {}
    for name, (dxu, dyu) in DIRECTIONS.items():
        x = cx0 + dxu * shift_x
        y = cy0 + dyu * shift_y
        x = max(0, min(x, W - cw))                # 이미지 밖으로 나가면 clamp
        y = max(0, min(y, H - ch))
        crops[name] = img.crop((x, y, x + cw, y + ch)).convert("RGB")
    return crops


def _save_debug(
    image_bytes: bytes,
    crops: dict[str, Image.Image],
    result: dict[str, Any],
    data_dir: Path | None = None,
) -> str:
    """9방향 크롭 + 원본 + 점수를 drone-data/nima-move/<타임스탬프>/에 저장(디버깅/검증용)."""
    base = data_dir or NIMA_MOVE_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder = base / ts
    suffix = 1
    while folder.exists():  # 같은 밀리초 충돌 방지
        folder = base / f"{ts}_{suffix}"
        suffix += 1
    folder.mkdir(parents=True)

    (folder / "original.jpg").write_bytes(image_bytes)
    for name, crop in crops.items():
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        (folder / f"{name}.jpg").write_bytes(buf.getvalue())

    payload = {
        "type": "directions",
        **{
            k: result[k]
            for k in ("scores", "best_direction", "is_center_best", "best_score", "center_score", "score_margin")
        },
    }
    (folder / "scores.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folder.name


def find_best_nima_direction(image_bytes: bytes, save: bool | None = None) -> dict[str, Any]:
    """현재 프레임의 9방향 2배율 크롭에 NIMA를 매겨 최고 방향을 반환한다.

    Returns:
        {best_direction, is_center_best, scores, best_score, center_score, score_margin, timing_ms}
    """
    t_start = time.perf_counter()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    t0 = time.perf_counter()
    crops = _make_direction_crops(img)
    crop_ms = (time.perf_counter() - t0) * 1000.0

    scores: dict[str, float] = {}
    t_n0 = time.perf_counter()
    for name, crop in crops.items():
        ti = time.perf_counter()
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        s = float(run_nima_score(buf.getvalue())["score"])
        scores[name] = round(s, 4)
        print(f"[nima-dir] {name:>10}: {s:.4f} ({(time.perf_counter() - ti) * 1000:.1f} ms)")
    nima_total_ms = (time.perf_counter() - t_n0) * 1000.0

    best_direction = max(scores, key=scores.__getitem__)
    best_score = scores[best_direction]
    center_score = scores["center"]
    score_margin = round(best_score - center_score, 4)
    total_ms = (time.perf_counter() - t_start) * 1000.0

    print(
        f"[nima-dir] DONE best={best_direction}({best_score}) center={center_score} "
        f"margin={score_margin} | crop={crop_ms:.1f}ms nima_total={nima_total_ms:.1f}ms "
        f"per_dir_avg={nima_total_ms / len(crops):.1f}ms total={total_ms:.1f}ms"
    )

    result = {
        "best_direction": best_direction,
        "is_center_best": best_direction == "center",
        "scores": scores,
        "best_score": best_score,
        "center_score": center_score,
        "score_margin": score_margin,
        "timing_ms": {
            "crop": round(crop_ms, 1),
            "nima_total": round(nima_total_ms, 1),
            "per_dir_avg": round(nima_total_ms / len(crops), 1),
            "total": round(total_ms, 1),
        },
    }

    # 디버깅 저장(응답 형식은 그대로, 저장만 부가). 실패해도 응답은 살린다.
    if SAVE_DEBUG if save is None else save:
        try:
            saved = _save_debug(image_bytes, crops, result)
            print(f"[nima-dir] debug saved: drone-data/nima-move/{saved}")
        except Exception as e:
            print(f"[WARN] nima-directions 디버깅 저장 실패(무시): {e}")

    return result
