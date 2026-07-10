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
from typing import Any

from PIL import Image

from models.nima import run_nima_score
from services.nima_log import append_session_nima
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_indexed_save_dir

# 중앙 크롭 창을 각 방향으로 얼마나(창 크기 대비 비율) 옮길지.
# 너무 작으면 방향별 그림이 거의 같아 NIMA가 구분 못 함 → 20% 유지.
CROP_SHIFT_RATIO = 0.20

# 디버깅 저장 기본 on/off (실서비스에선 False로). 함수 인자로도 덮어쓸 수 있음.
SAVE_DEBUG = True

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


def _make_direction_windows(img: Image.Image) -> dict[str, tuple[int, int, int, int]]:
    """9방향 1.5배율 크롭 창(중앙 2/3 크기 창을 방향별로 이동)의 픽셀 박스를 만든다.

    반환: {방향: (x1, y1, x2, y2)}. 밖으로 나가면 이미지 안으로 clamp.
    """
    W, H = img.size
    cw, ch = 2 * W // 3, 2 * H // 3              # 1.5배율 뷰 = 중앙 2/3 크기
    cx0, cy0 = (W - cw) // 2, (H - ch) // 2       # 중앙 창의 좌상단
    shift_x = round(cw * CROP_SHIFT_RATIO)
    shift_y = round(ch * CROP_SHIFT_RATIO)

    windows: dict[str, tuple[int, int, int, int]] = {}
    for name, (dxu, dyu) in DIRECTIONS.items():
        x = cx0 + dxu * shift_x
        y = cy0 + dyu * shift_y
        x = max(0, min(x, W - cw))                # 이미지 밖으로 나가면 clamp
        y = max(0, min(y, H - ch))
        windows[name] = (x, y, x + cw, y + ch)
    return windows


def _object_pixel_boxes(bbox: Any, W: int, H: int) -> list[tuple[float, float, float, float]]:
    """bbox → 픽셀 (x1,y1,x2,y2) 목록.

    bbox 형식: 정규화 [cx,cy,w,h](0~1) 하나, 또는 그런 박스들의 리스트.
    값이 1.5보다 크면 이미 픽셀 좌표로 간주.
    """
    if bbox is None:
        return []
    boxes = bbox
    # 단일 [cx,cy,w,h] 인지, 박스들의 리스트인지 판별
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        boxes = [bbox]

    out: list[tuple[float, float, float, float]] = []
    for b in boxes:
        cx, cy, w, h = [float(v) for v in b]
        if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:  # 정규화 → 픽셀
            cx *= W; w *= W; cy *= H; h *= H
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def _fully_contains(win: tuple[int, int, int, int],
                    obj: tuple[float, float, float, float], eps: float = 1.0) -> bool:
    """크롭 창 win이 객체 박스 obj를 (eps 여유 내에서) 완전히 포함하면 True."""
    wx1, wy1, wx2, wy2 = win
    ox1, oy1, ox2, oy2 = obj
    return (ox1 >= wx1 - eps and oy1 >= wy1 - eps
            and ox2 <= wx2 + eps and oy2 <= wy2 + eps)


def _save_debug(
    image_bytes: bytes,
    crops: dict[str, Image.Image],
    result: dict[str, Any],
    session_id: str | None = None,
    data_dir=None,
) -> str:
    """9방향 크롭 + 원본 + 점수를 저장(디버깅/검증용).

    세션 없으면 drone-data/nima-move/<ts>/, 있으면 drone-data/<session_id>/3_detail-move/lateral/<순번>/.
    (반복 호출 시 lateral/1, lateral/2 ... 순서대로 쌓임)
    """
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_indexed_save_dir(session_id, "3_detail-move/lateral", legacy=base / "nima-move" / make_ts(), data_dir=base)

    (folder / "original.jpg").write_bytes(image_bytes)
    for name, crop in crops.items():
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        (folder / f"{name}.jpg").write_bytes(buf.getvalue())

    payload = {
        "type": "directions",
        **{
            k: result[k]
            for k in ("scores", "best_direction", "is_center_best", "best_score",
                      "center_score", "score_margin", "excluded_directions")
        },
    }
    (folder / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 단계별 NIMA 누적 로그(선택 = 최고 방향 크롭, 이미 계산된 best_score 재사용).
    append_session_nima(session_id, f"3_detail-move/lateral/{folder.name}", score=result.get("best_score"), data_dir=base)
    return str(folder.relative_to(base))


def find_best_nima_direction(
    image_bytes: bytes,
    bbox: Any = None,
    session_id: str | None = None,
    save: bool | None = None,
) -> dict[str, Any]:
    """현재 프레임의 9방향 1.5배율 크롭에 NIMA를 매겨 최고 방향을 반환한다.

    bbox(선택): 대상 객체의 정규화 [cx,cy,w,h](또는 그 리스트). 주어지면 그 객체가
    창 밖으로 잘리는 방향은 후보에서 제외한다. 모든 방향이 잘리면 제외를 무시(전체 사용).

    Returns:
        {best_direction, is_center_best, scores, best_score, center_score, score_margin,
         excluded_directions, timing_ms}
    """
    t_start = time.perf_counter()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size

    t0 = time.perf_counter()
    windows = _make_direction_windows(img)
    crops = {name: img.crop(win).convert("RGB") for name, win in windows.items()}  # 저장/디버깅용 전부
    crop_ms = (time.perf_counter() - t0) * 1000.0

    # bbox가 있으면 객체가 잘리는 방향 제외. 남는 게 없으면 제외 무시(전체).
    obj_boxes = _object_pixel_boxes(bbox, W, H)
    excluded: dict[str, str] = {}
    candidates = list(windows.keys())
    if obj_boxes:
        kept = [n for n, win in windows.items()
                if all(_fully_contains(win, ob) for ob in obj_boxes)]
        if kept:
            candidates = kept
            excluded = {n: "object_cut" for n in windows if n not in kept}
            print(f"[nima-dir] 객체 잘림 제외: {sorted(excluded)} → 후보 {candidates}")
        else:
            print("[nima-dir] WARN 모든 방향에서 객체 잘림 → 제외 무시(전체 사용)")

    scores: dict[str, float] = {}
    t_n0 = time.perf_counter()
    for name in candidates:                       # 잘린 방향은 채점도 생략
        ti = time.perf_counter()
        buf = io.BytesIO()
        crops[name].save(buf, format="JPEG")
        s = float(run_nima_score(buf.getvalue())["score"])
        scores[name] = round(s, 4)
        print(f"[nima-dir] {name:>10}: {s:.4f} ({(time.perf_counter() - ti) * 1000:.1f} ms)")
    nima_total_ms = (time.perf_counter() - t_n0) * 1000.0

    best_direction = max(scores, key=scores.__getitem__)
    best_score = scores[best_direction]
    center_score = scores.get("center")           # center가 제외됐을 수 있음
    score_margin = round(best_score - center_score, 4) if center_score is not None else None
    total_ms = (time.perf_counter() - t_start) * 1000.0

    print(
        f"[nima-dir] DONE best={best_direction}({best_score}) center={center_score} "
        f"margin={score_margin} excluded={sorted(excluded)} | crop={crop_ms:.1f}ms "
        f"nima_total={nima_total_ms:.1f}ms per_dir_avg={nima_total_ms / max(1, len(scores)):.1f}ms "
        f"total={total_ms:.1f}ms"
    )

    result = {
        "best_direction": best_direction,
        "is_center_best": best_direction == "center",
        "scores": scores,                         # 채점된(=잘리지 않은) 방향만
        "best_score": best_score,
        "center_score": center_score,
        "score_margin": score_margin,
        "excluded_directions": sorted(excluded),  # 객체가 잘려 제외된 방향
        "timing_ms": {
            "crop": round(crop_ms, 1),
            "nima_total": round(nima_total_ms, 1),
            "per_dir_avg": round(nima_total_ms / max(1, len(scores)), 1),
            "total": round(total_ms, 1),
        },
    }

    # 디버깅 저장(응답 형식은 그대로, 저장만 부가). 실패해도 응답은 살린다.
    if SAVE_DEBUG if save is None else save:
        try:
            saved = _save_debug(image_bytes, crops, result, session_id=session_id)
            print(f"[nima-dir] debug saved: drone-data/{saved}")
        except Exception as e:
            print(f"[WARN] nima-directions 디버깅 저장 실패(무시): {e}")

    return result
