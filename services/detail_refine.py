"""3단계 세부조정(정밀 구도) — GAIC + TOPIQ로 최적 crop을 골라 이동 방향을 낸다.

기존 NIMA 9방향 반복(/nima-lateral) 대신, 한 번의 호출로:
  1. 현재 프레임 → 1.4배 48분할 후보 생성(services.crop_candidates 재사용)
  2. 주피사체 잘린 후보 제거(기존 좌표 필터 재사용)
  3. GAIC로 남은 후보 채점 → top3
  4. top3 각각을 9변형(중앙 + 8방향 20% 밀기)으로 확장 → 27개
  5. GAIC로 27개 채점 → 그룹별 top1 → 3장
  6. TOPIQ로 3장 채점 → 최고 1장 = 최종 crop
  7. 최종 crop 중심 vs 프레임 중심(=1.4배 중앙 크롭 중심) → dx, dy, theta_deg

GAIC 로드 실패 시 TOPIQ로 대체(score_source="topiq"). 후보 0개면 best=null.
저장: drone-data/<session_id>/3_detail-refine/
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from models import gaic
from models.topiq import run_topiq_score
from services.crop_candidates import filter_candidates_by_boxes, generate_candidates
from services.drone_offset import compute_offset
from services.nima_directions import CROP_SHIFT_RATIO, DIRECTIONS
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir

SAVE_DEBUG = True

ZOOM_RATIO = 1.4     # 1.4배 중앙 크롭(48분할 후보 창 크기 = W/1.4)
COLS, ROWS = 8, 6    # 8×6 = 48분할
CONTAIN_THRES = 0.90
TOP_K = 3            # GAIC top3


def _targets_to_boxes(targets: list[dict] | None, W: int, H: int) -> list[dict[str, Any]]:
    """targets([{class, bbox:[cx,cy,w,h]}]) → 필터용 boxes([{class, box:[x1,y1,x2,y2]}])."""
    boxes: list[dict[str, Any]] = []
    for t in targets or []:
        bb = t.get("bbox")
        if not bb or len(bb) != 4:
            continue
        cx, cy, w, h = [float(v) for v in bb]
        if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:   # 정규화 → 픽셀
            cx *= W; w *= W; cy *= H; h *= H
        boxes.append({
            "class": t.get("class", "target"),
            "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        })
    return boxes


def _expand_variants(source_box: tuple[int, int, int, int], W: int, H: int) -> dict[str, tuple[int, int, int, int]]:
    """crop 하나를 중앙 + 8방향(창 크기의 20%씩 밀기)으로 9변형. 배율 유지, 경계 clamp."""
    x1, y1, x2, y2 = source_box
    cw, ch = x2 - x1, y2 - y1
    sx, sy = round(cw * CROP_SHIFT_RATIO), round(ch * CROP_SHIFT_RATIO)
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, (dxu, dyu) in DIRECTIONS.items():
        nx = max(0, min(x1 + dxu * sx, W - cw))
        ny = max(0, min(y1 + dyu * sy, H - ch))
        out[name] = (nx, ny, nx + cw, ny + ch)
    return out


def _topiq_of_box(frame: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = frame.crop(tuple(int(v) for v in box))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG")
    return float(run_topiq_score(buf.getvalue())["score"])


def _score_boxes(frame: Image.Image, boxes: list[tuple[int, int, int, int]], use_gaic: bool) -> list[float]:
    """박스들을 GAIC(우선) 또는 TOPIQ로 채점. use_gaic=False면 TOPIQ."""
    if use_gaic:
        return [float(s) for s in gaic.score_boxes(frame, [list(b) for b in boxes])]
    return [_topiq_of_box(frame, b) for b in boxes]


def _save(
    folder: Path,
    frame: Image.Image,
    top3: list[dict],
    variants: list[dict],
    report: dict,
    final_crop: Image.Image,
) -> None:
    # 원본 프레임만 이미지로 저장. 48분할 후보는 사진 대신 좌표(report["candidates"])로만 보관.
    (folder / "original_frame.jpg").write_bytes(_jpeg(frame))

    tdir = folder / "top3"; tdir.mkdir(parents=True, exist_ok=True)
    for r, t in enumerate(top3):
        (tdir / f"top{r}_idx{t['index']:02d}_gaic{t['score']:.3f}.jpg").write_bytes(
            _jpeg(frame.crop(tuple(int(v) for v in t["source_box"])))
        )

    vdir = folder / "variants"; vdir.mkdir(parents=True, exist_ok=True)
    for v in variants:
        (vdir / f"g{v['group']}_{v['direction']}_{v['score']:.3f}.jpg").write_bytes(
            _jpeg(frame.crop(tuple(int(v2) for v2 in v["box"])))
        )

    (folder / "final.jpg").write_bytes(_jpeg(final_crop))
    (folder / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def refine_detail(
    image_bytes: bytes,
    targets: list[dict] | None = None,
    session_id: str | None = None,
    save: bool | None = None,
) -> dict[str, Any]:
    """GAIC+TOPIQ 정밀 crop 선택 + 이동 방향(dx,dy,theta) 반환. 위 파일 상단 파이프라인 참고."""
    t_start = time.perf_counter()
    frame = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = frame.size

    # 1) 48분할 후보
    candidates = generate_candidates(
        img_1x=frame, img_2x_size=frame.size, zoom_ratio=ZOOM_RATIO, cols=COLS, rows=ROWS,
    )
    # 2) 주피사체 잘린 후보 제거(fallback_topk=0 → 다 잘리면 빈 리스트)
    boxes = _targets_to_boxes(targets, W, H)
    filtered, filter_status = filter_candidates_by_boxes(
        candidates=candidates, boxes=boxes, contain_thres=CONTAIN_THRES,
        edge_margin_ratio=0.0, selected_require="all", fallback_topk=0,
    )

    folder = None
    if SAVE_DEBUG if save is None else save:
        folder = resolve_save_dir(session_id, "3_detail-refine",
                                  legacy=DRONE_DATA_DIR / f"detail-refine_{make_ts()}", data_dir=DRONE_DATA_DIR)
    saved_rel = str(folder.relative_to(DRONE_DATA_DIR)) if folder else None

    if not filtered:
        print("[detail-refine] 통과 후보 0개(다 잘림) → best=null")
        result = {
            "best_crop": None, "target_image": None,
            "dx": None, "dy": None, "theta_deg": None,
            "score_source": None, "saved": saved_rel,
            "reason": "no_candidate_after_filter", "filter_status": filter_status,
        }
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "original_frame.jpg").write_bytes(image_bytes)
            (folder / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # 3) GAIC(실패 시 TOPIQ)로 남은 후보 채점 → top3
    use_gaic = True
    t0 = time.perf_counter()
    try:
        cand_scores = _score_boxes(frame, [c.source_box for c in filtered], use_gaic=True)
    except Exception as e:
        use_gaic = False
        print(f"[detail-refine] WARN GAIC 실패 → TOPIQ 대체: {e}")
        cand_scores = _score_boxes(frame, [c.source_box for c in filtered], use_gaic=False)
    gaic48_ms = (time.perf_counter() - t0) * 1000.0
    score_source = "gaic+topiq" if use_gaic else "topiq"

    ranked = sorted(zip(filtered, cand_scores), key=lambda p: p[1], reverse=True)
    top3 = [{"index": c.index, "source_box": list(c.source_box), "score": round(float(s), 4)}
            for c, s in ranked[:TOP_K]]

    # 48분할 후보는 사진 대신 좌표만 저장(잘림 필터 통과 여부 + 통과분의 GAIC/TOPIQ 점수).
    score_by_idx = {c.index: float(s) for c, s in zip(filtered, cand_scores)}
    candidates_meta = [
        {
            "index": c.index,
            "source_box": list(c.source_box),
            "passed_filter": c.index in score_by_idx,
            "score": round(score_by_idx[c.index], 4) if c.index in score_by_idx else None,
        }
        for c in candidates
    ]
    print(f"[detail-refine] 48채점({score_source}) {gaic48_ms:.1f}ms → top{len(top3)}: "
          f"{[(t['index'], t['score']) for t in top3]}")

    # 4) top3 → 각 9변형(27개)
    variant_boxes: list[tuple[int, tuple[int, int, int, int], str]] = []  # (group, box, direction)
    for gi, t in enumerate(top3):
        for name, box in _expand_variants(tuple(t["source_box"]), W, H).items():
            variant_boxes.append((gi, box, name))

    # 5) 27개 채점 → 그룹별 top1
    t0 = time.perf_counter()
    v_scores = _score_boxes(frame, [vb[1] for vb in variant_boxes], use_gaic=use_gaic)
    gaic27_ms = (time.perf_counter() - t0) * 1000.0

    variants = [{"group": g, "direction": d, "box": list(b), "score": round(float(s), 4)}
                for (g, b, d), s in zip(variant_boxes, v_scores)]

    group_best: dict[int, dict] = {}
    for v in variants:
        g = v["group"]
        if g not in group_best or v["score"] > group_best[g]["score"]:
            group_best[g] = v
    finalists = [group_best[g] for g in sorted(group_best)]
    print(f"[detail-refine] 27변형 채점({score_source}) {gaic27_ms:.1f}ms → 그룹 top1 "
          f"{[(f['group'], f['direction'], f['score']) for f in finalists]}")

    # 6) 3장 → TOPIQ → 최고 1장
    t0 = time.perf_counter()
    for f in finalists:
        f["topiq"] = round(_topiq_of_box(frame, tuple(f["box"])), 4)
    topiq3_ms = (time.perf_counter() - t0) * 1000.0
    best = max(finalists, key=lambda f: f["topiq"])
    print(f"[detail-refine] TOPIQ 3장 {topiq3_ms:.1f}ms → best "
          f"group={best['group']} dir={best['direction']} gaic={best['score']} topiq={best['topiq']}")

    # 7) 이동 방향: 최종 crop 중심 vs 프레임 중심(=1.4배 중앙 크롭 중심)
    offset = compute_offset(best["box"], (W, H))
    final_crop = frame.crop(tuple(int(v) for v in best["box"]))
    total_ms = (time.perf_counter() - t_start) * 1000.0

    best_crop = {
        "source_box": best["box"],       # 1x 프레임 픽셀 좌표
        "gaic": best["score"],           # (또는 폴백 시 topiq로 계산된 crop점수)
        "topiq": best["topiq"],
        "group": best["group"],
        "direction": best["direction"],
    }
    report = {
        "score_source": score_source,
        "n_candidates": len(candidates),
        "n_filtered": len(filtered),
        "filter_status": filter_status,
        "candidates": candidates_meta,        # 48분할 좌표(사진 대신)
        "top3": top3,
        "variants": variants,                 # 27
        "finalists": finalists,               # 그룹 top1 3장(+topiq)
        "best_crop": best_crop,
        "offset": offset,
        "timing_ms": {
            "gaic_48": round(gaic48_ms, 1),
            "gaic_27": round(gaic27_ms, 1),
            "topiq_3": round(topiq3_ms, 1),
            "total": round(total_ms, 1),
        },
    }

    if folder is not None:
        try:
            _save(folder, frame, top3, variants, report, final_crop)
            print(f"[detail-refine] saved: drone-data/{saved_rel}")
        except Exception as e:
            print(f"[detail-refine] WARN 저장 실패(무시): {e}")

    return {
        "dx": offset["dx"] if offset else None,
        "dy": offset["dy"] if offset else None,
        "theta_deg": offset["theta_deg"] if offset else None,
        "best_crop": best_crop,
        "target_image": "data:image/jpeg;base64," + base64.b64encode(_jpeg(final_crop)).decode(),
        "score_source": score_source,
        "offset": offset,
        "saved": saved_rel,
        "timing_ms": report["timing_ms"],
    }
