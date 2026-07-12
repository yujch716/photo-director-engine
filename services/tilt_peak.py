"""틸트 sweep 최고 프레임 선택 (SAMP 구도 점수).

짐벌 틸트를 위→아래로 훑으며 찍은 프레임들을 받아 각 장에 SAMP-Net 구도(composition)
점수를 매기고, 최고 점수 프레임의 index와 실제 각도를 반환한다. (/scan-peak과 같은
"묶음→최고" 구조, 비교 기준이 SSIM/NIMA가 아니라 무참조 SAMP 구도 점수)

기존 models.sampnet.score_image 재사용. 결과/프레임은
drone-data/<session_id>/4_tilt/에 저장.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from models.sampnet import score_image
from services.nima_directions import _fully_contains, _object_pixel_boxes  # 잘림 판정 재사용
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir

SAVE_DEBUG = True


def _center_2x_jpeg(image_bytes: bytes) -> bytes:
    """이미지의 중앙 5/7(1.4배율 뷰)를 잘라 JPEG 바이트로 반환(색상 유지). scan_peak과 동일 방식."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    crop = img.crop((W // 7, H // 7, W - W // 7, H - H // 7))  # 1.4배: 중앙 5/7
    buf = io.BytesIO()
    crop.save(buf, format="JPEG")
    return buf.getvalue()


def _per_frame_bboxes(bboxes: Any, n: int) -> list[Any]:
    """bboxes를 프레임별 리스트로 정규화.

    - None → 전부 None(제외 안 함)
    - 단일 [cx,cy,w,h] → 모든 프레임에 동일 적용
    - 길이 n 리스트(프레임별 [cx,cy,w,h] 또는 null) → 그대로
    """
    if bboxes is None:
        return [None] * n
    if (isinstance(bboxes, (list, tuple)) and len(bboxes) == 4
            and all(isinstance(v, (int, float)) for v in bboxes)):
        return [bboxes] * n
    if isinstance(bboxes, (list, tuple)) and len(bboxes) == n:
        return list(bboxes)
    raise ValueError(f"bboxes 길이({len(bboxes) if hasattr(bboxes,'__len__') else '?'})가 "
                     f"frames({n})와 같거나, 단일 [cx,cy,w,h] 여야 함")


def _frame_object_kept(frame_bytes: bytes, bbox: Any) -> bool:
    """이 프레임의 중앙 1.4배 크롭 창이 객체(bbox)를 완전히 담으면 True(잘리지 않음)."""
    if bbox is None:
        return True
    img = Image.open(io.BytesIO(frame_bytes))
    W, H = img.size
    win = (W // 7, H // 7, W - W // 7, H - H // 7)   # _center_2x_jpeg와 동일 창
    objs = _object_pixel_boxes(bbox, W, H)
    if not objs:
        return True
    return all(_fully_contains(win, ob) for ob in objs)


def _auto_person_bboxes(frame_bytes_list: list[bytes], session_id: str | None) -> list[Any] | None:
    """bboxes 미제공 시 폴백: 세션 1_original/report.json의 선택 객체가 person이면
    각 틸트 프레임에 YOLO를 다시 돌려 '가장 큰 사람'의 bbox를 프레임별로 반환한다.

    - person 타깃이 없거나 report.json이 없으면 None(자동 제외 생략).
    - 프레임에서 사람을 못 찾으면 그 프레임은 None(=사람이 프레임 밖/잘림 → 나중에 제외 처리).
    (COCO YOLO는 person만 재검출 가능. 랜드마크 타깃은 재검출 불가라 생략.)
    """
    if not session_id:
        return None
    rp = DRONE_DATA_DIR / session_id / "1_original" / "report.json"
    if not rp.exists():
        return None
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return None
    items = data.get("targets") or data.get("results") or []
    classes = {str(t.get("class", "")).lower() for t in items if isinstance(t, dict)}
    if "person" not in classes:
        print(f"[tilt-peak] auto-bbox: 선택 객체 {sorted(classes)}에 person 없음 → YOLO 재검출 생략")
        return None

    from models.yolo import run_yolo   # person 전용(COCO)
    out: list[Any] = []
    for fb in frame_bytes_list:
        try:
            dets = run_yolo(fb)
        except Exception as e:
            print(f"[tilt-peak] auto-bbox: YOLO 실패 → 자동 제외 생략: {e}")
            return None
        if dets:
            best = max(dets, key=lambda d: d["bbox"][2] * d["bbox"][3])  # 가장 큰 사람 = 주 피사체
            out.append(best["bbox"])                                     # [cx,cy,w,h] 정규화
        else:
            out.append(None)                                            # 못 찾음 → 제외 대상
    print(f"[tilt-peak] auto-bbox(YOLO person): {sum(1 for b in out if b)}/{len(frame_bytes_list)} 프레임 검출")
    return out


def _save_tilt(
    frame_bytes_list: list[bytes],
    angles: list[float],
    result: dict[str, Any],
    session_id: str | None = None,
    data_dir: Path | None = None,
) -> str:
    """중앙 1.4배 크롭 프레임들(각도 파일명) + best + scores.json을 <session_id>/4_tilt/에 저장."""
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_save_dir(session_id, "4_tilt", legacy=base / "tilt" / make_ts(), data_dir=base)

    for i, (fb, ang) in enumerate(zip(frame_bytes_list, angles)):
        (folder / f"{i:02d}_{ang:+.1f}deg.jpg").write_bytes(fb)
    # 최고 점수 프레임
    peak = result["peak_index"]
    (folder / "best.jpg").write_bytes(frame_bytes_list[peak])
    (folder / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(folder.relative_to(base))


def find_tilt_peak(
    frame_bytes_list: list[bytes],
    angles: list[float],
    bboxes: Any = None,
    session_id: str | None = None,
    save: bool | None = None,
) -> dict[str, Any]:
    """프레임들에 SAMP 구도 점수를 매겨 최고 점수 프레임(각도)을 반환한다.

    bboxes(선택): 프레임별 대상 객체 [cx,cy,w,h](정규화). 프레임마다 각도가 달라 객체 위치도
    다르므로 **프레임별로** 준다(길이=frames). 단일 [cx,cy,w,h]면 전 프레임 동일 적용. 주어지면
    그 프레임의 중앙 1.4배 크롭이 객체를 자르는 프레임은 후보에서 제외한다. 다 잘리면 전체 사용.

    Returns (슬림): {peak_angle, peak_score, saved}
        상세(scores/angles/excluded/timing 등)는 scores.json 저장 + 서버 콘솔 로그로 확인.
    Raises:
        ValueError: frames가 비었거나 angles 개수가 frames와 다를 때.
    """
    t_start = time.perf_counter()
    n = len(frame_bytes_list)
    if n == 0:
        raise ValueError("frames가 비어 있음")
    if angles is None or len(angles) != n:
        raise ValueError(f"angles 개수({0 if angles is None else len(angles)})가 frames({n})와 달라야 함")

    # 각 프레임을 중앙 1.4배 크롭한 뒤 SAMP 구도 점수를 매긴다(구도 스케일 정합). 저장도 이 크롭본으로.
    cropped = [_center_2x_jpeg(fb) for fb in frame_bytes_list]

    # bbox로 객체가 잘리는 프레임 제외. 다 잘리면 제외 무시(전체 사용).
    per_bbox = _per_frame_bboxes(bboxes, n)
    bbox_source = "manual" if any(bb is not None for bb in per_bbox) else "none"
    # 앱이 bbox를 안 줬으면 → 세션 report.json의 person 타깃 기준으로 YOLO 재검출 폴백.
    if bbox_source == "none":
        auto = _auto_person_bboxes(frame_bytes_list, session_id)
        if auto is not None:
            per_bbox = auto
            bbox_source = "yolo_report"

    def _kept(i: int) -> bool:
        bb = per_bbox[i]
        if bb is None:
            # auto(yolo) 모드에서 '못 찾음'은 사람이 프레임 밖/잘림 → 제외. 그 외엔 정보없음 → 유지.
            return bbox_source != "yolo_report"
        return _frame_object_kept(frame_bytes_list[i], bb)

    kept = [i for i in range(n) if _kept(i)]
    excluded = [i for i in range(n) if i not in kept]
    if not kept:
        print("[tilt-peak] WARN 모든 프레임에서 객체 잘림 → 제외 무시(전체 사용)")
        kept, excluded = list(range(n)), []
    elif excluded:
        print(f"[tilt-peak] 객체 잘림 제외({bbox_source}): {[(i, round(angles[i], 1)) for i in excluded]}")

    scores: list[float | None] = [None] * n
    t_n0 = time.perf_counter()
    for i in kept:                                  # 잘린 프레임은 채점도 생략
        ti = time.perf_counter()
        s = float(score_image(cropped[i])["score"])
        scores[i] = round(s, 4)
        print(f"[tilt-peak] frame {i:>2} (angle={angles[i]:+.1f}): samp={s:.4f} ({(time.perf_counter() - ti) * 1000:.1f} ms)")
    samp_total_ms = (time.perf_counter() - t_n0) * 1000.0

    peak_index = int(max(kept, key=lambda i: scores[i]))
    peak_angle = angles[peak_index]
    peak_score = scores[peak_index]
    total_ms = (time.perf_counter() - t_start) * 1000.0

    print(
        f"[tilt-peak] DONE n={n} scored={len(kept)} excluded={[round(angles[i],1) for i in excluded]} "
        f"→ peak_index={peak_index}, peak_angle={peak_angle:+.1f}, peak_score={peak_score} | "
        f"samp_total={samp_total_ms:.1f}ms total={total_ms:.1f}ms"
    )

    result = {
        "metric": "samp",           # 구도 점수 기준(1~5). 이전엔 nima였음.
        "peak_index": peak_index,
        "peak_angle": peak_angle,
        "peak_score": peak_score,
        "scores": scores,           # 잘린 프레임은 null
        "angles": angles,
        "frame_count": n,
        "scored_count": len(kept),
        "bbox_source": bbox_source,   # "manual"(앱 제공) | "yolo_report"(person 자동재검출) | "none"
        "bboxes_in": per_bbox,        # 프레임별 사용된 bbox(디버그용, null=없음/미검출)
        "kept_indices": kept,                               # 채점(=잘리지 않음) 프레임
        "excluded_indices": excluded,                       # 객체 잘려 제외된 프레임 index
        "excluded_angles": [angles[i] for i in excluded],
        "timing_ms": {
            "samp_total": round(samp_total_ms, 1),
            "per_frame_avg": round(samp_total_ms / max(1, len(kept)), 1),
            "total": round(total_ms, 1),
        },
    }

    # scores.json에는 full result(scores/angles/timing/metric/peak_index 전부) 저장.
    saved = None
    if SAVE_DEBUG if save is None else save:
        try:
            # 저장/로그도 중앙 1.4배 크롭본으로.
            saved = _save_tilt(cropped, angles, result, session_id=session_id)
            print(f"[tilt-peak] saved: drone-data/{saved}")
        except Exception as e:
            print(f"[tilt-peak] WARN 저장 실패(무시): {e}")

    # 응답은 앱이 쓰는 것만 슬림하게(상세는 scores.json + 콘솔 로그로 확인).
    return {
        "peak_angle": peak_angle,   # 앱이 짐벌을 맞출 각도(핵심)
        "peak_score": peak_score,   # 토스트·로그용
        "saved": saved,             # 상세 확인용 폴더(scores.json 위치)
    }
