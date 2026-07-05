"""후보 크롭 생성 + 좌표 기반 솎아내기.

1x 원본에서 2x 크기의 창을 격자(cols x rows) 위치로 옮겨가며 후보를 만들고,
result.json의 객체 bbox를 기준으로 객체가 잘린 후보를 걸러낸다.

- generate_candidates(...): cols x rows개 크롭 후보 생성 (기본 8x6 = 48개)
- filter_candidates_by_boxes(...): 객체가 충분히 포함된 후보만 통과
  (사람 person은 하반신 잘림 허용)
- save_candidates(...): 디버그용으로 후보 크롭들을 파일로 저장
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Candidate:
    index: int
    image: Image.Image
    source_box: tuple[int, int, int, int]  # 1x 원본 좌표계
    coord_score: float = 0.0
    coord_results: list[dict[str, Any]] | None = None
    tags: dict[str, Any] | None = None  # 후보 crop을 CLIP으로 태깅한 결과
    patches: np.ndarray | None = None
    best_ref: str | None = None
    best_ref_index: int | None = None
    best_sim: float = -1.0


def generate_candidates(
    img_1x: Image.Image,
    img_2x_size: tuple[int, int],
    zoom_ratio: float,
    cols: int,
    rows: int,
) -> list[Candidate]:
    """1x 이미지에서 2x 창을 cols x rows 위치로 슬라이딩하며 후보를 만든다."""
    W, H = img_1x.size
    out_w, out_h = img_2x_size

    crop_w = int(round(W / zoom_ratio))
    crop_h = int(round(H / zoom_ratio))

    crop_w = max(1, min(crop_w, W))
    crop_h = max(1, min(crop_h, H))

    max_x = max(0, W - crop_w)
    max_y = max(0, H - crop_h)

    xs = [round(i * max_x / (cols - 1)) for i in range(cols)] if cols > 1 else [max_x // 2]
    ys = [round(i * max_y / (rows - 1)) for i in range(rows)] if rows > 1 else [max_y // 2]

    print(
        f"[INFO] candidate window: 1x={W}x{H}, "
        f"window={crop_w}x{crop_h}, output={out_w}x{out_h}, grid={cols}x{rows}"
    )

    candidates = []
    idx = 0

    for y1 in ys:
        for x1 in xs:
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            crop = img_1x.crop((x1, y1, x2, y2)).convert("RGB")

            if crop.size != (out_w, out_h):
                crop = crop.resize((out_w, out_h), Image.BICUBIC)

            candidates.append(
                Candidate(
                    index=idx,
                    image=crop,
                    source_box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
            idx += 1

    return candidates


def containment(candidate_box: tuple[int, int, int, int], target_box: list[float]) -> float:
    cx1, cy1, cx2, cy2 = [float(x) for x in candidate_box]
    tx1, ty1, tx2, ty2 = [float(x) for x in target_box]

    ix1 = max(cx1, tx1)
    iy1 = max(cy1, ty1)
    ix2 = min(cx2, tx2)
    iy2 = min(cy2, ty2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    area = max(1e-6, (tx2 - tx1) * (ty2 - ty1))

    return inter / area


def filter_candidates_by_boxes(
    candidates: list[Candidate],
    boxes: list[dict[str, Any]],
    contain_thres: float,
    edge_margin_ratio: float,
    selected_require: str,
    fallback_topk: int,
) -> tuple[list[Candidate], str]:
    """
    result.json의 bbox_pixel 좌표로 후보들을 솎는다.

    사람(person)의 경우:
    - 사람 전체 bbox가 다 들어올 필요 없음
    - 하반신은 어느 정도 잘려도 허용
    - bbox의 위쪽 60% 정도, 즉 얼굴/상체/중심부가 후보 안에 들어오면 통과 가능

    일반 객체의 경우:
    - 기존처럼 bbox 전체가 후보 안에 들어와야 함
    """
    if not boxes:
        print("[WARN] result.json에서 bbox를 못 찾음 → 후보 전체 사용")
        return candidates, "no_box_fallback_all"

    passed = []

    PERSON_VISIBLE_RATIO = 0.60  # 사람 bbox에서 위쪽 60%만 필수 포함 영역으로 봄

    def required_box_for_filter(box_item: dict[str, Any]) -> list[float]:
        cls_name = str(box_item.get("class", "")).lower()
        x1, y1, x2, y2 = [float(v) for v in box_item["box"]]

        if cls_name == "person":
            h = y2 - y1
            # 하반신 잘림 허용: 위쪽 60%만 필수로 포함되어야 하는 박스로 축소
            y2_req = y1 + h * PERSON_VISIBLE_RATIO
            return [x1, y1, x2, y2_req]

        return [x1, y1, x2, y2]

    def edge_ok_relaxed(candidate_box, required_box, original_box, cls_name):
        if edge_margin_ratio <= 0:
            return True

        cx1, cy1, cx2, cy2 = [float(x) for x in candidate_box]
        rx1, ry1, rx2, ry2 = [float(x) for x in required_box]

        cw = cx2 - cx1
        ch = cy2 - cy1

        mx = cw * edge_margin_ratio
        my = ch * edge_margin_ratio

        # 사람은 아래쪽 잘림 허용이므로 bottom edge는 검사하지 않음
        if cls_name == "person":
            return (
                rx1 > cx1 + mx and
                ry1 > cy1 + my and
                rx2 < cx2 - mx
            )

        return (
            rx1 > cx1 + mx and
            ry1 > cy1 + my and
            rx2 < cx2 - mx and
            ry2 < cy2 - my
        )

    for c in candidates:
        results = []
        ok_count = 0
        scores = []

        for b in boxes:
            cls_name = str(b.get("class", "")).lower()

            original_box = b["box"]
            req_box = required_box_for_filter(b)

            score = containment(c.source_box, req_box)
            safe = edge_ok_relaxed(c.source_box, req_box, original_box, cls_name)

            ok = score >= contain_thres and safe

            if ok:
                ok_count += 1

            scores.append(score)

            results.append({
                "class": b["class"],
                "original_target_box": original_box,
                "required_box_for_filter": req_box,
                "candidate_box": list(c.source_box),
                "containment": score,
                "edge_ok": safe,
                "ok": ok,
                "note": "person lower body may be cut" if cls_name == "person" else "full box required",
            })

        if selected_require == "all":
            candidate_ok = ok_count == len(boxes)
            c.coord_score = min(scores) if scores else 0.0
        else:
            candidate_ok = ok_count >= 1
            c.coord_score = max(scores) if scores else 0.0

        c.coord_results = results

        if candidate_ok:
            passed.append(c)

    ranked = sorted(candidates, key=lambda x: x.coord_score, reverse=True)

    print("[INFO] coordinate filter top candidates:")
    for c in ranked[:10]:
        print(f"  candidate{c.index:02d}: coord_score={c.coord_score:.4f}, source_box={c.source_box}")

    status = "strict"

    if not passed and fallback_topk > 0:
        k = min(fallback_topk, len(ranked))
        print(f"[WARN] strict coordinate filter 통과 0개 → 상위 {k}개 fallback 사용")
        passed = ranked[:k]
        status = "fallback_topk"

    print(f"[INFO] coordinate filter: {len(candidates)} -> {len(passed)} ({status})")
    return passed, status


def save_candidates(output_dir: Path, prefix: str, candidates: list[Candidate]):
    cand_dir = output_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        name = f"{prefix}_candidate{c.index:02d}_coord{c.coord_score:.4f}_ref{c.best_sim:.4f}.jpg"
        c.image.save(cand_dir / name, quality=95)
