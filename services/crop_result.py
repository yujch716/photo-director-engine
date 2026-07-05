"""result.json 해석.

capture 단계가 남긴 result.json에서
- load_result_boxes(...): 대상 객체들의 bbox_pixel(1x 좌표) 목록을 뽑고
- result_json_uses_hand_bank(...): 잡힌 대상이 호미곶 "상생의손"인지 판정해
  일반 bank / hand bank 중 무엇을 쓸지 알려준다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_result_boxes(json_path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    W, H = image_size

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []

    def add_box(cls_name: str, box: list[float], source: str):
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(W), float(x1)))
        y1 = max(0.0, min(float(H), float(y1)))
        x2 = max(0.0, min(float(W), float(x2)))
        y2 = max(0.0, min(float(H), float(y2)))

        if x2 <= x1 or y2 <= y1:
            return

        boxes.append({
            "class": cls_name,
            "box": [x1, y1, x2, y2],
            "source": source,
        })

    # 우선 results[*].bbox_pixel 사용
    for i, item in enumerate(data.get("results", [])):
        bp = item.get("bbox_pixel")
        cls_name = str(item.get("class", f"target{i}"))

        if isinstance(bp, dict):
            if all(k in bp for k in ["left", "top", "right", "bottom"]):
                add_box(
                    cls_name,
                    [bp["left"], bp["top"], bp["right"], bp["bottom"]],
                    f"results[{i}].bbox_pixel",
                )
                continue

        # 없으면 bbox normalized center xywh 사용
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            cx, cy, bw, bh = [float(x) for x in bbox]
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
                cx *= W
                bw *= W
                cy *= H
                bh *= H
            add_box(
                cls_name,
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                f"results[{i}].bbox",
            )

    # results가 없으면 targets 사용
    if not boxes:
        for i, item in enumerate(data.get("targets", [])):
            bbox = item.get("bbox")
            cls_name = str(item.get("class", f"target{i}"))

            if isinstance(bbox, list) and len(bbox) == 4:
                cx, cy, bw, bh = [float(x) for x in bbox]
                if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
                    cx *= W
                    bw *= W
                    cy *= H
                    bh *= H
                add_box(
                    cls_name,
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                    f"targets[{i}].bbox",
                )

    print(f"[INFO] loaded boxes from {json_path}: {len(boxes)}")
    for b in boxes:
        print(f"  {b['class']}: {b['box']} ({b['source']})")

    return boxes


# ---------------------------------------------------------------------------
# hand bank 사용 여부 판정
# ---------------------------------------------------------------------------

HAND_LANDMARK_KEYWORDS = [
    "상생의손",
    "상생의 손",
    "호미곶 상생",
    "hand of coexistence",
    "coexistence hand",
    "homigot hand",
    "homigot",
    "hand sculpture",
    "giant hand",
]


def _is_hand_landmark_text(value: Any) -> bool:
    if value is None:
        return False

    text = str(value).strip().lower()
    compact = text.replace(" ", "")

    for kw in HAND_LANDMARK_KEYWORDS:
        kw_l = kw.lower()
        kw_c = kw_l.replace(" ", "")

        if kw_l in text or kw_c in compact:
            return True

    return False


def result_json_uses_hand_bank(json_path: Path) -> tuple[bool, str]:
    """
    result.json에서 잡힌 대상이 호미곶 상생의손인지 판단한다.

    우선순위:
    1. results[*].landmark / class / name / label 등에 상생의손 키워드가 있으면 hand bank
    2. results[*].scores에서 가장 높은 label이 상생의손 계열이면 hand bank
    3. targets[*]도 같은 방식으로 확인
    4. 그 외에는 일반 bank
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return False, f"json_load_failed:{type(exc).__name__}"

    check_keys = [
        "landmark",
        "landmark_name",
        "matched_landmark",
        "target_landmark",
        "selected_landmark",
        "name",
        "label",
        "class",
        "category",
    ]

    # 전역 필드 확인
    for key in check_keys + ["place", "place_name", "location_name"]:
        if isinstance(data, dict) and key in data:
            if _is_hand_landmark_text(data.get(key)):
                return True, f"top_level.{key}={data.get(key)}"

    # results / targets 확인
    items = []
    if isinstance(data, dict):
        for group_key in ["results", "targets", "objects", "detections"]:
            group = data.get(group_key)
            if isinstance(group, list):
                for i, item in enumerate(group):
                    if isinstance(item, dict):
                        items.append((f"{group_key}[{i}]", item))

    for prefix, item in items:
        for key in check_keys:
            if key in item and _is_hand_landmark_text(item.get(key)):
                return True, f"{prefix}.{key}={item.get(key)}"

        scores = item.get("scores")
        if isinstance(scores, dict) and scores:
            try:
                best_label = max(scores.items(), key=lambda kv: float(kv[1]))[0]
                best_score = scores[best_label]
                if _is_hand_landmark_text(best_label):
                    return True, f"{prefix}.scores_best={best_label}:{best_score}"
            except Exception:
                pass

    # candidate_labels만으로는 너무 넓어서 원칙적으로 hand 확정하지 않음.
    # 단, candidate_labels가 전부 상생의손 계열뿐인 특수한 경우만 hand로 봄.
    labels = data.get("candidate_labels") if isinstance(data, dict) else None
    if isinstance(labels, list) and labels:
        hand_labels = [x for x in labels if _is_hand_landmark_text(x)]
        if len(hand_labels) == len(labels):
            return True, f"candidate_labels_all_hand={hand_labels}"

    return False, "no_hand_landmark_detected"
