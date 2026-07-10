"""capture json(report.json) 해석.

- load_result_boxes(...): 대상 객체들의 bbox_pixel(1x 좌표) 목록을 뽑는다.
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
