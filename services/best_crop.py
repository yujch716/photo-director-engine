"""1x → 2x 최적 크롭 선택 파이프라인 (전체 실행 오케스트레이션).

입력 폴더에서 original_1x / original_2x / result.json을 찾아
후보 생성 → 좌표 필터 → DINOv2 매칭 → best 선택까지 수행하고 report.json을 쓴다.

- models.dinov2          : DINO 임베딩 / reference bank 매칭 (모델 호출)
- services.crop_candidates : 후보 생성 + 좌표 필터
- services.crop_result     : result.json 해석 (bbox 로드 + hand bank 판정)

CLI 진입점은 저장소 루트의 run_best_crop.py 이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from models import dinov2
from services.crop_candidates import (
    Candidate,
    filter_candidates_by_boxes,
    generate_candidates,
    save_candidates,
)
from services.crop_result import load_result_boxes, result_json_uses_hand_bank


# ---------------------------------------------------------------------------
# 입력 파일 탐색
# ---------------------------------------------------------------------------

def find_file(folder: Path, names: list[str], patterns: list[str]) -> Path:
    for name in names:
        p = folder / name
        if p.exists():
            return p

    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"파일을 찾지 못함: folder={folder}, names={names}, patterns={patterns}")


def find_json(folder: Path) -> Path:
    preferred = [
        folder / "result.json",
        folder / "results.json",
        folder / "result" / "result.json",
        folder / "result" / "results.json",
    ]

    for p in preferred:
        if p.exists():
            return p

    matches = sorted(folder.rglob("*result*.json"))
    if matches:
        return matches[0]

    matches = sorted(folder.glob("*.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"result json을 찾지 못함: {folder}")


def _assign_embeddings(candidates: list[Candidate], batch_size: int):
    cls_list, patch_list = dinov2.embed_images_batch([c.image for c in candidates], batch_size=batch_size)

    for c, cls, patches in zip(candidates, cls_list, patch_list):
        c.cls = cls
        c.patches = patches


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def select_best_crop(
    input_dir: Path,
    output_dir: Path,
    general_embedding_dir: Path,
    hand_embedding_dir: Path,
    embedding_dir: Path | None = None,
    zoom_ratio: float = 2.0,
    cols: int = 8,
    rows: int = 6,
    contain_thres: float = 0.90,
    edge_margin_ratio: float = 0.0,
    selected_require: str = "all",
    fallback_topk: int = 12,
    dino_batch_size: int = 16,
    patch_topk: int = 20,
    save_all_candidates: bool = False,
) -> dict[str, Any]:
    """입력 폴더 하나에 대해 best 크롭을 골라 저장하고 report dict를 반환한다."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    img_1x_path = find_file(input_dir, ["original_1x.jpg", "original_1x.png", "1x.jpg", "1x.png"], ["*original*1x*.jpg", "*original*1x*.png", "*1x*.jpg", "*1x*.png"])
    img_2x_path = find_file(input_dir, ["original_2x.jpg", "original_2x.png", "2x.jpg", "2x.png"], ["*original*2x*.jpg", "*original*2x*.png", "*2x*.jpg", "*2x*.png"])
    json_path = find_json(input_dir)

    prefix = input_dir.name

    img_1x = Image.open(img_1x_path).convert("RGB")
    img_2x = Image.open(img_2x_path).convert("RGB")

    print(f"[INFO] input_dir  = {input_dir}")
    print(f"[INFO] output_dir = {output_dir}")
    print(f"[INFO] 1x         = {img_1x_path} {img_1x.size}")
    print(f"[INFO] 2x         = {img_2x_path} {img_2x.size}")
    print(f"[INFO] result     = {json_path}")

    boxes = load_result_boxes(json_path, img_1x.size)

    candidates = generate_candidates(
        img_1x=img_1x,
        img_2x_size=img_2x.size,
        zoom_ratio=zoom_ratio,
        cols=cols,
        rows=rows,
    )

    filtered, filter_status = filter_candidates_by_boxes(
        candidates=candidates,
        boxes=boxes,
        contain_thres=contain_thres,
        edge_margin_ratio=edge_margin_ratio,
        selected_require=selected_require,
        fallback_topk=fallback_topk,
    )

    if not filtered:
        report = {
            "status": "no_candidate_after_coordinate_filter",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "boxes": boxes,
            "candidate_total": len(candidates),
            "candidate_after_filter": 0,
        }

        with open(output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("[WARN] 통과 후보 없음")
        return report

    use_hand_bank, bank_reason = result_json_uses_hand_bank(json_path)

    if embedding_dir is not None:
        bank_dir = embedding_dir.resolve()
        bank_type = "manual_embedding_dir"
        bank_reason = f"manual_override:{bank_dir}"
    elif use_hand_bank:
        bank_dir = hand_embedding_dir.resolve()
        bank_type = "hand_dinov2imbedding"
    else:
        bank_dir = general_embedding_dir.resolve()
        bank_type = "dinov2imbedding"

    print(f"[INFO] selected reference bank = {bank_type}")
    print(f"[INFO] bank reason             = {bank_reason}")
    print(f"[INFO] bank dir                = {bank_dir}")

    bank = dinov2.load_reference_bank(bank_dir)

    _assign_embeddings(filtered, batch_size=dino_batch_size)

    best = None

    for c in filtered:
        idx, ref_path, sim = dinov2.search_bank(c.cls, c.patches, bank, topk=patch_topk)
        c.best_ref_index = idx
        c.best_ref = ref_path
        c.best_sim = sim

        if best is None or c.best_sim > best.best_sim:
            best = c

    if best is None:
        raise RuntimeError("DINO reference match 실패")

    best_path = output_dir / "best.jpg"
    best.image.save(best_path, quality=95)

    if save_all_candidates:
        save_candidates(output_dir, prefix, filtered)

    report = {
        "status": "ok",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_1x": str(img_1x_path),
        "image_2x": str(img_2x_path),
        "json": str(json_path),
        "boxes": boxes,
        "selected_reference_bank": bank_type,
        "selected_reference_bank_reason": bank_reason,
        "selected_reference_bank_dir": str(bank_dir),
        "zoom_ratio": zoom_ratio,
        "candidate_total": len(candidates),
        "candidate_after_filter": len(filtered),
        "filter_status": filter_status,
        "contain_thres": contain_thres,
        "edge_margin_ratio": edge_margin_ratio,
        "selected_require": selected_require,
        "best": str(best_path),
        "best_candidate_index": best.index,
        "best_source_box_in_1x": list(best.source_box),
        "best_coord_score": best.coord_score,
        "best_coord_results": best.coord_results,
        "best_reference": best.best_ref,
        "best_reference_index": best.best_ref_index,
        "best_similarity_patch": best.best_sim,
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] best saved: {best_path}")
    print(f"[INFO] report saved: {output_dir / 'report.json'}")
    print(f"[INFO] best_candidate_index={best.index}, source_box={best.source_box}, coord_score={best.coord_score:.4f}, patch_sim={best.best_sim:.4f}")

    return report
