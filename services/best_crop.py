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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from models import dinov2
from models.landmark_clip import ensure_loaded as clip_ensure_loaded
from models.landmark_clip import tag_pil
from services.crop_candidates import (
    Candidate,
    filter_candidates_by_boxes,
    generate_candidates,
    save_candidates,
)
from services.crop_result import load_result_boxes
from services.tag_match import REQUIRED_DEFAULT, compute_tag_bonus, required_filter_indices

# 단일 patch 임베딩 DB는 항상 config/ 아래에서 읽는다.
#   config/dinov2_patch_embeddings.npy   (N, P, D)
#   config/metadata.json                 (길이 N, 각 항목의 "path"를 라벨로 사용)
DEFAULT_BANK_DIR = Path("config")


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


def _load_capture_tags(json_path: Path) -> dict[str, Any] | None:
    """result.json의 최상위 "tags"(schema.yaml 형식)를 읽는다. 없으면 None."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    tags = data.get("tags") if isinstance(data, dict) else None
    return tags if isinstance(tags, dict) else None


def _tag_candidates(candidates: list[Candidate], scene_person_count: Any) -> None:
    """솎기 후 남은 후보 각각을 CLIP으로 태깅(schema.yaml 형식)해 c.tags에 저장.

    - CLIP은 person_count를 못 세므로, 장면(타깃) 인물 수를 각 후보 태그에 주입한다.
    - 후보 수가 적어 스레드풀로 병렬 태깅한다(모델은 미리 로드해 레이스 방지).
    - 개별 태깅 실패 시 그 후보의 tags는 None(→ 그 후보는 patch-only로 매칭).
    """
    if not candidates:
        return

    clip_ensure_loaded()  # 병렬 진입 전 CLIP/스키마 1회 로드

    def work(c: Candidate) -> None:
        try:
            tags = tag_pil(c.image)
        except Exception:
            tags = None
        if isinstance(tags, dict):
            tags["person_count"] = scene_person_count
        c.tags = tags

    if len(candidates) == 1:
        work(candidates[0])
        return

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as ex:
        list(ex.map(work, candidates))


def _assign_embeddings(candidates: list[Candidate], batch_size: int):
    patch_list = dinov2.embed_images_batch([c.image for c in candidates], batch_size=batch_size)

    for c, patches in zip(candidates, patch_list):
        c.patches = patches


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def select_best_crop(
    input_dir: Path,
    output_dir: Path,
    bank_dir: Path | None = None,
    zoom_ratio: float = 2.0,
    cols: int = 8,
    rows: int = 6,
    contain_thres: float = 0.90,
    edge_margin_ratio: float = 0.0,
    selected_require: str = "all",
    fallback_topk: int = 12,
    dino_batch_size: int = 16,
    dino_chunk_size: int = 64,
    dino_shortlist_k: int = 200,
    tag_weight: float = 0.3,
    required_tags: list[str] | tuple[str, ...] | None = None,
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

    # 분기 없이 항상 단일 DB(config/)만 사용한다.
    bank_dir = (bank_dir or DEFAULT_BANK_DIR).resolve()
    print(f"[INFO] reference bank dir = {bank_dir}")

    bank = dinov2.load_reference_bank(bank_dir)

    # 태그 가중을 쓸지: DB에 태그가 있고 tag_weight>0일 때만.
    # (장면 person_count는 후보 태그에 주입해 인물 수 근접도까지 반영)
    capture_tags = _load_capture_tags(json_path)
    scene_person_count = capture_tags.get("person_count") if capture_tags else None
    use_tags = tag_weight > 0 and bank.meta is not None
    print(
        f"[INFO] tag weighting = {'on' if use_tags else 'off'} "
        f"(tag_weight={tag_weight}, bank_tags={'yes' if bank.meta else 'no'})"
    )

    # 필수 태그 하드 필터(top-k 전). 장면 태그 기준으로 1회만 계산해 모든 후보에 공유.
    # 통과 0개면 필터를 무시하고 전체를 쓴다(0개 방지 안전장치).
    req_fields = REQUIRED_DEFAULT if required_tags is None else tuple(required_tags)
    allowed_idx = None
    if req_fields and bank.meta is not None and capture_tags is not None:
        idxs = required_filter_indices(capture_tags, bank.meta, req_fields)
        if idxs is not None:
            if len(idxs) > 0:
                allowed_idx = idxs
                print(f"[INFO] 필수 태그 필터 {list(req_fields)}: {len(bank.meta)} -> {len(idxs)}")
            else:
                print(f"[WARN] 필수 태그 필터가 0개 남김 → 필터 무시(전체 사용): {list(req_fields)}")

    # 솎기 후 남은 후보 각각을 CLIP으로 태깅(병렬). 그 후보 태그로 DB와 매칭한다.
    if use_tags:
        _tag_candidates(filtered, scene_person_count)

    _assign_embeddings(filtered, batch_size=dino_batch_size)

    best = None

    for c in filtered:
        # 후보별 태그 일치도. 태그가 없으면(태깅 실패 등) None → 그 후보는 patch-only.
        tag_bonus = compute_tag_bonus(c.tags, bank.meta) if use_tags else None
        idx, ref_path, sim = dinov2.search_bank(
            c.patches,
            bank,
            chunk_size=dino_chunk_size,
            tag_bonus=tag_bonus,
            tag_weight=tag_weight,
            shortlist_k=dino_shortlist_k,
            allowed_indices=allowed_idx,
        )
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
        "reference_bank_dir": str(bank_dir),
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
        # 태그 가중이 켜져 있으면 patch_sim + tag_weight*tag_bonus의 결합 점수
        "best_score": best.best_sim,
        "tag_weighting": use_tags,
        "tag_weight": tag_weight,
        "required_tags": list(req_fields),
        "required_filter_kept": (int(len(allowed_idx)) if allowed_idx is not None else None),
        "capture_tags": capture_tags,
        "best_candidate_tags": best.tags,
        "best_reference_tags": (
            bank.meta[best.best_ref_index]
            if bank.meta and best.best_ref_index is not None and 0 <= best.best_ref_index < len(bank.meta)
            else None
        ),
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] best saved: {best_path}")
    print(f"[INFO] report saved: {output_dir / 'report.json'}")
    print(f"[INFO] best_candidate_index={best.index}, source_box={best.source_box}, coord_score={best.coord_score:.4f}, best_score={best.best_sim:.4f}")

    return report
