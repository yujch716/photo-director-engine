"""DINOv2 임베딩 + reference bank 매칭 (patch-only).

best_crop 파이프라인에서 쓰는 DINO 관련 "모델 호출"만 담는다.
- embed_images_batch(images, batch_size): 이미지 리스트 -> patch 임베딩 리스트
- load_reference_bank(embedding_dir): 사전 임베딩된 reference bank(patch) 로드
- search_bank(patches, bank): 후보 patch 임베딩을 bank 전체와 비교해 best 매칭 반환

CLS 임베딩은 사용하지 않는다. 예전엔 CLS로 top-k 사전필터를 했으나,
지금은 patch 유사도만으로 매칭하며 reference를 청크로 나눠 전체 순회한다.

모델은 lazy singleton으로 로드한다(yolo.py / landmark_clip.py와 동일 패턴).
백엔드는 transformers DINOv2(facebook/dinov2-base).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_DINO_MODEL = "facebook/dinov2-base"

_dino_processor: Any | None = None
_dino_model: Any | None = None
_dino_device: Any | None = None


# ---------------------------------------------------------------------------
# 벡터 정규화 유틸
# ---------------------------------------------------------------------------

def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass
class ReferenceBank:
    root: Path
    patches: np.ndarray
    paths: list[str]
    meta: list[dict] | None = None  # metadata.json 태그 레코드 (없으면 None)
    coarse: np.ndarray | None = None  # (N, D) 평균풀링 코스 벡터 (2단계 검색 1차 추림용)


# ---------------------------------------------------------------------------
# DINOv2 로더 (lazy singleton)
# ---------------------------------------------------------------------------

def get_dino():
    global _dino_processor, _dino_model, _dino_device

    if _dino_model is not None:
        return _dino_processor, _dino_model, _dino_device

    import torch
    from transformers import AutoImageProcessor, AutoModel

    _dino_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] loading DINOv2: {DEFAULT_DINO_MODEL}, device={_dino_device}")
    _dino_processor = AutoImageProcessor.from_pretrained(DEFAULT_DINO_MODEL)
    _dino_model = AutoModel.from_pretrained(DEFAULT_DINO_MODEL).to(_dino_device)
    _dino_model.eval()

    return _dino_processor, _dino_model, _dino_device


def embed_images_batch(images: list[Image.Image], batch_size: int) -> list[np.ndarray]:
    """이미지 리스트 -> patch 임베딩 리스트. 각 원소 shape (P, D), L2 정규화됨."""
    import torch

    if not images:
        return []

    processor, model, device = get_dino()

    patch_out = []

    batch_size = max(1, int(batch_size))

    for start in range(0, len(images), batch_size):
        batch = [img.convert("RGB") for img in images[start:start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state

        # CLS(index 0)는 버리고 patch 토큰만 사용
        patches = hidden[:, 1:, :]
        patch_np = normalize_rows(patches.detach().cpu().float().numpy())

        for p in patch_np:
            patch_out.append(p.astype(np.float32))

    return patch_out


# ---------------------------------------------------------------------------
# reference bank 로딩
# ---------------------------------------------------------------------------

def _find_file(folder: Path, names: list[str], patterns: list[str]) -> Path:
    for name in names:
        p = folder / name
        if p.exists():
            return p

    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"파일을 찾지 못함: folder={folder}, names={names}, patterns={patterns}")


def _load_paths_and_meta(path: Path | None, n: int) -> tuple[list[str], list[dict] | None]:
    """경로 라벨(list[str])과 태그 레코드(list[dict] | None)를 함께 로드한다.

    metadata.json처럼 dict 리스트면 각 레코드를 meta로 보존하고, "path"를 라벨로 쓴다.
    형식이 안 맞거나 길이가 N과 다르면 meta는 None으로 두고 라벨은 idx로 fallback.
    """
    fallback = [f"idx{i}" for i in range(n)]

    if path is None or not path.exists():
        return fallback, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return fallback, None

    meta: list[dict] | None = None

    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        # metadata.json (태그 레코드 리스트)
        meta = obj
        paths = [str(x.get("path", x.get("id", f"idx{i}"))) for i, x in enumerate(obj)]
    elif isinstance(obj, list):
        paths = [str(x) for x in obj]
    elif isinstance(obj, dict):
        paths = None
        for key in ["paths", "image_paths", "files", "images"]:
            if key in obj and isinstance(obj[key], list):
                paths = [str(x) for x in obj[key]]
                break
        if paths is None:
            paths = list(fallback)
    else:
        paths = list(fallback)

    if len(paths) != n:
        # 정렬이 어긋나면 라벨/메타 모두 신뢰할 수 없으므로 안전하게 버린다.
        return fallback, None
    if meta is not None and len(meta) != n:
        meta = None

    return paths, meta


def _build_coarse(patches: np.ndarray) -> np.ndarray:
    """patch 임베딩 (N,P,D)를 이미지당 벡터 (N,D)로 평균풀링한다.

    후보와 동일한 방식으로: 패치별 L2정규화 → P축 평균 → 다시 L2정규화.
    N이 커도 청크로 나눠 읽어 메모리 폭주를 막는다.
    """
    n, _p, d = patches.shape
    out = np.empty((n, d), dtype=np.float32)
    chunk = 256
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = normalize_rows(np.asarray(patches[s:e], dtype=np.float32))  # (k,P,D)
        out[s:e] = blk.mean(axis=1)
    return normalize_rows(out).astype(np.float32)


def _load_or_build_coarse(patch_path: Path, patches: np.ndarray) -> np.ndarray:
    """코스 벡터를 캐시(<patch_dir>/coarse.npy)에서 읽고, 없거나 낡았으면 새로 만들어 저장.

    patch 파일보다 캐시가 오래됐으면(mtime) 재생성한다.
    (patch .npy를 교체하면 자동으로 다시 계산됨)
    """
    n, _p, d = patches.shape
    cache = patch_path.parent / "coarse.npy"

    if cache.exists() and cache.stat().st_mtime >= patch_path.stat().st_mtime:
        try:
            c = np.load(cache)
            if c.shape == (n, d):
                print(f"[INFO] coarse cache 사용: {cache}")
                return c.astype(np.float32)
        except Exception:
            pass

    print(f"[INFO] coarse 벡터 생성 중... (N={n})")
    coarse = _build_coarse(patches)
    try:
        np.save(cache, coarse)
        print(f"[INFO] coarse cache 저장: {cache}")
    except Exception as e:
        print(f"[WARN] coarse cache 저장 실패(무시): {e}")
    return coarse


def load_reference_bank(embedding_dir: Path) -> ReferenceBank:
    """단일 patch 임베딩 DB를 로드한다 (CLS 미사용).

    기대 파일(config/ 아래):
      dinov2_patch_embeddings.npy   shape (N, P, D)   patch 임베딩
      metadata.json                 길이 N            각 항목의 "path"를 라벨로 사용
    """
    patch_path = _find_file(
        embedding_dir,
        ["dinov2_patch_embeddings.npy", "patch_embedding.npy"],
        ["*patch*embedding*.npy"],
    )

    path_json = None
    for name in ["metadata.json", "tagging.json", "dinov2_image_paths.json", "image_paths.json"]:
        p = embedding_dir / name
        if p.exists():
            path_json = p
            break

    if path_json is None:
        matches = sorted(embedding_dir.glob("*path*.json"))
        path_json = matches[0] if matches else None

    patches = np.load(patch_path, mmap_mode="r")

    if patches.ndim != 3:
        raise ValueError(f"Patch embedding shape 이상함: {patches.shape}")

    paths, meta = _load_paths_and_meta(path_json, patches.shape[0])
    coarse = _load_or_build_coarse(patch_path, patches)

    print(
        f"[INFO] reference bank(patch-only): patches={patches.shape}, "
        f"n={patches.shape[0]}, tags={'yes' if meta else 'no'}, coarse={coarse.shape}"
    )

    return ReferenceBank(
        root=embedding_dir,
        patches=patches,
        paths=paths,
        meta=meta,
        coarse=coarse,
    )


# ---------------------------------------------------------------------------
# reference bank 검색 (OOM 방지)
# ---------------------------------------------------------------------------

def _chunk_scores(cand: np.ndarray, ref_patches_np: np.ndarray) -> np.ndarray:
    """후보 patch(cand: (P,D), 정규화됨)와 reference 청크(refs: (K,P,D))의 양방향
    patch 유사도 점수 (K,)를 반환한다. CUDA가 있으면 GPU로, 없으면 numpy로 계산."""
    import torch
    import torch.nn.functional as F

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cand_t = torch.as_tensor(cand, dtype=torch.float32, device=device)
        refs_t = torch.as_tensor(ref_patches_np, dtype=torch.float32, device=device)

        cand_t = F.normalize(cand_t, dim=-1).half()
        refs_t = F.normalize(refs_t, dim=-1).half()

        # cand: (P,D), refs: (K,P,D)
        sims = torch.einsum("pd,kqd->kpq", cand_t, refs_t)
        cand_to_ref = sims.max(dim=2).values.mean(dim=1)
        ref_to_cand = sims.max(dim=1).values.mean(dim=1)
        scores = ((cand_to_ref + ref_to_cand) * 0.5).float().cpu().numpy()

        del cand_t, refs_t, sims, cand_to_ref, ref_to_cand
        torch.cuda.empty_cache()
        return scores

    refs = normalize_rows(ref_patches_np)
    sims = np.einsum("pd,kqd->kpq", cand, refs, optimize=True)
    return (sims.max(axis=2).mean(axis=1) + sims.max(axis=1).mean(axis=1)) * 0.5


def _shortlist_indices(
    cand: np.ndarray,
    bank: ReferenceBank,
    base: np.ndarray,
    shortlist_k: int,
    use_tags: bool,
    tag_bonus: np.ndarray | None,
    tag_weight: float,
) -> np.ndarray:
    """1단계(코스): base 후보군에서 이미지당 벡터 코사인(+태그 가중)으로 상위 shortlist_k개만 추린다.

    coarse 벡터가 없거나 shortlist_k가 base 크기 이상이면 base를 그대로 반환(추림 없음).
    top-k라 결과가 0개가 되는 일은 없다.
    """
    m = len(base)
    if bank.coarse is None or shortlist_k <= 0 or shortlist_k >= m:
        return base

    cand_coarse = normalize_vec(cand.mean(axis=0))          # (D,)
    coarse_scores = bank.coarse[base] @ cand_coarse         # (m,) 코사인, 매우 쌈
    if use_tags:
        coarse_scores = coarse_scores + float(tag_weight) * np.asarray(tag_bonus[base], dtype=np.float32)

    k = min(int(shortlist_k), m)
    loc = np.argpartition(-coarse_scores, k - 1)[:k]
    return base[loc]


def search_bank(
    candidate_patches: np.ndarray,
    bank: ReferenceBank,
    chunk_size: int = 64,
    tag_bonus: np.ndarray | None = None,
    tag_weight: float = 0.0,
    shortlist_k: int = 200,
    allowed_indices: np.ndarray | None = None,
):
    """후보 patch 임베딩을 reference bank와 비교해 best 1개를 반환한다 (2단계).

    0단계(필수필터): allowed_indices가 주어지면 그 후보군 안에서만 검색한다.
                     (호출부에서 필수 태그가 일치하는 인덱스만 미리 골라 넘긴다)
    1단계(코스): 평균풀링 벡터 코사인(+태그 가중)으로 상위 shortlist_k개만 추림.
                 top-k라 0개가 되지 않는다.
    2단계(정밀): 추려진 후보에 대해서만 patch 레벨 유사도(+태그 가중)로 best 선택.

    태그 가중치:
      tag_bonus: reference별 태그 일치도 (N,) [0,1]. None이면 patch-only.
      tag_weight: 최종 점수 = patch_sim + tag_weight * tag_bonus (두 단계 모두 적용).
      (tag_weight=0 또는 tag_bonus=None이면 순수 임베딩 매칭)
    shortlist_k: 정밀 비교할 후보 수. base 이상이면 추림 없이 전체 정밀 비교.

    반환: (idx, path, score) — score는 태그 가중이 포함된 최종 점수.
    """
    n = int(bank.patches.shape[0])
    if n == 0:
        raise RuntimeError("reference bank가 비어 있음")

    use_tags = tag_bonus is not None and bool(tag_weight) and len(tag_bonus) == n

    cand = normalize_rows(np.asarray(candidate_patches, dtype=np.float32))  # (P, D)

    # 0단계: 필수필터 통과 후보군(없으면 전체)
    base = np.arange(n) if allowed_indices is None else np.asarray(allowed_indices, dtype=np.int64)
    if len(base) == 0:
        base = np.arange(n)

    # 1단계: 코스 추림
    idxs = _shortlist_indices(cand, bank, base, shortlist_k, use_tags, tag_bonus, tag_weight)

    # 2단계: 추려진 인덱스에만 patch 정밀 비교
    chunk_size = max(1, int(chunk_size))
    best_idx = int(idxs[0])
    best_score = -1.0

    for start in range(0, len(idxs), chunk_size):
        sub = idxs[start:start + chunk_size]
        # mmap fancy-index → 해당 후보 patch만 읽는다.
        ref_patches_np = np.asarray(bank.patches[sub], dtype=np.float32)
        scores = _chunk_scores(cand, ref_patches_np)

        if use_tags:
            scores = scores + float(tag_weight) * np.asarray(tag_bonus[sub], dtype=np.float32)

        local = int(np.argmax(scores))
        if float(scores[local]) > best_score:
            best_score = float(scores[local])
            best_idx = int(sub[local])

    return best_idx, bank.paths[best_idx], best_score
