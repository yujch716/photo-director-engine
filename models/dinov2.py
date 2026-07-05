"""DINOv2 임베딩 + reference bank 매칭.

best_crop 파이프라인에서 쓰는 DINO 관련 "모델 호출"만 담는다.
- embed_images_batch(images, batch_size): 이미지 리스트 -> (cls, patch) 임베딩
- load_reference_bank(embedding_dir): 사전 임베딩된 reference bank 로드
- search_bank(cls, patches, bank, topk): 후보 임베딩과 bank를 비교해 best 매칭 반환

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
    cls: np.ndarray
    patches: np.ndarray
    paths: list[str]


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


def embed_images_batch(images: list[Image.Image], batch_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import torch

    if not images:
        return [], []

    processor, model, device = get_dino()

    cls_out = []
    patch_out = []

    batch_size = max(1, int(batch_size))

    for start in range(0, len(images), batch_size):
        batch = [img.convert("RGB") for img in images[start:start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state

        cls = hidden[:, 0, :]
        patches = hidden[:, 1:, :]

        cls_np = normalize_rows(cls.detach().cpu().float().numpy())
        patch_np = normalize_rows(patches.detach().cpu().float().numpy())

        for c, p in zip(cls_np, patch_np):
            cls_out.append(c.astype(np.float32))
            patch_out.append(p.astype(np.float32))

    return cls_out, patch_out


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


def _load_image_paths(path: Path | None, n: int) -> list[str]:
    if path is None or not path.exists():
        return [f"idx{i}" for i in range(n)]

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            paths = [str(x) for x in obj]
        elif isinstance(obj, dict):
            paths = None
            for key in ["paths", "image_paths", "files", "images"]:
                if key in obj and isinstance(obj[key], list):
                    paths = [str(x) for x in obj[key]]
                    break
            if paths is None:
                paths = [f"idx{i}" for i in range(n)]
        else:
            paths = [f"idx{i}" for i in range(n)]

    except Exception:
        paths = [f"idx{i}" for i in range(n)]

    if len(paths) != n:
        paths = [f"idx{i}" for i in range(n)]

    return paths


def load_reference_bank(embedding_dir: Path) -> ReferenceBank:
    cls_path = _find_file(
        embedding_dir,
        ["dinov2_cls_embeddings.npy"],
        ["*cls*embedding*.npy"],
    )

    patch_path = _find_file(
        embedding_dir,
        ["dinov2_patch_embeddings.npy"],
        ["*patch*embedding*.npy"],
    )

    path_json = None
    for name in ["dinov2_image_paths.json", "image_paths.json"]:
        p = embedding_dir / name
        if p.exists():
            path_json = p
            break

    if path_json is None:
        matches = sorted(embedding_dir.glob("*path*.json"))
        path_json = matches[0] if matches else None

    cls = np.load(cls_path)
    patches = np.load(patch_path, mmap_mode="r")

    if cls.ndim != 2:
        raise ValueError(f"CLS embedding shape 이상함: {cls.shape}")

    if patches.ndim != 3:
        raise ValueError(f"Patch embedding shape 이상함: {patches.shape}")

    if cls.shape[0] != patches.shape[0] or cls.shape[-1] != patches.shape[-1]:
        raise ValueError(f"CLS/PATCH shape mismatch: cls={cls.shape}, patches={patches.shape}")

    cls = normalize_rows(cls).astype(np.float32)
    paths = _load_image_paths(path_json, cls.shape[0])

    print(f"[INFO] reference bank: cls={cls.shape}, patches={patches.shape}")

    return ReferenceBank(
        root=embedding_dir,
        cls=cls,
        patches=patches,
        paths=paths,
    )


# ---------------------------------------------------------------------------
# reference bank 검색 (OOM 방지)
# ---------------------------------------------------------------------------

def search_bank(candidate_cls: np.ndarray, candidate_patches: np.ndarray, bank: ReferenceBank, topk: int):
    """
    OOM 방지 버전.

    1. CLS embedding은 CPU에서 top-k reference만 고름
    2. 선택된 top-k patch만 GPU로 올림
    3. candidate patch와 top-k reference patch만 GPU에서 비교

    이렇게 하면 (6841,256,768) 전체를 GPU에 올리지 않고,
    예: topk=20이면 (20,256,768)만 GPU에 올린다.
    """
    import torch
    import torch.nn.functional as F

    cand_cls = normalize_vec(candidate_cls)

    # CLS top-k는 CPU에서 계산. 매우 가벼움.
    cls_sims = bank.cls @ cand_cls
    n = cls_sims.shape[0]

    if n == 0:
        raise RuntimeError("reference bank가 비어 있음")

    k = min(int(topk), int(n))

    if k <= 0:
        k = n

    if k >= n:
        idxs = np.arange(n)
        idxs = idxs[np.argsort(-cls_sims[idxs])]
    else:
        idxs = np.argpartition(-cls_sims, k - 1)[:k]
        idxs = idxs[np.argsort(-cls_sims[idxs])]

    # top-k patch만 읽음. bank.patches는 mmap이라 필요한 부분만 가져옴.
    ref_patches_np = np.asarray(bank.patches[idxs], dtype=np.float32)
    cand_patches_np = np.asarray(candidate_patches, dtype=np.float32)

    if torch.cuda.is_available():
        device = torch.device("cuda")

        # top-k만 GPU로 이동
        cand = torch.as_tensor(cand_patches_np, dtype=torch.float32, device=device)
        refs = torch.as_tensor(ref_patches_np, dtype=torch.float32, device=device)

        cand = F.normalize(cand, dim=-1).half()
        refs = F.normalize(refs, dim=-1).half()

        # cand: (P,D), refs: (K,P,D)
        sims = torch.einsum("pd,kqd->kpq", cand, refs)

        cand_to_ref = sims.max(dim=2).values.mean(dim=1)
        ref_to_cand = sims.max(dim=1).values.mean(dim=1)

        scores = (cand_to_ref + ref_to_cand) * 0.5

        local = int(torch.argmax(scores).item())
        score = float(scores[local].float().item())

        # 메모리 즉시 해제 힌트
        del cand, refs, sims, cand_to_ref, ref_to_cand, scores
        torch.cuda.empty_cache()

    else:
        cand = normalize_rows(cand_patches_np)
        refs = normalize_rows(ref_patches_np)

        sims = np.einsum("pd,kqd->kpq", cand, refs, optimize=True)
        scores = (sims.max(axis=2).mean(axis=1) + sims.max(axis=1).mean(axis=1)) * 0.5

        local = int(np.argmax(scores))
        score = float(scores[local])

    idx = int(idxs[local])
    return idx, bank.paths[idx], score
