from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_dino_processor = None
_dino_model = None
_dino_device = None
_GPU_BANK_CACHE: dict[str, Any] = {}


@dataclass
class Candidate:
    index: int
    image: Image.Image
    source_box: tuple[int, int, int, int]  # 1x 원본 좌표계
    coord_score: float = 0.0
    coord_results: list[dict[str, Any]] | None = None
    cls: np.ndarray | None = None
    patches: np.ndarray | None = None
    best_ref: str | None = None
    best_ref_index: int | None = None
    best_sim: float = -1.0


@dataclass
class ReferenceBank:
    root: Path
    cls: np.ndarray
    patches: np.ndarray
    paths: list[str]


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


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



def generate_48_candidates(
    img_1x: Image.Image,
    img_2x_size: tuple[int, int],
    zoom_ratio: float,
    cols: int,
    rows: int,
) -> list[Candidate]:
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


def edge_ok(candidate_box: tuple[int, int, int, int], target_box: list[float], edge_margin_ratio: float) -> bool:
    if edge_margin_ratio <= 0:
        return True

    cx1, cy1, cx2, cy2 = [float(x) for x in candidate_box]
    tx1, ty1, tx2, ty2 = [float(x) for x in target_box]

    cw = cx2 - cx1
    ch = cy2 - cy1

    mx = cw * edge_margin_ratio
    my = ch * edge_margin_ratio

    return (
        tx1 > cx1 + mx and
        ty1 > cy1 + my and
        tx2 < cx2 - mx and
        ty2 < cy2 - my
    )


def filter_candidates_by_boxes(
    candidates: list[Candidate],
    boxes: list[dict[str, Any]],
    contain_thres: float,
    edge_margin_ratio: float,
    selected_require: str,
    fallback_topk: int,
) -> tuple[list[Candidate], str]:
    """
    result.json의 bbox_pixel 좌표로 48개 후보를 솎는다.

    사람(person)의 경우:
    - 사람 전체 bbox가 다 들어올 필요 없음
    - 하반신은 어느 정도 잘려도 허용
    - bbox의 위쪽 60% 정도, 즉 얼굴/상체/중심부가 후보 안에 들어오면 통과 가능

    일반 객체의 경우:
    - 기존처럼 bbox 전체가 후보 안에 들어와야 함
    """
    if not boxes:
        print("[WARN] result.json에서 bbox를 못 찾음 → 48개 전체 사용")
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


def get_dino():
    global _dino_processor, _dino_model, _dino_device

    if _dino_model is not None:
        return _dino_processor, _dino_model, _dino_device

    import torch
    from transformers import AutoImageProcessor, AutoModel

    model_name = "facebook/dinov2-base"
    _dino_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] loading DINOv2: {model_name}, device={_dino_device}")
    _dino_processor = AutoImageProcessor.from_pretrained(model_name)
    _dino_model = AutoModel.from_pretrained(model_name).to(_dino_device)
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


def load_image_paths(path: Path | None, n: int) -> list[str]:
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
    cls_path = find_file(
        embedding_dir,
        ["dinov2_cls_embeddings.npy"],
        ["*cls*embedding*.npy"],
    )

    patch_path = find_file(
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
    paths = load_image_paths(path_json, cls.shape[0])

    print(f"[INFO] reference bank: cls={cls.shape}, patches={patches.shape}")

    return ReferenceBank(
        root=embedding_dir,
        cls=cls,
        patches=patches,
        paths=paths,
    )


def get_bank_gpu(bank: ReferenceBank):
    """
    더 이상 reference patch 전체를 GPU에 올리지 않음.
    전체 bank를 올리면 OOM이 나므로 search_bank에서 top-k patch만 GPU로 올린다.
    """
    return None


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


def assign_embeddings(candidates: list[Candidate], batch_size: int):
    cls_list, patch_list = embed_images_batch([c.image for c in candidates], batch_size=batch_size)

    for c, cls, patches in zip(candidates, cls_list, patch_list):
        c.cls = cls
        c.patches = patches


def save_candidates(output_dir: Path, prefix: str, candidates: list[Candidate]):
    cand_dir = output_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        name = f"{prefix}_candidate{c.index:02d}_coord{c.coord_score:.4f}_ref{c.best_sim:.4f}.jpg"
        c.image.save(cand_dir / name, quality=95)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/drone-data/20260702_222942_010"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/drone-data/20260702_222942_010_result"))
    parser.add_argument("--general-embedding-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/imbeddingdata/dinov2imbedding"))
    parser.add_argument("--hand-embedding-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/imbeddingdata/hand_dinov2imbedding"))
    parser.add_argument("--embedding-dir", type=Path, default=None)  # 수동 override용

    parser.add_argument("--zoom-ratio", type=float, default=2.0)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)

    parser.add_argument("--contain-thres", type=float, default=0.90)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.0)
    parser.add_argument("--selected-require", choices=["all", "any"], default="all")
    parser.add_argument("--fallback-topk", type=int, default=12)

    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--patch-topk", type=int, default=20)

    parser.add_argument("--save-candidates", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
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

    candidates = generate_48_candidates(
        img_1x=img_1x,
        img_2x_size=img_2x.size,
        zoom_ratio=args.zoom_ratio,
        cols=args.cols,
        rows=args.rows,
    )

    filtered, filter_status = filter_candidates_by_boxes(
        candidates=candidates,
        boxes=boxes,
        contain_thres=args.contain_thres,
        edge_margin_ratio=args.edge_margin_ratio,
        selected_require=args.selected_require,
        fallback_topk=args.fallback_topk,
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
        return

    use_hand_bank, bank_reason = result_json_uses_hand_bank(json_path)

    if args.embedding_dir is not None:
        bank_dir = args.embedding_dir.resolve()
        bank_type = "manual_embedding_dir"
        bank_reason = f"manual_override:{bank_dir}"
    elif use_hand_bank:
        bank_dir = args.hand_embedding_dir.resolve()
        bank_type = "hand_dinov2imbedding"
    else:
        bank_dir = args.general_embedding_dir.resolve()
        bank_type = "dinov2imbedding"

    print(f"[INFO] selected reference bank = {bank_type}")
    print(f"[INFO] bank reason             = {bank_reason}")
    print(f"[INFO] bank dir                = {bank_dir}")

    bank = load_reference_bank(bank_dir)

    assign_embeddings(filtered, batch_size=args.dino_batch_size)

    best = None

    for c in filtered:
        idx, ref_path, sim = search_bank(c.cls, c.patches, bank, topk=args.patch_topk)
        c.best_ref_index = idx
        c.best_ref = ref_path
        c.best_sim = sim

        if best is None or c.best_sim > best.best_sim:
            best = c

    if best is None:
        raise RuntimeError("DINO reference match 실패")

    best_path = output_dir / "best.jpg"
    best.image.save(best_path, quality=95)

    if args.save_candidates:
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
        "zoom_ratio": args.zoom_ratio,
        "candidate_total": len(candidates),
        "candidate_after_filter": len(filtered),
        "filter_status": filter_status,
        "contain_thres": args.contain_thres,
        "edge_margin_ratio": args.edge_margin_ratio,
        "selected_require": args.selected_require,
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


if __name__ == "__main__":
    main()
