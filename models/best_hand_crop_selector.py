from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
YOLO_IMPORT_CANDIDATES = (
    "homigot_hand_yolo8n",
    "homogot_hand_yolo8n",
    "homigot_hand_yolo",
    "homogot_hand_yolo",
)


@dataclass
class BoxDetection:
    label: str
    confidence: float
    xywhn: list[float]
    xyxy: tuple[float, float, float, float]


@dataclass
class Candidate:
    index: int
    image: Image.Image
    xyxy_in_wide: tuple[int, int, int, int]
    detection: BoxDetection | None = None
    object_similarity: float | None = None
    db_similarity: float | None = None
    db_ref: str | None = None
    reason: str = ""


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def xywhn_to_xyxy(xywhn: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in xywhn]
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    return x1, y1, x2, y2


def clamp_box_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def box_area_ratio(box: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)


def edge_margin_ratio(box: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    margin = min(x1, y1, width - x2, height - y2)
    return float(margin / max(1.0, min(width, height)))


def normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        n = np.linalg.norm(arr) + eps
        return arr / n
    n = np.linalg.norm(arr, axis=-1, keepdims=True) + eps
    return arr / n


def embedding_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    DINO cls/vector embedding과 patch embedding을 둘 다 처리하는 cosine similarity.
    - 1D vs 1D: cosine
    - 2D vs 2D: patch token mean-max similarity + aligned similarity 일부 반영
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim == 0 or b.ndim == 0:
        return -1.0

    if a.ndim == 1 and b.ndim == 1:
        aa = normalize_rows(a)
        bb = normalize_rows(b)
        return float(np.dot(aa, bb))

    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]

    if a.ndim != 2 or b.ndim != 2 or a.shape[-1] != b.shape[-1]:
        return -1.0

    aa = normalize_rows(a)
    bb = normalize_rows(b)
    sim = aa @ bb.T

    meanmax = 0.5 * (float(sim.max(axis=1).mean()) + float(sim.max(axis=0).mean()))

    if aa.shape[0] == bb.shape[0]:
        aligned = float(np.diag(sim).mean())
        return 0.7 * meanmax + 0.3 * aligned

    return meanmax


class DinoV2Extractor:
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 임베딩 계산에 torch/transformers가 필요합니다. "
                "설치: pip install torch transformers"
            ) from exc

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_patch(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state[0]
            patches = hidden[1:]  # CLS 제외 patch tokens
        return patches.detach().cpu().float().numpy()

    def encode_cls(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[0, 0]
        return cls.detach().cpu().float().numpy()


def _extract_embeddings_from_obj(obj: Any, source_name: str) -> list[tuple[str, np.ndarray]]:
    items: list[tuple[str, np.ndarray]] = []

    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.dtype == object:
            try:
                obj = arr.item()
                return _extract_embeddings_from_obj(obj, source_name)
            except Exception:
                return []
        if arr.ndim == 1 or arr.ndim == 2:
            items.append((source_name, arr.astype(np.float32)))
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                items.append((f"{source_name}#{i}", arr[i].astype(np.float32)))
        return items

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return _extract_embeddings_from_obj(obj.detach().cpu().numpy(), source_name)
    except Exception:
        pass

    if isinstance(obj, dict):
        emb = None
        for key in ("embeddings", "embedding", "features", "feats", "image_embeddings", "patch_embeddings", "dino_patch"):
            if key in obj:
                emb = obj[key]
                break
        paths = None
        for key in ("paths", "image_paths", "files", "filenames", "names"):
            if key in obj:
                paths = obj[key]
                break

        if emb is not None:
            try:
                import torch
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
            except Exception:
                pass
            emb = np.asarray(emb)
            if paths is not None:
                paths = [str(p) for p in list(paths)]
            if emb.ndim == 3 and paths is not None and len(paths) == emb.shape[0]:
                return [(paths[i], emb[i].astype(np.float32)) for i in range(emb.shape[0])]
            if emb.ndim == 2 and paths is not None and len(paths) == emb.shape[0]:
                return [(paths[i], emb[i].astype(np.float32)) for i in range(emb.shape[0])]
            return _extract_embeddings_from_obj(emb, source_name)

        for key, value in obj.items():
            if isinstance(value, (dict, list, tuple)):
                items.extend(_extract_embeddings_from_obj(value, f"{source_name}:{key}"))
            else:
                try:
                    arr = np.asarray(value)
                    if arr.ndim in (1, 2, 3) and np.issubdtype(arr.dtype, np.number):
                        items.extend(_extract_embeddings_from_obj(arr, f"{source_name}:{key}"))
                except Exception:
                    pass
        return items

    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            items.extend(_extract_embeddings_from_obj(value, f"{source_name}#{i}"))
        return items

    return items


def load_embedding_db(embedding_dir: Path) -> list[tuple[str, np.ndarray]]:
    if not embedding_dir.exists():
        raise FileNotFoundError(f"임베딩 폴더가 없습니다: {embedding_dir}")

    db: list[tuple[str, np.ndarray]] = []
    files = sorted(
        [p for p in embedding_dir.rglob("*") if p.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle"}]
    )

    for path in files:
        try:
            if path.suffix.lower() == ".npy":
                obj = np.load(path, allow_pickle=True)
                db.extend(_extract_embeddings_from_obj(obj, str(path)))
            elif path.suffix.lower() == ".npz":
                data = np.load(path, allow_pickle=True)
                obj = {k: data[k] for k in data.files}
                db.extend(_extract_embeddings_from_obj(obj, str(path)))
            elif path.suffix.lower() in {".pt", ".pth"}:
                import torch
                obj = torch.load(path, map_location="cpu")
                db.extend(_extract_embeddings_from_obj(obj, str(path)))
            elif path.suffix.lower() in {".pkl", ".pickle"}:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                db.extend(_extract_embeddings_from_obj(obj, str(path)))
        except Exception as exc:
            print(f"[WARN] 임베딩 파일 로드 실패: {path} / {exc}")

    # 마지막 차원이 512/768/1024 등인 유효 embedding만 남김
    clean = []
    for name, emb in db:
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim in (1, 2) and emb.size > 0:
            clean.append((name, emb))

    if not clean:
        raise RuntimeError(
            f"{embedding_dir}에서 사용할 수 있는 임베딩을 찾지 못했습니다. "
            "지원 형식: .npy/.npz/.pt/.pth/.pkl, key는 embeddings/paths 등"
        )

    return clean


def load_yolo_runner(models_dir: Path, module_name: str | None = None):
    sys.path.insert(0, str(models_dir))

    candidates = [module_name] if module_name else []
    candidates += [m for m in YOLO_IMPORT_CANDIDATES if m not in candidates]

    last_error = None
    for name in candidates:
        if not name:
            continue
        try:
            module = importlib.import_module(name)
            if not hasattr(module, "run_yolo"):
                raise AttributeError(f"{name}.run_yolo 함수가 없습니다.")
            print(f"[INFO] YOLO runner loaded: {name}.run_yolo")
            return module.run_yolo
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "상생의손 YOLO runner를 import하지 못했습니다. "
        f"models_dir={models_dir}, tried={candidates}, last_error={last_error}"
    )


def run_hand_yolo(run_yolo, img: Image.Image, min_conf: float) -> list[BoxDetection]:
    w, h = img.size
    detections_raw = run_yolo(image_to_jpeg_bytes(img))
    detections: list[BoxDetection] = []
    for det in detections_raw:
        conf = float(det.get("confidence", det.get("conf", 0.0)))
        if conf < min_conf:
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        label = str(det.get("class", det.get("label", "homigot_hand")))
        xywhn = [float(v) for v in bbox]
        xyxy = xywhn_to_xyxy(xywhn, w, h)
        detections.append(BoxDetection(label=label, confidence=conf, xywhn=xywhn, xyxy=xyxy))
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def generate_48_crops(wide_img: Image.Image, target_size: tuple[int, int], cols: int = 8, rows: int = 6) -> list[Candidate]:
    wide_w, wide_h = wide_img.size
    crop_w, crop_h = target_size

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"잘못된 target_size: {target_size}")

    # 0.6x 이미지가 1x보다 작으면 검증이 불가능하므로, 패딩 후 crop한다.
    if wide_w < crop_w or wide_h < crop_h:
        canvas_w = max(wide_w, crop_w)
        canvas_h = max(wide_h, crop_h)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        ox = (canvas_w - wide_w) // 2
        oy = (canvas_h - wide_h) // 2
        canvas.paste(wide_img.convert("RGB"), (ox, oy))
        wide_img = canvas
        wide_w, wide_h = wide_img.size
        print("[WARN] 06x 이미지가 1x보다 작아서 검은 패딩을 적용했습니다.")

    max_x = max(0, wide_w - crop_w)
    max_y = max(0, wide_h - crop_h)
    xs = [round(i * max_x / max(1, cols - 1)) for i in range(cols)]
    ys = [round(i * max_y / max(1, rows - 1)) for i in range(rows)]

    candidates: list[Candidate] = []
    idx = 0
    for y in ys:
        for x in xs:
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + crop_w, y1 + crop_h
            crop = wide_img.crop((x1, y1, x2, y2)).convert("RGB")
            candidates.append(Candidate(index=idx, image=crop, xyxy_in_wide=(x1, y1, x2, y2)))
            idx += 1
    return candidates


def find_pair_image(data_dir: Path, base: str, suffix: str = "_1x") -> Path | None:
    for ext in IMAGE_EXTS:
        p = data_dir / f"{base}{suffix}{ext}"
        if p.exists():
            return p
    matches = sorted(data_dir.glob(f"{base}{suffix}.*"))
    for p in matches:
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def find_selected_crop_images(data_dir: Path, base: str) -> list[Path]:
    paths = []
    for p in sorted(data_dir.glob(f"{base}_crop*.*")):
        if p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return paths


def get_base_from_06x(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_06x"):
        return stem[:-4]
    return re.sub(r"_0?6x$", "", stem)


def best_db_match(candidate_emb: np.ndarray, embedding_db: list[tuple[str, np.ndarray]]) -> tuple[float, str]:
    best_score = -1.0
    best_name = ""
    for name, ref_emb in embedding_db:
        score = embedding_similarity(candidate_emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name
    return best_score, best_name


def draw_debug_box(img: Image.Image, det: BoxDetection | None, text: str = "") -> Image.Image:
    out = img.copy().convert("RGB")
    if det is not None:
        draw = ImageDraw.Draw(out)
        x1, y1, x2, y2 = clamp_box_xyxy(det.xyxy, out.size[0], out.size[1])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 149, 0), width=4)
        draw.text((max(0, x1), max(0, y1 - 16)), text or f"{det.label} {det.confidence:.2f}", fill=(255, 149, 0))
    return out


def process_one_pair(
    wide_path: Path,
    one_x_path: Path,
    data_dir: Path,
    output_dir: Path,
    run_yolo,
    dino: DinoV2Extractor,
    embedding_db: list[tuple[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = get_base_from_06x(wide_path)
    wide_img = Image.open(wide_path).convert("RGB")
    one_x_img = Image.open(one_x_path).convert("RGB")
    candidates = generate_48_crops(wide_img, one_x_img.size, cols=args.grid_cols, rows=args.grid_rows)

    selected_paths = find_selected_crop_images(data_dir, base)
    selected_embs = []
    for p in selected_paths:
        try:
            selected_embs.append((str(p), dino.encode_patch(Image.open(p).convert("RGB"))))
        except Exception as exc:
            print(f"[WARN] 선택 crop 임베딩 실패: {p} / {exc}")

    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []

    for cand in candidates:
        detections = run_hand_yolo(run_yolo, cand.image, min_conf=args.yolo_conf)
        if not detections:
            cand.reason = "no_yolo_detection"
            rejected.append({"index": cand.index, "reason": cand.reason})
            continue

        det = detections[0]
        cand.detection = det
        crop_w, crop_h = cand.image.size
        area_ratio = box_area_ratio(det.xyxy, crop_w, crop_h)
        margin_ratio = edge_margin_ratio(det.xyxy, crop_w, crop_h)

        if area_ratio < args.min_area_ratio:
            cand.reason = f"too_small_area:{area_ratio:.4f}"
            rejected.append({"index": cand.index, "reason": cand.reason})
            continue
        if area_ratio > args.max_area_ratio:
            cand.reason = f"too_large_area:{area_ratio:.4f}"
            rejected.append({"index": cand.index, "reason": cand.reason})
            continue
        if margin_ratio < args.edge_margin_ratio:
            cand.reason = f"too_close_to_edge:{margin_ratio:.4f}"
            rejected.append({"index": cand.index, "reason": cand.reason})
            continue

        # 사용자가 선택한 객체 crop이 있으면, 후보 내 YOLO bbox crop과 DINO 유사도를 비교한다.
        if selected_embs and args.selected_crop_sim_thres > 0:
            x1, y1, x2, y2 = clamp_box_xyxy(det.xyxy, crop_w, crop_h)
            obj_img = cand.image.crop((x1, y1, x2, y2)).convert("RGB")
            obj_emb = dino.encode_patch(obj_img)
            best_obj_sim = max(embedding_similarity(obj_emb, emb) for _, emb in selected_embs)
            cand.object_similarity = best_obj_sim
            if best_obj_sim < args.selected_crop_sim_thres:
                cand.reason = f"selected_object_similarity_low:{best_obj_sim:.4f}"
                rejected.append({"index": cand.index, "reason": cand.reason})
                continue

        accepted.append(cand)

    if not accepted:
        print(f"[WARN] {base}: 통과 후보가 없습니다. best 저장 생략")
        return {
            "base": base,
            "wide": str(wide_path),
            "one_x": str(one_x_path),
            "selected_crop_count": len(selected_paths),
            "accepted_count": 0,
            "rejected_count": len(rejected),
            "best_path": None,
            "best_index": None,
            "status": "no_accepted_candidate",
            "rejected": rejected,
        }

    for cand in accepted:
        cand_emb = dino.encode_patch(cand.image)
        score, ref_name = best_db_match(cand_emb, embedding_db)
        cand.db_similarity = score
        cand.db_ref = ref_name

    accepted.sort(
        key=lambda c: (
            c.db_similarity if c.db_similarity is not None else -1.0,
            c.object_similarity if c.object_similarity is not None else -1.0,
            c.detection.confidence if c.detection else -1.0,
        ),
        reverse=True,
    )
    best = accepted[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{base}_best.jpg"
    best.image.save(best_path, quality=95)

    if args.save_debug:
        debug_path = output_dir / f"{base}_best_debug.jpg"
        draw_debug_box(best.image, best.detection, f"sim={best.db_similarity:.3f}").save(debug_path, quality=95)

    if args.save_candidates:
        cand_dir = output_dir / f"{base}_candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        for cand in accepted:
            score = cand.db_similarity if cand.db_similarity is not None else -1
            cand.image.save(cand_dir / f"{base}_ok_crop{cand.index:02d}_sim{score:.3f}.jpg", quality=90)

    return {
        "base": base,
        "wide": str(wide_path),
        "one_x": str(one_x_path),
        "selected_crop_count": len(selected_paths),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "best_path": str(best_path),
        "best_index": best.index,
        "best_xyxy_in_wide": list(best.xyxy_in_wide),
        "best_yolo_conf": best.detection.confidence if best.detection else None,
        "best_object_similarity": best.object_similarity,
        "best_db_similarity": best.db_similarity,
        "best_db_ref": best.db_ref,
        "status": "saved",
        "top_candidates": [
            {
                "index": c.index,
                "xyxy_in_wide": list(c.xyxy_in_wide),
                "yolo_conf": c.detection.confidence if c.detection else None,
                "object_similarity": c.object_similarity,
                "db_similarity": c.db_similarity,
                "db_ref": c.db_ref,
            }
            for c in accepted[: min(10, len(accepted))]
        ],
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="06x 이미지를 1x 크기로 48분할한 뒤, 상생의손 YOLO + DINOv2 유사도로 best crop 저장")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "drone-data")
    parser.add_argument("--embedding-dir", type=Path, default=project_root / "imbeddingdata" / "hand_dinov2imbedding")
    parser.add_argument("--output-dir", type=Path, default=None, help="기본값: data-dir")
    parser.add_argument("--models-dir", type=Path, default=script_dir)
    parser.add_argument("--yolo-module", type=str, default=None, help="예: homigot_hand_yolo8n")
    parser.add_argument("--pattern", type=str, default="*_06x.jpg")
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--yolo-conf", type=float, default=0.20)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--max-area-ratio", type=float, default=0.65)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.035)
    parser.add_argument("--selected-crop-sim-thres", type=float, default=0.25, help="0이면 선택 crop 유사도 필터 비활성화")
    parser.add_argument("--dino-model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="테스트용 처리 개수 제한. 0이면 전체")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir or data_dir

    if args.grid_cols * args.grid_rows != 48:
        print(f"[WARN] 현재 grid는 {args.grid_cols}x{args.grid_rows}={args.grid_cols * args.grid_rows}개입니다. 48개를 원하면 8x6을 사용하세요.")

    print(f"[INFO] data_dir      = {data_dir}")
    print(f"[INFO] embedding_dir = {args.embedding_dir}")
    print(f"[INFO] output_dir    = {output_dir}")
    print(f"[INFO] models_dir    = {args.models_dir}")

    run_yolo = load_yolo_runner(args.models_dir, module_name=args.yolo_module)
    print("[INFO] Loading DINOv2 extractor...")
    dino = DinoV2Extractor(args.dino_model)
    print("[INFO] Loading reference embedding DB...")
    embedding_db = load_embedding_db(args.embedding_dir)
    print(f"[INFO] reference embeddings: {len(embedding_db)}")

    wide_paths = sorted(data_dir.glob(args.pattern))
    if args.limit and args.limit > 0:
        wide_paths = wide_paths[: args.limit]
    if not wide_paths:
        raise FileNotFoundError(f"06x 이미지를 찾지 못했습니다: {data_dir / args.pattern}")

    reports = []
    for i, wide_path in enumerate(wide_paths, start=1):
        base = get_base_from_06x(wide_path)
        one_x_path = find_pair_image(data_dir, base, suffix="_1x")
        if one_x_path is None:
            print(f"[WARN] {base}: 1x 이미지가 없어 건너뜀")
            reports.append({"base": base, "wide": str(wide_path), "status": "missing_1x"})
            continue

        print(f"\n[INFO] ({i}/{len(wide_paths)}) processing: {base}")
        report = process_one_pair(
            wide_path=wide_path,
            one_x_path=one_x_path,
            data_dir=data_dir,
            output_dir=output_dir,
            run_yolo=run_yolo,
            dino=dino,
            embedding_db=embedding_db,
            args=args,
        )
        reports.append(report)
        print(f"[INFO] status={report.get('status')}, accepted={report.get('accepted_count')}, best={report.get('best_path')}")

    report_path = output_dir / "best_hand_crop_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] report saved: {report_path}")


if __name__ == "__main__":
    main()
