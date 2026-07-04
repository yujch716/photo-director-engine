"""상생의 손 이미지용 로컬 VLM + RAG 추론.

CLIP의 이미지/텍스트 임베딩을 이용해 이미지와 가장 가까운 지식 문서를
검색하고, 검색된 근거만으로 판정 문장을 만든다.

실행 예:
    python -m models.vlm_rag
    python -m models.vlm_rag --input "C:\images" --output results.json
    python -m models.vlm_rag --input image.jpg --knowledge knowledge.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_VLM_MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_INDEX = Path("clip_reference_embeddings.npy")
DEFAULT_TAGS = Path("data/hand_of_harmony_tags.tsv")
DEFAULT_INPUT = Path(
    r"C:\Users\김동준\Desktop\취업준비\포스코 빅데이터 아카데미\추론용 상생의 손"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 검색 단위가 되는 작은 지식 베이스. image_description은 CLIP 검색에,
# content는 최종 답변의 근거에 사용한다.
DEFAULT_KNOWLEDGE: list[dict[str, str]] = [
    {
        "id": "sea_hand",
        "label": "상생의 손(바다)",
        "image_description": (
            "A large bronze hand sculpture rising vertically from the sea at "
            "Homigot, Pohang, often photographed with waves, horizon and sunrise."
        ),
        "content": (
            "포항 호미곶의 상생의 손 가운데 바다에 설치된 오른손 조형물이다. "
            "바다 수면에서 손목과 손바닥이 솟아 있고 수평선이나 일출이 함께 보이는 "
            "장면이 대표적이다."
        ),
    },
    {
        "id": "land_hand",
        "label": "상생의 손(육지)",
        "image_description": (
            "A large bronze hand sculpture standing on land in Homigot Sunrise "
            "Square, Pohang, with a plaza, people or buildings around it."
        ),
        "content": (
            "호미곶 해맞이광장 육지에 설치된 왼손 조형물이다. 바다가 아닌 광장 바닥 "
            "위에 서 있으며 주변에 관람객, 광장 시설이나 건물이 나타날 수 있다."
        ),
    },
    {
        "id": "generic_hand_statue",
        "label": "기타 손 조형물",
        "image_description": (
            "A generic hand-shaped statue, sculpture, artwork or monument that is "
            "not the bronze Hand of Harmony landmark at Homigot."
        ),
        "content": (
            "손 모양 조형물이더라도 호미곶의 청동색 상생의 손과 장소적 특징이 "
            "일치하지 않으면 기타 손 조형물로 구분한다."
        ),
    },
    {
        "id": "not_hand_of_harmony",
        "label": "상생의 손 아님",
        "image_description": (
            "An ordinary travel photo, landscape, building, lighthouse, person or "
            "object with no large bronze hand sculpture."
        ),
        "content": "큰 청동색 손 조형물이 확인되지 않는 일반 풍경·건물·인물 사진이다.",
    },
]

_model: Any | None = None
_processor: Any | None = None
_device: Any | None = None
_loaded_model_name: str | None = None


def _load_clip(model_name: str = DEFAULT_MODEL) -> tuple[Any, Any, Any]:
    """모델은 최초 추론 때 한 번만 로드한다."""
    global _model, _processor, _device, _loaded_model_name

    if _model is not None and _loaded_model_name == model_name:
        return _model, _processor, _device

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "torch, transformers, Pillow가 필요합니다. "
            "설치: pip install torch transformers pillow"
        ) from exc

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _processor = CLIPProcessor.from_pretrained(model_name)
    _model = CLIPModel.from_pretrained(model_name).to(_device)
    _model.eval()
    _loaded_model_name = model_name
    return _model, _processor, _device


def load_knowledge(path: str | os.PathLike[str] | None = None) -> list[dict[str, str]]:
    """기본 지식과 선택적인 JSON 지식을 합친다."""
    documents = [dict(item) for item in DEFAULT_KNOWLEDGE]
    if path is None:
        return documents

    with open(path, "r", encoding="utf-8") as file:
        extra = json.load(file)
    if not isinstance(extra, list):
        raise ValueError("knowledge JSON의 최상위 값은 문서 배열이어야 합니다.")

    required = {"id", "label", "image_description", "content"}
    for index, document in enumerate(extra):
        if not isinstance(document, dict) or not required <= document.keys():
            raise ValueError(
                f"knowledge[{index}]에 필수 키가 없습니다: {sorted(required)}"
            )
        documents.append({key: str(document[key]) for key in required})
    return documents


def _open_image(path: str | os.PathLike[str]) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}") from exc


def retrieve(
    image: Image.Image,
    documents: list[dict[str, str]],
    *,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """이미지를 질의로 사용해 CLIP 유사도가 높은 문서를 검색한다."""
    import torch

    if not documents:
        raise ValueError("검색할 지식 문서가 없습니다.")

    model, processor, device = _load_clip(model_name)
    texts = [document["image_description"] for document in documents]
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).squeeze(0)
        probabilities = similarities.softmax(dim=0)

    count = min(max(1, top_k), len(documents))
    indices = torch.topk(probabilities, k=count).indices.detach().cpu().tolist()
    scores = probabilities.detach().cpu().tolist()
    return [
        {
            **documents[index],
            "score": round(float(scores[index]), 6),
        }
        for index in indices
    ]


def infer_image(
    image_path: str | os.PathLike[str],
    *,
    documents: list[dict[str, str]] | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 3,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """단일 이미지에 VLM 검색과 근거 기반 판정을 수행한다."""
    path = Path(image_path)
    documents = documents or load_knowledge()
    retrieved = retrieve(
        _open_image(path), documents, model_name=model_name, top_k=top_k
    )
    best = retrieved[0]
    is_target = best["id"] in {"sea_hand", "land_hand"} and best["score"] >= threshold

    if is_target:
        answer = (
            f"이 이미지는 '{best['label']}'로 판단됩니다. "
            f"검색 신뢰도는 {best['score']:.1%}입니다."
        )
    else:
        answer = (
            "이 이미지는 상생의 손이라고 확신하기 어렵습니다. "
            f"가장 가까운 검색 결과는 '{best['label']}'"
            f"({best['score']:.1%})입니다."
        )

    return {
        "image": str(path.resolve()),
        "is_hand_of_harmony": is_target,
        "prediction": best["label"],
        "confidence": best["score"],
        "answer": answer,
        "evidence": best["content"],
        "retrieved_documents": retrieved,
        "model": model_name,
        "threshold": threshold,
    }


def _iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"지원하지 않는 이미지 확장자입니다: {input_path.suffix}")
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"입력 경로가 없습니다: {input_path}")
    yield from (
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _console_print(message: str) -> None:
    """Windows CP949 콘솔에서도 유니코드 파일명 때문에 추론이 중단되지 않게 한다."""
    encoding = sys.stdout.encoding or "utf-8"
    safe_message = message.encode(encoding, errors="replace").decode(encoding)
    print(safe_message)


def infer_path(
    input_path: str | os.PathLike[str],
    *,
    knowledge_path: str | os.PathLike[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 3,
    threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """파일 또는 폴더의 모든 이미지를 추론한다."""
    documents = load_knowledge(knowledge_path)
    images = list(_iter_images(Path(input_path)))
    if not images:
        raise ValueError(f"입력 폴더에 이미지가 없습니다: {input_path}")

    results: list[dict[str, Any]] = []
    for image_path in images:
        try:
            result = infer_image(
                image_path,
                documents=documents,
                model_name=model_name,
                top_k=top_k,
                threshold=threshold,
            )
            results.append(result)
            _console_print(
                f"[OK] {image_path.name}: "
                f"{result['prediction']} ({result['confidence']:.1%})"
            )
        except Exception as exc:
            results.append({"image": str(image_path.resolve()), "error": str(exc)})
            _console_print(f"[FAIL] {image_path.name}: {exc}")
    return results


def save_image_tags(
    results: list[dict[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    input_path: str | os.PathLike[str],
) -> None:
    """CLIP 추론 결과를 ``이미지 상대경로<TAB>태그`` 형식으로 저장한다."""
    input_root = Path(input_path)
    if input_root.is_file():
        input_root = input_root.parent

    lines = ["image_name\ttag"]
    for result in results:
        image_path = Path(result["image"])
        try:
            image_name = str(image_path.relative_to(input_root.resolve()))
        except ValueError:
            image_name = image_path.name

        tag = "ERROR" if "error" in result else str(result["prediction"])
        # 탭/개행이 TSV 열 구조를 깨뜨리지 않도록 정리한다.
        image_name = image_name.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        tag = tag.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        lines.append(f"{image_name}\t{tag}")

    tag_path = Path(output_path)
    tag_path.parent.mkdir(parents=True, exist_ok=True)
    tag_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_clip_index(
    dataset_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str] = DEFAULT_INDEX,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    tags_path: str | os.PathLike[str] | None = DEFAULT_TAGS,
) -> dict[str, Any]:
    """레퍼런스 이미지 전체를 CLIP 임베딩하여 .npy와 메타데이터로 저장한다."""
    import numpy as np
    import torch

    root = Path(dataset_path).resolve()
    image_paths = list(_iter_images(root))
    if not image_paths:
        raise ValueError(f"데이터셋에 이미지가 없습니다: {root}")

    model, processor, device = _load_clip(model_name)
    batches: list[Any] = []
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start : start + batch_size]
        images = [_open_image(path) for path in paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
            # transformers 버전에 따라 tensor 또는 ModelOutput을 반환한다.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        batches.append(features.float().cpu().numpy())
        _console_print(f"[INDEX] {min(start + batch_size, len(image_paths))}/{len(image_paths)}")

    embeddings = np.concatenate(batches, axis=0).astype(np.float32)
    output = Path(index_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)

    tag_map: dict[str, list[str]] = {}
    if tags_path is not None:
        tag_file = Path(tags_path)
        if not tag_file.exists():
            raise FileNotFoundError(f"태그 파일이 없습니다: {tag_file}")
        for line_number, line in enumerate(
            tag_file.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip() or line_number == 1:
                continue
            try:
                relative_path, raw_tags = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"태그 파일 {line_number}행의 탭 구분이 잘못됐습니다.") from exc
            normalized = relative_path.replace("/", "\\")
            tag_map[normalized] = [
                tag.strip() for tag in raw_tags.split(",") if tag.strip()
            ]

    relative_paths = [str(path.relative_to(root)) for path in image_paths]
    metadata = {
        "dataset_root": str(root),
        "clip_model": model_name,
        "count": len(image_paths),
        "dimension": int(embeddings.shape[1]),
        "images": relative_paths,
        "categories": [
            path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "미분류"
            for path in image_paths
        ],
        "tags": [tag_map.get(path.replace("/", "\\"), []) for path in relative_paths],
        "tag_file": str(Path(tags_path).resolve()) if tags_path is not None else None,
        "tagged_count": sum(bool(tag_map.get(path.replace("/", "\\"))) for path in relative_paths),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"index": str(output.resolve()), "metadata": str(metadata_path.resolve()), **metadata}


def retrieve_reference_images(
    query_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str] = DEFAULT_INDEX,
    *,
    top_k: int = 3,
    reference_category: str | None = None,
) -> list[dict[str, Any]]:
    """현재 드론 프레임과 저장된 CLIP 벡터를 비교해 top-k 레퍼런스를 찾는다."""
    import numpy as np
    import torch

    index_file = Path(index_path)
    metadata = json.loads(index_file.with_suffix(".json").read_text(encoding="utf-8"))
    embeddings = np.load(index_file)
    if len(embeddings) != len(metadata["images"]):
        raise ValueError("임베딩 개수와 메타데이터 이미지 개수가 다릅니다.")

    model, processor, device = _load_clip(metadata["clip_model"])
    inputs = processor(images=[_open_image(query_path)], return_tensors="pt")
    with torch.inference_mode():
        query = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
        if not isinstance(query, torch.Tensor):
            query = query.pooler_output
        query = query / query.norm(dim=-1, keepdim=True)
    query_vector = query.float().cpu().numpy()[0]

    # 저장 시 L2 정규화했으므로 내적이 cosine similarity와 같다.
    similarities = embeddings @ query_vector
    root = Path(metadata["dataset_root"])
    query_resolved = Path(query_path).resolve()
    ranked_indices = np.argsort(similarities)[::-1]
    stored_tags = metadata.get("tags")
    categories = metadata.get("categories", ["미분류"] * len(embeddings))
    indices = [
        int(index)
        for index in ranked_indices
        if (root / metadata["images"][int(index)]).resolve() != query_resolved
        and (
            reference_category is not None
            or stored_tags is None
            or bool(stored_tags[int(index)])
        )
        and (
            reference_category is None
            or categories[int(index)] == reference_category
        )
    ][: min(max(1, top_k), len(embeddings))]
    if not indices:
        category_message = (
            f"'{reference_category}' 카테고리에 " if reference_category else ""
        )
        raise ValueError(f"{category_message}사용 가능한 레퍼런스 이미지가 없습니다.")
    return [
        {
            "rank": rank,
            "image": str((root / metadata["images"][int(index)]).resolve()),
            "relative_path": metadata["images"][int(index)],
            "category": metadata.get("categories", ["미분류"] * len(embeddings))[int(index)],
            "tags": metadata.get("tags", [[] for _ in embeddings])[int(index)],
            "similarity": round(float(similarities[int(index)]), 6),
        }
        for rank, index in enumerate(indices, start=1)
    ]


def generate_drone_direction(
    query_path: str | os.PathLike[str],
    references: list[dict[str, Any]],
    *,
    vlm_model_name: str = DEFAULT_VLM_MODEL,
    max_new_tokens: int = 128,
) -> str:
    """현재 프레임과 검색 레퍼런스를 VLM에 함께 넣어 이동 지시를 생성한다."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(vlm_model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm_model_name, dtype=dtype
    ).to(device)
    model.eval()

    images = [_open_image(query_path)] + [_open_image(item["image"]) for item in references]
    content: list[dict[str, str]] = [
        {"type": "text", "text": "CURRENT DRONE FRAME:"},
        {"type": "image"},
    ]
    for reference in references:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"REFERENCE {reference['rank']} | "
                        f"category={reference['category']} | "
                        f"tags={', '.join(reference['tags']) or 'none'} | "
                        f"CLIP similarity={reference['similarity']}"
                    ),
                },
                {"type": "image"},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "첫 번째 이미지는 현재 드론 프레임이고, 나머지는 CLIP으로 검색한 "
                "기준 사진이다. 현재 프레임을 기준 사진과 비슷한 구도로 촬영하려면 "
                "드론을 어떻게 이동해야 하는지 분석하라. 반드시 다음 형식의 한국어로 "
                "답하라.\n"
                "- 좌우 이동: 왼쪽/오른쪽/유지, 추정 이동량\n"
                "- 전후 이동: 전진/후진/유지, 추정 이동량\n"
                "- 고도: 상승/하강/유지, 추정 이동량\n"
                "- Yaw: 좌회전/우회전/유지, 추정 각도\n"
                "- 짐벌: 위/아래/유지, 추정 각도\n"
                "- 근거: 피사체 위치와 크기 차이\n"
                "- 안전: 장애물이나 불확실성\n"
                "영상만으로 확실하지 않은 수치는 반드시 '추정'이라고 표시하고, "
                "충돌 위험이 보이면 이동하지 말라고 명시하라."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {
        key: value.to(device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    prompt_length = inputs["input_ids"].shape[1]
    generated_text = processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )[0].strip()
    return f"기준 사진처럼 담으려면 다음과 같이 이동하세요.\n{generated_text}"


def run_reference_rag(
    query_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str] = DEFAULT_INDEX,
    *,
    top_k: int = 3,
    vlm_model_name: str = DEFAULT_VLM_MODEL,
    skip_generation: bool = False,
    reference_category: str | None = None,
) -> dict[str, Any]:
    """R(쿼리/검색) → A(멀티이미지 입력) → G(이동 지시) 전체 파이프라인."""
    references = retrieve_reference_images(
        query_path,
        index_path,
        top_k=top_k,
        reference_category=reference_category,
    )
    direction = None
    if not skip_generation:
        direction = generate_drone_direction(
            query_path, references, vlm_model_name=vlm_model_name
        )
    return {
        "query_image": str(Path(query_path).resolve()),
        "reference_category": reference_category,
        "references": references,
        "vlm_model": None if skip_generation else vlm_model_name,
        "drone_direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="상생의 손 VLM + RAG 일괄 추론")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="이미지 또는 폴더")
    parser.add_argument("--output", default="vlm_rag_results.json", help="결과 JSON")
    parser.add_argument(
        "--tag-output",
        default="clip_image_tags.txt",
        help="이미지 이름과 CLIP 태그를 저장할 TSV 텍스트 파일",
    )
    parser.add_argument("--knowledge", help="추가 지식 문서 JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face CLIP 모델")
    parser.add_argument("--top-k", type=int, default=3, help="검색 문서 수")
    parser.add_argument("--threshold", type=float, default=0.45, help="양성 판정 임계값")
    parser.add_argument("--build-index", metavar="DATASET", help="레퍼런스 CLIP 인덱스 생성")
    parser.add_argument("--query", metavar="FRAME", help="현재 드론 프레임으로 RAG 실행")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help=".npy CLIP 인덱스")
    parser.add_argument("--batch-size", type=int, default=16, help="인덱싱 배치 크기")
    parser.add_argument(
        "--tags",
        default=str(DEFAULT_TAGS),
        help="relative_path와 tags 열을 가진 UTF-8 TSV",
    )
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="생성형 VLM")
    parser.add_argument(
        "--reference-category",
        help="특정 폴더만 레퍼런스로 사용 (예: 기준 사진)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="top-k 검색까지만 실행하고 VLM 생성은 생략",
    )
    args = parser.parse_args()

    if args.build_index:
        report = build_clip_index(
            args.build_index,
            args.index,
            model_name=args.model,
            batch_size=args.batch_size,
            tags_path=args.tags,
        )
        _console_print(
            f"인덱스 저장: {report['index']} "
            f"({report['count']}개, {report['dimension']}차원, "
            f"태그 {report['tagged_count']}개)"
        )
        return

    if args.query:
        result = run_reference_rag(
            args.query,
            args.index,
            top_k=args.top_k,
            vlm_model_name=args.vlm_model,
            skip_generation=args.skip_generation,
            reference_category=args.reference_category,
        )
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _console_print(f"RAG 결과 저장: {output_path.resolve()}")
        return

    results = infer_path(
        args.input,
        knowledge_path=args.knowledge,
        model_name=args.model,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    save_image_tags(results, args.tag_output, input_path=args.input)
    _console_print(f"\n결과 저장: {output_path.resolve()} ({len(results)}개)")
    _console_print(f"태그 저장: {Path(args.tag_output).resolve()}")


if __name__ == "__main__":
    main()
