import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

BASE_DIR = "VLM_RGA"
DATABASE_FILE = os.path.join(BASE_DIR, "rag_database.npy")
QUERY_DIR = os.path.join(BASE_DIR, "dataset", "기준 사진")

TOP_K = 10
MIN_FILTER_MATCH = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_query_image_path():
    valid_ext = (".jpg", ".jpeg", ".png", ".webp")
    images = [
        os.path.join(QUERY_DIR, f)
        for f in os.listdir(QUERY_DIR)
        if f.lower().endswith(valid_ext)
    ]

    if not images:
        raise FileNotFoundError(f"기준 사진이 없습니다: {QUERY_DIR}")

    return images[0]


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


def get_text_embedding(text):
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def collect_all_tags(database):
    all_tags = set()
    for item in database:
        for tag in item["tags"]:
            all_tags.add(tag.strip())
    return sorted(all_tags)


def extract_filter_words_by_clip_text(query_embedding, all_tags, top_n=8, threshold=0.20):
    tag_scores = []

    for tag in all_tags:
        text_embedding = get_text_embedding(tag)
        score = cosine_similarity(query_embedding, text_embedding)

        tag_scores.append({
            "tag": tag,
            "score": score
        })

    tag_scores = sorted(tag_scores, key=lambda x: x["score"], reverse=True)

    selected = [
        x["tag"]
        for x in tag_scores[:top_n]
        if x["score"] >= threshold
    ]

    return selected, tag_scores


def tag_match_score(query_filters, item_tags):
    query_set = set(query_filters)
    item_set = set(item_tags)

    matched = sorted(list(query_set & item_set))
    return len(matched), matched


def main():
    database = np.load(DATABASE_FILE, allow_pickle=True)
    query_image_path = get_query_image_path()

    print("[기준 사진]")
    print(query_image_path)
    print()

    query_embedding = get_image_embedding(query_image_path)

    all_tags = collect_all_tags(database)

    print("[tags.txt에서 추출된 전체 태그 후보]")
    print(", ".join(all_tags))
    print()

    filter_words, tag_scores = extract_filter_words_by_clip_text(
        query_embedding=query_embedding,
        all_tags=all_tags,
        top_n=8,
        threshold=0.20
    )

    print("[CLIP으로 선택된 필터 단어]")
    print(", ".join(filter_words) if filter_words else "선택된 필터 단어 없음")
    print()

    print("[태그 후보 점수 TOP-15]")
    for item in tag_scores[:15]:
        print(f"{item['tag']}: {item['score']:.4f}")
    print()

    passed = []
    excluded = []

    for item in database:
        match_count, matched_tags = tag_match_score(filter_words, item["tags"])

        if match_count >= MIN_FILTER_MATCH:
            image_sim = cosine_similarity(query_embedding, item["image_embedding"])

            passed.append({
                "filename": item["filename"],
                "image_path": item["image_path"],
                "group": item["group"],
                "tags": item["tags"],
                "matched_tags": matched_tags,
                "filter_match_count": match_count,
                "image_similarity": image_sim,
            })
        else:
            excluded.append({
                "filename": item["filename"],
                "image_path": item["image_path"],
                "group": item["group"],
                "tags": item["tags"],
                "matched_tags": matched_tags,
                "filter_match_count": match_count,
                "reason": "필터 단어 매칭 부족",
            })

    ranked = sorted(passed, key=lambda x: x["image_similarity"], reverse=True)

    print(f"[필터 통과 사진 수] {len(passed)}장")
    print(f"[제외 사진 수] {len(excluded)}장")
    print()

    print(f"[TOP-{TOP_K} 유사 사진 순위]")
    for idx, item in enumerate(ranked[:TOP_K], start=1):
        print(f"{idx}위")
        print(f"파일명: {item['filename']}")
        print(f"그룹: {item['group']}")
        print(f"이미지 유사도: {item['image_similarity']:.4f}")
        print(f"매칭 태그: {', '.join(item['matched_tags'])}")
        print(f"경로: {item['image_path']}")
        print("-" * 60)

    print()
    print("[제외된 사진]")
    for item in excluded:
        print(f"파일명: {item['filename']}")
        print(f"그룹: {item['group']}")
        print(f"매칭 태그 수: {item['filter_match_count']}")
        print(f"매칭 태그: {', '.join(item['matched_tags']) if item['matched_tags'] else '없음'}")
        print(f"이유: {item['reason']}")
        print("-" * 60)


if __name__ == "__main__":
    main()

from PIL import Image

img = Image.open(ranked[0]["image_path"])
img.show()