import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

BASE_DIR = "VLM_RGA"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TAGS_FILE = os.path.join(BASE_DIR, "tags.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "rag_database.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_tags(tags_file):
    tag_map = {}
    current_group = None

    with open(tags_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("[") and line.endswith("]"):
            current_group = line[1:-1]
            i += 1
            continue

        filename = line

        if i + 1 >= len(lines):
            print(f"[경고] 태그 줄 없음: {filename}")
            break

        tag_line = lines[i + 1]
        tags = [tag.strip() for tag in tag_line.split(",") if tag.strip()]

        tag_map[filename] = {
            "group": current_group,
            "tags": tags,
            "tag_text": ", ".join(tags)
        }

        i += 2

    return tag_map


def find_image_path(filename):
    for root, _, files in os.walk(DATASET_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        features = outputs.pooler_output

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
        outputs = model.text_model(**inputs)
        features = outputs.pooler_output

    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


def main():
    tag_map = load_tags(TAGS_FILE)
    database = []

    print(f"태그 개수: {len(tag_map)}")
    print(f"데이터셋 폴더: {DATASET_DIR}")

    for filename, info in tqdm(tag_map.items()):
        image_path = find_image_path(filename)

        if image_path is None:
            print(f"[경고] 이미지 파일을 찾지 못함: {filename}")
            continue

        image_embedding = get_image_embedding(image_path)
        text_embedding = get_text_embedding(info["tag_text"])

        database.append({
            "filename": filename,
            "image_path": image_path,
            "group": info["group"],
            "tags": info["tags"],
            "tag_text": info["tag_text"],
            "image_embedding": image_embedding,
            "text_embedding": text_embedding
        })

    np.save(OUTPUT_FILE, np.array(database, dtype=object))

    print()
    print(f"저장 완료: {OUTPUT_FILE}")
    print(f"저장된 데이터 수: {len(database)}")


if __name__ == "__main__":
    main()