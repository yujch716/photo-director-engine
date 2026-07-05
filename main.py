from __future__ import annotations

import base64
import json
import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.nima import run_nima_score
from models.yolo import run_yolo
from models.yolo_world import run_yolo_world
from models.homigot_hand_yolo8n import run_homigot_hand_yolo
from services.capture import process_capture
from services.kakao_places import get_landmarks_by_keyword

app = FastAPI()

DRONE_DATA_DIR = pathlib.Path("drone-data")
DRONE_DATA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/drone-data", StaticFiles(directory="drone-data"), name="drone-data")

# reference(임베딩 원본) 이미지 폴더가 있으면 /refimages로 서빙한다.
# metadata.json의 path(예: "ver_1/50.jpeg")가 이 폴더 기준 상대경로.
# 폴더가 없으면 마운트하지 않음(뷰어는 안내 문구만 표시).
REFERENCE_IMAGES_DIR = pathlib.Path(os.environ.get("REFERENCE_IMAGES_DIR", "reference_images"))
if REFERENCE_IMAGES_DIR.is_dir():
    app.mount("/refimages", StaticFiles(directory=str(REFERENCE_IMAGES_DIR)), name="refimages")
    print(f"[INFO] reference images mounted: /refimages -> {REFERENCE_IMAGES_DIR}")

# API Test 페이지
@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/captures")
def list_captures():
    names = [p.name for p in DRONE_DATA_DIR.iterdir() if p.is_dir()]
    names.sort(reverse=True)  # 최신 폴더 먼저
    return {"captures": names}


@app.get("/captures/{name}")
def get_capture(name: str):
    folder = DRONE_DATA_DIR / name
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="capture not found")

    images = sorted(
        p.name for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    # 원본 1x: "1x"가 든 파일 우선, 없으면 crop/2x/best가 아닌 첫 이미지
    original = next(
        (f for f in images if "1x" in f.lower()), None
    ) or next(
        (f for f in images if all(k not in f.lower() for k in ("crop", "2x", "best"))), None
    )
    original_2x = next((f for f in images if "2x" in f.lower()), None)
    crops = [f for f in images if "crop" in f.lower()]
    best = "best.jpg" if "best.jpg" in images else None

    result = None
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))

    return {
        "name": name,
        "original": original,
        "original_2x": original_2x,
        "crops": crops,
        "best": best,
        "result": result,
    }

# YOLO 객체 탐지 API
@app.post("/detect-image")
async def detect_image(image: UploadFile):
    contents = await image.read()
    detections = run_yolo(contents) + run_homigot_hand_yolo(contents)
    return {"detections": detections}

@app.post("/nima-score")
async def nima_score(image: UploadFile):
    contents = await image.read()

    try:
        result = run_nima_score(contents, image.filename or "image.jpg")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result

@app.post("/capture")
async def capture(
    image: UploadFile,
    targets: str = Form(default="[]"),
    lat: float | None = Form(default=None),
    lng: float | None = Form(default=None),
):
    contents = await image.read()
    return process_capture(contents, json.loads(targets), lat, lng)


@app.get("/nearby-landmarks")
def nearby_landmarks(lat: float, lng: float, radius_m: int = 1000):
    return {"landmarks": get_landmarks_by_keyword(lat, lng, radius_m)}
