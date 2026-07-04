import base64
import json
import pathlib

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.image_preprocess import run_quality_normalization
from models.image_preprocess_logo import remove_logo_arrows
from models.landmark_clip import classify_landmark
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
    original = next((f for f in images if "crop" not in f.lower()), None)
    crops = [f for f in images if "crop" in f.lower()]

    result = None
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))

    return {"name": name, "original": original, "crops": crops, "result": result}

# YOLO 객체 탐지 API
@app.post("/detect-image")
async def detect_image(image: UploadFile):
    contents = await image.read()
    detections = run_yolo(contents) + run_yolo_world(contents) + run_homigot_hand_yolo(contents)
    return {"detections": detections}


@app.post("/preprocess-image")
async def preprocess_image(image: UploadFile):
    contents = await image.read()

    try:
        jpg_bytes, report = run_quality_normalization(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_b64 = base64.b64encode(jpg_bytes).decode("utf-8")
    return {"image": image_b64, "report": report}


@app.post("/nima-score")
async def nima_score(image: UploadFile):
    contents = await image.read()

    try:
        result = run_nima_score(contents, image.filename or "image.jpg")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.post("/remove-logo-arrows")
async def remove_logo_arrows_image(image: UploadFile):
    contents = await image.read()

    try:
        png_bytes, mask_bytes, report = remove_logo_arrows(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_b64 = base64.b64encode(png_bytes).decode("utf-8")
    mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")
    return {"image": image_b64, "mask": mask_b64, "report": report}


@app.post("/classify-landmark")
async def classify_landmark_image(image: UploadFile):
    contents = await image.read()

    try:
        png_bytes, report = classify_landmark(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"image": image_b64, "report": report}


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
