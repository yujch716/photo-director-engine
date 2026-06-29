import base64

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.image_preprocess import run_quality_normalization
from models.image_preprocess_logo import remove_logo_arrows
from models.landmark_clip import classify_landmark
from models.nima import run_nima_score
from models.yolo import run_detection

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# API Test 페이지
@app.get("/")
def index():
    return FileResponse("static/index.html")

# YOLO 객체 탐지 API
@app.post("/detect-image")
async def detect_image(image: UploadFile):
    contents = await image.read()
    detections = run_detection(contents)
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
