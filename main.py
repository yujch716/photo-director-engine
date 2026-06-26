import base64
import io

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

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
    img = Image.open(io.BytesIO(contents))
    png_bytes, detections = run_detection(img)
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"image": image_b64, "detections": detections}
