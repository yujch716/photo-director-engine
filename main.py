from __future__ import annotations

import base64
import json
import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.nima import run_nima_score
from models.yolo import run_yolo
from models.yolo_world import run_yolo_world
from models.homigot_hand_yolo8n import run_homigot_hand_yolo
from services.capture import process_capture
from services.kakao_places import get_landmarks_by_keyword
from services.nima_directions import find_best_nima_direction
from services.nima_zoom import evaluate_zoom
from services.scan_peak import find_scan_peak, save_scan_peak_inputs
from services.scan_result import save_scan_result

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

@app.get("/nearby-landmarks")
def nearby_landmarks(lat: float, lng: float, radius_m: int = 1000):
    return {"landmarks": get_landmarks_by_keyword(lat, lng, radius_m)}



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


@app.post("/nima-directions")
async def nima_directions(image: UploadFile):
    """현재 프레임의 9방향(중앙+상하좌우+대각선) 2배율 크롭에 NIMA를 매겨 최고 방향을 반환한다."""
    contents = await image.read()
    try:
        return find_best_nima_direction(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/nima-zoom")
async def nima_zoom(image: UploadFile, direction: str | None = Form(default=None)):
    """현재 프레임의 배율 크롭 NIMA로 전진/후진을 판단한다.

    direction 없음=첫 회차(1.0/1.1/0.9 양방향 비교로 방향 확정),
    direction=forward|backward=이후 회차(그 방향 배율 + 1.0만 평가).
    """
    contents = await image.read()
    try:
        return evaluate_zoom(contents, direction)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/capture")
async def capture(
    image: UploadFile,
    targets: str = Form(default="[]"),
    lat: float | None = Form(default=None),
    lng: float | None = Form(default=None),
):
    contents = await image.read()
    return process_capture(contents, json.loads(targets), lat, lng)


@app.post("/scan-peak")
async def scan_peak(frames: list[UploadFile] = File(...), target: UploadFile = File(...)):
    """스캔 프레임들(frames, 시간순) 중 target 구도와 SSIM이 가장 높은 정점 프레임을 반환한다.

    입력: multipart — frames(여러 장, 업로드 순서 보존), target(1장).
    반환: {peak_index, peak_ssim, scores, frame_count, timing_ms}
    """
    frame_bytes = [await f.read() for f in frames]
    tgt_bytes = await target.read()
    try:
        result = find_scan_peak(frame_bytes, tgt_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 받은 프레임 전부 + target(+결과)을 drone-data/scan-peak/<ts>/에 저장.
    result["saved"] = save_scan_peak_inputs(frame_bytes, tgt_bytes, result)
    return result


@app.post("/scan-result")
async def scan_result(
    image: UploadFile,
    target: UploadFile | None = File(default=None),
    meta: str | None = Form(default=None),
):
    """정점 도착 후 촬영 사진을 저장한다(+선택 target/meta). drone-data/<타임스탬프>/ 아래.

    입력: multipart — image(필수), target(선택), meta(선택 JSON 문자열).
    반환: {"saved": "<폴더명>", "message": "저장 완료"}
    """
    img_bytes = await image.read()
    tgt_bytes = await target.read() if target is not None else None
    return save_scan_result(img_bytes, tgt_bytes, meta)

