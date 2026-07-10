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
from services.final_shot import save_final_shot
from services.kakao_places import get_landmarks_by_keyword
from services.nima_directions import find_best_nima_direction
from services.nima_zoom import evaluate_zoom
from services.scan_peak import find_scan_peak, save_scan_peak_inputs
from services.session_paths import session_root
from services.tilt_peak import find_tilt_peak
from services.scan_result import save_scan_result
from services.vlm_select import select_final

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


# ── 세션 로그 뷰어용 ─────────────────────────────────────────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_SESSION_STEPS = ["1_original", "2_scan", "3_nima-move", "4_final"]


def _rel_url(f: pathlib.Path) -> str:
    return "/drone-data/" + str(f.resolve().relative_to(DRONE_DATA_DIR.resolve())).replace("\\", "/")


def _dir_sort_key(p: pathlib.Path):
    # 숫자 폴더는 숫자순 먼저, 그 외는 이름순.
    return (0, int(p.name), "") if p.name.isdigit() else (1, 0, p.name)


def _collect_group(folder: pathlib.Path, label: str) -> dict:
    """폴더 하나의 이미지(url) + json(파싱 내용)을 모은다."""
    images, jsons = [], {}
    for f in sorted(folder.iterdir(), key=lambda x: x.name):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf in _IMG_EXTS:
            images.append({"url": _rel_url(f), "name": f.name})
        elif suf == ".json":
            try:
                jsons[f.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                jsons[f.name] = None
    return {"label": label, "images": images, "jsons": jsons}


def _leaf_groups(folder: pathlib.Path) -> list[dict]:
    """중첩 폴더(3_nima-move)에서 파일이 들어있는 leaf 폴더들을 그룹으로 모은다."""
    groups: list[dict] = []

    def walk(d: pathlib.Path, label: str):
        entries = list(d.iterdir())
        if any(x.is_file() for x in entries):
            groups.append(_collect_group(d, label))
        for sd in sorted([x for x in entries if x.is_dir()], key=_dir_sort_key):
            walk(sd, f"{label}/{sd.name}" if label else sd.name)

    walk(folder, "")
    return groups


@app.get("/sessions")
def list_sessions():
    """세션 폴더(1_original 등 단계 폴더를 가진 top-level 디렉토리) 목록. 최신순."""
    out = []
    for p in DRONE_DATA_DIR.iterdir():
        if p.is_dir() and any((p / s).is_dir() for s in _SESSION_STEPS):
            out.append((p.stat().st_mtime, p.name))
    out.sort(reverse=True)
    return {"sessions": [name for _, name in out]}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """세션의 단계별 이미지 + json + nima-score.json 을 반환한다(뷰어용)."""
    root = (DRONE_DATA_DIR / session_id).resolve()
    if not str(root).startswith(str(DRONE_DATA_DIR.resolve()) + os.sep) or not root.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    nima_score = None
    nsp = root / "nima-score.json"
    if nsp.exists():
        try:
            nima_score = json.loads(nsp.read_text(encoding="utf-8"))
        except Exception:
            nima_score = None

    menus: dict[str, list] = {}
    for step in _SESSION_STEPS:
        sd = root / step
        if not sd.is_dir():
            menus[step] = []
        elif step == "3_nima-move":
            menus[step] = _leaf_groups(sd)
        else:
            menus[step] = [_collect_group(sd, step)]

    # 4_final 비교: 원본(1_original/original_1x) vs 최종(4_final/final.jpg) + 각 NIMA.
    final_compare = None
    fin = root / "4_final" / "final.jpg"
    if fin.exists():
        def _nima_of(p: pathlib.Path):
            try:
                return round(float(run_nima_score(p.read_bytes())["score"]), 4)
            except Exception:
                return None

        # 최종 점수는 nima-score.json의 4_final 항목 재사용, 없으면 계산.
        fin_nima = None
        for e in (nima_score or []):
            if e.get("stage") == "4_final":
                fin_nima = e.get("nima")
        if fin_nima is None:
            fin_nima = _nima_of(fin)

        orig = root / "1_original" / "original_1x.jpg"
        final_compare = {
            "original": ({"url": _rel_url(orig), "nima": _nima_of(orig)} if orig.exists() else None),
            "final": {"url": _rel_url(fin), "nima": fin_nima},
        }

    return {
        "session_id": session_id,
        "nima_score": nima_score,
        "menus": menus,
        "final_compare": final_compare,
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


@app.post("/nima-lateral")
async def nima_lateral(image: UploadFile, session_id: str | None = Form(default=None)):
    """현재 프레임의 9방향(중앙+상하좌우+대각선) 2배율 크롭에 NIMA를 매겨 최고 방향을 반환한다."""
    contents = await image.read()
    try:
        return find_best_nima_direction(contents, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/nima-depth")
async def nima_depth(
    image: UploadFile,
    direction: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
):
    """현재 프레임의 배율 크롭 NIMA로 전진/후진을 판단한다.

    direction 없음=첫 회차(1.0/1.1/0.9 양방향 비교로 방향 확정),
    direction=forward|backward=이후 회차(그 방향 배율 + 1.0만 평가).
    """
    contents = await image.read()
    try:
        return evaluate_zoom(contents, direction, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/capture")
async def capture(
    image: UploadFile,
    targets: str = Form(default="[]"),
    lat: float | None = Form(default=None),
    lng: float | None = Form(default=None),
    session_id: str | None = Form(default=None),
):
    contents = await image.read()
    return process_capture(contents, json.loads(targets), lat, lng, session_id=session_id)


@app.post("/scan-peak")
async def scan_peak(
    frames: list[UploadFile] = File(...),
    session_id: str = Form(...),
):
    """스캔 프레임들(frames, 시간순) 중 최종구도와 SSIM이 가장 높은 정점 프레임을 반환한다.

    target은 따로 받지 않고, 이 세션의 /capture 결과인
    drone-data/<session_id>/1_original/best.jpg 를 비교 기준으로 자동 사용한다.

    입력: multipart — frames(여러 장, 업로드 순서 보존), session_id(필수).
    반환: {peak_index, peak_ssim, scores, frame_count, timing_ms, saved}
    """
    frame_bytes = [await f.read() for f in frames]

    best_path = session_root(session_id) / "1_original" / "best.jpg"
    if not best_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"target 없음: {session_id}/1_original/best.jpg — /capture를 먼저 호출하세요(best 생성).",
        )
    tgt_bytes = best_path.read_bytes()

    try:
        result = find_scan_peak(frame_bytes, tgt_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 프레임(2배 크롭)+정점 best+결과를 drone-data/<session_id>/2_scan/에 저장.
    result["saved"] = save_scan_peak_inputs(frame_bytes, result, session_id=session_id)
    return result


@app.post("/tilt-peak")
async def tilt_peak(
    frames: list[UploadFile] = File(...),
    angles: str = Form(...),
    session_id: str | None = Form(default=None),
):
    """틸트 sweep 프레임들에 NIMA를 매겨 최고 점수 프레임(각도)을 반환한다.

    입력: multipart — frames(여러 장, 위→아래 시간순), angles(JSON 각도 배열, 프레임과 같은 순서·개수), session_id.
    반환: {peak_index, peak_angle, peak_score, scores, angles, frame_count, timing_ms}
    저장: drone-data/<session_id>/tilt/
    """
    frame_bytes = [await f.read() for f in frames]
    try:
        parsed_angles = json.loads(angles)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"angles JSON 파싱 실패: {exc}") from exc
    try:
        return find_tilt_peak(frame_bytes, parsed_angles, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/vlm-select")
async def vlm_select(
    frames: list[UploadFile] = File(...),
    angles: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
):
    """틸트 sweep 프레임들(frames) 중 NIMA+VLM으로 최종 1장을 선택한다.

    입력: multipart — frames(여러 장), angles(선택, JSON 리스트 예 [10,8,...,-10]), session_id(선택).
    반환: {selected_index, selected_angle, reason, nima_scores, defect_flag, source, timing_ms}
    """
    frame_bytes = [await f.read() for f in frames]
    parsed_angles = None
    if angles:
        try:
            parsed_angles = json.loads(angles)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"angles JSON 파싱 실패: {exc}") from exc
    try:
        return select_final(frame_bytes, parsed_angles, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/scan-result")
async def scan_result(
    image: UploadFile,
    meta: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
):
    """정점 도착 후 촬영 사진을 저장한다(+선택 meta). target은 받지 않는다.

    세션 있으면 drone-data/<session_id>/2_scan/(scan-peak와 공유), 없으면 drone-data/result/<ts>/.
    입력: multipart — image(필수), meta(선택 JSON 문자열), session_id(선택).
    반환: {"saved": "<폴더경로>", "message": "저장 완료"}
    """
    img_bytes = await image.read()
    return save_scan_result(img_bytes, meta_json=meta, session_id=session_id)


@app.post("/final-shot")
async def final_shot(
    image: UploadFile,
    session_id: str | None = Form(default=None),
    meta: str | None = Form(default=None),
):
    """최종 촬영 이미지(초점 조정 후)를 drone-data/<session_id>/final/ 에 저장한다.

    입력: multipart — image(필수), session_id, meta(선택 JSON 문자열).
    반환: {"saved": "<session_id>", "path": "/drone-data/.../final.jpg", "message": "최종 저장 완료"}
    """
    img_bytes = await image.read()
    return save_final_shot(img_bytes, session_id=session_id, meta_json=meta)

