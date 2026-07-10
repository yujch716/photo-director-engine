from __future__ import annotations

import base64
import io
import json
import pathlib
from pathlib import Path
from typing import Any

from PIL import Image

from models.landmark_clip import identify_object, tag_image
from services.best_crop import select_best_crop
from services.drone_offset import compute_offset
from services.kakao_places import get_landmarks_by_keyword
from services.nima_log import append_session_nima
from services.session_paths import make_ts, resolve_save_dir

TEST_DATA_DIR = pathlib.Path("drone-data")


def _run_best_crop_pipeline(
    capture_dir: Path,
    image_1x_size: tuple[int, int],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """result.json이 만들어진 캡처 폴더에서 best 크롭 선택 + 드론 오프셋을 계산한다.

    단일 DB(config/dinov2_patch_embeddings.npy, config/metadata.json)가 없거나
    로드에 실패하면 (None, None)을 반환하고 서버는 죽지 않는다.
    (임베딩 데이터를 옮기기 전에도 /capture 앞단은 정상 동작하도록.)

    반환: (best, drone_offset) — 실패 시 각각 None.
    """
    try:
        report = select_best_crop(
            input_dir=capture_dir,
            output_dir=capture_dir,     # best.jpg를 캡처 폴더에 생성
            write_report=False,          # report.json은 capture가 통합본으로 직접 관리
        )
    except FileNotFoundError as e:
        print(
            f"[WARN] 임베딩 DB 로드 실패 → best crop 건너뜀: {e}\n"
            "       config/ 아래 필요한 파일: dinov2_patch_embeddings.npy (N,P,D), metadata.json (길이 N)"
        )
        return None, None
    except Exception as e:  # 매칭 중 임의 실패 시에도 앞단 응답은 살린다
        print(f"[WARN] best crop 단계 실패 → 건너뜀: {e}")
        return None, None

    if report.get("status") != "ok":
        print(f"[WARN] best crop status={report.get('status')} → best 없음")
        return None, None

    best = {
        "path": report.get("best"),
        "method": report.get("method"),            # "dino"(랜드마크) | "gaic"(랜드마크 없음)
        "candidate_index": report.get("best_candidate_index"),
        "source_box_in_1x": report.get("best_source_box_in_1x"),
        "similarity": report.get("best_score"),    # dino: 유사도 / gaic: 구도점수
        "reference": report.get("best_reference"),
        "reference_index": report.get("best_reference_index"),
        "reference_tags": report.get("best_reference_tags"),
        "scene_tags": report.get("capture_tags"),  # 모든 후보가 공유한 원본 장면 태그
    }
    drone_offset = compute_offset(
        best_source_box=report.get("best_source_box_in_1x"),
        image_1x_size=image_1x_size,
    )
    return best, drone_offset


def process_capture(
    image_bytes: bytes,
    target_list: list[dict],
    lat: float | None,
    lng: float | None,
    session_id: str | None = None,
) -> dict[str, Any]:
    # 세션 없으면 drone-data/<ts>/, 있으면 drone-data/<session_id>/1_original/
    capture_dir = resolve_save_dir(
        session_id, "1_original", legacy=TEST_DATA_DIR / make_ts(), data_dir=TEST_DATA_DIR
    )
    name = str(capture_dir.relative_to(TEST_DATA_DIR))  # "<ts>" 또는 "<session_id>/1_original"

    (capture_dir / "original_1x.jpg").write_bytes(image_bytes)

    # 정가운데를 2배율로 줌 땡긴 사진 저장
    base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = base_img.size
    zoom_box = (W // 4, H // 4, W - W // 4, H - H // 4)  # 중앙 절반 영역
    zoomed = base_img.crop(zoom_box).resize((W, H), Image.LANCZOS)
    zoom_buf = io.BytesIO()
    zoomed.save(zoom_buf, format="JPEG")
    (capture_dir / "original_2x.jpg").write_bytes(zoom_buf.getvalue())

    nearby_places = get_landmarks_by_keyword(lat, lng) if lat is not None and lng is not None else []
    nearby_names = [p["name"] for p in nearby_places]

    # 전체 이미지 태깅 (schema.yaml 형식)
    try:
        tags = tag_image(image_bytes)
    except Exception:
        tags = None

    # person_count는 CLIP이 못 세므로, 선택된 타깃 중 person 라벨 개수로 채운다.
    # (schema: "사진의 타깃 인물 수 (관중 제외)" — 관중 bystander는 타깃이 아니라 제외됨)
    person_count = sum(1 for t in target_list if str(t.get("class", "")).lower() == "person")
    if isinstance(tags, dict):
        tags["person_count"] = person_count
    else:
        tags = {"person_count": person_count}

    results = []
    if target_list:
        img = base_img
        for i, t in enumerate(target_list):
            cx, cy, w, h = t["bbox"]
            box = (
                max(0, int((cx - w / 2) * W)),
                max(0, int((cy - h / 2) * H)),
                min(W, int((cx + w / 2) * W)),
                min(H, int((cy + h / 2) * H)),
            )
            crop = img.crop(box)

            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            (capture_dir / f"crop{i}.jpg").write_bytes(buf.getvalue())

            landmark = None
            confidence = None
            scores = None
            try:
                clip_result = identify_object(image_bytes, t["bbox"])
                landmark = clip_result["landmark"]
                confidence = clip_result["best_score"]
                scores = clip_result["scores"]
            except Exception:
                pass

            results.append({
                "class": t["class"],
                "bbox": t["bbox"],
                "bbox_pixel": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
                "landmark": landmark,
                "confidence": confidence,
                "scores": scores,
            })

    # 캡처 산출물은 report.json 하나로 통합한다(내용 전부 여기 담음).
    # best_crop이 이 파일을 읽어 후보 필터/태그에 쓰므로, best 이전에 먼저 저장.
    report_path = capture_dir / "report.json"
    report = {
        "location": {"lat": lat, "lng": lng},
        "nearby_landmarks": nearby_places,
        "candidate_labels": nearby_names,
        "tags": tags,
        "targets": target_list,
        # original_2x.jpg가 원본에서 잘라낸 영역(중앙 절반)의 픽셀 좌표
        "original_2x": {
            "left": zoom_box[0],
            "top": zoom_box[1],
            "right": zoom_box[2],
            "bottom": zoom_box[3],
        },
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 앞단(2배율/크롭/CLIP/report.json)이 끝난 뒤 best 크롭 + 드론 오프셋으로 이어붙임.
    # best_crop은 report.json을 읽고, report.json은 여기서만 관리한다(write_report=False).
    best, drone_offset = _run_best_crop_pipeline(capture_dir, (W, H))

    # best/오프셋을 통합 report.json에 합쳐 저장.
    if best is not None:
        try:
            # best 선택에 뭘 썼는지 한눈에: 랜드마크 有 → clip_dino, 無 → gaic
            method = best.get("method")
            selection_method = {"dino": "clip_dino", "gaic": "gaic"}.get(method, method)

            data = json.loads(report_path.read_text(encoding="utf-8"))
            data["selection_method"] = selection_method  # "clip_dino" | "gaic"
            data["has_landmark"] = (method == "dino")    # 랜드마크 타깃 존재 여부
            data["best"] = best
            data["drone_offset"] = drone_offset
            report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] report.json에 best 저장 실패(무시): {e}")

    # 단계별 NIMA 누적 로그(선택 = best 크롭, 없으면 원본 1x). 파이프라인 시작점.
    best_path = capture_dir / "best.jpg"
    sel_bytes = best_path.read_bytes() if best_path.exists() else image_bytes
    append_session_nima(session_id, "1_original", image_bytes=sel_bytes, data_dir=TEST_DATA_DIR)

    # HTTP 응답은 드론이 실제로 쓰는 것만 슬림하게 반환한다.
    # (location/nearby_landmarks/tags/results/reference 태그 등 상세는 result.json에 다 저장돼 있고
    #  /captures/{saved} 로 언제든 조회 가능)
    # best 이미지(best.jpg)를 base64(JPEG)로 응답에 실어, 앱이 디코드해 저장했다가
    # 이후 유사도 비교(예: /scan-peak)의 target(목표 구도)으로 다시 보낼 수 있게 한다.
    best_image_b64 = None
    if best is not None:
        try:
            best_image_b64 = base64.b64encode((capture_dir / "best.jpg").read_bytes()).decode()
        except Exception as e:
            print(f"[WARN] best.jpg base64 인코딩 실패(무시): {e}")

    best_slim = None if best is None else {
        "method": best.get("method"),          # "dino"(랜드마크) | "gaic"(랜드마크 없음)
        "source_box_in_1x": best.get("source_box_in_1x"),
        "candidate_index": best.get("candidate_index"),
        "similarity": best.get("similarity"),
        "image_base64": best_image_b64,  # 목표 구도 이미지(JPEG) base64 — 앱이 디코드해 저장 후 재전송
    }
    return {
        "saved": name,            # 상세 조회용 폴더 id (/captures/{saved})
        "best": best_slim,        # 최종 구도 박스(1x 좌표) + 유사도
        "drone_offset": drone_offset,  # 이동 명령: dr, theta_deg (+ dx,dy)
    }
