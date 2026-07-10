"""스캔 정점 탐지: 프레임 묶음을 target과 SSIM 비교해 가장 잘 맞은 프레임을 찾는다.

드론이 연속 이동(스캔)하며 찍은 프레임들을 시간순으로 받고, 최종구도 이미지(target)와
각 프레임의 구조적 유사도(SSIM)를 재서 최대인 지점(정점)을 반환한다.

- SSIM은 두 이미지가 같은 크기여야 하므로 grayscale + size×size로 리사이즈 후 계산.
- 디스크 저장 없이 메모리에서 처리.
- 각 단계 소요시간을 로그로 출력.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from services.nima_log import append_session_nima
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir

DEFAULT_SSIM_SIZE = 256


def _center_2x_jpeg(image_bytes: bytes) -> bytes:
    """이미지의 중앙 5/7(1.4배율 뷰)를 잘라 JPEG 바이트로 반환(색상 유지)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    crop = img.crop((W // 7, H // 7, W - W // 7, H - H // 7))  # 1.4배: 중앙 5/7
    buf = io.BytesIO()
    crop.save(buf, format="JPEG")
    return buf.getvalue()


def save_scan_peak_inputs(
    frame_bytes_list: list[bytes],
    result: dict[str, Any] | None = None,
    session_id: str | None = None,
    data_dir: Path | None = None,
) -> str:
    """스캔 프레임(중앙 1.4배 크롭) + 정점 best + 결과를 저장한다.

    세션 없으면 drone-data/scan-peak/<ts>/, 있으면 drone-data/<session_id>/2_scan/.
    저장 파일:
        frame_000.jpg, ...  : 받은 프레임을 중앙 1.4배 크롭한 이미지(시간순, SSIM 비교와 동일 스케일)
        best.jpg            : SSIM 정점 프레임(중앙 1.4배 크롭)
        report.json         : (선택) 각 프레임 SSIM / 선택 index·이름 / timing_ms
    Returns:
        저장 폴더 경로(drone-data 이하).
    """
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_save_dir(session_id, "2_scan", legacy=base / "scan-peak" / make_ts(), data_dir=base)
    name = str(folder.relative_to(base))

    # 프레임은 SSIM 비교와 동일하게 중앙 1.4배 크롭해서 저장.
    for i, fb in enumerate(frame_bytes_list):
        (folder / f"frame_{i:03d}.jpg").write_bytes(_center_2x_jpeg(fb))

    peak_index = result.get("peak_index") if result else None
    if peak_index is not None and 0 <= peak_index < len(frame_bytes_list):
        best_bytes = _center_2x_jpeg(frame_bytes_list[peak_index])
        (folder / "best.jpg").write_bytes(best_bytes)
        # 단계별 NIMA 누적 로그(선택 = SSIM 정점 프레임).
        append_session_nima(session_id, "2_scan/peak", image_bytes=best_bytes, data_dir=base)

    if result is not None:
        report = {
            **result,
            "selected_index": peak_index,
            "selected_name": (f"frame_{peak_index:03d}.jpg" if peak_index is not None else None),
        }
        (folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return name


def _load_gray(image_bytes: bytes, size: int, crop_center: bool = False) -> np.ndarray:
    """이미지 바이트 → (선택 중앙 1.4배율 크롭) grayscale + (size×size) uint8 배열. SSIM 입력용.

    crop_center=True면 중앙 5/7(= 1.4배율 뷰)만 잘라서 사용한다.
    (capture의 original_2x와 동일한 crop box)
    """
    img = Image.open(io.BytesIO(image_bytes))
    if crop_center:
        W, H = img.size
        img = img.crop((W // 7, H // 7, W - W // 7, H - H // 7))  # 1.4배: 중앙 5/7
    img = img.convert("L").resize((size, size))
    return np.asarray(img, dtype=np.uint8)


def _moving_average(x: list[float], window: int) -> list[float]:
    """가장자리를 보존하는 단순 이동평균(정점 탐색 스무딩용)."""
    if window <= 1 or len(x) < window:
        return list(x)
    arr = np.asarray(x, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return list(np.convolve(arr, kernel, mode="same"))


def find_scan_peak(
    frame_bytes_list: list[bytes],
    target_bytes: bytes,
    size: int = DEFAULT_SSIM_SIZE,
    smooth_window: int = 1,
    crop_center: bool = True,
) -> dict[str, Any]:
    """프레임들 vs target SSIM을 재서 정점(최대 유사도) 프레임을 찾는다.

    각 프레임은 기본적으로 중앙 1.4배율 크롭 후 비교한다(target은 이미 1.4배율 구도라 크롭 안 함).
    → 프레임과 target의 스케일을 맞춰 비교.

    Args:
        frame_bytes_list: 프레임 이미지 바이트들(시간순, 0번=스캔 시작). 드론 전체 프레임.
        target_bytes: 최종구도 이미지 바이트(비교 기준). 이미 1.4배율 크롭된 상태 가정 → 크롭 안 함.
        size: SSIM 계산용 리사이즈 크기(정사각). 작을수록 빠름.
        smooth_window: 정점 탐색 전 이동평균 창(1이면 스무딩 없음=단순 최대).
        crop_center: True(기본)면 각 프레임을 중앙 1.4배율 크롭 후 비교. False면 프레임 전체 비교.

    Returns:
        {peak_index, peak_ssim, scores, frame_count, timing_ms}
        (peak_ssim/scores는 원본 SSIM 값. 정점 index만 스무딩 결과로 고른다.)
    """
    t_start = time.perf_counter()
    n = len(frame_bytes_list)

    if n == 0:
        raise ValueError("frames가 비어 있음")

    # target 로드 (이미 1.4배율 구도라 크롭하지 않음)
    t0 = time.perf_counter()
    target_gray = _load_gray(target_bytes, size, crop_center=False)
    target_load_ms = (time.perf_counter() - t0) * 1000.0

    # 프레임별 SSIM (각 프레임은 중앙 1.4배율 크롭 후 비교)
    scores: list[float] = []
    per_frame_ms: list[float] = []
    t_ssim0 = time.perf_counter()
    for i, fb in enumerate(frame_bytes_list):
        ti = time.perf_counter()
        frame_gray = _load_gray(fb, size, crop_center=crop_center)
        s = float(ssim(target_gray, frame_gray, data_range=255))
        scores.append(s)
        dt = (time.perf_counter() - ti) * 1000.0
        per_frame_ms.append(dt)
        print(f"[scan-peak] frame {i:>3}/{n}: ssim={s:.4f} ({dt:.1f} ms)")
    ssim_total_ms = (time.perf_counter() - t_ssim0) * 1000.0

    # 정점 탐색(원본 또는 스무딩)
    curve = _moving_average(scores, smooth_window)
    peak_index = int(np.argmax(curve))
    peak_ssim = float(scores[peak_index])

    total_ms = (time.perf_counter() - t_start) * 1000.0
    per_frame_avg = (sum(per_frame_ms) / len(per_frame_ms)) if per_frame_ms else 0.0

    print(
        f"[scan-peak] DONE frames={n}, size={size}, smooth={smooth_window} → "
        f"peak_index={peak_index}, peak_ssim={peak_ssim:.4f} | "
        f"target_load={target_load_ms:.1f}ms, ssim_total={ssim_total_ms:.1f}ms, "
        f"per_frame_avg={per_frame_avg:.1f}ms, total={total_ms:.1f}ms"
    )

    return {
        "peak_index": peak_index,
        "peak_ssim": peak_ssim,
        "scores": [round(s, 6) for s in scores],
        "frame_count": n,
        "timing_ms": {
            "target_load": round(target_load_ms, 1),
            "ssim_total": round(ssim_total_ms, 1),
            "per_frame_avg": round(per_frame_avg, 1),
            "total": round(total_ms, 1),
        },
    }
