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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

DEFAULT_SSIM_SIZE = 256

# 스캔 프레임/타깃 저장 루트. 기존 폴더들과 섞이지 않게 scan-peak/ 아래로 분리.
SCAN_PEAK_DIR = Path("drone-data/scan-peak")


def save_scan_peak_inputs(
    frame_bytes_list: list[bytes],
    target_bytes: bytes,
    result: dict[str, Any] | None = None,
    data_dir: Path | None = None,
) -> str:
    """스캔 프레임 전부 + target(+정점 결과)을 drone-data/scan-peak/<타임스탬프>/에 저장.

    저장 파일:
        frame_000.jpg, frame_001.jpg, ...  : 받은 프레임 전부(업로드 순서)
        target.jpg                          : 비교 기준(최종구도)
        scan.json                           : (선택) find_scan_peak 결과(정점/scores/타이밍)
    Returns:
        저장 폴더명(saved).
    """
    base = data_dir or SCAN_PEAK_DIR
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder = base / name
    # 같은 밀리초 중복 호출에도 덮어쓰지 않도록 유일한 폴더명 보장.
    suffix = 1
    while folder.exists():
        folder = base / f"{name}_{suffix}"
        suffix += 1
    name = folder.name
    folder.mkdir(parents=True)

    for i, fb in enumerate(frame_bytes_list):
        (folder / f"frame_{i:03d}.jpg").write_bytes(fb)
    (folder / "target.jpg").write_bytes(target_bytes)
    if result is not None:
        (folder / "scan.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return name


def _load_gray(image_bytes: bytes, size: int, crop_center: bool = False) -> np.ndarray:
    """이미지 바이트 → (선택 중앙 2배율 크롭) grayscale + (size×size) uint8 배열. SSIM 입력용.

    crop_center=True면 중앙 절반(= 2배율 뷰)만 잘라서 사용한다.
    (capture의 original_2x와 동일한 crop box)
    """
    img = Image.open(io.BytesIO(image_bytes))
    if crop_center:
        W, H = img.size
        img = img.crop((W // 4, H // 4, W - W // 4, H - H // 4))
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

    각 프레임은 기본적으로 중앙 2배율 크롭 후 비교한다(target은 이미 2배율 구도라 크롭 안 함).
    → 프레임과 target의 스케일을 맞춰 비교.

    Args:
        frame_bytes_list: 프레임 이미지 바이트들(시간순, 0번=스캔 시작). 드론 전체 프레임.
        target_bytes: 최종구도 이미지 바이트(비교 기준). 이미 2배율 크롭된 상태 가정 → 크롭 안 함.
        size: SSIM 계산용 리사이즈 크기(정사각). 작을수록 빠름.
        smooth_window: 정점 탐색 전 이동평균 창(1이면 스무딩 없음=단순 최대).
        crop_center: True(기본)면 각 프레임을 중앙 2배율 크롭 후 비교. False면 프레임 전체 비교.

    Returns:
        {peak_index, peak_ssim, scores, frame_count, timing_ms}
        (peak_ssim/scores는 원본 SSIM 값. 정점 index만 스무딩 결과로 고른다.)
    """
    t_start = time.perf_counter()
    n = len(frame_bytes_list)

    if n == 0:
        raise ValueError("frames가 비어 있음")

    # target 로드 (이미 2배율 구도라 크롭하지 않음)
    t0 = time.perf_counter()
    target_gray = _load_gray(target_bytes, size, crop_center=False)
    target_load_ms = (time.perf_counter() - t0) * 1000.0

    # 프레임별 SSIM (각 프레임은 중앙 2배율 크롭 후 비교)
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
