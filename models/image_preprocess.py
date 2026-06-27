from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


TARGET_PIXELS = 9_144_576
RESOLUTION_SIMILARITY_TOLERANCE = 0.15
TARGET_SHARPNESS = 292.00286771520047
TARGET_NOISE_SIGMA = 0.8339511
TARGET_BPP = 0.5406343607401809


def _image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read image bytes with PIL.") from exc

    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise ValueError("JPEG encoding failed.")
    return encoded.tobytes()


def _jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode JPEG bytes with OpenCV.")
    return img


def resize_long_side(img_bgr: np.ndarray, long_side: int = 1024) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("img_bgr is None. Check the image path or bytes.")

    h, w = img_bgr.shape[:2]
    scale = long_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)


def calc_sharpness(img_bgr: np.ndarray, long_side: int = 1024) -> float:
    img_small = resize_long_side(img_bgr, long_side=long_side)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calc_noise_sigma(img_bgr: np.ndarray, long_side: int = 1024) -> float:
    img_small = resize_long_side(img_bgr, long_side=long_side)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - smooth
    sigma = np.median(np.abs(residual - np.median(residual))) / 0.6745
    return float(sigma)


def calc_bpp_from_bytes(image_bytes: bytes, width: int, height: int) -> float:
    return float(len(image_bytes) / (width * height))


def calc_bpp_from_path(image_path: str | os.PathLike[str]) -> float:
    img = Image.open(image_path)
    w, h = img.size
    return float(os.path.getsize(image_path) / (w * h))


def resize_keep_ratio_by_pixels(
    img_bgr: np.ndarray,
    target_pixels: int = TARGET_PIXELS,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    current_pixels = w * h
    if current_pixels <= 0:
        raise ValueError("Invalid image size.")

    scale = math.sqrt(target_pixels / current_pixels)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)


def classify_resolution_similarity(
    width: int,
    height: int,
    target_pixels: int = TARGET_PIXELS,
    tolerance: float = RESOLUTION_SIMILARITY_TOLERANCE,
) -> dict[str, Any]:
    pixels = width * height
    if pixels <= 0:
        raise ValueError("Invalid image size.")

    lower_bound = target_pixels * (1.0 - tolerance)
    upper_bound = target_pixels * (1.0 + tolerance)

    if lower_bound <= pixels <= upper_bound:
        label = "similar"
    elif pixels < lower_bound:
        label = "lower_not_similar"
    else:
        label = "higher_not_similar"

    return {
        "classification": label,
        "pixels": pixels,
        "target_pixels": target_pixels,
        "pixel_ratio": pixels / target_pixels,
        "tolerance": tolerance,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def resize_by_resolution_policy(
    img_bgr: np.ndarray,
    target_pixels: int = TARGET_PIXELS,
    similarity_tolerance: float = RESOLUTION_SIMILARITY_TOLERANCE,
    use_pretrained_cnn_sr: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    resolution = classify_resolution_similarity(
        w,
        h,
        target_pixels=target_pixels,
        tolerance=similarity_tolerance,
    )
    classification = resolution["classification"]
    current_pixels = resolution["pixels"]
    working = img_bgr
    sr_report: dict[str, Any] = {
        "backend": None,
        "model": None,
        "applied": False,
        "error": None,
    }

    if classification == "lower_not_similar":
        super_resolution_candidate = True
        interpolation = cv2.INTER_CUBIC

        if use_pretrained_cnn_sr:
            try:
                from models.super_resolution import apply_pretrained_cnn_sr

                working, sr_report = apply_pretrained_cnn_sr(img_bgr)
                action = "pretrained_cnn_sr_then_resize"
            except Exception as exc:
                action = "sr_failed_fallback_resize"
                sr_report = {
                    "backend": "real_esrgan",
                    "model": "RealESRGAN_x4plus",
                    "applied": False,
                    "error": str(exc),
                }
        else:
            action = "sr_candidate_then_resize"
    elif classification == "higher_not_similar":
        action = "downscale"
        interpolation = cv2.INTER_AREA
        super_resolution_candidate = False
    else:
        action = "resize"
        super_resolution_candidate = False

    wh, ww = working.shape[:2]
    working_pixels = ww * wh
    scale = math.sqrt(target_pixels / working_pixels)
    if classification == "similar":
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if working is not img_bgr:
        new_w = max(1, int(round(ww * scale)))
        new_h = max(1, int(round(wh * scale)))
    out = cv2.resize(working, (new_w, new_h), interpolation=interpolation)

    report = {
        **resolution,
        "original_width": w,
        "original_height": h,
        "sr_width": ww,
        "sr_height": wh,
        "sr_pixels": working_pixels,
        "output_width": new_w,
        "output_height": new_h,
        "output_pixels": new_w * new_h,
        "scale": scale,
        "action": action,
        "super_resolution_candidate": super_resolution_candidate,
        "pretrained_cnn_sr_enabled": use_pretrained_cnn_sr,
        "pretrained_cnn_sr_applied": sr_report.get("applied", False),
        "pretrained_cnn_sr_backend": sr_report.get("backend"),
        "pretrained_cnn_sr_model": sr_report.get("model"),
        "pretrained_cnn_sr_error": sr_report.get("error"),
        "aspect_ratio_policy": "keep_original_ratio",
        "crop": False,
        "padding": False,
        "force_16_9": False,
    }

    return out, report


def unsharp_mask(
    img_bgr: np.ndarray,
    amount: float = 0.35,
    radius: float = 1.0,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def normalize_sharpness(
    img_bgr: np.ndarray,
    target_sharpness: float = TARGET_SHARPNESS,
    tolerance: float = 0.15,
    max_iter: int = 3,
    allow_soften: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    lower = target_sharpness * (1.0 - tolerance)
    upper = target_sharpness * (1.0 + tolerance)
    before = calc_sharpness(img_bgr)
    action = "none"
    out = img_bgr

    for _ in range(max_iter):
        current = calc_sharpness(out)
        if current < lower:
            out = unsharp_mask(out, amount=0.35, radius=1.0)
            action = "sharpen"
        elif current > upper and allow_soften:
            out = cv2.GaussianBlur(out, (0, 0), 0.35)
            action = "soften"
        else:
            break

    after = calc_sharpness(out)
    return out, {
        "target": target_sharpness,
        "before": before,
        "after": after,
        "tolerance": tolerance,
        "action": action,
        "within_range": lower <= after <= upper,
    }


def _seed_from_bytes(image_bytes: bytes) -> int:
    digest = hashlib.md5(image_bytes).hexdigest()
    return int(digest[:8], 16)


def add_gaussian_noise(
    img_bgr: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    if sigma <= 0:
        return img_bgr

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_noise(
    img_bgr: np.ndarray,
    target_noise: float = TARGET_NOISE_SIGMA,
    tolerance: float = 0.20,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    lower = target_noise * (1.0 - tolerance)
    upper = target_noise * (1.0 + tolerance)
    before = calc_noise_sigma(img_bgr)
    action = "none"
    out = img_bgr

    if before > upper:
        strength = int(np.clip((before - target_noise) * 2.0, 3, 10))
        out = cv2.fastNlMeansDenoisingColored(
            out,
            None,
            h=strength,
            hColor=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        action = f"denoise_h{strength}"
    elif before < lower:
        sigma_to_add = max(0.0, target_noise - before)
        out = add_gaussian_noise(out, sigma=sigma_to_add, seed=seed)
        action = f"add_noise_sigma{sigma_to_add:.4f}"

    after = calc_noise_sigma(out)
    return out, {
        "target": target_noise,
        "before": before,
        "after": after,
        "tolerance": tolerance,
        "action": action,
        "within_range": lower <= after <= upper,
    }


def encode_jpeg_match_bpp(
    img_bgr: np.ndarray,
    target_bpp: float = TARGET_BPP,
    quality_min: int = 50,
    quality_max: int = 95,
    step: int = 2,
) -> tuple[bytes, dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    best_bytes: bytes | None = None
    best_quality = quality_min
    best_bpp = -1.0
    best_diff = float("inf")

    for q in range(quality_min, quality_max + 1, step):
        jpg_bytes = _bgr_to_jpeg_bytes(img_bgr, quality=q)
        bpp = calc_bpp_from_bytes(jpg_bytes, w, h)
        diff = abs(bpp - target_bpp)
        if diff < best_diff:
            best_diff = diff
            best_bytes = jpg_bytes
            best_quality = q
            best_bpp = bpp

    if best_bytes is None:
        raise ValueError("JPEG quality search failed.")

    return best_bytes, {
        "target_bpp": target_bpp,
        "after_bpp": best_bpp,
        "quality": best_quality,
        "diff": best_diff,
    }


def run_quality_normalization(
    image_bytes: bytes,
    *,
    target_pixels: int = TARGET_PIXELS,
    resolution_similarity_tolerance: float = RESOLUTION_SIMILARITY_TOLERANCE,
    use_pretrained_cnn_sr: bool = True,
    target_sharpness: float = TARGET_SHARPNESS,
    target_noise: float = TARGET_NOISE_SIGMA,
    target_bpp: float = TARGET_BPP,
    sharpness_tolerance: float = 0.15,
    noise_tolerance: float = 0.20,
    allow_soften: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    seed = _seed_from_bytes(image_bytes)
    original = _image_bytes_to_bgr(image_bytes)
    oh, ow = original.shape[:2]

    original_report = {
        "width": ow,
        "height": oh,
        "pixels": ow * oh,
        "sharpness": calc_sharpness(original),
        "noise_sigma": calc_noise_sigma(original),
    }

    out, resolution_report = resize_by_resolution_policy(
        original,
        target_pixels=target_pixels,
        similarity_tolerance=resolution_similarity_tolerance,
        use_pretrained_cnn_sr=use_pretrained_cnn_sr,
    )
    rh, rw = out.shape[:2]

    out, noise_report = normalize_noise(
        out,
        target_noise=target_noise,
        tolerance=noise_tolerance,
        seed=seed,
    )
    out, sharpness_report = normalize_sharpness(
        out,
        target_sharpness=target_sharpness,
        tolerance=sharpness_tolerance,
        allow_soften=allow_soften,
    )
    normalized_jpeg_bytes, compression_report = encode_jpeg_match_bpp(
        out,
        target_bpp=target_bpp,
    )

    final_img = _jpeg_bytes_to_bgr(normalized_jpeg_bytes)
    fh, fw = final_img.shape[:2]

    final_report = {
        "width": fw,
        "height": fh,
        "pixels": fw * fh,
        "sharpness": calc_sharpness(final_img),
        "noise_sigma": calc_noise_sigma(final_img),
        "bpp": calc_bpp_from_bytes(normalized_jpeg_bytes, fw, fh),
    }

    report = {
        "target": {
            "pixels": target_pixels,
            "resolution_similarity_tolerance": resolution_similarity_tolerance,
            "use_pretrained_cnn_sr": use_pretrained_cnn_sr,
            "sharpness": target_sharpness,
            "noise_sigma": target_noise,
            "bpp": target_bpp,
            "aspect_ratio_policy": "keep_original_ratio",
        },
        "original": original_report,
        "after_resize": {
            "width": rw,
            "height": rh,
            "pixels": rw * rh,
        },
        "resolution": resolution_report,
        "noise": noise_report,
        "sharpness": sharpness_report,
        "compression": compression_report,
        "final": final_report,
    }

    return normalized_jpeg_bytes, report


def preprocess_file(input_path: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> dict[str, Any]:
    input_path = str(input_path)
    output_path = str(output_path)

    with open(input_path, "rb") as f:
        image_bytes = f.read()

    normalized_bytes, report = run_quality_normalization(image_bytes)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(normalized_bytes)

    report["input_path"] = input_path
    report["output_path"] = output_path
    return report


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess_folder(input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or not _is_image_file(path):
            continue

        out_path = output_dir / f"{path.stem}_normalized.jpg"
        try:
            report = preprocess_file(path, out_path)
            reports.append(report)
            print(f"[OK] {path.name} -> {out_path.name}")
        except Exception as exc:
            print(f"[FAIL] {path.name}: {exc}")

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize SNS image quality metrics for NIMA fine-tuning.")
    parser.add_argument("--input", required=True, help="Input image file or folder")
    parser.add_argument("--output", required=True, help="Output image file or folder")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        preprocess_folder(input_path, output_path)
    else:
        preprocess_file(input_path, output_path)


if __name__ == "__main__":
    main()
