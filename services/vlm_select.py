"""틸트 sweep 최종 선택: 프레임들에 NIMA를 매기고 VLM으로 최종 1장을 고른다.

VLM(Qwen2.5-VL)이 없거나 응답 파싱이 실패하면 NIMA 최고점 장으로 fallback한다.
선택 결과와 전체 프레임을 drone-data/final_<타임스탬프>/에 저장(검증용).
"""

from __future__ import annotations

import io
import json
import re
import time
from typing import Any

from PIL import Image

from models import qwen_vl
from models.nima import run_nima_score
from services.session_paths import DRONE_DATA_DIR, make_ts, resolve_save_dir

SAVE_DEBUG = True


def build_prompt(items: list[dict[str, Any]]) -> str:
    """VLM에게 줄 지시문. items = [{index, angle, nima}]."""
    lines = []
    for it in items:
        angle = it.get("angle")
        angle_s = f"{angle:+g}도" if isinstance(angle, (int, float)) else "미상"
        lines.append(f"- index {it['index']}: 짐벌 각도 {angle_s}, NIMA 미학점수 {it['nima']:.3f}")
    photo_block = "\n".join(lines)

    return (
        "당신은 여러 짐벌 틸트 각도로 촬영한 사진 중 최고의 한 장을 고르는 사진 심사위원입니다.\n"
        "아래 사진들은 위 목록 순서대로 index 0부터 주어집니다. 각 사진의 각도와 NIMA 점수는 다음과 같습니다:\n"
        f"{photo_block}\n\n"
        "판단 기준(우선순위 높은 순):\n"
        "1. 주 피사체(인물/랜드마크)가 잘리지 않았는가 — 머리·발·핵심부가 프레임 밖으로 나가면 감점. "
        "잘린 사진은 아무리 예뻐도 우선순위 낮음.\n"
        "2. 구도 — 주 피사체가 적절히 배치됐는가(삼분할, 과도한 치우침·중앙박힘 아닌지), "
        "수평이 자연스러운지, 머리 위 여백(headroom)이 적절한지.\n"
        "3. 미학 — NIMA 점수는 참고만, 맹신하지 말고 위 기준과 종합.\n"
        "4. 틸트 각도가 피사체를 자연스럽게 담는가(과도하게 올려다보거나 내려다보지 않는지).\n\n"
        "출력은 JSON만, 다른 말 없이. 형식:\n"
        '{"selected_index": <정수>, "reason": "<간단한 선택 이유>", '
        '"rejected_notes": {"<index>": "<탈락 이유(잘림 등)>"}, "defect_flag": <true|false>}\n'
        "모든 사진이 주 피사체 잘림 등 근본 결함이면 defect_flag를 true로 하되, "
        "그래도 그중 최선의 selected_index는 반드시 고르세요.\n"
    )


def parse_vlm_json(text: str) -> dict[str, Any] | None:
    """VLM 응답에서 JSON 객체를 추출·파싱. 실패 시 None."""
    if not text:
        return None
    # ```json ... ``` 펜스 제거
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # 첫 { 부터 마지막 } 까지
        s, e = text.find("{"), text.rfind("}")
        candidate = text[s : e + 1] if (s != -1 and e != -1 and e > s) else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _save_final(
    frames_bytes: list[bytes],
    selected_index: int,
    meta: dict[str, Any],
    session_id: str | None = None,
    data_dir=None,
) -> str:
    # 세션 없으면 drone-data/final_<ts>/, 있으면 drone-data/<session_id>/vlm/
    base = data_dir or DRONE_DATA_DIR
    folder = resolve_save_dir(session_id, "vlm", legacy=base / f"final_{make_ts()}", data_dir=base)

    for i, fb in enumerate(frames_bytes):
        (folder / f"frame_{i:03d}.jpg").write_bytes(fb)
    # 선택된 최종 사진
    (folder / "final.jpg").write_bytes(frames_bytes[selected_index])
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(folder.relative_to(base))


def select_final(
    frames_bytes: list[bytes],
    angles: list[float | int | None] | None = None,
    session_id: str | None = None,
    save: bool | None = None,
) -> dict[str, Any]:
    """프레임들에 NIMA를 매기고 VLM으로 최종 1장을 선택한다.

    Returns:
        {selected_index, selected_angle, reason, nima_scores, defect_flag, source, timing_ms}
    Raises:
        ValueError: frames가 비어 있을 때.
    """
    t_start = time.perf_counter()
    n = len(frames_bytes)
    if n == 0:
        raise ValueError("frames가 비어 있음")
    if angles is None:
        angles = [None] * n

    images = [Image.open(io.BytesIO(fb)).convert("RGB") for fb in frames_bytes]

    # 1) NIMA
    nima_scores: list[float] = []
    t_n0 = time.perf_counter()
    for i, fb in enumerate(frames_bytes):
        ti = time.perf_counter()
        s = float(run_nima_score(fb)["score"])
        nima_scores.append(round(s, 4))
        print(f"[vlm-select] NIMA frame {i:>2}: {s:.4f} ({(time.perf_counter() - ti) * 1000:.1f} ms)")
    nima_total_ms = (time.perf_counter() - t_n0) * 1000.0

    items = [
        {"index": i, "angle": (angles[i] if i < len(angles) else None), "nima": nima_scores[i]}
        for i in range(n)
    ]
    prompt = build_prompt(items)

    # 2) VLM 호출 → 파싱 (실패 시 fallback)
    source = "vlm"
    reason = ""
    defect_flag = False
    rejected_notes: dict[str, Any] | None = None
    vlm_raw = None
    parsed = None
    vlm_ms = 0.0

    try:
        t_v0 = time.perf_counter()
        vlm_raw = qwen_vl.run_vlm(images, prompt)
        vlm_ms = (time.perf_counter() - t_v0) * 1000.0
        parsed = parse_vlm_json(vlm_raw)
        if parsed is None or not isinstance(parsed.get("selected_index"), int):
            raise ValueError(f"VLM JSON 파싱 실패 또는 selected_index 없음: {vlm_raw!r}")
        selected_index = int(parsed["selected_index"])
        reason = str(parsed.get("reason", ""))
        defect_flag = bool(parsed.get("defect_flag", False))
        rejected_notes = parsed.get("rejected_notes")
    except Exception as e:
        source = "fallback"
        selected_index = int(max(range(n), key=lambda i: nima_scores[i]))
        reason = "fallback: VLM 사용 불가/파싱 실패 → NIMA 최고점 선택"
        print(f"[vlm-select] WARN VLM fallback: {e}")

    # index clamp
    if not (0 <= selected_index < n):
        print(f"[vlm-select] WARN selected_index 범위 밖({selected_index}) → NIMA 최고점으로 보정")
        selected_index = int(max(range(n), key=lambda i: nima_scores[i]))
        source = "fallback"

    selected_angle = angles[selected_index] if selected_index < len(angles) else None
    total_ms = (time.perf_counter() - t_start) * 1000.0

    print(
        f"[vlm-select] DONE n={n} source={source} → selected_index={selected_index}, "
        f"angle={selected_angle}, defect={defect_flag} | "
        f"nima_total={nima_total_ms:.1f}ms vlm={vlm_ms:.1f}ms total={total_ms:.1f}ms"
    )

    result = {
        "selected_index": selected_index,
        "selected_angle": selected_angle,
        "reason": reason,
        "nima_scores": nima_scores,
        "defect_flag": defect_flag,
        "source": source,
        "timing_ms": {
            "nima_total": round(nima_total_ms, 1),
            "vlm": round(vlm_ms, 1),
            "total": round(total_ms, 1),
        },
    }

    if SAVE_DEBUG if save is None else save:
        try:
            meta = {
                **result,
                "angles": angles,
                "rejected_notes": rejected_notes,
                "vlm_raw": vlm_raw,
                "vlm_parsed": parsed,
            }
            saved = _save_final(frames_bytes, selected_index, meta, session_id=session_id)
            print(f"[vlm-select] saved: drone-data/{saved}")
        except Exception as e:
            print(f"[vlm-select] WARN 저장 실패(무시): {e}")

    return result
