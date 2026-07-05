"""캡처 장면 태그 ↔ reference DB(metadata) 태그 매칭 점수.

capture가 만든 result.json의 "tags"(schema.yaml 형식)와
config/metadata.json 각 reference 레코드의 태그를 필드별로 비교해
[0,1] 범위의 태그 일치도를 계산한다.

best_crop 매칭에서 patch 유사도에 이 점수를 tag_weight로 가중해 더한다.
캡처 태그가 없거나 metadata가 없으면 None을 돌려 patch-only로 degrade한다.

두 태그 dict는 같은 스키마를 공유한다:
  time, weather, location  : 단일 값
  facing, landmark         : 단일 값 또는 null
  flags                    : 리스트
  person_count             : 정수. 캡처는 선택된 person 타깃 개수로 채움(차이 기반 근접도)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 단일값 필드 가중치 (landmark를 가장 강하게)
FIELD_WEIGHTS: dict[str, float] = {
    "landmark": 3.0,
    "time": 1.0,
    "weather": 1.0,
    "location": 1.0,
    "facing": 1.0,
}
# flags 리스트(자카드 유사도)에 주는 가중치
FLAG_WEIGHT = 2.0
# person_count 근접도(카운트라 정확일치 대신 차이 기반)에 주는 가중치
PERSON_COUNT_WEIGHT = 1.0

# top-k 추림 전에 "반드시 일치해야" 하는 필수 태그(하드 필터)의 기본값.
# 캡처가 값을 아는 필드만 강제한다(모르는 필드는 강제 못 함). person_count는 정확일치.
REQUIRED_DEFAULT = ("landmark", "location", "person_count")


def _passes_required(cap: dict[str, Any], ref: dict[str, Any], fields) -> bool:
    """ref가 필수 태그를 모두 만족하는지. 캡처값이 None인 필드는 강제하지 않는다."""
    for f in fields:
        cap_v = cap.get(f)
        if cap_v is None:
            continue
        ref_v = ref.get(f)
        if f == "person_count":
            if ref_v is None or int(ref_v) != int(cap_v):
                return False
        elif ref_v != cap_v:
            return False
    return True


def required_filter_indices(
    capture_tags: dict[str, Any] | None,
    meta_records: list[dict[str, Any]] | None,
    fields,
) -> np.ndarray | None:
    """필수 태그가 모두 일치하는 reference 인덱스 배열을 반환한다.

    - 적용 불가(캡처 태그/metadata/필드 없음) → None
    - 적용했으나 통과 0개 → 빈 배열 (호출부에서 필터 무시 판단)
    """
    if not capture_tags or not meta_records or not fields:
        return None
    keep = [i for i, r in enumerate(meta_records) if _passes_required(capture_tags, r, fields)]
    return np.array(keep, dtype=np.int64)


def _record_score(cap: dict[str, Any], ref: dict[str, Any]) -> float:
    """캡처 태그 cap과 reference 레코드 ref의 태그 일치도 [0,1].

    캡처가 값을 가진 필드만 분모에 넣어(누락 필드가 점수를 희석하지 않게) 정규화한다.
    """
    total = 0.0
    got = 0.0

    for field, w in FIELD_WEIGHTS.items():
        cap_v = cap.get(field)
        if cap_v is None:
            continue  # 캡처에 값이 없으면 이 필드는 비교 대상에서 제외
        total += w
        if ref.get(field) == cap_v:
            got += w

    cap_flags = set(cap.get("flags") or [])
    if cap_flags:
        ref_flags = set(ref.get("flags") or [])
        union = cap_flags | ref_flags
        jaccard = len(cap_flags & ref_flags) / len(union) if union else 0.0
        total += FLAG_WEIGHT
        got += FLAG_WEIGHT * jaccard

    cap_pc = cap.get("person_count")
    ref_pc = ref.get("person_count")
    if cap_pc is not None and ref_pc is not None:
        # 카운트 차이가 클수록 감점: 차이0→1.0, 차이1→0.5, 차이2→0.33 ...
        closeness = 1.0 / (1.0 + abs(int(cap_pc) - int(ref_pc)))
        total += PERSON_COUNT_WEIGHT
        got += PERSON_COUNT_WEIGHT * closeness

    if total <= 0:
        return 0.0
    return got / total


def compute_tag_bonus(
    capture_tags: dict[str, Any] | None,
    meta_records: list[dict[str, Any]] | None,
) -> np.ndarray | None:
    """reference N개 각각에 대한 태그 일치도 벡터 (N,) [0,1]을 반환.

    캡처 태그나 metadata가 없으면 None(→ patch-only).
    """
    if not capture_tags or not meta_records:
        return None
    return np.array(
        [_record_score(capture_tags, r) for r in meta_records],
        dtype=np.float32,
    )
