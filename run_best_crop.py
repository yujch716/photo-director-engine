"""best_crop 파이프라인 CLI 진입점.

실제 로직은 services.best_crop.select_best_crop 에 있다.
이 파일은 argparse로 인자만 받아 파이프라인을 호출하는 얇은 래퍼다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from services.best_crop import select_best_crop


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # 단일 patch 임베딩 DB. 기본값 None → config/ 아래에서 읽는다.
    parser.add_argument("--bank-dir", type=Path, default=None)

    parser.add_argument("--zoom-ratio", type=float, default=1.5)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)

    parser.add_argument("--contain-thres", type=float, default=0.90)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.0)
    parser.add_argument("--selected-require", choices=["all", "any"], default="all")
    parser.add_argument("--fallback-topk", type=int, default=12)

    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--dino-chunk-size", type=int, default=64)
    # 2단계 검색: 코스 벡터로 상위 K개만 추린 뒤 patch 정밀 비교 (K>=N이면 추림 없음)
    parser.add_argument("--dino-shortlist-k", type=int, default=200)
    # 태그 가중치. 0이면 순수 patch 매칭. 최종점수 = patch_sim + tag_weight*tag_bonus
    parser.add_argument("--tag-weight", type=float, default=0.3)
    # 필수 태그(top-k 전 하드 필터). 미지정→기본(landmark location person_count),
    # "--required-tags" 만 주면 빈 리스트→필터 끔. 통과 0개면 자동으로 전체 사용.
    parser.add_argument("--required-tags", nargs="*", default=None)

    parser.add_argument("--save-candidates", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    select_best_crop(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bank_dir=args.bank_dir,
        zoom_ratio=args.zoom_ratio,
        cols=args.cols,
        rows=args.rows,
        contain_thres=args.contain_thres,
        edge_margin_ratio=args.edge_margin_ratio,
        selected_require=args.selected_require,
        fallback_topk=args.fallback_topk,
        dino_batch_size=args.dino_batch_size,
        dino_chunk_size=args.dino_chunk_size,
        dino_shortlist_k=args.dino_shortlist_k,
        tag_weight=args.tag_weight,
        required_tags=args.required_tags,
        save_all_candidates=args.save_candidates,
    )


if __name__ == "__main__":
    main()
