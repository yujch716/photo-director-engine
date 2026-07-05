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

    parser.add_argument("--input-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/drone-data/20260702_222942_010"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/drone-data/20260702_222942_010_result"))
    parser.add_argument("--general-embedding-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/imbeddingdata/dinov2imbedding"))
    parser.add_argument("--hand-embedding-dir", type=Path, default=Path("/data/MyGit/photo-director-engine/imbeddingdata/hand_dinov2imbedding"))
    parser.add_argument("--embedding-dir", type=Path, default=None)  # 수동 override용

    parser.add_argument("--zoom-ratio", type=float, default=2.0)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)

    parser.add_argument("--contain-thres", type=float, default=0.90)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.0)
    parser.add_argument("--selected-require", choices=["all", "any"], default="all")
    parser.add_argument("--fallback-topk", type=int, default=12)

    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--patch-topk", type=int, default=20)

    parser.add_argument("--save-candidates", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    select_best_crop(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        general_embedding_dir=args.general_embedding_dir,
        hand_embedding_dir=args.hand_embedding_dir,
        embedding_dir=args.embedding_dir,
        zoom_ratio=args.zoom_ratio,
        cols=args.cols,
        rows=args.rows,
        contain_thres=args.contain_thres,
        edge_margin_ratio=args.edge_margin_ratio,
        selected_require=args.selected_require,
        fallback_topk=args.fallback_topk,
        dino_batch_size=args.dino_batch_size,
        patch_topk=args.patch_topk,
        save_all_candidates=args.save_candidates,
    )


if __name__ == "__main__":
    main()
