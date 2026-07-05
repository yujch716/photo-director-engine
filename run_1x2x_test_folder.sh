#!/usr/bin/env bash
set -e

cd /data/MyGit/photo-director-engine

FOLDER_NAME="${1:-20260702_222942_010}"

INPUT_DIR="/data/MyGit/photo-director-engine/drone-data/${FOLDER_NAME}"
OUTPUT_DIR="/data/MyGit/photo-director-engine/drone-data/${FOLDER_NAME}_result"

mkdir -p "$OUTPUT_DIR"

echo "[RUN] folder = $FOLDER_NAME"
echo "[RUN] input  = $INPUT_DIR"
echo "[RUN] output = $OUTPUT_DIR"

python models/best_crop_1x2x_folder.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --general-embedding-dir /data/MyGit/photo-director-engine/imbeddingdata/dinov2imbedding \
  --hand-embedding-dir /data/MyGit/photo-director-engine/imbeddingdata/hand_dinov2imbedding \
  --zoom-ratio 2.0 \
  --cols 8 \
  --rows 6 \
  --contain-thres 0.80 \
  --edge-margin-ratio 0.0 \
  --selected-require all \
  --fallback-topk 12 \
  --dino-batch-size 16 \
  --patch-topk 50 \
  --save-candidates

echo "[DONE] result files:"
find "$OUTPUT_DIR" -maxdepth 2 -type f -print
