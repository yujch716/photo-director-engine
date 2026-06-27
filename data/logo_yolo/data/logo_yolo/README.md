# Synthetic YouTube-logo YOLO dataset

This dataset is generated for a small logo detector used before mask-based inpainting.

- Class: `youtube_logo`
- Train images: 180
- Val images: 45
- YOLO labels: `labels/train`, `labels/val`
- Optional binary masks for inpainting experiments: `masks/train`, `masks/val`

The graphics are synthetic YouTube-like play-button overlays on generated backgrounds.
Use real collected screenshots later to improve robustness.

Training command with the provided `logo_yolo_train.py`:

```bash
pip install ultralytics pyyaml
python logo_yolo_train.py --data data/logo_yolo/dataset.yaml --epochs 80 --imgsz 640 --batch 8
```
