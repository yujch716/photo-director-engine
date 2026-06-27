from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


DEFAULT_DATASET = Path("data/logo_yolo/dataset.yaml")
DEFAULT_BASE_MODEL = Path("yolov8n.pt")
DEFAULT_OUTPUT = Path("models/logo_yolo.pt")


def _validate_dataset_yaml(dataset_yaml: Path) -> None:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_keys = {"path", "train", "val", "names"}
    missing = required_keys - set(data)
    if missing:
        raise ValueError(f"Dataset yaml is missing required keys: {sorted(missing)}")

    names = data["names"]
    if not isinstance(names, (list, dict)) or len(names) == 0:
        raise ValueError("Dataset yaml must define at least one class name.")

    dataset_root = Path(data["path"])
    train_dir = dataset_root / data["train"]
    val_dir = dataset_root / data["val"]
    if not train_dir.exists() or not any(train_dir.glob("*")):
        raise FileNotFoundError(f"Training images not found: {train_dir}")
    if not val_dir.exists() or not any(val_dir.glob("*")):
        raise FileNotFoundError(f"Validation images not found: {val_dir}")


def train_logo_yolo(
    dataset_yaml: str | Path = DEFAULT_DATASET,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 20,
) -> Path:
    dataset_yaml = Path(dataset_yaml)
    base_model = Path(base_model)
    output_path = Path(output_path)

    _validate_dataset_yaml(dataset_yaml)

    from ultralytics import YOLO

    model = YOLO(str(base_model))
    result = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project="runs/logo_yolo",
        name="train",
        exist_ok=True,
    )

    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished, but best weights were not found: {best_weights}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO logo/arrow detector for mask-based inpainting.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Path to YOLO dataset.yaml")
    parser.add_argument("--base", default=str(DEFAULT_BASE_MODEL), help="Base YOLO weights, e.g. yolov8n.pt")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output weights path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    output_path = train_logo_yolo(
        dataset_yaml=args.data,
        base_model=args.base,
        output_path=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
    )
    print(f"[OK] Logo YOLO weights saved to {output_path}")


if __name__ == "__main__":
    main()
