#!/usr/bin/env python3
"""
Update dataset paths inside configs/dataset/drone_detection.yml.

Example:
    python tools/dataset/update_dataset_paths.py \
        --train_dir coco_dataset/coco_dataset_train \
        --val_dir coco_dataset/coco_dataset_val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ruamel.yaml import YAML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update dataset paths in the drone config.")
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Train split directory containing images/ and annotations/instances_train.json.",
    )
    parser.add_argument(
        "--val_dir",
        required=True,
        help="Validation split directory containing images/ and annotations/instances_val.json.",
    )
    parser.add_argument(
        "--config",
        default="configs/dataset/drone_detection.yml",
        help="Path to the dataset configuration file to update.",
    )
    return parser.parse_args()


def validate_split(split_dir: Path, split_name: str) -> tuple[Path, Path]:
    images_dir = split_dir / "images"
    annotations_file = split_dir / "annotations" / f"instances_{split_name}.json"
    if not images_dir.exists():
        raise FileNotFoundError(f"{split_name} images directory not found: {images_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"{split_name} annotations not found: {annotations_file}")
    return images_dir, annotations_file


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    train_dir = Path(args.train_dir).expanduser().resolve()
    val_dir = Path(args.val_dir).expanduser().resolve()

    train_images, train_annotations = validate_split(train_dir, "train")
    val_images, val_annotations = validate_split(val_dir, "val")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(config_path, "r") as f:
        config = yaml.load(f)

    train_dataset = config.get("train_dataloader", {}).get("dataset")
    val_dataset = config.get("val_dataloader", {}).get("dataset")
    if train_dataset is None or val_dataset is None:
        raise KeyError("train_dataloader.dataset or val_dataloader.dataset missing from config.")

    train_dataset["img_folder"] = str(train_images)
    train_dataset["ann_file"] = str(train_annotations)
    val_dataset["img_folder"] = str(val_images)
    val_dataset["ann_file"] = str(val_annotations)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print("Updated dataset config:")
    print(f"  train img_folder -> {train_images}")
    print(f"  train ann_file   -> {train_annotations}")
    print(f"  val img_folder   -> {val_images}")
    print(f"  val ann_file     -> {val_annotations}")


if __name__ == "__main__":
    main()
