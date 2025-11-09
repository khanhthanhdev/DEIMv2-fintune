#!/usr/bin/env python3
"""
Update dataset paths in drone_detection.yml

Usage:
    python update_dataset_paths.py --train_dir /path/to/train --val_dir /path/to/val
"""

import argparse
import yaml
from pathlib import Path


def update_dataset_config(train_dir, val_dir, config_file='configs/dataset/drone_detection.yml'):
    """Update the dataset paths in the drone detection config"""

    # Read the config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update train paths
    if train_dir:
        train_images = Path(train_dir) / "images"
        train_annotations = Path(train_dir) / "annotations" / "instances_train.json"

        config['train_dataloader']['dataset']['img_folder'] = str(train_images)
        config['train_dataloader']['dataset']['ann_file'] = str(train_annotations)

    # Update val paths
    if val_dir:
        val_images = Path(val_dir) / "images"
        val_annotations = Path(val_dir) / "annotations" / "instances_val.json"

        config['val_dataloader']['dataset']['img_folder'] = str(val_images)
        config['val_dataloader']['dataset']['ann_file'] = str(val_annotations)

    # Write back the updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {config_file}")
    if train_dir:
        print(f"Train images: {config['train_dataloader']['dataset']['img_folder']}")
        print(f"Train annotations: {config['train_dataloader']['dataset']['ann_file']}")
    if val_dir:
        print(f"Val images: {config['val_dataloader']['dataset']['img_folder']}")
        print(f"Val annotations: {config['val_dataloader']['dataset']['ann_file']}")


def main():
    parser = argparse.ArgumentParser(description="Update dataset paths in drone_detection.yml")
    parser.add_argument("--train_dir", help="Path to training dataset directory")
    parser.add_argument("--val_dir", help="Path to validation dataset directory")
    parser.add_argument("--config_file", default="configs/dataset/drone_detection.yml",
                       help="Path to config file to update")

    args = parser.parse_args()

    if not args.train_dir and not args.val_dir:
        print("Error: Must provide at least --train_dir or --val_dir")
        return

    update_dataset_config(args.train_dir, args.val_dir, args.config_file)


if __name__ == "__main__":
    main()