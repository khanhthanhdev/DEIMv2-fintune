#!/usr/bin/env python3
"""
Validate dataset configuration before training

Usage:
    python validate_dataset.py --config configs/dataset/drone_detection.yml
"""

import argparse
import json
import yaml
from pathlib import Path


def validate_dataset_config(config_file):
    """Validate the dataset configuration"""

    print(f"Validating config: {config_file}")

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    errors = []
    warnings = []

    # Check train dataset
    train_config = config.get('train_dataloader', {}).get('dataset', {})
    train_img_folder = train_config.get('img_folder')
    train_ann_file = train_config.get('ann_file')

    if train_img_folder:
        train_img_path = Path(train_img_folder)
        if not train_img_path.exists():
            errors.append(f"Train image folder does not exist: {train_img_path}")
        else:
            img_count = len(list(train_img_path.glob('*.jpg'))) + len(list(train_img_path.glob('*.png')))
            print(f"Train images found: {img_count}")
            if img_count == 0:
                warnings.append("No images found in train image folder")

    if train_ann_file:
        train_ann_path = Path(train_ann_file)
        if not train_ann_path.exists():
            errors.append(f"Train annotation file does not exist: {train_ann_path}")
        else:
            try:
                with open(train_ann_path, 'r') as f:
                    train_data = json.load(f)
                print(f"Train annotations loaded: {len(train_data.get('images', []))} images, {len(train_data.get('annotations', []))} annotations")
            except Exception as e:
                errors.append(f"Error loading train annotations: {e}")

    # Check val dataset
    val_config = config.get('val_dataloader', {}).get('dataset', {})
    val_img_folder = val_config.get('img_folder')
    val_ann_file = val_config.get('ann_file')

    if val_img_folder:
        val_img_path = Path(val_img_folder)
        if not val_img_path.exists():
            errors.append(f"Val image folder does not exist: {val_img_path}")
        else:
            img_count = len(list(val_img_path.glob('*.jpg'))) + len(list(val_img_path.glob('*.png')))
            print(f"Val images found: {img_count}")
            if img_count == 0:
                warnings.append("No images found in val image folder")

    if val_ann_file:
        val_ann_path = Path(val_ann_file)
        if not val_ann_path.exists():
            errors.append(f"Val annotation file does not exist: {val_ann_path}")
        else:
            try:
                with open(val_ann_path, 'r') as f:
                    val_data = json.load(f)
                print(f"Val annotations loaded: {len(val_data.get('images', []))} images, {len(val_data.get('annotations', []))} annotations")
            except Exception as e:
                errors.append(f"Error loading val annotations: {e}")

    # Check num_classes
    num_classes = config.get('num_classes')
    if num_classes != 1:
        warnings.append(f"num_classes is {num_classes}, expected 1 for drone dataset")

    # Check remap_mscoco_category
    remap = config.get('remap_mscoco_category')
    if remap != False:
        warnings.append(f"remap_mscoco_category is {remap}, should be False for custom dataset")

    # Report results
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n✅ Dataset configuration is valid!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate dataset configuration")
    parser.add_argument("--config", default="configs/dataset/drone_detection.yml",
                       help="Path to dataset config file")

    args = parser.parse_args()
    validate_dataset_config(args.config)


if __name__ == "__main__":
    main()