#!/usr/bin/env python3
"""
Dataset Splitter for Drone Dataset

This script splits the converted COCO dataset into train and validation sets.

Usage:
    python split_dataset.py --input_dir /path/to/coco_dataset --train_ratio 0.8
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(input_dir, train_ratio=0.8, random_state=42):
    """
    Split the dataset into train and validation sets based on video IDs
    to ensure frames from the same video stay together.
    """
    input_dir = Path(input_dir)
    annotations_file = input_dir / "annotations" / "instances_train.json"

    # Load annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Group images by video_id
    video_groups = {}
    for image in coco_data['images']:
        video_id = image['video_id']
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(image)

    # Split video IDs
    video_ids = list(video_groups.keys())
    train_videos, val_videos = train_test_split(
        video_ids,
        train_size=train_ratio,
        random_state=random_state
    )

    print(f"Total videos: {len(video_ids)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")

    # Create train/val splits
    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []

    # Create image ID mappings
    train_image_id_map = {}
    val_image_id_map = {}

    new_train_image_id = 1
    new_val_image_id = 1
    new_train_ann_id = 1
    new_val_ann_id = 1

    # Process train videos
    for video_id in train_videos:
        for image in video_groups[video_id]:
            old_image_id = image['id']
            train_image_id_map[old_image_id] = new_train_image_id

            new_image = image.copy()
            new_image['id'] = new_train_image_id
            train_images.append(new_image)
            new_train_image_id += 1

    # Process val videos
    for video_id in val_videos:
        for image in video_groups[video_id]:
            old_image_id = image['id']
            val_image_id_map[old_image_id] = new_val_image_id

            new_image = image.copy()
            new_image['id'] = new_val_image_id
            val_images.append(new_image)
            new_val_image_id += 1

    # Process annotations
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in train_image_id_map:
            new_ann = ann.copy()
            new_ann['id'] = new_train_ann_id
            new_ann['image_id'] = train_image_id_map[image_id]
            train_annotations.append(new_ann)
            new_train_ann_id += 1
        elif image_id in val_image_id_map:
            new_ann = ann.copy()
            new_ann['id'] = new_val_ann_id
            new_ann['image_id'] = val_image_id_map[image_id]
            val_annotations.append(new_ann)
            new_val_ann_id += 1

    # Create output directories
    train_dir = input_dir.parent / f"{input_dir.name}_train"
    val_dir = input_dir.parent / f"{input_dir.name}_val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    (train_dir / "images").mkdir(exist_ok=True)
    (train_dir / "annotations").mkdir(exist_ok=True)
    (val_dir / "images").mkdir(exist_ok=True)
    (val_dir / "annotations").mkdir(exist_ok=True)

    # Copy images to respective directories
    print("Copying images...")
    for image in train_images:
        src_path = input_dir / "images" / image['file_name']
        dst_path = train_dir / "images" / image['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)

    for image in val_images:
        src_path = input_dir / "images" / image['file_name']
        dst_path = val_dir / "images" / image['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)

    # Save annotation files
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data['categories']
    }

    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data['categories']
    }

    with open(train_dir / "annotations" / "instances_train.json", 'w') as f:
        json.dump(train_coco, f, indent=2)

    with open(val_dir / "annotations" / "instances_val.json", 'w') as f:
        json.dump(val_coco, f, indent=2)

    print(f"Train set: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Val set: {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split drone dataset into train/val sets")
    parser.add_argument("--input_dir", required=True, help="Input COCO dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")

    args = parser.parse_args()
    split_dataset(args.input_dir, args.train_ratio)


if __name__ == "__main__":
    main()