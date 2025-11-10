#!/usr/bin/env python3
"""
Split a converted COCO-style dataset into train/validation folders.

Example:
    python tools/dataset/split_dataset.py \
        --input_dir coco_dataset/coco_dataset \
        --train_ratio 0.8
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split COCO dataset into train/val splits.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the COCO dataset that contains `images/` and `annotations/instances.json`.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of videos assigned to the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <input_dir>_train / <input_dir>_val directories.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy strategy for images when creating the split directories.",
    )
    return parser.parse_args()


def prepare_split_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "annotations").mkdir(parents=True, exist_ok=True)


def determine_video_id(image: Dict) -> str:
    if "video_id" in image and image["video_id"]:
        return str(image["video_id"])
    file_name = image.get("file_name", "")
    return file_name.split("/")[0] if "/" in file_name else file_name


def copy_image(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source image: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)


def build_split(
    video_ids: Sequence[str],
    video_to_images: Dict[str, List[Dict]],
    image_to_annotations: Dict[int, List[Dict]],
    src_images_root: Path,
    dest_root: Path,
    split_name: str,
    copy_mode: str,
) -> Tuple[int, int]:
    next_image_id = 1
    next_ann_id = 1
    split_images: List[Dict] = []
    split_annotations: List[Dict] = []
    id_map: Dict[int, int] = {}

    for video_id in video_ids:
        for image in video_to_images[video_id]:
            new_image = dict(image)
            new_image["id"] = next_image_id
            split_images.append(new_image)
            id_map[image["id"]] = next_image_id

            src_path = src_images_root / image["file_name"]
            dst_path = dest_root / "images" / image["file_name"]
            copy_image(src_path, dst_path, copy_mode)

            next_image_id += 1

    for ann in image_to_annotations.values():
        for item in ann:
            old_image_id = item["image_id"]
            if old_image_id not in id_map:
                continue
            new_ann = dict(item)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[old_image_id]
            split_annotations.append(new_ann)
            next_ann_id += 1

    annotations_path = dest_root / "annotations" / f"instances_{split_name}.json"
    dataset = {
        "images": split_images,
        "annotations": split_annotations,
        "categories": [],
        "videos": [],
    }
    # categories and videos will be filled by caller
    with open(annotations_path, "w") as f:
        json.dump(dataset, f, indent=2)

    return len(split_images), len(split_annotations)


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be within (0, 1).")

    input_dir = Path(args.input_dir).expanduser().resolve()
    images_root = input_dir / "images"
    annotations_path = input_dir / "annotations" / "instances.json"

    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

    with open(annotations_path, "r") as f:
        dataset = json.load(f)

    video_to_images: Dict[str, List[Dict]] = defaultdict(list)
    for image in dataset.get("images", []):
        video_to_images[determine_video_id(image)].append(image)

    if not video_to_images:
        raise ValueError("No images found in dataset. Did you run convert_drone_dataset.py?")

    image_to_annotations: Dict[int, List[Dict]] = defaultdict(list)
    for ann in dataset.get("annotations", []):
        image_to_annotations[ann["image_id"]].append(ann)

    all_videos = sorted(video_to_images.keys())
    if len(all_videos) < 2:
        raise ValueError("Need at least two videos to create a train/val split.")

    rng = random.Random(args.seed)
    rng.shuffle(all_videos)

    train_count = max(1, int(round(len(all_videos) * args.train_ratio)))
    train_videos = all_videos[:train_count]
    val_videos = all_videos[train_count:]
    if not val_videos:
        val_videos = [train_videos.pop()]

    train_dir = input_dir.parent / f"{input_dir.name}_train"
    val_dir = input_dir.parent / f"{input_dir.name}_val"
    prepare_split_dir(train_dir, args.overwrite)
    prepare_split_dir(val_dir, args.overwrite)

    train_image_count, train_ann_count = build_split(
        train_videos,
        video_to_images,
        image_to_annotations,
        images_root,
        train_dir,
        "train",
        args.copy_mode,
    )
    val_image_count, val_ann_count = build_split(
        val_videos,
        video_to_images,
        image_to_annotations,
        images_root,
        val_dir,
        "val",
        args.copy_mode,
    )

    # Add shared metadata (categories/videos) to the annotation files
    for split_dir, split_name in [(train_dir, "train"), (val_dir, "val")]:
        ann_path = split_dir / "annotations" / f"instances_{split_name}.json"
        with open(ann_path, "r") as f:
            data = json.load(f)
        data["categories"] = dataset.get("categories", [])
        data["videos"] = dataset.get("videos", [])
        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)

    print("=== Split Summary ===")
    print(f"Total videos : {len(all_videos)}")
    print(f"Train videos : {len(train_videos)} ({train_image_count} images, {train_ann_count} annots)")
    print(f"Val videos   : {len(val_videos)} ({val_image_count} images, {val_ann_count} annots)")
    print(f"Train dir    : {train_dir}")
    print(f"Val dir      : {val_dir}")


if __name__ == "__main__":
    main()
