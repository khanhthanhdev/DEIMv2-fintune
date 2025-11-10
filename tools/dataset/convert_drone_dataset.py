#!/usr/bin/env python3
"""
Convert the custom drone dataset (videos + frame-level annotations) to COCO format.

Example:
    python tools/dataset/convert_drone_dataset.py \
        --input_dir /home/25thanh.tk/DEIMv2/train \
        --output_dir coco_dataset/coco_dataset \
        --global_frames_root /home/25thanh.tk/DEIMv2/train/frames
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert drone videos and annotations to COCO detection format."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to dataset root containing `samples/` and `annotations/annotations.json`.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the COCO-style dataset will be written.",
    )
    parser.add_argument(
        "--annotations_file",
        help="Optional annotations JSON path. Defaults to <input_dir>/annotations/annotations.json.",
    )
    parser.add_argument(
        "--video_filename",
        default="drone_video.mp4",
        help="Video filename inside each sample directory.",
    )
    parser.add_argument(
        "--frame_dir_name",
        default="frames",
        help="Folder name that may contain pre-extracted frames inside each sample.",
    )
    parser.add_argument(
        "--global_frames_root",
        help="Optional root directory containing pre-extracted frames for all videos.",
    )
    parser.add_argument(
        "--image_extension",
        default=".jpg",
        choices=[".jpg", ".png"],
        help="Image extension for extracted frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video FPS (used only for storing timestamps in the COCO metadata).",
    )
    parser.add_argument(
        "--min_box_area",
        type=float,
        default=4.0,
        help="Minimum bbox area (in pixels^2) to keep after conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing new data.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse previously extracted frames when they already exist in output_dir.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        help="Debug option to limit how many videos are processed.",
    )
    return parser.parse_args()


def infer_category(video_id: str) -> str:
    if "_" not in video_id:
        return video_id
    return video_id.rsplit("_", 1)[0]


def find_video_file(sample_dir: Path, preferred_name: str) -> Optional[Path]:
    preferred = sample_dir / preferred_name
    if preferred.exists():
        return preferred
    mp4_files = list(sample_dir.glob("*.mp4"))
    if len(mp4_files) == 1:
        return mp4_files[0]
    return None


def find_preextracted_frame(frames_dir: Path, frame_idx: int) -> Optional[Path]:
    patterns = [
        f"frame_{frame_idx}.jpg",
        f"frame_{frame_idx}.png",
        f"frame_{frame_idx:05d}.jpg",
        f"frame_{frame_idx:05d}.png",
        f"frame_{frame_idx:06d}.jpg",
        f"frame_{frame_idx:06d}.png",
    ]
    for pattern in patterns:
        candidate = frames_dir / pattern
        if candidate.exists():
            return candidate
    return None


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists. Use --overwrite to replace its contents."
            )
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)


def gather_frames(record: Dict) -> List[int]:
    frames = set()
    for ann in record.get("annotations", []):
        for box in ann.get("bboxes", []):
            frame = box.get("frame")
            if frame is None:
                continue
            frames.add(int(frame))
    return sorted(frames)


def extract_frames(
    video_path: Optional[Path],
    frames_dir: Optional[Path],
    frames_needed: List[int],
    out_images_root: Path,
    video_id: str,
    image_extension: str,
    skip_existing: bool,
) -> Dict[int, Tuple[Path, int, int]]:
    """Extract or copy the requested frames and return metadata keyed by frame index."""

    results: Dict[int, Tuple[Path, int, int]] = {}
    if not frames_needed:
        return results

    dest_dir = out_images_root / video_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    cap = None
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

    for frame_idx in frames_needed:
        file_name = f"{video_id}_frame_{frame_idx:06d}{image_extension}"
        rel_path = Path(video_id) / file_name
        out_path = out_images_root / rel_path

        if skip_existing and out_path.exists():
            image = cv2.imread(str(out_path))
            if image is None:
                out_path.unlink()
            else:
                h, w = image.shape[:2]
                results[frame_idx] = (rel_path, w, h)
                continue

        copied = False
        if frames_dir and frames_dir.exists():
            src = find_preextracted_frame(frames_dir, frame_idx)
            if src is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_path)
                image = cv2.imread(str(out_path))
                if image is not None:
                    h, w = image.shape[:2]
                    results[frame_idx] = (rel_path, w, h)
                    copied = True
                else:
                    out_path.unlink(missing_ok=True)

        if copied:
            continue

        if cap is None:
            raise RuntimeError(
                f"No video available for {video_id} and no pre-extracted frames found."
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success or frame is None:
            print(f"[WARN] Could not read frame {frame_idx} from {video_id}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame):
            print(f"[WARN] Failed to save frame {frame_idx} for {video_id}")
            continue
        h, w = frame.shape[:2]
        results[frame_idx] = (rel_path, w, h)

    if cap is not None:
        cap.release()

    return results


def clip_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(x1), width - 1))
    y1 = max(0.0, min(float(y1), height - 1))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return x1, y1, w, h


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    samples_dir = input_dir / "samples"
    annotations_path = (
        Path(args.annotations_file).expanduser().resolve()
        if args.annotations_file
        else input_dir / "annotations" / "annotations.json"
    )

    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    frames_root = (
        Path(args.global_frames_root).expanduser().resolve()
        if args.global_frames_root
        else None
    )

    with open(annotations_path, "r") as f:
        raw_annotations = json.load(f)

    if isinstance(raw_annotations, list):
        annotations_data = raw_annotations
    elif isinstance(raw_annotations, dict):
        if raw_annotations.get("video_id"):
            annotations_data = [raw_annotations]
        else:
            candidate = None
            for value in raw_annotations.values():
                if (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], dict)
                    and value[0].get("video_id")
                ):
                    candidate = value
                    break
            if candidate is None:
                raise ValueError(
                    "annotations.json should contain a list of video records or a dict with such a list."
                )
            annotations_data = candidate
    else:
        raise ValueError("annotations.json must be a list or dict containing video records.")

    if args.max_videos:
        annotations_data = annotations_data[: args.max_videos]

    out_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output_dir(out_dir, args.overwrite)
    out_images_root = out_dir / "images"
    annotations_out_path = out_dir / "annotations" / "instances.json"

    categories = sorted({infer_category(item["video_id"]) for item in annotations_data})
    category_map = {name: idx + 1 for idx, name in enumerate(categories)}

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    processed_videos: List[str] = []
    image_id = 1
    annotation_id = 1
    skipped_frames = 0
    skipped_bboxes = 0

    for record in tqdm(annotations_data, desc="Processing videos"):
        video_id = record["video_id"]
        sample_dir = samples_dir / video_id
        video_path = find_video_file(sample_dir, args.video_filename)
        sample_frames_dir = sample_dir / args.frame_dir_name
        global_frames_dir = (
            frames_root / video_id if frames_root else None
        )
        if global_frames_dir and not global_frames_dir.exists():
            global_frames_dir = None
        frames_dir = sample_frames_dir if sample_frames_dir.exists() else global_frames_dir
        if video_path is None and not frames_dir:
            print(f"[WARN] No video or frames directory found for {video_id}, skipping.")
            continue

        frames_needed = gather_frames(record)
        frame_metadata = extract_frames(
            video_path,
            frames_dir,
            frames_needed,
            out_images_root,
            video_id,
            args.image_extension,
            args.skip_existing,
        )
        if not frame_metadata:
            print(f"[WARN] No frames extracted for {video_id}, skipping annotations.")
            continue

        processed_videos.append(video_id)

        for frame_idx, (rel_path, width, height) in frame_metadata.items():
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": str(rel_path).replace("\\", "/"),
                    "width": width,
                    "height": height,
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "timestamp": round(frame_idx / args.fps, 3),
                }
            )
            frame_metadata[frame_idx] = (rel_path, width, height, image_id)
            image_id += 1

        cat_id = category_map[infer_category(video_id)]
        for ann in record.get("annotations", []):
            for box in ann.get("bboxes", []):
                frame_idx = box.get("frame")
                if frame_idx not in frame_metadata:
                    skipped_frames += 1
                    continue
                rel_path, width, height, img_id = frame_metadata[frame_idx]
                clipped = clip_bbox(
                    box.get("x1", 0),
                    box.get("y1", 0),
                    box.get("x2", width),
                    box.get("y2", height),
                    width,
                    height,
                )
                if clipped is None:
                    skipped_bboxes += 1
                    continue
                x, y, w, h = clipped
                area = round(w * h, 3)
                if area < args.min_box_area:
                    skipped_bboxes += 1
                    continue
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                        "area": area,
                        "iscrowd": 0,
                        "video_id": video_id,
                        "frame_index": frame_idx,
                    }
                )
                annotation_id += 1

    coco_dataset = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": cid, "name": name} for name, cid in category_map.items()
        ],
        "videos": [{"id": idx + 1, "name": name} for idx, name in enumerate(processed_videos)],
    }

    with open(annotations_out_path, "w") as f:
        json.dump(coco_dataset, f, indent=2)

    print("\n=== Conversion Summary ===")
    print(f"Videos processed : {len(coco_dataset['videos'])}")
    print(f"Images written   : {len(coco_images)}")
    print(f"Annotations kept : {len(coco_annotations)}")
    if skipped_frames:
        print(f"Frames skipped   : {skipped_frames}")
    if skipped_bboxes:
        print(f"BBoxes skipped   : {skipped_bboxes}")
    print(f"Output directory : {out_dir}")


if __name__ == "__main__":
    main()
