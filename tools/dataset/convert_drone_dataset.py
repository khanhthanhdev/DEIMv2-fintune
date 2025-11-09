#!/usr/bin/env python3
"""
Drone Video to COCO Dataset Converter for DEIMv2

This script converts drone video datasets with frame-level annotations
to COCO format compatible with DEIMv2 training.

Usage:
    python convert_drone_dataset.py --input_dir /path/to/train --output_dir /path/to/coco_dataset
"""

import os
import json
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


class DroneToCocoConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"

        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        # COCO dataset structure
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "target_object", "supercategory": "object"}]
        }

        self.image_id = 0
        self.annotation_id = 0

    def load_annotations(self):
        """Load the custom annotation file"""
        annotations_file = self.input_dir / "annotations" / "annotations.json"
        with open(annotations_file, 'r') as f:
            return json.load(f)

    def extract_frames_from_video(self, video_path, frame_numbers):
        """Extract specific frames from video"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []

        extracted_frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        for frame_num in frame_numbers:
            # Convert frame number to time (assuming 25 fps as mentioned)
            time_sec = frame_num / 25.0
            cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

            ret, frame = cap.read()
            if ret:
                frame_filename = f"{video_path.stem}_frame_{frame_num:06d}.jpg"
                frame_path = self.images_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append((frame_num, frame_path, frame.shape))
            else:
                print(f"Failed to extract frame {frame_num} from {video_path}")

        cap.release()
        return extracted_frames

    def convert_bbox_format(self, bbox, img_width, img_height):
        """Convert bbox from (x1,y1,x2,y2) to COCO format (x,y,width,height)"""
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        width = x2 - x1
        height = y2 - y1

        return [x1, y1, width, height]

    def process_video(self, video_id, annotations):
        """Process a single video and its annotations"""
        video_path = None

        # Find the video file
        for sample_dir in (self.input_dir / "samples").iterdir():
            if sample_dir.is_dir() and video_id in sample_dir.name:
                video_file = sample_dir / "drone_video.mp4"
                if video_file.exists():
                    video_path = video_file
                    break

        if not video_path:
            print(f"Video file not found for {video_id}")
            return

        print(f"Processing video: {video_path}")

        # Collect all frame numbers that have annotations
        frame_numbers = set()
        for annotation_group in annotations:
            for bbox in annotation_group['bboxes']:
                frame_numbers.add(bbox['frame'])

        frame_numbers = sorted(list(frame_numbers))

        # Extract frames
        extracted_frames = self.extract_frames_from_video(video_path, frame_numbers)

        # Create frame to image mapping
        frame_to_image = {}
        current_width, current_height = None, None
        for frame_num, frame_path, (height, width, _) in extracted_frames:
            # Note: cv2 returns (height, width, channels), but COCO expects width, height
            current_width, current_height = width, height
            self.image_id += 1
            image_info = {
                "id": self.image_id,
                "file_name": frame_path.name,
                "height": height,
                "width": width,
                "frame_number": frame_num,
                "video_id": video_id
            }
            self.coco_data["images"].append(image_info)
            frame_to_image[frame_num] = self.image_id

        # Convert annotations
        for annotation_group in annotations:
            for bbox in annotation_group['bboxes']:
                frame_num = bbox['frame']
                if frame_num not in frame_to_image:
                    continue

                image_id = frame_to_image[frame_num]
                coco_bbox = self.convert_bbox_format(bbox, current_width, current_height)

                # Skip invalid bboxes
                if coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                    continue

                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # target_object
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                    "frame_number": frame_num,
                    "video_id": video_id
                }

                self.coco_data["annotations"].append(annotation)
                self.annotation_id += 1

    def convert(self):
        """Main conversion function"""
        print("Loading annotations...")
        annotations_data = self.load_annotations()

        print(f"Found {len(annotations_data)} video annotations")

        # Process each video
        for video_data in tqdm(annotations_data, desc="Processing videos"):
            video_id = video_data['video_id']
            video_annotations = video_data['annotations']
            self.process_video(video_id, video_annotations)

        # Save COCO annotations
        output_file = self.annotations_dir / "instances_train.json"
        with open(output_file, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

        print(f"Conversion complete!")
        print(f"Images saved to: {self.images_dir}")
        print(f"Annotations saved to: {output_file}")
        print(f"Total images: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert drone video dataset to COCO format")
    parser.add_argument("--input_dir", required=True, help="Input directory containing train/ folder")
    parser.add_argument("--output_dir", required=True, help="Output directory for COCO dataset")

    args = parser.parse_args()

    converter = DroneToCocoConverter(args.input_dir, args.output_dir)
    converter.convert()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/home/thanhkt/DEIMv2-fintune/tools/dataset/convert_drone_dataset.py