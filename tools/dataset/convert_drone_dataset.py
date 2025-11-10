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

        # COCO dataset structure - 7 classes for ground object detection
        self.category_mapping = {
            'Backpack': 1,
            'Jacket': 2,
            'Laptop': 3,
            'Lifering': 4,
            'MobilePhone': 5,
            'Person1': 6,
            'WaterBottle': 7
        }
        
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "Backpack", "supercategory": "ground_object"},
                {"id": 2, "name": "Jacket", "supercategory": "ground_object"},
                {"id": 3, "name": "Laptop", "supercategory": "ground_object"},
                {"id": 4, "name": "Lifering", "supercategory": "ground_object"},
                {"id": 5, "name": "MobilePhone", "supercategory": "ground_object"},
                {"id": 6, "name": "Person1", "supercategory": "ground_object"},
                {"id": 7, "name": "WaterBottle", "supercategory": "ground_object"}
            ]
        }

        self.image_id = 0
        self.annotation_id = 0

    def load_annotations(self):
        """Load the custom annotation file"""
        annotations_file = self.input_dir / "annotations" / "annotations.json"
        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded annotations: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
        return data

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

    def find_existing_frames(self, frames_dir, frame_numbers, video_id):
        """Find existing frame files in the frames directory"""
        print(f"Looking for frames in: {frames_dir}")
        print(f"Annotation frame numbers: {frame_numbers}")

        existing_frames = []

        # Get all frame files in the directory
        frame_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpeg"))
        print(f"Found {len(frame_files)} frame files in directory")

        # Extract frame numbers from filenames
        available_frames = {}
        for frame_file in frame_files:
            # Try to extract frame number from filename
            filename = frame_file.name
            # Look for patterns like frame_XXXX.jpg or frame_XXXX.png
            if 'frame_' in filename:
                try:
                    # Extract number after 'frame_'
                    parts = filename.split('frame_')[1]
                    frame_num = int(parts.split('.')[0])  # Remove extension
                    available_frames[frame_num] = frame_file
                except (ValueError, IndexError):
                    continue

        print(f"Available frame numbers from files: {sorted(available_frames.keys())}")

        # For each annotation frame number, find the closest available frame
        # This handles cases where annotation frame numbers don't exactly match file frame numbers
        for target_frame in frame_numbers:
            # Find the closest frame number that's available
            closest_frame = None
            min_diff = float('inf')

            for available_frame_num in available_frames.keys():
                diff = abs(available_frame_num - target_frame)
                if diff < min_diff:
                    min_diff = diff
                    closest_frame = available_frame_num

            if closest_frame is not None and min_diff <= 10:  # Allow small differences
                frame_path = available_frames[closest_frame]
                print(f"Using frame {closest_frame} for annotation frame {target_frame} (diff: {min_diff})")

                # Read image to get dimensions
                img = cv2.imread(str(frame_path))
                if img is not None:
                    height, width = img.shape[:2]
                    # Copy frame to output directory
                    output_frame_path = self.images_dir / f"{video_id}_frame_{target_frame:06d}.jpg"
                    shutil.copy2(frame_path, output_frame_path)
                    existing_frames.append((target_frame, output_frame_path, (height, width, 3)))
                else:
                    print(f"Failed to read frame: {frame_path}")
            else:
                print(f"No suitable frame found for annotation frame {target_frame}")

        print(f"Total frames matched: {len(existing_frames)}")
        return existing_frames

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
        frames_dir = None

        print(f"Looking for video_id: {video_id}")

        # Find the video file or frames directory
        sample_dirs = list((self.input_dir / "samples").iterdir())
        print(f"Available sample directories: {[d.name for d in sample_dirs if d.is_dir()]}")

        for sample_dir in sample_dirs:
            if sample_dir.is_dir():
                print(f"Checking directory: {sample_dir.name}")
                # More flexible matching - check if video_id is contained in directory name
                if video_id in sample_dir.name or sample_dir.name in video_id:
                    print(f"Found matching directory: {sample_dir.name}")

                    # Priority: Check for frames directory first, then video file
                    frames_dir_check = sample_dir / "frames"
                    if frames_dir_check.exists() and frames_dir_check.is_dir():
                        frames_dir = frames_dir_check
                        print(f"Found frames directory: {frames_dir}")
                        break

                    # Check for video file
                    video_file = sample_dir / "drone_video.mp4"
                    if video_file.exists():
                        video_path = video_file
                        print(f"Found video file: {video_path}")
                        break

        if not video_path and not frames_dir:
            print(f"No video file or frames directory found for {video_id}")
            return

        print(f"Processing: {video_path or frames_dir}")

        # Collect all frame numbers that have annotations
        frame_numbers = set()
        for annotation_group in annotations:
            print(f"Processing annotation group: {annotation_group}")
            if 'bboxes' in annotation_group:
                for bbox in annotation_group['bboxes']:
                    frame_num = bbox.get('frame')
                    if frame_num is not None:
                        frame_numbers.add(frame_num)
                        print(f"Found frame number: {frame_num}, bbox: {bbox}")

        frame_numbers = sorted(list(frame_numbers))
        print(f"Total unique frame numbers to process: {len(frame_numbers)}")
        print(f"Frame numbers: {frame_numbers}")

        if not frame_numbers:
            print("No frame numbers found in annotations!")
            return

        if video_path:
            # Extract frames from video
            print(f"Extracting {len(frame_numbers)} frames from video: {frame_numbers}")
            extracted_frames = self.extract_frames_from_video(video_path, frame_numbers)
        elif frames_dir:
            # Try to find existing frames
            print(f"Trying to find existing frames in: {frames_dir}")
            extracted_frames = self.find_existing_frames(frames_dir, frame_numbers, video_id)

            # If no frames found, try to extract from video if it exists
            if not extracted_frames:
                video_file = frames_dir.parent / "drone_video.mp4"
                if video_file.exists():
                    print(f"No existing frames found, extracting from video: {video_file}")
                    extracted_frames = self.extract_frames_from_video(video_file, frame_numbers)
                else:
                    print(f"No frames found and no video file available for {video_id}")
        else:
            print(f"No video file or frames directory found for {video_id}")
            return

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

                # Get category_id from bbox annotation or map from video_id prefix
                category_id = bbox.get('category_id', None)
                
                if category_id is None:
                    # Extract category from video_id (e.g., "Backpack_0" -> "Backpack")
                    video_prefix = video_id.split('_')[0]  # Get part before first underscore
                    category_id = self.category_mapping.get(video_prefix, 1)  # Default to 1 if not found
                
                # Ensure category_id is valid (1-7)
                category_id = max(1, min(7, int(category_id)))

                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
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

        print(f"Raw annotations data type: {type(annotations_data)}")
        print(f"Raw annotations keys: {annotations_data.keys() if isinstance(annotations_data, dict) else 'Not a dict'}")

        # Handle both single video object and list of video objects
        if isinstance(annotations_data, dict):
            # Single video object
            video_list = [annotations_data]
            print("Found 1 video annotation (single object format)")
        elif isinstance(annotations_data, list):
            # List of video objects
            video_list = annotations_data
            print(f"Found {len(video_list)} video annotations (list format)")
        else:
            raise ValueError("Invalid annotations format. Expected dict or list.")

        # Process each video
        for video_data in tqdm(video_list, desc="Processing videos"):
            if not isinstance(video_data, dict):
                print(f"Warning: Skipping invalid video data: {video_data}")
                continue

            video_id = video_data.get('video_id')
            if not video_id:
                print(f"Warning: Skipping video data without video_id: {video_data}")
                continue

            print(f"Processing video_id: {video_id}")
            print(f"Full video_data: {video_data}")
            video_annotations = video_data.get('annotations', [])
            print(f"Found {len(video_annotations)} annotation groups for {video_id}")
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
    main()