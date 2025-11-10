#!/usr/bin/env python3
"""
Extract frames from every drone video so that the converter can reuse them later.

The script targets the same folder structure used by `train/`:

train/
├── samples/
│   ├── Backpack_0/
│   │   ├── drone_video.mp4
│   │   └── object_images/
│   └── ...
└── annotations/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from each drone sample (uses ffmpeg if available)."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root of dataset containing `samples/` and `annotations/`.",
    )
    parser.add_argument(
        "--video_name",
        default="drone_video.mp4",
        help="Video filename inside each sample directory.",
    )
    parser.add_argument(
        "--frames_dir",
        default="frames",
        help="Name given to the generated frames folder inside each sample.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Enforce a fixed FPS for the exported frames (FFmpeg only).",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        help="Limit the number of videos that get processed (debug).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already contain a non-empty frames folder.",
    )
    parser.add_argument(
        "--use_opencv",
        action="store_true",
        help="Always use OpenCV instead of ffmpeg.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show informational logs during extraction.",
    )
    return parser.parse_args()


def find_samples(root: Path) -> Iterator[Path]:
    samples_dir = root / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples/ directory not found inside {root}")
    for child in sorted(samples_dir.iterdir()):
        if child.is_dir():
            yield child


def has_frames(sample_dir: Path, frames_dir: str) -> bool:
    target = sample_dir / frames_dir
    if not target.exists():
        return False
    if any(target.iterdir()):
        return True
    return False


def extract_with_opencv(
    video_path: Path, dest_dir: Path, start_frame: int = 0, end_frame: Optional[int] = None
) -> Tuple[int, List[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_written = 0
    written_frames: List[int] = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if end_frame is not None and frame_idx > end_frame:
            break

        path = dest_dir / f"frame_{frame_idx:06d}.jpg"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            logger.warning("Failed to write frame %s", path)
        else:
            total_written += 1
            written_frames.append(frame_idx)
        frame_idx += 1

    cap.release()
    return total_written, written_frames


def extract_with_ffmpeg(video_path: Path, dest_dir: Path, fps: Optional[float]) -> Tuple[int, List[int]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-q:v",
        "2",
    ]
    if fps:
        cmd += ["-r", str(fps)]
    cmd.append(str(dest_dir / "frame_%06d.jpg"))

    dest_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\nstdout:\n%s\n\nstderr:\n%s" % (process.stdout, process.stderr)
        )

    written = list(dest_dir.glob("frame_*.jpg"))
    frame_numbers = [int(p.stem.split("_")[-1]) for p in written]
    return len(written), sorted(frame_numbers)


def extract_frames_for_sample(
    sample_dir: Path,
    video_name: str,
    frames_dir: str,
    force_opencv: bool,
    fps: Optional[float],
) -> Tuple[int, List[int]]:
    video_path = sample_dir / video_name
    if not video_path.exists():
        mp4_candidates = list(sample_dir.glob("*.mp4"))
        if len(mp4_candidates) == 1:
            video_path = mp4_candidates[0]
        else:
            raise FileNotFoundError(f"No video file found in {sample_dir}")

    dest_dir = sample_dir / frames_dir
    ffmpeg_available = shutil.which("ffmpeg") is not None and not force_opencv
    if ffmpeg_available:
        return extract_with_ffmpeg(video_path, dest_dir, fps)
    return extract_with_opencv(video_path, dest_dir)


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    root = Path(args.input_dir).expanduser().resolve()
    samples = list(find_samples(root))
    if args.max_videos:
        samples = samples[: args.max_videos]

    extracted_videos = 0
    for sample_dir in samples:
        if args.skip_existing and has_frames(sample_dir, args.frames_dir):
            logger.info("Skipping %s (frames already exist)", sample_dir.name)
            continue
        logger.info("Extracting %s", sample_dir.name)
        try:
            count, frames = extract_frames_for_sample(
                sample_dir,
                args.video_name,
                args.frames_dir,
                args.use_opencv,
                args.fps,
            )
        except Exception as exc:
            logger.error("Failed to extract %s: %s", sample_dir.name, exc)
            continue
        extracted_videos += 1
        logger.info("Wrote %d frames for %s", count, sample_dir.name)

    print("Finished extracting frames from %d videos." % extracted_videos)


if __name__ == "__main__":
    main()
