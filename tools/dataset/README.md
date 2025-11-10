# Drone Dataset Conversion for DEIMv2

This folder contains helper scripts that turn the raw drone challenge data (videos + bounding boxes) into a COCO-style dataset that can be used for fine-tuning DEIMv2.

## Raw Input Structure

```
train/
├── samples/
│   ├── Backpack_0/
│   │   ├── object_images/
│   │   └── drone_video.mp4
│   └── ...
└── annotations/
    └── annotations.json

public_test/
└── samples/
    └── ...
```

- `train/` contains 7 semantic labels (Backpack, Jacket, Laptop, Lifering, MobilePhone, Person1, WaterBottle). Each subfolder bundles its reference images and a 3‑5 minute drone video.
- `public_test/` contains only videos/object references (no ground truth). Use it for inference only—the scripts below focus on the labeled `train/` split.

## Step 1 — Install Dependencies

All scripts rely on OpenCV, tqdm, PyYAML, and `ruamel.yaml`:

```bash
pip install -r requirements.txt
```

## Step 2 — Convert the Training Split to COCO

This extracts only the annotated frames and creates `instances.json` with 7 categories:

```bash
python tools/dataset/convert_drone_dataset.py \
  --input_dir /home/25thanh.tk/DEIMv2/train \
  --output_dir coco_dataset/coco_dataset \
  --overwrite
```

Key features:

- Works directly on MP4 files or previously extracted `frames/` directories.
- Automatically infers the category name from the video ID prefix (`Backpack_0` → `Backpack`).
- Clips and filters invalid bounding boxes; stores timestamps and frame indices for traceability.

Resulting layout:

```
coco_dataset/coco_dataset/
├── images/
│   ├── Backpack_0/
│   │   ├── Backpack_0_frame_003483.jpg
│   │   └── ...
└── annotations/
    └── instances.json
```

## Step 3 — Create Train/Validation Splits

Split the converted dataset while keeping entire videos in a single split:

```bash
python tools/dataset/split_dataset.py \
  --input_dir coco_dataset/coco_dataset \
  --train_ratio 0.8 \
  --overwrite
```

This produces sibling folders:

```
coco_dataset/
├── coco_dataset_train/
│   ├── images/
│   └── annotations/instances_train.json
└── coco_dataset_val/
    ├── images/
    └── annotations/instances_val.json
```

## Step 4 — Update the Dataset Config

Inject the new paths into `configs/dataset/drone_detection.yml` (num_classes is already set to 7 and `remap_mscoco_category` is disabled):

```bash
python tools/dataset/update_dataset_paths.py \
  --train_dir coco_dataset/coco_dataset_train \
  --val_dir coco_dataset/coco_dataset_val
```

## Step 5 — Validate Before Training

```bash
python tools/dataset/validate_dataset.py --config configs/dataset/drone_detection.yml
```

You should see counts for images/annotations plus warnings if anything is missing.

## Notes & Tips

- **public_test**: keep it untouched for now; once ground-truth annotations are released you can rerun `convert_drone_dataset.py` on that folder as well.
- **Re-running conversion**: pass `--overwrite` to rebuild everything or `--skip_existing` to reuse extracted frames.
- **Performance**: on the provided dataset (~20k annotated frames) conversion takes a few minutes and uses ~6 GB of disk space for JPEGs.
- **Custom splits**: adjust `--train_ratio`/`--seed` in `split_dataset.py` to try different train/val partitions without recomputing frames.
