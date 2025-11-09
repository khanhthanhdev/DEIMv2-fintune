# Drone Dataset Conversion for DEIMv2

This guide explains how to convert your drone video dataset with frame-level annotations to COCO format for training DEIMv2.

## Dataset Structure

Your original dataset should be organized as:

```
train/
├── samples/
│   ├── drone_video_001/
│   │   ├── object_images/
│   │   │   ├── img_1.jpg
│   │   │   ├── img_2.jpg
│   │   │   └── img_3.jpg
│   │   └── drone_video.mp4
│   └── ...
└── annotations/
    └── annotations.json
```

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Convert Dataset to COCO Format

Run the conversion script to extract frames and convert annotations:

```bash
python tools/dataset/convert_drone_dataset.py \
    --input_dir /path/to/your/train \
    --output_dir /path/to/coco_dataset
```

This will:
- Extract frames from videos at annotated timestamps
- Convert bounding box format from (x1,y1,x2,y2) to COCO format (x,y,width,height)
- Create COCO-formatted annotation files
- Organize images in the required structure

## Step 3: Split Dataset (Optional)

If you want to split the dataset into train/validation sets:

```bash
python tools/dataset/split_dataset.py \
    --input_dir /path/to/coco_dataset \
    --train_ratio 0.8
```

This creates `coco_dataset_train/` and `coco_dataset_val/` directories.

## Step 4: Configure DEIMv2

Update the dataset configuration in `configs/dataset/drone_detection.yml`:

```yaml
# Update paths to your dataset
train_dataloader:
  dataset:
    img_folder: /path/to/coco_dataset_train/images
    ann_file: /path/to/coco_dataset_train/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: /path/to/coco_dataset_val/images
    ann_file: /path/to/coco_dataset_val/annotations/instances_val.json
```

## Step 5: Update Model Configuration

Create or modify a model configuration file (e.g., based on `deimv2_hgnetv2_s_coco.yml`):

```yaml
# Include your dataset config
dataset: drone_detection.yml

# Set num_classes to 1 (single object class)
num_classes: 1
```

## Step 6: Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
    -c configs/deimv2/deimv2_hgnetv2_s_coco.yml \
    --use-amp --seed=0
```

## Notes

- The conversion script assumes 25 FPS videos (as mentioned in your dataset description)
- Only frames with annotations are extracted to save disk space
- Bounding boxes are validated to ensure they are within image bounds
- The dataset uses a single category: "target_object" with ID 1
- Video IDs and frame numbers are preserved in the COCO annotations for reference

## Troubleshooting

1. **Video not found**: Ensure video files are named consistently with video_id in annotations
2. **Frame extraction fails**: Check video codec compatibility with OpenCV
3. **Memory issues**: For large datasets, process videos individually or reduce batch size
4. **Invalid bboxes**: The script automatically clips bboxes to image boundaries