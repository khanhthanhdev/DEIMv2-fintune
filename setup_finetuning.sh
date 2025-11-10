#!/bin/bash
# DEIMv2 Fine-tuning Setup Script
# Server path: /home/25thanh.tk/DEIMv2-finetune/

set -e  # Exit on any error

echo "=== DEIMv2 Fine-tuning Setup Script ==="
echo "Server path: /home/25thanh.tk/DEIMv2-finetune/"
echo ""

# Configuration
PROJECT_DIR="/home/25thanh.tk/DEIMv2-finetune"
DATA_DIR="/home/25thanh.tk/drone_dataset"  # Change this to your actual data directory
COCO_DATASET_DIR="$PROJECT_DIR/coco_dataset"
TRAIN_DIR="$COCO_DATASET_DIR/coco_dataset_train"
VAL_DIR="$COCO_DATASET_DIR/coco_dataset_val"

echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "COCO dataset directory: $COCO_DATASET_DIR"
echo ""

# Function to check if directory exists
check_directory() {
    local dir_path=$1
    local dir_name=$2

    if [ ! -d "$dir_path" ]; then
        echo "❌ ERROR: $dir_name directory not found: $dir_path"
        echo "Please create the directory or update the path in this script."
        exit 1
    else
        echo "✅ Found $dir_name: $dir_path"
    fi
}

# Function to check if file exists
check_file() {
    local file_path=$1
    local file_name=$2

    if [ ! -f "$file_path" ]; then
        echo "❌ ERROR: $file_name not found: $file_path"
        exit 1
    else
        echo "✅ Found $file_name: $file_path"
    fi
}

echo "=== Step 1: Directory Structure Check ==="
check_directory "$PROJECT_DIR" "Project"
check_directory "$DATA_DIR" "Data"

echo ""
echo "=== Step 2: Validate Data Structure ==="
echo "Expected structure in $DATA_DIR:"
echo "  $DATA_DIR/"
echo "  ├── samples/"
echo "  │   ├── drone_video_001/"
echo "  │   │   ├── object_images/"
echo "  │   │   └── drone_video.mp4"
echo "  │   └── ..."
echo "  └── annotations/"
echo "      └── annotations.json"
echo ""

# Check if samples directory exists
if [ -d "$DATA_DIR/samples" ]; then
    echo "✅ Found samples directory"
    SAMPLE_COUNT=$(find "$DATA_DIR/samples" -name "drone_video.mp4" | wc -l)
    echo "Found $SAMPLE_COUNT video files"
else
    echo "❌ ERROR: samples directory not found in $DATA_DIR"
    exit 1
fi

# Check if annotations file exists
if [ -f "$DATA_DIR/annotations/annotations.json" ]; then
    echo "✅ Found annotations.json"
else
    echo "❌ ERROR: annotations.json not found in $DATA_DIR/annotations/"
    exit 1
fi

echo ""
echo "=== Step 3: Convert Dataset to COCO Format ==="
echo "Converting drone dataset to COCO format..."

cd "$PROJECT_DIR"

# Create output directory
mkdir -p "$COCO_DATASET_DIR"

# Run conversion
echo "Running conversion script..."
python tools/dataset/convert_drone_dataset.py \
    --input_dir "$DATA_DIR" \
    --output_dir "$COCO_DATASET_DIR/coco_dataset"

if [ $? -eq 0 ]; then
    echo "✅ Dataset conversion completed successfully"
else
    echo "❌ Dataset conversion failed"
    exit 1
fi

echo ""
echo "=== Step 4: Split Dataset ==="
echo "Splitting dataset into train/validation sets..."

# Run dataset splitting
python tools/dataset/split_dataset.py \
    --input_dir "$COCO_DATASET_DIR/coco_dataset" \
    --train_ratio 0.8

if [ $? -eq 0 ]; then
    echo "✅ Dataset splitting completed successfully"
else
    echo "❌ Dataset splitting failed"
    exit 1
fi

echo ""
echo "=== Step 5: Update Dataset Paths ==="
echo "Updating dataset configuration with correct paths..."

# Update dataset paths
python tools/dataset/update_dataset_paths.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Dataset paths updated successfully"
else
    echo "❌ Failed to update dataset paths"
    exit 1
fi

echo ""
echo "=== Step 6: Validate Dataset Configuration ==="
echo "Validating dataset configuration..."

# Validate configuration
python tools/dataset/validate_dataset.py \
    --config configs/dataset/drone_detection.yml

if [ $? -eq 0 ]; then
    echo "✅ Dataset validation passed"
else
    echo "❌ Dataset validation failed"
    exit 1
fi

echo ""
echo "=== Step 7: Setup Complete ==="
echo ""
echo "🎉 All setup steps completed successfully!"
echo ""
echo "=== Next Steps ==="
echo "To start fine-tuning, run:"
echo ""
echo "cd $PROJECT_DIR"
echo "CUDA_VISIBLE_DEVICES=6,7 torchrun --master_port=7777 --nproc_per_node=2 train.py \\"
echo "    -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0"
echo ""
echo "Or for single GPU:"
echo "python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0"
echo ""
echo "=== Output Locations ==="
echo "Fine-tuned models will be saved to:"
echo "$PROJECT_DIR/outputs/deimv2_dinov3_l_drone_finetune/"
echo ""
echo "Dataset locations:"
echo "Train: $TRAIN_DIR"
echo "Val: $VAL_DIR"
