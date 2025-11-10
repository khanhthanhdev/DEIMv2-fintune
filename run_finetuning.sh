#!/bin/bash
# Automated DEIMv2 Fine-tuning Pipeline
# Server path: /home/25thanh.tk/DEIMv2-finetune/

set -e  # Exit on any error

# Configuration - Update these paths for your setup
PROJECT_DIR="/home/25thanh.tk/DEIMv2-finetune"
DATA_DIR="/home/25thanh.tk/DEIMv2/train"  # Update this to your data directory
COCO_DATASET_DIR="$PROJECT_DIR/coco_dataset"
TRAIN_RATIO=0.8
GPUS="6,7"  # Update based on your GPU setup

echo "=== DEIMv2 Automated Fine-tuning Pipeline ==="
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_DIR"
echo "Train ratio: $TRAIN_RATIO"
echo "GPUs: $GPUS"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check requirements
check_requirements() {
    print_status "Checking requirements..."

    # Check if in correct directory
    if [ ! -f "train.py" ]; then
        print_error "Not in DEIMv2 project directory. Please run from: $PROJECT_DIR"
        exit 1
    fi

    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        print_error "Please update DATA_DIR in this script or create the directory"
        exit 1
    fi

    # Check Python and required modules
    if ! python -c "import torch, torchvision, cv2, tqdm, sklearn, yaml" 2>/dev/null; then
        print_error "Missing required Python modules"
        print_error "Please run: pip install -r requirements.txt"
        exit 1
    fi

    print_status "Requirements check passed"
}

# Function to validate data structure
validate_data() {
    print_status "Validating data structure..."

    # Check samples directory
    if [ ! -d "$DATA_DIR/samples" ]; then
        print_error "samples/ directory not found in $DATA_DIR"
        exit 1
    fi

    # Check annotations
    if [ ! -f "$DATA_DIR/annotations/annotations.json" ]; then
        print_error "annotations.json not found in $DATA_DIR/annotations/"
        exit 1
    fi

    # Count videos
    VIDEO_COUNT=$(find "$DATA_DIR/samples" -name "*.mp4" | wc -l)
    print_status "Found $VIDEO_COUNT video files"

    if [ "$VIDEO_COUNT" -eq 0 ]; then
        print_error "No video files found"
        exit 1
    fi
}

# Main pipeline
main() {
    print_status "Starting DEIMv2 fine-tuning pipeline..."

    # Change to project directory
    cd "$PROJECT_DIR"

    # Run checks
    check_requirements
    validate_data

    echo ""
    print_status "=== Step 1: Converting Dataset ==="

    # Create output directory
    mkdir -p "$COCO_DATASET_DIR"

    # Convert dataset
    if python tools/dataset/convert_drone_dataset.py \
        --input_dir "$DATA_DIR" \
        --output_dir "$COCO_DATASET_DIR/coco_dataset"; then
        print_status "Dataset conversion completed"
    else
        print_error "Dataset conversion failed"
        exit 1
    fi

    echo ""
    print_status "=== Step 2: Splitting Dataset ==="

    # Split dataset
    if python tools/dataset/split_dataset.py \
        --input_dir "$COCO_DATASET_DIR/coco_dataset" \
        --train_ratio $TRAIN_RATIO; then
        print_status "Dataset splitting completed"
    else
        print_error "Dataset splitting failed"
        exit 1
    fi

    echo ""
    print_status "=== Step 3: Updating Configuration ==="

    # Update dataset paths
    TRAIN_DIR="$COCO_DATASET_DIR/coco_dataset_train"
    VAL_DIR="$COCO_DATASET_DIR/coco_dataset_val"

    if python tools/dataset/update_dataset_paths.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR"; then
        print_status "Configuration updated"
    else
        print_error "Configuration update failed"
        exit 1
    fi

    echo ""
    print_status "=== Step 4: Validating Setup ==="

    # Validate configuration
    if python tools/dataset/validate_dataset.py \
        --config configs/dataset/drone_detection.yml; then
        print_status "Validation passed"
    else
        print_error "Validation failed"
        exit 1
    fi

    echo ""
    print_status "=== Step 5: Starting Fine-tuning ==="

    # Check GPU availability
    if ! nvidia-smi &>/dev/null; then
        print_warning "nvidia-smi not available, falling back to CPU training"
        GPU_CMD=""
    else
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ "$GPU_COUNT" -gt 1 ]; then
            print_status "Using $GPU_COUNT GPUs for training"
            GPU_CMD="CUDA_VISIBLE_DEVICES=$GPUS torchrun --master_port=7777 --nproc_per_node=$GPU_COUNT"
        else
            print_status "Using single GPU for training"
            GPU_CMD=""
        fi
    fi

    # Start training
    print_status "Starting training with command:"
    echo "$GPU_CMD python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0"

    if $GPU_CMD python train.py \
        -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml \
        --use-amp \
        --seed=0; then
        print_status "Training completed successfully!"
    else
        print_error "Training failed"
        exit 1
    fi

    echo ""
    print_status "=== Pipeline Completed Successfully! ==="
    echo ""
    echo "Model saved to: $PROJECT_DIR/outputs/deimv2_dinov3_l_drone_finetune/"
    echo "To evaluate: python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --test-only -r outputs/deimv2_dinov3_l_drone_finetune/model_final.pth"
}

# Show usage if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "DEIMv2 Automated Fine-tuning Pipeline"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --dry-run          Show what would be done without executing"
    echo ""
    echo "Configuration (edit script to change):"
    echo "  PROJECT_DIR: $PROJECT_DIR"
    echo "  DATA_DIR: $DATA_DIR"
    echo "  TRAIN_RATIO: $TRAIN_RATIO"
    echo "  GPUS: $GPUS"
    exit 0
fi

# Run main pipeline
main "$@"