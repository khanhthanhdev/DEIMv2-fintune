# Fine-tuning DEIMv2 on Drone Dataset

This guide explains how to fine-tune the pre-trained DEIMv2_DINOv3_L_COCO model on your custom drone dataset.

## Prerequisites

1. **Downloaded Model Weights**: You should have downloaded `Intellindust/DEIMv2_DINOv3_L_COCO` from Hugging Face
2. **Converted Dataset**: Your drone dataset should be converted to COCO format using the conversion scripts

## Step 1: Prepare Your Dataset

If you haven't already converted your dataset:

```bash
# Convert your drone dataset to COCO format
python tools/dataset/convert_drone_dataset.py \
    --input_dir /path/to/your/train \
    --output_dir /path/to/coco_dataset

# Split into train/val sets
python tools/dataset/split_dataset.py \
    --input_dir /path/to/coco_dataset \
    --train_ratio 0.8
```

## Step 2: Update Dataset Paths

Update the dataset configuration with your actual paths:

```bash
python tools/dataset/update_dataset_paths.py \
    --train_dir /path/to/coco_dataset_train \
    --val_dir /path/to/coco_dataset_val
```

## Step 3: Validate Dataset Configuration

Before training, validate your dataset setup:

```bash
python tools/dataset/validate_dataset.py --config configs/dataset/drone_detection.yml
```

This will check:
- Dataset paths exist
- Annotation files are valid JSON
- Image counts match expectations
- Configuration parameters are correct

## Step 4: Download Pre-trained Weights (if not already done)

The config will automatically download the weights from Hugging Face when training starts. Alternatively, you can pre-download them:

```bash
# The weights will be downloaded automatically during training
# Or you can manually download from:
# https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO
```

## Step 5: Fine-tune the Model

Run the fine-tuning training:

```bash
# Single GPU
python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0

# Multi-GPU (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
    -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0
```

## Fine-tuning Configuration Details

### Key Differences from Training from Scratch:

1. **Pre-trained Weights**: Loads from `Intellindust/DEIMv2_DINOv3_L_COCO`
2. **Reduced Learning Rates**: 50% of original training rates
3. **Fewer Epochs**: 24 epochs instead of 68
4. **Lighter Augmentation**: Reduced augmentation probabilities
5. **Single Class**: Configured for your target object detection

### Training Schedule:
- **Epochs**: 24 total
- **Warmup**: 500 iterations
- **Learning Rate Schedule**: Flat cosine with gamma=0.5
- **Data Augmentation**: Mosaic, Mixup, CopyBlend with adjusted probabilities

## Monitoring Training

```bash
# View training logs
tail -f outputs/deimv2_dinov3_l_drone_finetune/train.log

# Use TensorBoard
tensorboard --logdir outputs/deimv2_dinov3_l_drone_finetune
```

## Expected Output

The fine-tuned model will be saved in:
```
outputs/deimv2_dinov3_l_drone_finetune/
├── model_final.pth      # Final model weights
├── best_model.pth       # Best model based on validation mAP
├── train.log           # Training logs
└── events.out.tfevents.* # TensorBoard logs
```

## Evaluation

After training, evaluate your model:

```bash
python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --test-only -r outputs/deimv2_dinov3_l_drone_finetune/model_final.pth
```

## Tips for Better Fine-tuning

1. **Dataset Size**: If your dataset is small (< 1K images), consider:
   - Further reduce learning rates
   - Increase weight decay
   - Use stronger data augmentation

2. **Convergence**: Monitor validation mAP. If it plateaus early, try:
   - Reducing learning rate further
   - Adding more epochs
   - Adjusting the learning rate schedule

3. **Memory Issues**: If you encounter GPU memory issues:
   - Reduce batch size in the dataloader config
   - Use gradient checkpointing (if available)
   - Use smaller input resolution

## Troubleshooting

1. **Weights not loading**: Ensure the Hugging Face model is accessible
2. **Dataset path errors**: Double-check paths in `drone_detection.yml`
3. **CUDA out of memory**: Reduce batch size or use fewer GPUs
4. **Poor performance**: Check annotation quality and dataset size