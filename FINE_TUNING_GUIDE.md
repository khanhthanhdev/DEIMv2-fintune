# DEIMv2 Fine-tuning Complete Guide
# Server Path: /home/25thanh.tk/DEIMv2-finetune/

## Prerequisites
- All required libraries are installed
- Your drone dataset is available at: `/home/25thanh.tk/DEIMv2/train/`
- Dataset structure follows the expected format

## Step 1: Verify Your Data Structure

Your drone dataset should be organized as:

```
/home/25thanh.tk/DEIMv2/train/
├── samples/
│   ├── Person1_0/  # or your video directory name
│   │   ├── frames/  # Pre-extracted frames
│   │   │   ├── frame_5605.jpg
│   │   │   ├── frame_5606.jpg
│   │   │   └── ...
│   │   └── drone_video.mp4  # Optional: original video
│   └── ... (other video directories)
└── annotations/
    └── annotations.json
```

**Verify your data:**
```bash
# Check if data directory exists
ls -la /home/25thanh.tk/DEIMv2/train/

# Count frame files
find /home/25thanh.tk/DEIMv2/train/samples/ -name "*.jpg" | wc -l

# Check annotations file
ls -la /home/25thanh.tk/DEIMv2/train/annotations/annotations.json
```

## Step 2: Extract Frames & Convert Dataset to COCO Format

Extract all annotated frames first so that the converter can reuse the JPGs instead of decoding the MP4s on-the-fly:

```bash
cd /home/25thanh.tk/DEIMv2-finetune/

python tools/dataset/extract_frames.py \
    --input_dir /home/25thanh.tk/DEIMv2/train \
    --output_root /home/25thanh.tk/DEIMv2/train/frames \
    --skip_existing
```

Now run the converter. It will detect the freshly created `frames/` folders and copy images instead of re-reading the videos.

```bash
python tools/dataset/convert_drone_dataset.py \
    --input_dir /home/25thanh.tk/DEIMv2/train \
    --output_dir coco_dataset \
    --overwrite
    --global_frames_root /home/25thanh.tk/DEIMv2/train/frames
```

**Expected output:**
- Creates `coco_dataset/coco_dataset/` directory
- Copies the extracted frames into the COCO `images/` directory
- Converts annotations to COCO format
- Shows conversion statistics

## Step 3: Prepare Your Custom Dataset (7 Labels)

### 3.1 Dataset Structure (COCO Layout)

After running the conversion script you should organize the extracted frames and labels into COCO format:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

In this project the default location is `coco_dataset/` (for the combined set) plus `coco_dataset_train/` and `coco_dataset_val/` after splitting. Each image folder will contain seven categories (Backpack, Jacket, Laptop, Lifering, MobilePhone, Person1, WaterBottle) derived from the `train/samples/*` prefixes. Keep the unlabeled `public_test/` split untouched—it is meant for leaderboard submission once ground-truth is released.

### 3.2 Custom Dataset Configuration

Update your dataset configuration so that DEIMv2 knows about the seven labels and the COCO paths:

```yaml
num_classes: 7
remap_mscoco_category: False

train_dataloader:
  dataset:
    img_folder: /ABS/PATH/TO/coco_dataset_train/images
    ann_file: /ABS/PATH/TO/coco_dataset_train/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: /ABS/PATH/TO/coco_dataset_val/images
    ann_file: /ABS/PATH/TO/coco_dataset_val/annotations/instances_val.json
```

Use `python tools/dataset/update_dataset_paths.py --train_dir ... --val_dir ...` to keep these paths in sync after every split.

## Step 4: Split Dataset into Train/Validation

Split the converted dataset:

```bash
# Split dataset (80% train, 20% validation)
python tools/dataset/split_dataset.py \
    --input_dir coco_dataset/coco_dataset \
    --train_ratio 0.8
```

**Expected output:**
- Creates `coco_dataset/coco_dataset_train/`
- Creates `coco_dataset/coco_dataset_val/`
- Maintains video-level separation

## Step 5: Update Dataset Configuration

Update the dataset paths in the configuration:

```bash
# Update paths in drone_detection.yml
python tools/dataset/update_dataset_paths.py \
    --train_dir coco_dataset/coco_dataset_train \
    --val_dir coco_dataset/coco_dataset_val
```

## Step 6: Validate Dataset Setup

Validate your dataset configuration:

```bash
# Validate dataset configuration
python tools/dataset/validate_dataset.py \
    --config configs/dataset/drone_detection.yml
```

**Expected output:**
- ✅ Dataset validation passed
- Shows image and annotation counts
- No errors or warnings

## Step 7: Start Fine-tuning

### Multi-GPU Training (Recommended)
```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --master_port=7777 --nproc_per_node=2 train.py \
    -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0
```

### Single GPU Training
```bash
python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml --use-amp --seed=0
```

## Step 8: Monitor Training

### View Training Logs
```bash
# Monitor training progress
tail -f outputs/deimv2_dinov3_l_drone_finetune/train.log
```

### Use TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir outputs/deimv2_dinov3_l_drone_finetune

# Access at: http://localhost:6006
```

## Step 9: Evaluate Trained Model

After training completes, evaluate your model:

```bash
python train.py -c configs/deimv2/deimv2_dinov3_l_drone_finetune.yml \
    --test-only \
    -r outputs/deimv2_dinov3_l_drone_finetune/model_final.pth
```

## Expected Output Structure

After successful fine-tuning:

```
/home/25thanh.tk/DEIMv2-finetune/
├── coco_dataset/
│   ├── coco_dataset_train/
│   │   ├── images/          # Training images
│   │   └── annotations/     # Training annotations
│   └── coco_dataset_val/
│       ├── images/          # Validation images
│       └── annotations/     # Validation annotations
├── outputs/
│   └── deimv2_dinov3_l_drone_finetune/
│       ├── model_final.pth      # Final model
│       ├── best_model.pth       # Best checkpoint
│       ├── train.log           # Training logs
│       └── events.out.tfevents.* # TensorBoard data
└── configs/
    └── dataset/
        └── drone_detection.yml  # Updated config
```

## Troubleshooting

### Common Issues

1. **"No such file or directory"**
   - Check if paths exist: `ls -la /home/25thanh.tk/drone_dataset/`
   - Verify file permissions

2. **"Module not found"**
   - Ensure all libraries are installed: `pip install -r requirements.txt`

3. **"CUDA out of memory"**
   - Reduce batch size or use fewer GPUs
   - Use `--use-amp` flag for mixed precision

4. **"No images found"**
   - Check if videos were processed correctly
   - Verify video codec compatibility

5. **Poor training performance**
   - Check annotation quality
   - Verify dataset split quality
   - Monitor learning rate and loss curves

### Getting Help

- Check training logs: `tail -f outputs/deimv2_dinov3_l_drone_finetune/train.log`
- Validate dataset: `python tools/dataset/validate_dataset.py`
- Test model loading: `python -c "from huggingface_hub import PyTorchModelHubMixin; print('HuggingFace OK')"`

## Performance Tips

1. **Multi-GPU**: Use all available GPUs for faster training
2. **Mixed Precision**: Always use `--use-amp` for memory efficiency
3. **Monitor Resources**: Use `nvidia-smi` to monitor GPU usage
4. **Backup**: Save important checkpoints manually if needed

## Next Steps

After successful fine-tuning:

1. **Deploy Model**: Use the trained model for inference
2. **Evaluate Performance**: Run evaluation on test set
3. **Fine-tune Further**: Adjust hyperparameters if needed
4. **Export Model**: Convert to deployment format if required

---

**Success Indicators:**
- Training loss decreases steadily
- Validation mAP improves over epochs
- No CUDA memory errors
- Model saves successfully

**Estimated Training Time:**
- Dataset conversion: 10-30 minutes (depending on video count)
- Fine-tuning: 2-6 hours (depending on GPU count and dataset size)
