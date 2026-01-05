#!/bin/bash

# Training script for ScanNet++ scenes
# Usage: bash train_scannetpp.sh <scene_name> <model_idx>

SCENE_NAME=$1
MODEL_IDX=${2:-0}
DATASET_ROOT=${DATASET_ROOT:-"./data/scannetpp"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./output"}

echo "Training LangSplatV2 on ScanNet++ scene: $SCENE_NAME"
echo "Model index: $MODEL_IDX"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"

# Paths
SOURCE_PATH="$DATASET_ROOT/$SCENE_NAME"
MODEL_PATH="$OUTPUT_ROOT/${SCENE_NAME}_${MODEL_IDX}"

# Check if source path exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source path does not exist: $SOURCE_PATH"
    exit 1
fi

# Step 1: Preprocess - Generate language features
echo "Step 1: Generating language features..."
python preprocess.py \
    --dataset_path "$SOURCE_PATH" \
    --resolution -1 \
    --sam_ckpt_path "ckpts/sam_vit_h_4b8939.pth"

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

# Step 2: Train
echo "Step 2: Training LangSplatV2..."
python train.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --images images \
    --eval \
    --include_feature \
    --feature_level 1 \
    --iterations 30000 \
    --position_lr_init 0.00016 \
    --position_lr_final 0.0000016 \
    --position_lr_delay_mult 0.01 \
    --position_lr_max_steps 30000 \
    --feature_lr 0.0025 \
    --opacity_lr 0.05 \
    --language_feature_lr 0.0025 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    --vq_layer_num 1 \
    --codebook_size 64 \
    --cos_loss \
    --l1_loss \
    --normalize \
    --accum_iter 1 \
    --topk 1 \
    --save_iterations 2000 4000 6000 8000 10000 30000 \
    --test_iterations 2000 4000 6000 8000 10000 30000 \
    --checkpoint_iterations 2000 4000 6000 8000 10000 30000

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "Training completed for scene: $SCENE_NAME"
echo "Model saved to: $MODEL_PATH"

