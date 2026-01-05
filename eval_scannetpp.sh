#!/bin/bash

# Evaluation script for ScanNet++ scenes - Extract 3D bounding boxes
# Usage: bash eval_scannetpp.sh <scene_name> <model_idx> <checkpoint> <semantic_queries...>
# Example: bash eval_scannetpp.sh scene0000_00 0 10000 "chair" "table" "sofa"

SCENE_NAME=$1
MODEL_IDX=${2:-0}
CHECKPOINT=${3:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-"./results/scannetpp"}
CKPT_ROOT=${CKPT_ROOT:-"./output"}

# Get semantic queries (all arguments after checkpoint)
shift 3
SEMANTIC_QUERIES="$@"

if [ -z "$SEMANTIC_QUERIES" ]; then
    echo "Error: No semantic queries provided"
    echo "Usage: bash eval_scannetpp.sh <scene_name> <model_idx> <checkpoint> <semantic_queries...>"
    echo "Example: bash eval_scannetpp.sh scene0000_00 0 10000 chair table sofa"
    exit 1
fi

echo "Extracting 3D bounding boxes for ScanNet++ scene: $SCENE_NAME"
echo "Model index: $MODEL_IDX"
echo "Checkpoint: $CHECKPOINT"
echo "Semantic queries: $SEMANTIC_QUERIES"
echo "Output directory: $OUTPUT_DIR"

# Paths
SOURCE_PATH="./data/scannetpp/$SCENE_NAME"
CKPT_PATHS="$CKPT_ROOT/${SCENE_NAME}_${MODEL_IDX}"

# Check if source path exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source path does not exist: $SOURCE_PATH"
    exit 1
fi

# Check if checkpoint paths exist
for level in 1 2 3; do
    ckpt_path="${CKPT_PATHS}_${level}"
    if [ ! -d "$ckpt_path" ]; then
        echo "Error: Checkpoint path does not exist: $ckpt_path"
        exit 1
    fi
done

# Run evaluation
python eval_scannetpp.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$CKPT_PATHS" \
    --images images \
    --eval \
    --include_feature \
    --feature_level 1 \
    --ckpt_root_path "$CKPT_ROOT" \
    --dataset_name "$SCENE_NAME" \
    --index "$MODEL_IDX" \
    --output_dir "$OUTPUT_DIR" \
    --mask_thresh 0.4 \
    --checkpoint "$CHECKPOINT" \
    --topk 1 \
    --quick_render \
    --semantic_queries $SEMANTIC_QUERIES

if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/${SCENE_NAME}_${MODEL_IDX}/3d_bboxes.json"
else
    echo "Error: Evaluation failed"
    exit 1
fi

