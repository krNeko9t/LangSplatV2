#!/bin/bash
# 可视化脚本用于LERF数据集
# 用法: bash visualize_lerf.sh <dataset_name> <index> <checkpoint> [query] [view_idx]
# 示例: bash visualize_lerf.sh teatime 0 10000 "teapot" 0

DATASET_NAME=$1 # teatime: 数据集名称
INDEX=$2 # 0: 模型索引
CHECKPOINT=$3 # 10000: checkpoint迭代次数
QUERY=${4:-"elephant"}  # 默认查询为 "teapot"
VIEW_IDX=${5:-0}      # 默认视图索引为 0
TOPK=4

# 数据集路径（根据实际情况修改）
DATASET_ROOT_PATH=./lerf_ovs

ROOT_PATH="."

python visualize_lerf.py \
    -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
    --dataset_name ${DATASET_NAME} \
    --index ${INDEX} \
    --ckpt_root_path ${ROOT_PATH}/output \
    --output_dir ${ROOT_PATH}/visualize_result \
    --checkpoint ${CHECKPOINT} \
    --include_feature \
    --topk ${TOPK} \
    --quick_render \
    --query "${QUERY}" \
    --view_idx ${VIEW_IDX}


