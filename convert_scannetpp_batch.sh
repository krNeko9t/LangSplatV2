#!/bin/bash
# ScanNet++ 批量转换脚本

# 1. 设置基础路径 (保持不变)
SCANNETPP_PATH="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_pick"
OUTPUT_PATH="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse"

# 2. 定义场景列表 (数组)
SCENE_LIST=(
    # "027cd6ea0f"
    "09d6e808b4"
    "0a7cc12c0e"
    "0b031f3119"
    "0d8ead0038"
    "116456116b"
    "17a5e7d36c"
    "1cefb55d50"
    "20871b98f3"
)

SCENE_LIST=(
    "924b364b9f"
)

# 3. 循环处理每一个场景
for SCENE_NAME in "${SCENE_LIST[@]}"; do
    echo "----------------------------------------"
    echo "正在处理场景: $SCENE_NAME"

    # 动态构建 GS_SCENE_PATH
    # 逻辑是：基础路径 + 场景名 + /3dgs/ + 场景名
    GS_SCENE_PATH="${SCANNETPP_PATH}/${SCENE_NAME}/output/${SCENE_NAME}"

    # 运行转换脚本
    python convert_scannetpp.py \
        --scannetpp_path "$SCANNETPP_PATH" \
        --scene_name "$SCENE_NAME" \
        --output_path "$OUTPUT_PATH" \
        --gs_scene_path "$GS_SCENE_PATH"
        
    echo "场景 $SCENE_NAME 处理完成"
done

echo "========================================"
echo "所有任务已完成"