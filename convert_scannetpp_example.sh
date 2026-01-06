#!/bin/bash
# ScanNet++转换脚本使用示例

# 设置路径（请根据实际情况修改
SCANNETPP_PATH="/home/bingxing2/ailab/liuyifei/lyj/Dataset/scannetpp"          # ScanNet++数据集根路径
SCENE_NAME="0a7cc12c0e"                  # 场景名称
OUTPUT_PATH="./scannetpp"               # 输出路径
GS_SCENE_PATH="/home/bingxing2/ailab/liuyifei/lyj/Dataset/scannetpp/0a7cc12c0e/3dgs/0a7cc12c0e"         # 3DGS场景路径

# 运行转换脚本
python convert_scannetpp.py \
    --scannetpp_path "$SCANNETPP_PATH" \
    --scene_name "$SCENE_NAME" \
    --output_path "$OUTPUT_PATH" \
    --gs_scene_path "$GS_SCENE_PATH"
    --skip_step1
    --skip_step2

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "转换成功！"
    echo "输出路径: $OUTPUT_PATH/$SCENE_NAME"
    echo ""
    echo "下一步："
    echo "1. 运行预处理: python preprocess.py --dataset_path $OUTPUT_PATH/$SCENE_NAME"
    echo "2. 运行训练: bash train.sh $OUTPUT_PATH $SCENE_NAME 0"
else
    echo ""
    echo "转换失败，请检查错误信息"
    exit 1
fi
