#!/bin/bash

# ================= 配置区域 =================
# 请将下面的路径替换为你实际的路径
SOURCE_DIR="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannetppv2/scannetppv2/data"  # 源目录位置
SOURCE_DIR="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/pgsr_all"  # pgsr目录位置
TARGET_DIR="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_pick"  # 目标目录位置
# ===========================================

# 定义文件夹列表
FOLDERS=(
"027cd6ea0f"
"09d6e808b4"
"0a7cc12c0e"
"0b031f3119"
"0d8ead0038"
"116456116b"
"17a5e7d36c"
"1cefb55d50"
"20871b98f3"
)

# 1. 确保目标目录存在，如果不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    echo "创建目标目录: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# 2. 开始循环复制
echo "开始复制..."

for folder in "${FOLDERS[@]}"; do
    SRC_PATH="$SOURCE_DIR/$folder"
    
    if [ -d "$SRC_PATH" ]; then
        # -r 表示递归复制 (复制文件夹及其内容)
        # 也可以使用 -a (归档模式) 来保留权限和时间戳
        # copy pgsr
        mkdir -p "$TARGET_DIR/$folder/output/$folder/"
        cp -r "$SRC_PATH" "$TARGET_DIR/$folder/output/"
        # copy scannetpp
        # cp -r "$SRC_PATH" "$TARGET_DIR/"
        echo "✅ 已复制: $folder"
    else
        echo "⚠️ 警告: 源目录中找不到文件夹 $folder，跳过。"
    fi
done

echo "🎉 所有操作已完成。"