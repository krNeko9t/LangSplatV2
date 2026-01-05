# ScanNet++ 数据准备完整指南

本指南详细说明如何准备ScanNet++数据用于LangSplatV2训练。

## 目录结构要求

LangSplatV2需要以下目录结构：

```
data/scannetpp/
├── scene0000_00/
│   ├── images/                    # RGB图像文件夹（必需）
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
│   ├── sparse/                    # COLMAP重建结果（必需）
│   │   └── 0/
│   │       ├── cameras.bin        # 相机参数（二进制格式）
│   │       ├── images.bin         # 图像和位姿信息（二进制格式）
│   │       ├── points3D.bin        # 3D点云（二进制格式）
│   │       └── project.ini        # COLMAP项目文件（可选）
│   └── language_features/          # 语言特征（预处理后生成，训练前必需）
│       ├── frame_000000_s.npy     # 分割mask
│       ├── frame_000000_f.npy     # CLIP特征
│       ├── frame_000001_s.npy
│       ├── frame_000001_f.npy
│       └── ...
├── scene0001_00/
│   └── ...
└── ...
```

## 方法1：使用COLMAP重建（推荐）

如果您有ScanNet++的原始RGB图像，可以使用COLMAP进行重建。

### 步骤1：准备图像

```bash
# 假设ScanNet++原始数据在 /path/to/scannetpp_raw
SCENE_NAME="scene0000_00"
SCANNETPP_RAW="/path/to/scannetpp_raw"
OUTPUT_DIR="./data/scannetpp"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}/${SCENE_NAME}/images

# 复制图像（根据您的ScanNet++数据格式调整路径）
# 示例：如果图像在 color/ 文件夹中
cp ${SCANNETPP_RAW}/${SCENE_NAME}/color/*.jpg ${OUTPUT_DIR}/${SCENE_NAME}/images/

# 或者重命名为frame_XXXXXX.jpg格式
cd ${OUTPUT_DIR}/${SCENE_NAME}/images
counter=0
for img in *.jpg; do
    mv "$img" "frame_$(printf '%06d' $counter).jpg"
    counter=$((counter+1))
done
```

### 步骤2：使用COLMAP重建

```bash
cd ${OUTPUT_DIR}/${SCENE_NAME}

# 创建数据库
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

# 特征匹配
colmap exhaustive_matcher \
    --database_path database.db

# 稀疏重建
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse

# 检查结果
ls sparse/0/
# 应该看到：cameras.bin, images.bin, points3D.bin
```

### 步骤3：验证COLMAP结果

```bash
# 使用COLMAP查看器验证（可选）
colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0 \
    --output_type TXT

# 检查文本文件
head sparse/0/cameras.txt
head sparse/0/images.txt
```

## 方法2：从ScanNet++已有重建结果转换

如果ScanNet++已经提供了相机参数和位姿，可以转换为COLMAP格式。

### 步骤1：准备数据

假设ScanNet++数据格式如下：
```
scannetpp_raw/
└── scene0000_00/
    ├── color/              # RGB图像
    ├── depth/              # 深度图（可选）
    ├── pose/               # 相机位姿
    └── intrinsic/          # 相机内参
```

### 步骤2：使用转换脚本

```bash
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp_raw \
    --output_root ./data/scannetpp \
    --scene_name scene0000_00
```

### 步骤3：手动转换（如果脚本不支持您的格式）

如果ScanNet++提供了JSON格式的相机参数和位姿，可以手动转换为COLMAP格式。

#### 创建转换脚本

```python
# convert_scannetpp_to_colmap.py
import numpy as np
import json
from pathlib import Path
import struct

def write_cameras_bin(cameras_file, output_file):
    """将相机参数写入COLMAP二进制格式"""
    with open(cameras_file, 'r') as f:
        cameras = json.load(f)
    
    with open(output_file, 'wb') as f:
        # 写入相机数量
        f.write(struct.pack('Q', len(cameras)))
        
        for cam_id, cam in cameras.items():
            # 相机ID
            f.write(struct.pack('I', int(cam_id)))
            # 模型类型 (1 = PINHOLE)
            f.write(struct.pack('I', 1))
            # 图像宽度
            f.write(struct.pack('Q', cam['width']))
            # 图像高度
            f.write(struct.pack('Q', cam['height']))
            # 参数：fx, fy, cx, cy
            params = [cam['fx'], cam['fy'], cam['cx'], cam['cy']]
            f.write(struct.pack('d' * len(params), *params))

def write_images_bin(poses_file, images_dir, output_file):
    """将图像和位姿写入COLMAP二进制格式"""
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    image_files = sorted(Path(images_dir).glob('*.jpg'))
    
    with open(output_file, 'wb') as f:
        # 写入图像数量
        f.write(struct.pack('Q', len(image_files)))
        
        for idx, img_file in enumerate(image_files):
            # 图像ID
            f.write(struct.pack('I', idx + 1))
            # 四元数 (qvec)
            qvec = poses[str(idx)]['qvec']  # [w, x, y, z]
            f.write(struct.pack('d' * 4, *qvec))
            # 平移向量 (tvec)
            tvec = poses[str(idx)]['tvec']  # [x, y, z]
            f.write(struct.pack('d' * 3, *tvec))
            # 相机ID
            f.write(struct.pack('I', 1))
            # 图像名称
            img_name = img_file.name
            f.write(struct.pack('Q', len(img_name)))
            f.write(img_name.encode('utf-8'))
            # 点2D数量（这里设为0，因为不存储2D点）
            f.write(struct.pack('Q', 0))

# 使用示例
write_cameras_bin('cameras.json', 'sparse/0/cameras.bin')
write_images_bin('poses.json', 'images', 'sparse/0/images.bin')
```

## 方法3：从已有3D Gaussian Splatting模型转换

如果您已经有训练好的3D Gaussian Splatting模型，可以直接使用。

### 所需文件

```
scene0000_00/
├── images/              # RGB图像
├── sparse/0/            # COLMAP重建结果
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── output/              # 3DGS模型输出（可选，用于初始化）
    └── scene0000_00/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply
        └── chkpnt30000.pth
```

## 完整数据准备流程

### 1. 批量准备多个场景

```bash
#!/bin/bash
# prepare_all_scenes.sh

SCANNETPP_RAW="/path/to/scannetpp_raw"
OUTPUT_DIR="./data/scannetpp"
SCENES=(
    "scene0000_00"
    "scene0001_00"
    "scene0002_00"
    "scene0003_00"
    "scene0004_00"
    "scene0005_00"
    "scene0006_00"
    "scene0007_00"
    "scene0008_00"
    "scene0009_00"
)

for scene in "${SCENES[@]}"; do
    echo "Processing $scene..."
    
    # 方法1：使用准备脚本
    python prepare_scannetpp.py \
        --scannetpp_root $SCANNETPP_RAW \
        --output_root $OUTPUT_DIR \
        --scene_name $scene
    
    # 方法2：或使用COLMAP重建
    # (参考上面的COLMAP重建步骤)
done
```

### 2. 验证数据完整性

```bash
# 检查脚本
python -c "
import os
from pathlib import Path

def check_scene(scene_path):
    scene_path = Path(scene_path)
    errors = []
    
    # 检查images文件夹
    images_dir = scene_path / 'images'
    if not images_dir.exists():
        errors.append('Missing images/ folder')
    else:
        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        if len(images) == 0:
            errors.append('No images found in images/ folder')
    
    # 检查sparse文件夹
    sparse_dir = scene_path / 'sparse' / '0'
    if not sparse_dir.exists():
        errors.append('Missing sparse/0/ folder')
    else:
        required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        for f in required_files:
            if not (sparse_dir / f).exists():
                errors.append(f'Missing {f}')
    
    return errors

# 检查所有场景
data_dir = Path('./data/scannetpp')
for scene_dir in data_dir.iterdir():
    if scene_dir.is_dir():
        errors = check_scene(scene_dir)
        if errors:
            print(f'{scene_dir.name}: {errors}')
        else:
            print(f'{scene_dir.name}: OK')
"
```

## 图像命名要求

LangSplatV2期望图像文件名与COLMAP中的图像名称匹配。常见格式：

- `frame_000000.jpg`, `frame_000001.jpg`, ...
- `000000.jpg`, `000001.jpg`, ...
- 或任何与COLMAP `images.bin`中记录的名称一致

## COLMAP文件格式说明

### cameras.bin
- 包含相机内参（焦距、主点等）
- 格式：PINHOLE模型（fx, fy, cx, cy）

### images.bin
- 包含每张图像的：
  - 位姿（四元数旋转 + 平移向量）
  - 相机ID
  - 图像文件名

### points3D.bin
- 包含稀疏3D点云
- 用于初始化Gaussian Splatting

## 常见问题

### Q1: 如何知道COLMAP重建是否成功？

```bash
# 检查点云数量
colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0 \
    --output_type TXT

# 查看points3D.txt，应该有大量3D点
wc -l sparse/0/points3D.txt
```

### Q2: 图像数量要求？

- 建议至少50-100张图像
- 图像应该覆盖场景的不同角度
- 避免图像过于相似

### Q3: 如果COLMAP重建失败？

1. 检查图像质量（模糊、过曝等）
2. 增加特征点数量：`--SiftExtraction.max_num_features 8192`
3. 调整匹配参数：`--SiftMatching.guided_matching 1`
4. 使用更宽松的匹配阈值

### Q4: 如何从ScanNet++官方数据获取COLMAP格式？

ScanNet++可能不直接提供COLMAP格式，需要：
1. 使用提供的相机参数和位姿转换为COLMAP格式
2. 或使用原始图像进行COLMAP重建

## 数据准备检查清单

- [ ] 图像文件夹存在且包含图像
- [ ] 图像命名格式正确
- [ ] `sparse/0/cameras.bin` 存在
- [ ] `sparse/0/images.bin` 存在
- [ ] `sparse/0/points3D.bin` 存在
- [ ] COLMAP文件可以正常读取
- [ ] 图像数量与COLMAP记录一致

## 下一步：预处理和训练

数据准备完成后，继续：

```bash
# 1. 生成语言特征
python preprocess.py \
    --dataset_path ./data/scannetpp/scene0000_00 \
    --resolution -1 \
    --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth

# 2. 训练模型
bash train_scannetpp.sh scene0000_00 0
```

## 参考资源

- [COLMAP文档](https://colmap.github.io/)
- [3D Gaussian Splatting数据准备](https://github.com/graphdeco-inria/gaussian-splatting)
- ScanNet++官方文档

