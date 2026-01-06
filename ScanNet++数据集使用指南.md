# ScanNet++数据集使用指南

本文档说明ScanNet++数据集的结构以及如何将其用于LangSplatV2项目。

## 一、ScanNet++数据集概述

### 1.1 数据集简介

ScanNet++是一个高保真的室内场景3D数据集，包含460个场景，提供以下数据：

- **高精度几何数据**：使用高端激光扫描仪以亚毫米级分辨率捕获的3D重建模型
- **高分辨率图像**：由3300万像素的DSLR相机拍摄的图像（约28万张）
- **RGB-D序列**：来自iPhone的RGB-D视频流（超过370万帧）
- **语义和实例标注**：开放词汇的语义标注，包括明确标注语义模糊的场景

### 1.2 每个Scene的数据结构

ScanNet++每个scene通常包含以下文件夹和文件：

```
<scene_name>/
├── dslr/                          # DSLR相机数据
│   ├── images/                    # 高分辨率RGB图像
│   │   ├── <image_0>.jpg
│   │   ├── <image_1>.jpg
│   │   └── ...
│   ├── cameras/                   # 相机参数
│   │   ├── intrinsics.txt         # 相机内参
│   │   └── extrinsics.txt         # 相机外参（位姿）
│   └── ...
├── iphone/                        # iPhone RGB-D数据（可选）
│   ├── color/                     # RGB图像
│   ├── depth/                     # 深度图
│   └── ...
├── mesh/                          # 3D网格模型
│   └── <scene_name>.ply
├── semantics/                     # 语义标注（可选）
│   └── ...
└── metadata/                      # 元数据
    └── ...
```

## 二、项目所需的数据格式

LangSplatV2项目需要以下数据结构：

```
<dataset_name>/
├── images/                        # RGB图像文件夹（必需）
│   ├── <image_0>.jpg
│   ├── <image_1>.jpg
│   └── ...
├── sparse/                        # COLMAP稀疏重建结果（必需）
│   └── 0/
│       ├── cameras.bin            # 相机内参（二进制格式）
│       ├── images.bin             # 相机外参和图像信息（二进制格式）
│       └── points3D.bin           # 稀疏3D点云（二进制格式）
└── output/                        # 预训练的RGB高斯点云模型（必需）
    └── <dataset_name>/
        ├── point_cloud/
        │   └── iteration_30000/
        │       └── point_cloud.ply
        ├── cameras.json
        ├── cfg_args
        ├── chkpnt30000.pth        # 预训练checkpoint（必需）
        └── input.ply
```

## 三、数据转换步骤

### Step 1: 准备图像数据

**方法1：使用DSLR图像（推荐）**

```bash
# 创建目标目录
mkdir -p <output_path>/<scene_name>/images

# 复制DSLR图像到images文件夹
cp <scannetpp_path>/<scene_name>/dslr/images/*.jpg <output_path>/<scene_name>/images/
```

**方法2：使用iPhone RGB图像**

```bash
# 如果DSLR图像不可用，可以使用iPhone的RGB图像
cp <scannetpp_path>/<scene_name>/iphone/color/*.jpg <output_path>/<scene_name>/images/
```

### Step 2: 生成COLMAP稀疏重建

ScanNet++提供了相机参数，但项目需要COLMAP格式的稀疏重建。有两种方法：

#### 方法A：使用ScanNet++提供的相机参数转换为COLMAP格式（推荐）

需要编写转换脚本，将ScanNet++的相机参数转换为COLMAP格式：

```python
# 转换脚本示例（需要根据实际数据格式调整）
import numpy as np
from scene.colmap_loader import write_extrinsics_binary, write_intrinsics_binary

# 读取ScanNet++的相机参数
# intrinsics: 相机内参（fx, fy, cx, cy）
# extrinsics: 相机外参（旋转矩阵R，平移向量t）

# 转换为COLMAP格式
# 1. 内参：COLMAP使用PINHOLE模型（fx, fy, cx, cy）
# 2. 外参：需要转换为四元数qvec和平移向量tvec
```

#### 方法B：使用COLMAP重新进行稀疏重建

如果ScanNet++的相机参数格式不兼容，可以使用COLMAP重新进行重建：

```bash
# 安装COLMAP（如果未安装）
# 参考：https://colmap.github.io/install.html

# 运行COLMAP稀疏重建
colmap feature_extractor \
    --database_path <output_path>/<scene_name>/database.db \
    --image_path <output_path>/<scene_name>/images

colmap exhaustive_matcher \
    --database_path <output_path>/<scene_name>/database.db

mkdir -p <output_path>/<scene_name>/sparse/0

colmap mapper \
    --database_path <output_path>/<scene_name>/database.db \
    --image_path <output_path>/<scene_name>/images \
    --output_path <output_path>/<scene_name>/sparse
```

**注意**：COLMAP重建可能需要较长时间，特别是对于高分辨率图像。

### Step 3: 训练RGB高斯点云模型

在运行LangSplatV2之前，需要先使用[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)训练RGB模型：

```bash
# 克隆3D Gaussian Splatting仓库
git clone https://github.com/graphdeco-inria/gaussian-splatting.git

# 训练RGB模型
cd gaussian-splatting
python train.py \
    -s <output_path>/<scene_name> \
    -m <output_path>/<scene_name>/output/<scene_name> \
    --iterations 30000

# 训练完成后，checkpoint会保存在：
# <output_path>/<scene_name>/output/<scene_name>/point_cloud/iteration_30000/point_cloud.ply
# <output_path>/<scene_name>/output/<scene_name>/chkpnt30000.pth
```

### Step 4: 验证数据结构

确保以下文件存在：

```bash
# 检查图像文件夹
ls <output_path>/<scene_name>/images/*.jpg

# 检查COLMAP稀疏重建
ls <output_path>/<scene_name>/sparse/0/cameras.bin
ls <output_path>/<scene_name>/sparse/0/images.bin
ls <output_path>/<scene_name>/sparse/0/points3D.bin

# 检查RGB模型checkpoint
ls <output_path>/<scene_name>/output/<scene_name>/chkpnt30000.pth
ls <output_path>/<scene_name>/output/<scene_name>/point_cloud/iteration_30000/point_cloud.ply
```

## 四、运行LangSplatV2流程

数据准备完成后，按照标准流程运行：

### Step 1: 预处理 - 生成语言特征

```bash
python preprocess.py --dataset_path <output_path>/<scene_name>
```

**输出**：
- `<output_path>/<scene_name>/language_features/`目录
- 每张图像生成`<image_name>_f.npy`和`<image_name>_s.npy`

### Step 2: 训练语言特征模型

```bash
bash train.sh <output_path> <scene_name> 0
```

**参数说明**：
- `<output_path>`: 数据集根路径（包含所有scene的目录）
- `<scene_name>`: 场景名称
- `0`: 模型索引

**输出**：
- `output/<scene_name>_0_1/chkpnt<iter>.pth` (level 1)
- `output/<scene_name>_0_2/chkpnt<iter>.pth` (level 2)
- `output/<scene_name>_0_3/chkpnt<iter>.pth` (level 3)

### Step 3: 评估（需要创建评估脚本）

由于ScanNet++不是项目默认支持的数据集，可能需要：

1. **创建评估脚本**：参考`eval_lerf.py`创建`eval_scannetpp.py`
2. **准备查询文本**：ScanNet++提供语义标注，可以用于评估

或者直接使用可视化脚本：

```bash
# 需要先创建visualize_scannetpp.sh脚本
bash visualize_scannetpp.sh <scene_name> 0 10000 "chair" 0
```

## 五、常见问题和解决方案

### 问题1: COLMAP重建失败

**可能原因**：
- 图像质量不足
- 图像重叠度不够
- 相机参数不正确

**解决方案**：
- 检查图像是否清晰
- 确保图像之间有足够的重叠（建议>60%）
- 如果使用ScanNet++提供的相机参数，确保格式转换正确

### 问题2: 图像分辨率过高导致内存不足

**解决方案**：
- 在`preprocess.py`中使用`--resolution`参数降低分辨率
- 或者在复制图像时先进行resize

```bash
# 使用ImageMagick批量resize
for img in <scannetpp_path>/<scene_name>/dslr/images/*.jpg; do
    convert "$img" -resize 1920x1080 <output_path>/<scene_name>/images/$(basename "$img")
done
```

### 问题3: 相机参数格式不匹配

**解决方案**：
- 检查ScanNet++的相机参数格式
- 编写转换脚本将格式转换为COLMAP格式
- 或者使用COLMAP重新进行重建

### 问题4: RGB模型训练失败

**解决方案**：
- 确保COLMAP稀疏重建成功
- 检查点云数量是否足够（建议>10万点）
- 调整训练参数（学习率、迭代次数等）

## 六、数据转换脚本示例

### 脚本1: 批量转换ScanNet++场景

```bash
#!/bin/bash
# convert_scannetpp.sh

SCANNETPP_ROOT=$1
OUTPUT_ROOT=$2

for scene_dir in $SCANNETPP_ROOT/*/; do
    scene_name=$(basename "$scene_dir")
    echo "Processing scene: $scene_name"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_ROOT/$scene_name/images"
    
    # 复制图像
    if [ -d "$scene_dir/dslr/images" ]; then
        cp "$scene_dir/dslr/images"/*.jpg "$OUTPUT_ROOT/$scene_name/images/"
    elif [ -d "$scene_dir/iphone/color" ]; then
        cp "$scene_dir/iphone/color"/*.jpg "$OUTPUT_ROOT/$scene_name/images/"
    fi
    
    # 运行COLMAP重建（需要先安装COLMAP）
    # ... COLMAP命令 ...
    
    echo "Scene $scene_name processed"
done
```

### 脚本2: 相机参数转换（Python示例）

```python
# convert_cameras.py
# 注意：这只是一个示例框架，需要根据ScanNet++的实际数据格式调整

import numpy as np
import os
from scene.colmap_loader import Camera, Image, write_intrinsics_binary, write_extrinsics_binary

def convert_scannetpp_to_colmap(scannetpp_path, output_path):
    """
    将ScanNet++的相机参数转换为COLMAP格式
    """
    # 读取ScanNet++的相机参数
    # 这里需要根据实际格式实现
    intrinsics_file = os.path.join(scannetpp_path, "dslr/cameras/intrinsics.txt")
    extrinsics_file = os.path.join(scannetpp_path, "dslr/cameras/extrinsics.txt")
    
    # 解析内参和外参
    # ... 实现解析逻辑 ...
    
    # 转换为COLMAP格式
    cameras = {}
    images = {}
    
    # 创建COLMAP格式的cameras和images
    # ... 转换逻辑 ...
    
    # 保存为二进制格式
    os.makedirs(os.path.join(output_path, "sparse/0"), exist_ok=True)
    write_intrinsics_binary(cameras, os.path.join(output_path, "sparse/0/cameras.bin"))
    write_extrinsics_binary(images, os.path.join(output_path, "sparse/0/images.bin"))
```

## 七、参考资料

1. **ScanNet++论文**: [arXiv:2308.11417](https://arxiv.org/abs/2308.11417)
2. **ScanNet++官网**: 访问官方网站获取数据集下载链接和详细文档
3. **COLMAP文档**: [https://colmap.github.io/](https://colmap.github.io/)
4. **3D Gaussian Splatting**: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## 八、总结

使用ScanNet++数据集运行LangSplatV2的主要步骤：

1. ✅ **准备图像**：从ScanNet++复制RGB图像到`images/`文件夹
2. ✅ **生成COLMAP重建**：将ScanNet++相机参数转换为COLMAP格式，或使用COLMAP重新重建
3. ✅ **训练RGB模型**：使用3D Gaussian Splatting训练RGB高斯点云模型
4. ✅ **预处理**：运行`preprocess.py`生成语言特征
5. ✅ **训练语言模型**：运行`train.sh`训练语言特征模型
6. ✅ **评估/可视化**：创建评估脚本或使用可视化工具

**关键点**：
- ScanNet++提供高质量的几何和图像数据，非常适合用于LangSplatV2
- 主要挑战是将ScanNet++的数据格式转换为项目所需的COLMAP格式
- 建议优先使用DSLR图像，因为它们具有更高的分辨率和质量
