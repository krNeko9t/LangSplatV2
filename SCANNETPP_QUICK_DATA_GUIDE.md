# ScanNet++ 数据准备快速指南

## 必需目录结构

```
data/scannetpp/
└── scene0000_00/
    ├── images/                    # RGB图像（必需）
    │   ├── frame_000000.jpg
    │   ├── frame_000001.jpg
    │   └── ...
    └── sparse/0/                  # COLMAP重建结果（必需）
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## 快速准备步骤

### 方法1：使用COLMAP重建（推荐）

```bash
# 1. 准备图像
SCENE_NAME="scene0000_00"
mkdir -p data/scannetpp/${SCENE_NAME}/images
# 复制您的RGB图像到 images/ 文件夹

# 2. COLMAP重建
cd data/scannetpp/${SCENE_NAME}
colmap feature_extractor --database_path database.db --image_path images
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images --output_path sparse
```

### 方法2：使用准备脚本

```bash
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp_raw \
    --output_root ./data/scannetpp \
    --scene_name scene0000_00
```

## 验证数据

```bash
# 检查数据完整性
python check_scannetpp_data.py --data_root ./data/scannetpp

# 或检查单个场景
python check_scannetpp_data.py --data_root ./data/scannetpp --scene scene0000_00
```

## 检查清单

- [ ] `images/` 文件夹存在且包含图像
- [ ] `sparse/0/cameras.bin` 存在
- [ ] `sparse/0/images.bin` 存在  
- [ ] `sparse/0/points3D.bin` 存在
- [ ] 图像文件名与COLMAP记录一致

## 下一步

数据准备完成后：

```bash
# 1. 预处理（生成语言特征）
python preprocess.py --dataset_path ./data/scannetpp/scene0000_00

# 2. 训练
bash train_scannetpp.sh scene0000_00 0
```

## 详细文档

更多详细信息请参考 `SCANNETPP_DATA_PREPARATION.md`

