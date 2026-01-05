# ScanNet++ Dataset Support for LangSplatV2

本指南说明如何在ScanNet++数据集的10个场景上运行LangSplatV2，并通过语义查询提取每个实例对应的3D边界框（3D bbox）。

## 目录

1. [数据准备](#数据准备)
2. [数据格式要求](#数据格式要求)
3. [训练流程](#训练流程)
4. [提取3D边界框](#提取3d边界框)
5. [使用CLIP进行语义查询](#使用clip进行语义查询)

## 数据准备

### 1. 下载ScanNet++数据集

首先，您需要下载ScanNet++数据集。数据集通常包含：
- RGB图像
- 相机参数
- 相机位姿
- 3D点云（可选）

### 2. 转换为COLMAP格式

ScanNet++数据需要转换为COLMAP格式才能被LangSplatV2使用。有两种方式：

#### 方式1：使用COLMAP重建（推荐）

如果您有原始图像，可以使用COLMAP进行重建：

```bash
# 使用COLMAP重建场景
colmap feature_extractor --database_path $SCENE_PATH/database.db --image_path $SCENE_PATH/images
colmap exhaustive_matcher --database_path $SCENE_PATH/database.db
colmap mapper --database_path $SCENE_PATH/database.db --image_path $SCENE_PATH/images --output_path $SCENE_PATH/sparse
```

#### 方式2：使用准备脚本

如果您的数据已经是COLMAP格式，可以使用准备脚本：

```bash
# 准备单个场景
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp \
    --output_root ./data/scannetpp \
    --scene_name scene0000_00

# 准备多个场景（从文件列表）
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp \
    --output_root ./data/scannetpp \
    --scene_list scenes.txt

# 准备多个场景（直接指定）
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp \
    --output_root ./data/scannetpp \
    --scenes scene0000_00 scene0001_00 scene0002_00
```

## 数据格式要求

转换后的数据应具有以下结构：

```
data/scannetpp/
├── scene0000_00/
│   ├── images/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   └── language_features/  (将在预处理后生成)
│       ├── frame_000000_s.npy
│       ├── frame_000000_f.npy
│       └── ...
├── scene0001_00/
└── ...
```

## 训练流程

### 步骤1：生成语言特征

对每个场景运行预处理脚本以生成CLIP特征：

```bash
# 单个场景
python preprocess.py \
    --dataset_path ./data/scannetpp/scene0000_00 \
    --resolution -1 \
    --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth

# 批量处理（使用循环）
for scene in scene0000_00 scene0001_00 scene0002_00; do
    python preprocess.py \
        --dataset_path ./data/scannetpp/$scene \
        --resolution -1 \
        --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth
done
```

### 步骤2：训练模型

使用训练脚本训练LangSplatV2：

```bash
# 训练单个场景
bash train_scannetpp.sh scene0000_00 0

# 批量训练多个场景
for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    bash train_scannetpp.sh $scene 0
done
```

训练脚本会自动：
1. 运行预处理（如果未完成）
2. 训练模型到30000次迭代
3. 保存检查点

## 提取3D边界框

训练完成后，可以使用评估脚本提取3D边界框：

```bash
# 基本用法
bash eval_scannetpp.sh scene0000_00 0 10000 "chair" "table" "sofa"

# 指定更多语义类别
bash eval_scannetpp.sh scene0000_00 0 10000 \
    "chair" "table" "sofa" "bed" "desk" "monitor" "keyboard" "mouse"
```

### 输出格式

评估脚本会生成 `3d_bboxes.json` 文件，包含每个语义类别的3D边界框信息：

```json
{
  "chair": {
    "center": [1.23, 0.45, 2.67],
    "size": [0.5, 0.8, 0.6],
    "min_corner": [0.98, 0.05, 2.37],
    "max_corner": [1.48, 0.85, 2.97],
    "num_views": 15,
    "all_bboxes": [...]
  },
  "table": {
    ...
  }
}
```

### 直接使用Python脚本

您也可以直接使用Python脚本进行更精细的控制：

```python
python eval_scannetpp.py \
    --source_path ./data/scannetpp/scene0000_00 \
    --ckpt_root_path ./output \
    --dataset_name scene0000_00 \
    --index 0 \
    --output_dir ./results/scannetpp \
    --mask_thresh 0.4 \
    --checkpoint 10000 \
    --semantic_queries chair table sofa bed desk
```

## 使用CLIP进行语义查询

LangSplatV2使用OpenCLIP进行语义查询。支持的查询包括：

### 常见室内物体类别

- 家具：`chair`, `table`, `sofa`, `bed`, `desk`, `cabinet`, `shelf`
- 电子设备：`monitor`, `keyboard`, `mouse`, `laptop`, `tv`, `phone`
- 装饰：`picture`, `vase`, `lamp`, `clock`
- 其他：`door`, `window`, `wall`, `floor`, `ceiling`

### 查询技巧

1. **使用具体名称**：`"office chair"` 比 `"chair"` 更具体
2. **使用同义词**：可以尝试 `"sofa"` 或 `"couch"`
3. **组合查询**：可以查询多个相关物体

### 调整参数

- `--mask_thresh`：控制mask的阈值（默认0.4）
  - 较低值：更宽松的mask，可能包含更多背景
  - 较高值：更严格的mask，可能遗漏部分物体

## 完整工作流程示例

以下是一个完整的工作流程，处理10个ScanNet++场景：

```bash
# 1. 准备数据
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp \
    --output_root ./data/scannetpp \
    --scenes scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00

# 2. 预处理所有场景
for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    python preprocess.py \
        --dataset_path ./data/scannetpp/$scene \
        --resolution -1 \
        --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth
done

# 3. 训练所有场景
for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    bash train_scannetpp.sh $scene 0
done

# 4. 提取3D边界框
SEMANTIC_QUERIES="chair table sofa bed desk monitor keyboard mouse"
for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    bash eval_scannetpp.sh $scene 0 10000 $SEMANTIC_QUERIES
done
```

## 故障排除

### 问题1：COLMAP格式不存在

**解决方案**：
- 使用COLMAP重建场景
- 或确保您的ScanNet++数据包含COLMAP重建结果

### 问题2：找不到语言特征

**解决方案**：
- 确保运行了 `preprocess.py`
- 检查 `language_features/` 文件夹是否存在

### 问题3：3D边界框为空

**可能原因**：
1. 语义查询不匹配场景中的物体
2. mask阈值过高
3. 训练不充分

**解决方案**：
- 降低 `--mask_thresh` 值
- 尝试不同的语义查询
- 检查训练是否完成

### 问题4：内存不足

**解决方案**：
- 减少图像分辨率（使用 `--resolution` 参数）
- 使用更小的batch size
- 使用更少的语义查询

## 注意事项

1. **数据格式**：确保ScanNet++数据已转换为COLMAP格式
2. **GPU内存**：训练需要足够的GPU内存（建议至少24GB）
3. **训练时间**：每个场景的训练可能需要数小时
4. **语义查询**：使用英文查询词，CLIP模型对英文支持最好

## 相关文件

- `prepare_scannetpp.py`：数据准备脚本
- `train_scannetpp.sh`：训练脚本
- `eval_scannetpp.sh`：评估脚本
- `eval_scannetpp.py`：3D边界框提取脚本

## 引用

如果您使用ScanNet++数据集，请引用相应的论文。

