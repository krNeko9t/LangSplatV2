# ScanNet++ 快速开始指南

本指南提供在ScanNet++数据集上运行LangSplatV2并提取3D边界框的快速开始步骤。

## 前提条件

1. 已安装LangSplatV2环境
2. 已下载ScanNet++数据集
3. 数据已转换为COLMAP格式（或使用COLMAP重建）

## 快速开始（3步）

### 步骤1：数据准备

```bash
# 准备10个场景的数据
python prepare_scannetpp.py \
    --scannetpp_root /path/to/scannetpp \
    --output_root ./data/scannetpp \
    --scenes scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00
```

### 步骤2：预处理和训练

```bash
# 对每个场景进行预处理和训练
for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    
    # 预处理
    python preprocess.py \
        --dataset_path ./data/scannetpp/$scene \
        --resolution -1 \
        --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth
    
    # 训练
    bash train_scannetpp.sh $scene 0
done
```

### 步骤3：提取3D边界框

```bash
# 定义要查询的语义类别
SEMANTIC_QUERIES="chair table sofa bed desk monitor keyboard mouse"

# 对每个场景提取3D边界框
for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    bash eval_scannetpp.sh $scene 0 10000 $SEMANTIC_QUERIES
done
```

## 输出结果

每个场景的结果保存在：
```
./results/scannetpp/{scene_name}_0/3d_bboxes.json
```

JSON格式示例：
```json
{
  "chair": {
    "center": [1.23, 0.45, 2.67],
    "size": [0.5, 0.8, 0.6],
    "min_corner": [0.98, 0.05, 2.37],
    "max_corner": [1.48, 0.85, 2.97],
    "num_views": 15
  }
}
```

## 常用语义查询词

### 家具类
- `chair`, `table`, `sofa`, `bed`, `desk`, `cabinet`, `shelf`, `bookshelf`

### 电子设备
- `monitor`, `keyboard`, `mouse`, `laptop`, `tv`, `phone`, `printer`

### 其他
- `door`, `window`, `picture`, `lamp`, `vase`, `clock`

## 参数调整

### 调整mask阈值

如果检测到的物体太少，降低阈值：
```bash
bash eval_scannetpp.sh scene0000_00 0 10000 --mask_thresh 0.3 chair table
```

如果检测到太多背景，提高阈值：
```bash
bash eval_scannetpp.sh scene0000_00 0 10000 --mask_thresh 0.5 chair table
```

### 使用不同的checkpoint

```bash
bash eval_scannetpp.sh scene0000_00 0 30000 chair table  # 使用30000迭代的checkpoint
```

## 故障排除

### 问题：找不到COLMAP文件

**解决**：确保数据已转换为COLMAP格式，或使用COLMAP重建场景。

### 问题：内存不足

**解决**：减少图像分辨率：
```bash
python preprocess.py --dataset_path ./data/scannetpp/scene0000_00 --resolution 640
```

### 问题：3D边界框为空

**解决**：
1. 检查语义查询是否匹配场景中的物体
2. 降低mask阈值
3. 确保训练已完成

## 详细文档

更多详细信息请参考 `SCANNETPP_README.md`。

