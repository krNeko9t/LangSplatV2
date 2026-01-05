# 直接3D空间边界框提取指南

本指南说明如何使用直接在3D空间提取边界框的方法，无需2D投影。

## 方法概述

与2D投影方法不同，本方法直接在3D空间操作：

1. **从训练好的模型中提取数据**：
   - 高斯点位置 (xyz)
   - 稀疏系数索引 (Top-K Indices)
   - 稀疏系数权重 (Top-K Values)
   - 全局码本 (Global Codebook)

2. **计算每个高斯点的语义特征**：
   - 使用稀疏系数从码本中提取特征
   - 对多层级特征进行残差连接

3. **计算与文本的余弦相似度**

4. **阈值过滤**

5. **聚类**：使用 DBSCAN 或 HDBSCAN 对保留的3D点进行聚类

6. **计算边界框**：
   - AABB (Axis-Aligned Bounding Box)
   - OBB (Oriented Bounding Box)

## 优势

- **更直接**：直接在3D空间操作，无需渲染2D图像
- **更高效**：避免了2D渲染的开销
- **更准确**：直接使用3D点云信息，不受视角限制
- **支持多实例**：通过聚类可以分离多个相同类别的实例

## 使用方法

### 基本用法

```bash
bash eval_scannetpp_direct.sh scene0000_00 0 10000 chair table sofa
```

### 使用Python脚本（更灵活）

```bash
python extract_3d_bbox_direct.py \
    --source_path ./data/scannetpp/scene0000_00 \
    --ckpt_root_path ./output \
    --dataset_name scene0000_00 \
    --index 0 \
    --output_dir ./results/scannetpp \
    --checkpoint 10000 \
    --topk 4 \
    --similarity_threshold 0.3 \
    --clustering_method dbscan \
    --dbscan_eps 0.1 \
    --dbscan_min_samples 10 \
    --semantic_queries chair table sofa
```

## 参数说明

### 核心参数

- `--topk`: Top-K稀疏系数数量（默认4）
- `--similarity_threshold`: 相似度阈值，用于过滤点（默认0.3）
- `--clustering_method`: 聚类方法，`dbscan` 或 `hdbscan`（默认`dbscan`）

### DBSCAN参数

- `--dbscan_eps`: DBSCAN的eps参数，控制邻域半径（默认0.1）
- `--dbscan_min_samples`: DBSCAN的最小样本数（默认10）

### HDBSCAN参数

- `--hdbscan_min_cluster_size`: HDBSCAN的最小簇大小（默认50）

## 输出格式

结果保存在 `3d_bboxes_direct.json`，格式如下：

```json
{
  "chair": {
    "num_instances": 2,
    "total_points": 1523,
    "instances": [
      {
        "cluster_id": 0,
        "num_points": 856,
        "aabb": {
          "center": [1.23, 0.45, 2.67],
          "size": [0.5, 0.8, 0.6],
          "min_corner": [0.98, 0.05, 2.37],
          "max_corner": [1.48, 0.85, 2.97]
        },
        "obb": {
          "center": [1.23, 0.45, 2.67],
          "size": [0.48, 0.82, 0.58],
          "rotation": [[...], [...], [...]],
          "eigenvalues": [0.12, 0.08, 0.05],
          "corners": [[...], [...], ...],
          "min_corner": [...],
          "max_corner": [...]
        }
      },
      {
        "cluster_id": 1,
        "num_points": 667,
        "aabb": {...},
        "obb": {...}
      }
    ]
  },
  "table": {...}
}
```

## 边界框类型

### AABB (Axis-Aligned Bounding Box)

轴对齐边界框，与坐标轴平行：
- `center`: 中心点
- `size`: 尺寸 [width, height, depth]
- `min_corner`: 最小角点
- `max_corner`: 最大角点

### OBB (Oriented Bounding Box)

定向边界框，使用PCA计算主方向：
- `center`: 中心点
- `size`: 在主方向上的尺寸
- `rotation`: 旋转矩阵（3x3）
- `eigenvalues`: 主成分的特征值
- `corners`: 8个角点的3D坐标

## 参数调优建议

### 相似度阈值

- **太低**（<0.2）：可能包含太多背景点
- **太高**（>0.5）：可能遗漏部分物体
- **推荐范围**：0.25-0.4

### DBSCAN参数

- **eps**: 
  - 太小：产生很多小簇
  - 太大：多个实例被合并
  - 建议：根据场景尺度调整（0.05-0.2）

- **min_samples**:
  - 太小：产生噪声点
  - 太大：遗漏小实例
  - 建议：10-20

### HDBSCAN参数

- **min_cluster_size**:
  - 根据期望的最小实例大小设置
  - 建议：30-100

## 批量处理多个场景

```bash
SEMANTIC_QUERIES="chair table sofa bed desk monitor keyboard mouse"

for scene in scene0000_00 scene0001_00 scene0002_00 scene0003_00 scene0004_00 \
             scene0005_00 scene0006_00 scene0007_00 scene0008_00 scene0009_00; do
    bash eval_scannetpp_direct.sh $scene 0 10000 $SEMANTIC_QUERIES
done
```

## 与2D投影方法对比

| 特性 | 2D投影方法 | 直接3D方法 |
|------|-----------|-----------|
| 计算开销 | 需要渲染 | 无需渲染 |
| 视角依赖 | 是 | 否 |
| 多实例分离 | 需要多视角 | 自动聚类 |
| 精度 | 受投影误差影响 | 直接使用3D信息 |
| 速度 | 较慢 | 较快 |

## 故障排除

### 问题1：没有检测到任何点

**可能原因**：
- 相似度阈值过高
- 语义查询不匹配

**解决方案**：
- 降低 `--similarity_threshold`
- 尝试不同的查询词

### 问题2：多个实例被合并

**可能原因**：
- DBSCAN eps过大
- 实例距离太近

**解决方案**：
- 降低 `--dbscan_eps`
- 使用HDBSCAN并调整 `--hdbscan_min_cluster_size`

### 问题3：产生太多小簇

**可能原因**：
- DBSCAN eps过小
- min_samples过小

**解决方案**：
- 增加 `--dbscan_eps`
- 增加 `--dbscan_min_samples`

## 依赖安装

确保安装了必要的依赖：

```bash
# sklearn (DBSCAN)
pip install scikit-learn

# hdbscan (可选，用于HDBSCAN)
pip install hdbscan
```

## 示例输出

```
Loading checkpoints from all levels...
Extracted 125430 Gaussian points
Codebook shape: torch.Size([192, 512])
Top-K indices shape: torch.Size([125430, 12])
Top-K weights shape: torch.Size([125430, 12])
Computing semantic features for each point...
Processing query: chair
  Filtered to 1523 points (threshold=0.3)
  Clustering points using dbscan...
  Found 2 clusters
  Extracted 2 instances for chair
Processing query: table
  Filtered to 2341 points (threshold=0.3)
  Clustering points using dbscan...
  Found 1 clusters
  Extracted 1 instances for table
Saved results to ./results/scannetpp/scene0000_00_0/3d_bboxes_direct.json
```

## 引用

如果使用本方法，请引用LangSplatV2论文。

