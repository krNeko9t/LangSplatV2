# 直接3D边界框提取 - 快速开始

## 三步快速开始

### 步骤1：确保已训练模型

```bash
# 如果还没有训练，先训练模型
bash train_scannetpp.sh scene0000_00 0
```

### 步骤2：提取3D边界框

```bash
# 使用shell脚本（简单）
bash eval_scannetpp_direct.sh scene0000_00 0 10000 chair table sofa

# 或使用Python脚本（更灵活）
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
    --semantic_queries chair table sofa
```

### 步骤3：查看结果

结果保存在：
```
./results/scannetpp/scene0000_00_0/3d_bboxes_direct.json
```

## 关键参数快速调整

### 如果检测不到物体
```bash
# 降低相似度阈值
--similarity_threshold 0.2
```

### 如果多个实例被合并
```bash
# 减小DBSCAN eps
--dbscan_eps 0.05
```

### 如果产生太多小簇
```bash
# 增加DBSCAN eps或使用HDBSCAN
--clustering_method hdbscan --hdbscan_min_cluster_size 100
```

## 批量处理

```bash
SEMANTIC_QUERIES="chair table sofa bed desk"

for scene in scene0000_00 scene0001_00 scene0002_00; do
    bash eval_scannetpp_direct.sh $scene 0 10000 $SEMANTIC_QUERIES
done
```

## 输出示例

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
          "size": [0.5, 0.8, 0.6]
        },
        "obb": {
          "center": [1.23, 0.45, 2.67],
          "size": [0.48, 0.82, 0.58],
          "rotation": [[...], [...], [...]]
        }
      }
    ]
  }
}
```

更多详细信息请参考 `DIRECT_3D_BBOX_README.md`。

