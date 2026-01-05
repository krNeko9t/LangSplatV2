# LangSplatV2 可视化使用说明

这个文档说明如何使用可视化脚本查看LERF数据集（如teatime场景）的输出结果。

## 前提条件

1. 已完成模型训练，checkpoint保存在 `output/` 目录下
2. 已准备好数据集（路径结构应与训练时一致）

## 快速开始

### 方法1: 使用shell脚本（推荐）

```bash
# 基本用法：可视化单个查询和单个视图
bash visualize_lerf.sh teatime 0 10000 "teapot" 0

# 参数说明：
# teatime: 数据集名称
# 0: 模型索引
# 10000: checkpoint迭代次数
# "teapot": 查询文本
# 0: 视图索引
```

### 方法2: 直接使用Python脚本

```bash
python visualize_lerf.py \
    -s ../../data/lerf_ovs/teatime \
    --dataset_name teatime \
    --index 0 \
    --ckpt_root_path output \
    --output_dir visualize_result \
    --checkpoint 10000 \
    --include_feature \
    --quick_render \
    --topk 4 \
    --query "teapot" \
    --view_idx 0
```

## 参数说明

### 必需参数
- `-s`: 数据集路径（源路径）
- `--dataset_name`: 数据集名称，如 `teatime`
- `--index`: 模型索引，如 `0`
- `--ckpt_root_path`: checkpoint根目录，默认为 `output`
- `--output_dir`: 可视化结果输出目录
- `--checkpoint`: checkpoint迭代次数，如 `10000`
- `--include_feature`: 启用语言特征
- `--quick_render`: 使用快速渲染模式

### 可选参数
- `--query`: 单个查询文本（如 `"teapot"`, `"book"`, `"cup"`）
- `--queries`: 多个查询文本（如 `--queries teapot book cup`）
- `--view_idx`: 单个视图索引（从0开始）
- `--view_indices`: 多个视图索引（如 `--view_indices 0 1 2 3 4`）
- `--topk`: topk值，默认为1

## 输出结果

可视化结果保存在 `visualize_result/<dataset_name>_<index>/<query>/view_<view_idx>/` 目录下，包含：

1. **rgb.png**: 渲染的RGB图像
2. **heatmap.png**: 查询相关性热力图（使用turbo colormap）
3. **overlay.png**: RGB图像与热力图的叠加（半透明）
4. **mask.png**: 二值化分割mask（阈值=0.4）
5. **mask_overlay.png**: RGB图像与分割mask的叠加

## 示例

### 可视化teatime场景的多个查询

```bash
python visualize_lerf.py \
    -s ../../data/lerf_ovs/teatime \
    --dataset_name teatime \
    --index 0 \
    --ckpt_root_path output \
    --output_dir visualize_result \
    --checkpoint 10000 \
    --include_feature \
    --quick_render \
    --topk 4 \
    --queries teapot book cup spoon \
    --view_idx 0
```

### 可视化多个视图

```bash
python visualize_lerf.py \
    -s ../../data/lerf_ovs/teatime \
    --dataset_name teatime \
    --index 0 \
    --ckpt_root_path output \
    --output_dir visualize_result \
    --checkpoint 10000 \
    --include_feature \
    --quick_render \
    --topk 4 \
    --query "teapot" \
    --view_indices 0 1 2 3 4 5
```

## 注意事项

1. 确保checkpoint路径正确：应为 `output/<dataset_name>_<index>_{1,2,3}/chkpnt<checkpoint>.pth`
2. 需要3个level的checkpoint（level 1, 2, 3）
3. 可视化过程会在GPU上运行，确保有足够的显存
4. 查询文本使用英文，大小写不敏感

## 故障排除

如果遇到错误：
1. 检查checkpoint路径是否正确
2. 检查数据集路径是否正确
3. 确保已启用 `--include_feature` 和 `--quick_render` 参数
4. 检查GPU显存是否充足


