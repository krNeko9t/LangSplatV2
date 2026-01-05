# Feature Map Reconstruction 模块使用说明

这个模块提取了 LangSplatV2 中 **codebook 与 weight map 矩阵乘法重建 CLIP feature map** 的核心功能，可以独立使用。

## 核心功能

从渲染得到的 `language_feature_weight_map`（稀疏系数图）和训练好的 `codebook`（全局码本），通过矩阵乘法重建出 512 维的 CLIP feature map。

## 输入输出格式

### 输入
- **language_feature_weight_map**: 渲染得到的稀疏系数图
  - Shape: `[D, H, W]` 其中 `D = layer_num * codebook_size`
  - 例如：单层时 `[64, 480, 640]`，3层时 `[192, 480, 640]`
  
- **codebook**: 全局码本（训练好的）
  - Shape: `[layer_num, codebook_size, 512]`
  - 例如：单层时 `[1, 64, 512]`，3层时 `[3, 64, 512]`

### 输出
- **language_feature_map**: 重建的 CLIP feature map
  - Shape: `[512, H, W]`

## 使用方法

### 方法 1: `compute_final_feature_map()` - 标准版本（推荐）

最简单直接的方法，适用于单层或多层码本：

```python
import torch
from feature_map_reconstruction import compute_final_feature_map

# 假设你已经有了渲染的 weight_map 和训练好的 codebook
weight_map = torch.randn(64, 480, 640)  # [codebook_size, H, W]
codebook = torch.randn(1, 64, 512)       # [layer_num, codebook_size, 512]

# 重建 feature map
feature_map = compute_final_feature_map(weight_map, codebook)
# 输出: [512, 480, 640]

# 可选：L2 归一化
feature_map = compute_final_feature_map(weight_map, codebook, normalize=True)
```

### 方法 2: `compute_layer_feature_map()` - 支持残差量化

适用于多层残差量化（Residual Vector Quantization），可以指定重建到哪一层：

```python
from feature_map_reconstruction import compute_layer_feature_map

# 3层码本
weight_map = torch.randn(192, 480, 640)  # [3*64, H, W]
codebook = torch.randn(3, 64, 512)       # [3, 64, 512]

# 重建到第 0 层
feature_map_layer0 = compute_layer_feature_map(
    weight_map, codebook, layer_idx=0
)

# 重建到第 2 层（会累加前两层的残差）
feature_map_layer2 = compute_layer_feature_map(
    weight_map, codebook, layer_idx=2
)
```

### 方法 3: `compute_feature_map_quick()` - 快速版本（einsum）

使用 einsum 进行批量计算，性能更好：

```python
from feature_map_reconstruction import compute_feature_map_quick

# 需要先将 weight_map reshape 为 [layer_num, codebook_size, H, W]
weight_map = torch.randn(192, 480, 640)  # [3*64, H, W]
weight_map_reshaped = weight_map.view(3, 64, 480, 640)  # [3, 64, H, W]
codebook = torch.randn(3, 64, 512)       # [3, 64, 512]

feature_map = compute_feature_map_quick(weight_map_reshaped, codebook)
# 输出: [512, 480, 640] (单层) 或 [3, 512, 480, 640] (多层)
```

## 与原仓库的对应关系

| 原仓库函数 | 本模块函数 | 说明 |
|-----------|-----------|------|
| `GaussianModel.compute_final_feature_map()` | `compute_final_feature_map()` | 标准版本 |
| `GaussianModel.compute_layer_feature_map()` | `compute_layer_feature_map()` | 支持残差量化 |
| `render_language_feature_map_quick()` | `compute_feature_map_quick()` | 快速版本（einsum） |

## 完整使用示例

```python
import torch
from feature_map_reconstruction import compute_final_feature_map

# 1. 从 LangSplatV2 模型加载 codebook
# 假设你已经有了训练好的模型
# codebook = gaussians._language_feature_codebooks  # [layer_num, codebook_size, 512]

# 2. 从渲染结果获取 weight_map
# 假设你已经有了渲染结果
# render_output = render(...)
# weight_map = render_output['language_feature_weight_map']  # [D, H, W]

# 3. 重建 feature map
feature_map = compute_final_feature_map(weight_map, codebook)

# 4. 使用 feature map（例如：CLIP 查询）
# query_text = "a red car"
# query_embedding = clip_model.encode_text(query_text)  # [512]
# similarity_map = (feature_map.permute(1, 2, 0) @ query_embedding)  # [H, W]
```

## 注意事项

1. **设备一致性**: 确保 `weight_map` 和 `codebook` 在同一个设备上（CPU 或 GPU）
2. **数据类型**: 建议使用 `float32` 或 `float16`
3. **归一化**: 如果需要与 CLIP 查询，建议对 feature map 进行 L2 归一化
4. **内存**: 对于大分辨率图像，注意内存占用

## 数学原理

核心公式：
```
Feature_map = Codebook^T @ Weight_map
```

其中：
- `Codebook`: `[layer_num * codebook_size, 512]` (展平后转置)
- `Weight_map`: `[layer_num * codebook_size, H*W]` (展平后)
- `Feature_map`: `[512, H*W]` → reshape 为 `[512, H, W]`

对于多层残差量化：
```
Feature_map_layer_i = Codebook_i^T @ Weight_map_i + Feature_map_layer_{i-1}
```

## 依赖

- PyTorch >= 1.8.0

## 测试

运行模块文件可以查看示例：

```bash
python feature_map_reconstruction.py
```

