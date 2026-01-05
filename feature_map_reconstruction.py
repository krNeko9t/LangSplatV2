"""
独立的功能模块：从渲染的 weight map 和 codebook 重建 CLIP feature map

这个模块提取了 LangSplatV2 中 codebook 与 weight map 矩阵乘法的核心功能，
可以独立使用，不依赖 GaussianModel 类。

核心功能：
1. compute_final_feature_map() - 标准版本：单层或多层码本重建
2. compute_layer_feature_map() - 支持残差量化的多层重建
3. compute_feature_map_quick() - 快速版本：使用 einsum，支持批量处理
"""

import torch


def compute_final_feature_map(
    language_feature_weight_map: torch.Tensor,
    codebook: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """
    从渲染的 weight map 和 codebook 重建最终的 CLIP feature map（标准版本）
    
    Args:
        language_feature_weight_map: 渲染得到的稀疏系数图
            Shape: [D, H, W] 其中 D = layer_num * codebook_size
        codebook: 全局码本
            Shape: [layer_num, codebook_size, 512]
        normalize: 是否对输出进行 L2 归一化
    
    Returns:
        language_feature_map: 重建的 CLIP feature map
            Shape: [512, H, W]
    
    Example:
        >>> weight_map = torch.randn(64, 480, 640)  # [codebook_size, H, W]
        >>> codebook = torch.randn(1, 64, 512)      # [layer_num, codebook_size, 512]
        >>> feature_map = compute_final_feature_map(weight_map, codebook)
        >>> print(feature_map.shape)  # [512, 480, 640]
    """
    D, H, W = language_feature_weight_map.shape
    # 将 weight map 展平为 [D, H*W]
    language_feature_weight_map = language_feature_weight_map.view(D, -1)
    
    # 将 codebook 展平为 [layer_num * codebook_size, 512]，然后转置
    # codebook.view(-1, 512) -> [layer_num * codebook_size, 512]
    # .T -> [512, layer_num * codebook_size]
    # 矩阵乘法: [512, D] @ [D, H*W] -> [512, H*W]
    language_feature = codebook.view(-1, 512).T @ language_feature_weight_map
    
    # 重塑为 [512, H, W]
    language_feature = language_feature.view(512, H, W)
    
    if normalize:
        language_feature = language_feature / (language_feature.norm(dim=0, keepdim=True) + 1e-10)
    
    return language_feature


def compute_layer_feature_map(
    language_feature_weight_map: torch.Tensor,
    codebook: torch.Tensor,
    layer_idx: int,
    normalize: bool = False
) -> torch.Tensor:
    """
    从渲染的 weight map 和 codebook 重建指定层的 CLIP feature map（支持残差量化）
    
    这个方法支持多层残差量化（Residual Vector Quantization），
    每一层的特征会累加到前一层（detached）的特征上。
    
    Args:
        language_feature_weight_map: 渲染得到的稀疏系数图
            Shape: [D, H, W] 其中 D = layer_num * codebook_size
        codebook: 全局码本
            Shape: [layer_num, codebook_size, 512]
        layer_idx: 要重建的层索引（从 0 开始）
        normalize: 是否对输出进行 L2 归一化
    
    Returns:
        language_feature_map: 重建的 CLIP feature map
            Shape: [512, H, W]
    
    Example:
        >>> weight_map = torch.randn(192, 480, 640)  # 3层，每层64
        >>> codebook = torch.randn(3, 64, 512)       # 3层码本
        >>> feature_map = compute_layer_feature_map(weight_map, codebook, layer_idx=2)
    """
    D, H, W = language_feature_weight_map.shape
    language_feature_weight_map = language_feature_weight_map.view(D, -1)
    layer_num, codebook_size, _ = codebook.shape
    
    language_feature_before = None
    for i in range(layer_idx + 1):
        # 提取第 i 层的 weight map: [codebook_size, H*W]
        layer_weights = language_feature_weight_map[i * codebook_size:(i+1)*codebook_size]
        # 矩阵乘法: [512, codebook_size] @ [codebook_size, H*W] -> [512, H*W]
        language_feature = codebook[i].T @ layer_weights
        language_feature = language_feature.view(512, H, W)
        
        # 残差累加：当前层特征 += 前一层特征（detached）
        if i > 0:
            language_feature = language_feature + language_feature_before.detach()
        
        language_feature_before = language_feature
    
    if normalize:
        language_feature = language_feature / (language_feature.norm(dim=0, keepdim=True) + 1e-10)
    
    return language_feature


def compute_feature_map_quick(
    language_feature_weight_map: torch.Tensor,
    codebook: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    快速版本：使用 einsum 进行矩阵乘法，支持批量处理和多层码本
    
    这个版本假设 weight_map 已经按照层组织好了形状。
    适用于多层码本的情况（如 3 层，每层 64）。
    
    Args:
        language_feature_weight_map: 渲染得到的稀疏系数图
            Shape: [layer_num, codebook_size, H, W] 或 [D, H, W]
        codebook: 全局码本
            Shape: [layer_num, codebook_size, 512]
        normalize: 是否对输出进行 L2 归一化
    
    Returns:
        language_feature_map: 重建的 CLIP feature map
            Shape: [layer_num, 512, H, W] 或 [512, H, W]
    
    Example:
        >>> # 单层情况
        >>> weight_map = torch.randn(64, 480, 640)  # [codebook_size, H, W]
        >>> codebook = torch.randn(1, 64, 512)      # [layer_num, codebook_size, 512]
        >>> # 需要先 reshape: weight_map.view(1, 64, 480, 640)
        >>> feature_map = compute_feature_map_quick(
        ...     weight_map.view(1, 64, 480, 640), codebook
        ... )
        
        >>> # 多层情况（3层）
        >>> weight_map = torch.randn(192, 480, 640)  # [3*64, H, W]
        >>> codebook = torch.randn(3, 64, 512)       # [3, 64, 512]
        >>> # 需要先 reshape: weight_map.view(3, 64, 480, 640)
        >>> feature_map = compute_feature_map_quick(
        ...     weight_map.view(3, 64, 480, 640), codebook
        ... )
    """
    if language_feature_weight_map.dim() == 3:
        # 如果是 [D, H, W] 格式，需要先 reshape
        D, H, W = language_feature_weight_map.shape
        layer_num, codebook_size, _ = codebook.shape
        # 假设 D = layer_num * codebook_size
        language_feature_weight_map = language_feature_weight_map.view(
            layer_num, codebook_size, H, W
        )
    
    layer_num, codebook_size, H, W = language_feature_weight_map.shape
    
    # 将 weight_map 展平为 [layer_num, codebook_size, H*W]
    language_feature_weight_map = language_feature_weight_map.view(
        layer_num, codebook_size, H * W
    )
    
    # 转置 codebook: [layer_num, codebook_size, 512] -> [layer_num, 512, codebook_size]
    language_codebooks = codebook.permute(0, 2, 1)
    
    # einsum: [layer_num, 512, codebook_size] @ [layer_num, codebook_size, H*W]
    # -> [layer_num, 512, H*W]
    language_feature_map = torch.einsum(
        'ldk,lkn->ldn', 
        language_codebooks, 
        language_feature_weight_map
    )
    
    # 重塑为 [layer_num, 512, H, W]
    language_feature_map = language_feature_map.view(layer_num, 512, H, W)
    
    if normalize:
        language_feature_map = language_feature_map / (
            language_feature_map.norm(dim=1, keepdim=True) + 1e-10
        )
    
    # 如果是单层，返回 [512, H, W]；否则返回 [layer_num, 512, H, W]
    if layer_num == 1:
        return language_feature_map.squeeze(0)
    else:
        return language_feature_map


def compute_feature_maps_all_layers(
    language_feature_weight_map: torch.Tensor,
    codebook: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """
    重建所有层的 feature map（用于多层残差量化）
    
    Args:
        language_feature_weight_map: 渲染得到的稀疏系数图
            Shape: [D, H, W] 其中 D = layer_num * codebook_size
        codebook: 全局码本
            Shape: [layer_num, codebook_size, 512]
        normalize: 是否对输出进行 L2 归一化
    
    Returns:
        language_feature_maps: 所有层的 feature map
            Shape: [layer_num, 512, H, W]
    """
    D, H, W = language_feature_weight_map.shape
    language_feature_weight_map = language_feature_weight_map.view(D, -1)
    layer_num, codebook_size, _ = codebook.shape
    
    language_features = []
    for i in range(layer_num):
        # 计算第 i 层的特征
        language_feature = codebook[i].T @ language_feature_weight_map[
            i * codebook_size:(i+1)*codebook_size
        ]
        language_feature = language_feature.view(512, H, W)
        
        # 残差累加
        if i > 0:
            language_feature = language_feature + language_features[-1].detach()
        
        language_features.append(language_feature)
    
    # 堆叠所有层: [layer_num, 512, H, W]
    result = torch.stack(language_features, dim=0)
    
    if normalize:
        result = result / (result.norm(dim=1, keepdim=True) + 1e-10)
    
    return result


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例 1: 单层码本（最常见的情况）
    print("示例 1: 单层码本")
    weight_map = torch.randn(64, 480, 640)  # [codebook_size, H, W]
    codebook = torch.randn(1, 64, 512)        # [layer_num, codebook_size, 512]
    
    feature_map = compute_final_feature_map(weight_map, codebook)
    print(f"Input weight_map shape: {weight_map.shape}")
    print(f"Input codebook shape: {codebook.shape}")
    print(f"Output feature_map shape: {feature_map.shape}")
    print()
    
    # 示例 2: 多层码本（残差量化）
    print("示例 2: 多层码本（3层）")
    weight_map_multi = torch.randn(192, 480, 640)  # [3*64, H, W]
    codebook_multi = torch.randn(3, 64, 512)       # [3, 64, 512]
    
    feature_map_layer0 = compute_layer_feature_map(
        weight_map_multi, codebook_multi, layer_idx=0
    )
    feature_map_layer2 = compute_layer_feature_map(
        weight_map_multi, codebook_multi, layer_idx=2
    )
    print(f"Layer 0 feature_map shape: {feature_map_layer0.shape}")
    print(f"Layer 2 feature_map shape: {feature_map_layer2.shape}")
    print()
    
    # 示例 3: 快速版本（einsum）
    print("示例 3: 快速版本（einsum）")
    weight_map_reshaped = weight_map_multi.view(3, 64, 480, 640)
    feature_map_quick = compute_feature_map_quick(
        weight_map_reshaped, codebook_multi
    )
    print(f"Quick version output shape: {feature_map_quick.shape}")
    print()
    
    # 示例 4: 所有层
    print("示例 4: 重建所有层")
    all_layers = compute_feature_maps_all_layers(
        weight_map_multi, codebook_multi
    )
    print(f"All layers shape: {all_layers.shape}")

