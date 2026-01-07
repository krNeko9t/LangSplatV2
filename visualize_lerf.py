#!/usr/bin/env python
"""
可视化脚本用于LangSplatV2的LERF数据集
用法示例:
    python visualize_lerf.py \
        -s ../../data/lerf_ovs/teatime \
        --dataset_name teatime \
        --index 0 \
        --ckpt_root_path output \
        --output_dir visualize_result \
        --checkpoint 10000 \
        --query "teapot" \
        --view_idx 0 \
        --include_feature \
        --quick_render \
        --topk 4
"""
import sklearn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import time
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
import eval.colormaps as colormaps
import sys
sys.path.append("eval")
from eval.utils import colormap_saving, vis_mask_save
from utils.vq_utils import get_weights_and_indices

import torch.nn.functional as F


def render_language_feature_map_quick(gaussians:GaussianModel, view, pipeline, background, args):
    """快速渲染语言特征图"""
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        rendered_rgb = output['render']  # RGB渲染结果
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(3, 64, H, W).view(3, 64, H*W)
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(3, 512, H, W)
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)

    return language_feature_map, rendered_rgb


def visualize_single_query(
    dataset: ModelParams, 
    pipeline: PipelineParams, 
    args,
    query_text: str,
    view_idx: int = 0
):
    """可视化单个文本查询的结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    output_path = Path(args.output_path)
    query_output_dir = output_path / query_text / f"view_{view_idx:05d}"
    query_output_dir.mkdir(exist_ok=True, parents=True)
    
    # 初始化CLIP模型
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives([query_text])
    
    # 加载场景和视图
    combined_gaussians = GaussianModel(dataset.sh_degree)
    dataset.model_path = args.ckpt_paths[0]
    scene = Scene(dataset, combined_gaussians, shuffle=False)
    views = scene.getTrainCameras()
    
    if view_idx >= len(views):
        print(f"警告: view_idx {view_idx} 超出范围，使用最后一个视图")
        view_idx = len(views) - 1
    
    view = views[view_idx]
    
    # 加载checkpoint
    checkpoint = os.path.join(args.ckpt_paths[0], f'chkpnt{args.checkpoint}.pth')
    (model_params, first_iter) = torch.load(checkpoint)
    combined_gaussians.restore(model_params, args, mode='test')
    
    # 组合3个level的codebooks
    language_feature_weights = []
    language_feature_indices = []
    language_feature_codebooks = []
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    for level_idx in range(3):
        gaussians = GaussianModel(dataset.sh_degree)
        checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        language_feature_codebooks.append(gaussians._language_feature_codebooks.view(-1, 512))
        weights, indices = get_weights_and_indices(gaussians._language_feature_logits, 4)
        language_feature_weights.append(weights)
        language_feature_indices.append(indices + int(level_idx * gaussians._language_feature_codebooks.shape[1]))
    
    language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
    language_feature_weights = torch.cat(language_feature_weights, dim=1)
    language_feature_indices = torch.cat(language_feature_indices, dim=1)
    combined_gaussians._language_feature_codebooks = language_feature_codebooks
    combined_gaussians._language_feature_weights = language_feature_weights
    combined_gaussians._language_feature_indices = torch.from_numpy(language_feature_indices.detach().cpu().numpy()).to(combined_gaussians._language_feature_weights.device)
    
    # 渲染语言特征图
    print(f"正在渲染视图 {view_idx} 的查询: '{query_text}'...")
    language_feature_image, rendered_rgb = render_language_feature_map_quick(
        combined_gaussians, view, pipeline, background, args
    )
    
    # 计算相关性热力图
    language_feature_image = language_feature_image.permute(0, 2, 3, 1)  # [3, H, W, 512]
    
    # 使用CLIP模型计算相关性
    relev_map = clip_model.get_max_across_quick(language_feature_image)  # [3, 1, H, W]
    
    # 选择最佳level（使用最大值）
    best_level = torch.argmax(relev_map.max(dim=2).values.max(dim=2).values).item()
    relev_heatmap = relev_map[best_level, 0].cpu().numpy()  # [H, W]
    
    # 归一化热力图
    relev_heatmap = (relev_heatmap - relev_heatmap.min()) / (relev_heatmap.max() - relev_heatmap.min() + 1e-10)
    
    # 转换为numpy
    rendered_rgb_np = rendered_rgb.permute(1, 2, 0).cpu().numpy()
    rendered_rgb_np = np.clip(rendered_rgb_np, 0, 1)
    
    # 保存RGB图像
    rgb_path = query_output_dir / "rgb.png"
    plt.imsave(rgb_path, rendered_rgb_np)
    print(f"已保存RGB图像到: {rgb_path}")
    
    # 保存热力图
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=False,
        colormap_min=0.0,
        colormap_max=1.0,
    )
    heatmap_tensor = torch.from_numpy(relev_heatmap).unsqueeze(-1).cpu()  # [H, W, 1]
    heatmap_path = query_output_dir / "heatmap.png"
    colormap_saving(heatmap_tensor, colormap_options, heatmap_path)
    print(f"已保存热力图到: {heatmap_path}")
    
    # 保存叠加图像（RGB + 热力图）
    overlay = rendered_rgb_np.copy()
    heatmap_colored = plt.cm.turbo(relev_heatmap)[..., :3]
    alpha = 0.5
    overlay = overlay * (1 - alpha * relev_heatmap[..., None]) + heatmap_colored * (alpha * relev_heatmap[..., None])
    overlay_path = query_output_dir / "overlay.png"
    plt.imsave(overlay_path, np.clip(overlay, 0, 1))
    print(f"已保存叠加图像到: {overlay_path}")
    
    # 生成二值化mask（阈值分割）
    threshold = 0.4
    mask = (relev_heatmap > threshold).astype(np.uint8)
    mask_path = query_output_dir / "mask.png"
    vis_mask_save(mask, mask_path)
    print(f"已保存分割mask到: {mask_path}")
    
    # 保存带mask的叠加图像
    mask_overlay = rendered_rgb_np.copy()
    mask_rgb = np.zeros_like(rendered_rgb_np)
    mask_rgb[mask == 1] = [1.0, 0.0, 0.0]  # 红色mask
    mask_overlay = mask_overlay * 0.7 + mask_rgb * 0.3
    mask_overlay_path = query_output_dir / "mask_overlay.png"
    plt.imsave(mask_overlay_path, np.clip(mask_overlay, 0, 1))
    print(f"已保存mask叠加图像到: {mask_overlay_path}")
    
    print(f"可视化完成！最佳level: {best_level}, 最大相关性: {relev_map[best_level, 0].max().item():.4f}")
    
    return {
        'rgb': rendered_rgb_np,
        'heatmap': relev_heatmap,
        'mask': mask,
        'best_level': best_level,
        'max_relevance': relev_map[best_level, 0].max().item()
    }


def visualize_multiple_queries(
    dataset: ModelParams, 
    pipeline: PipelineParams, 
    args,
    query_texts: list,
    view_indices: list = None
):
    """可视化多个文本查询的结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载场景以获取视图数量
    combined_gaussians = GaussianModel(dataset.sh_degree)
    dataset.model_path = args.ckpt_paths[0]
    scene = Scene(dataset, combined_gaussians, shuffle=False)
    views = scene.getTrainCameras()
    
    if view_indices is None:
        view_indices = list(range(min(10, len(views))))  # 默认可视化前10个视图
    
    print(f"将可视化 {len(query_texts)} 个查询，{len(view_indices)} 个视图")
    
    for query_text in query_texts:
        print(f"\n处理查询: '{query_text}'")
        for view_idx in tqdm(view_indices, desc=f"查询 '{query_text}'"):
            try:
                visualize_single_query(
                    dataset, pipeline, args, 
                    query_text, view_idx
                )
            except Exception as e:
                print(f"错误: 处理查询 '{query_text}' 视图 {view_idx} 时出错: {e}")
                continue


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    # 设置命令行参数
    parser = ArgumentParser(description="可视化脚本参数")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # 高斯模型参数
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # 可视化参数
    parser.add_argument("--dataset_name", type=str, default=None, help="数据集名称，如 'teatime'")
    parser.add_argument("--index", type=str, default=None, help="模型索引，如 '0'")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--checkpoint", type=int, default=10000, help="checkpoint迭代次数")
    parser.add_argument("--topk", type=int, default=1, help="topk值")
    
    # 查询参数
    parser.add_argument("--query", type=str, default=None, help="单个查询文本，如 'teapot'")
    parser.add_argument("--queries", type=str, nargs="+", default=None, help="多个查询文本，如 --queries teapot book cup")
    parser.add_argument("--view_idx", type=int, default=0, help="视图索引")
    parser.add_argument("--view_indices", type=int, nargs="+", default=None, help="多个视图索引")
    
    args = get_combined_args(parser)
    
    # 构建路径
    args.ckpt_paths = [os.path.join(args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}") for level in [1, 2, 3]]
    args.output_path = os.path.join(args.output_dir, args.dataset_name + f"_{args.index}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    safe_state(args.quiet)
    print(f"配置:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  模型索引: {args.index}")
    print(f"  Checkpoint路径: {args.ckpt_paths}")
    print(f"  输出路径: {args.output_path}")
    print(f"  Checkpoint: {args.checkpoint}")
    
    # 确定查询列表
    if getattr(args, 'queries', None):
        query_texts = args.queries
    elif args.query:
        query_texts = [args.query]
    else:
        # 默认查询（针对teatime场景）
        query_texts = ["teapot", "book", "cup", "spoon"]
        print(f"未指定查询，使用默认查询: {query_texts}")
    
    # 确定视图列表
    view_indices = args.view_indices if getattr(args, 'view_indices', None) is not None else [args.view_idx]
    
    print(f"  查询列表: {query_texts}")
    print(f"  视图索引: {view_indices}")
    
    # 提取参数
    model_params = model.extract(args)
    pipeline_params = pipeline.extract(args)
    
    with torch.no_grad():
        if len(query_texts) == 1 and len(view_indices) == 1:
            # 单个查询，单个视图
            visualize_single_query(
                model_params, 
                pipeline_params, 
                args,
                query_texts[0],
                view_indices[0]
            )
        else:
            # 多个查询或多个视图
            visualize_multiple_queries(
                model_params, 
                pipeline_params, 
                args,
                query_texts,
                view_indices
            )
    
    print("\n可视化完成！")

