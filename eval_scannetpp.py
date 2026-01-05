#
# Evaluation script for ScanNet++ dataset
# Extracts 3D bounding boxes for instances based on semantic queries
#

import numpy as np
import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

from gaussian_renderer import render, GaussianModel
from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from eval.openclip_encoder import OpenCLIPNetwork
from utils.general_utils import safe_state
from utils.vq_utils import get_weights_and_indices


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def smooth_cuda(mask_pred: torch.Tensor):
    """Smooth mask using average pooling"""
    scale = 7
    avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=3, count_include_pad=False).to(mask_pred.device)
    avg_filtered = avg_pool(mask_pred.float().unsqueeze(0).unsqueeze(0))
    mask = (avg_filtered > 0.5).type(torch.uint8).squeeze(0).squeeze(0)
    return mask


def get_semantic_mask(sem_map: torch.Tensor, clip_model, query_text: str, thresh: float = 0.4):
    """
    Extract 2D semantic mask from semantic feature map using CLIP query
    
    Args:
        sem_map: [n_levels, H, W, 512] semantic feature map
        clip_model: OpenCLIPNetwork model
        query_text: Text query for the semantic class
        thresh: Threshold for mask binarization
    
    Returns:
        mask: [H, W] binary mask
        best_level: Best level index
    """
    device = sem_map.device
    clip_model.set_positives([query_text])
    
    valid_map = clip_model.get_max_across_quick(sem_map)  # [n_levels, 1, H, W]
    n_levels, n_prompt, h, w = valid_map.shape
    
    # Smooth the heatmap
    scale = 29
    avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
    
    best_score = -1
    best_mask = None
    best_level = 0
    
    for i in range(n_levels):
        avg_filtered = avg_pool(valid_map[i, 0].unsqueeze(0).unsqueeze(0))
        valid_map[i, 0] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i, 0])
        
        # Normalize to [0, 1]
        output = valid_map[i, 0]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = torch.clip(output, 0, 1)
        
        # Binarize
        mask_pred = (output > thresh).type(torch.uint8)
        mask_pred = smooth_cuda(mask_pred)
        
        score = output.max().item()
        if score > best_score:
            best_score = score
            best_mask = mask_pred
            best_level = i
    
    return best_mask, best_level


def mask_to_3d_points(mask: torch.Tensor, gaussians: GaussianModel, camera, 
                      depth_threshold: float = 0.1) -> np.ndarray:
    """
    Convert 2D mask to 3D points by finding Gaussians that project into the mask
    
    Args:
        mask: [H, W] binary mask
        gaussians: GaussianModel with 3D points
        camera: Camera object with projection parameters
        depth_threshold: Threshold for depth filtering
    
    Returns:
        points_3d: [N, 3] array of 3D points
    """
    device = mask.device
    H, W = mask.shape
    
    # Get 3D Gaussian positions
    gaussian_points = gaussians.get_xyz  # [N, 3]
    
    # Project 3D points to 2D
    points_homogeneous = torch.cat([
        gaussian_points,
        torch.ones(gaussian_points.shape[0], 1, device=device)
    ], dim=1)  # [N, 4]
    
    # Transform to camera space
    world_to_cam = camera.world_view_transform  # [4, 4]
    cam_points = (world_to_cam @ points_homogeneous.T).T  # [N, 4]
    
    # Project to image space
    proj_matrix = camera.projection_matrix  # [4, 4]
    proj_points = (proj_matrix @ cam_points.T).T  # [N, 4]
    
    # Normalize by w
    proj_points = proj_points / (proj_points[:, 3:4] + 1e-8)
    
    # Convert to pixel coordinates
    x = (proj_points[:, 0] + 1) * 0.5 * W
    y = (1 - proj_points[:, 1]) * 0.5 * H  # Flip Y axis
    
    # Filter points that are in front of camera and within image bounds
    valid = (
        (cam_points[:, 2] > 0) &  # In front of camera
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )
    
    if valid.sum() == 0:
        return np.array([]).reshape(0, 3)
    
    x_valid = x[valid].long()
    y_valid = y[valid].long()
    
    # Clamp to image bounds
    x_valid = torch.clamp(x_valid, 0, W - 1)
    y_valid = torch.clamp(y_valid, 0, H - 1)
    
    # Check if projected points fall within mask
    mask_values = mask[y_valid, x_valid]
    in_mask = mask_values > 0
    
    if in_mask.sum() == 0:
        return np.array([]).reshape(0, 3)
    
    # Get 3D points that project into mask
    points_3d = gaussian_points[valid][in_mask].cpu().numpy()
    
    return points_3d


def compute_3d_bbox(points_3d: np.ndarray) -> Dict:
    """
    Compute 3D bounding box from 3D points
    
    Args:
        points_3d: [N, 3] array of 3D points
    
    Returns:
        bbox_dict: Dictionary with bbox parameters
            - center: [3] center of bbox
            - size: [3] size of bbox (width, height, depth)
            - min_corner: [3] minimum corner
            - max_corner: [3] maximum corner
    """
    if points_3d.shape[0] == 0:
        return {
            'center': np.array([0, 0, 0]),
            'size': np.array([0, 0, 0]),
            'min_corner': np.array([0, 0, 0]),
            'max_corner': np.array([0, 0, 0])
        }
    
    min_corner = points_3d.min(axis=0)
    max_corner = points_3d.max(axis=0)
    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    
    return {
        'center': center.tolist(),
        'size': size.tolist(),
        'min_corner': min_corner.tolist(),
        'max_corner': max_corner.tolist()
    }


def render_language_feature_map_quick(gaussians: GaussianModel, view, pipeline, background, args):
    """Render language feature map quickly"""
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(3, 64, H, W).view(3, 64, H*W)
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(3, 512, H, W)
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)
        language_feature_map = language_feature_map.permute(0, 2, 3, 1)  # [3, H, W, 512]

    return language_feature_map


def extract_3d_bboxes_for_scene(
    dataset: ModelParams,
    pipeline: PipelineParams,
    args,
    semantic_queries: List[str],
    output_dir: Path
):
    """
    Extract 3D bounding boxes for all semantic queries in a scene
    
    Args:
        dataset: Model parameters
        pipeline: Pipeline parameters
        args: Additional arguments
        semantic_queries: List of semantic class names to extract
        output_dir: Output directory for results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = OpenCLIPNetwork(device)
    
    # Load scene - first load one checkpoint to get the scene structure
    dataset.model_path = args.ckpt_paths[0]
    temp_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, temp_gaussians, shuffle=False, load_iteration=args.checkpoint)
    views = scene.getTrainCameras()
    
    # Load checkpoints and combine
    language_feature_codebooks = []
    language_feature_weights = []
    language_feature_indices = []
    combined_gaussians = None
    
    for level_idx in range(3):
        gaussians = GaussianModel(dataset.sh_degree)
        checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        # Use first level's gaussians as base
        if combined_gaussians is None:
            combined_gaussians = gaussians
        
        language_feature_codebooks.append(gaussians._language_feature_codebooks.view(-1, 512))
        weights, indices = get_weights_and_indices(gaussians._language_feature_logits, 4)
        language_feature_weights.append(weights)
        language_feature_indices.append(indices + int(level_idx * gaussians._language_feature_codebooks.shape[1]))
    
    language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
    language_feature_weights = torch.cat(language_feature_weights, dim=1)
    language_feature_indices = torch.cat(language_feature_indices, dim=1)
    combined_gaussians._language_feature_codebooks = language_feature_codebooks
    combined_gaussians._language_feature_weights = language_feature_weights
    combined_gaussians._language_feature_indices = torch.from_numpy(
        language_feature_indices.detach().cpu().numpy()
    ).to(combined_gaussians._language_feature_weights.device)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Store results for each semantic query
    all_results = defaultdict(list)
    
    logger.info(f"Processing {len(views)} views for {len(semantic_queries)} semantic queries...")
    
    for view_idx, view in enumerate(tqdm(views, desc="Processing views")):
        # Render language feature map
        language_feature_map = render_language_feature_map_quick(
            combined_gaussians, view, pipeline, background, args
        )
        
        # Process each semantic query
        for query_text in semantic_queries:
            # Get 2D mask
            mask_2d, best_level = get_semantic_mask(
                language_feature_map, clip_model, query_text, args.mask_thresh
            )
            
            if mask_2d.sum() == 0:
                continue
            
            # Convert mask to 3D points
            points_3d = mask_to_3d_points(mask_2d, combined_gaussians, view)
            
            if points_3d.shape[0] < 10:  # Too few points, skip
                continue
            
            # Compute 3D bbox
            bbox_3d = compute_3d_bbox(points_3d)
            
            # Store result
            all_results[query_text].append({
                'view_idx': view_idx,
                'view_name': view.image_name,
                'bbox_3d': bbox_3d,
                'num_points': points_3d.shape[0],
                'mask_area': mask_2d.sum().item()
            })
    
    # Aggregate bboxes across views (merge overlapping bboxes)
    final_bboxes = {}
    for query_text, bbox_list in all_results.items():
        if len(bbox_list) == 0:
            continue
        
        # Simple aggregation: take the union of all bboxes
        all_centers = np.array([bbox['bbox_3d']['center'] for bbox in bbox_list])
        all_sizes = np.array([bbox['bbox_3d']['size'] for bbox in bbox_list])
        all_min_corners = np.array([bbox['bbox_3d']['min_corner'] for bbox in bbox_list])
        all_max_corners = np.array([bbox['bbox_3d']['max_corner'] for bbox in bbox_list])
        
        # Compute overall bbox
        overall_min = all_min_corners.min(axis=0)
        overall_max = all_max_corners.max(axis=0)
        overall_center = (overall_min + overall_max) / 2
        overall_size = overall_max - overall_min
        
        final_bboxes[query_text] = {
            'center': overall_center.tolist(),
            'size': overall_size.tolist(),
            'min_corner': overall_min.tolist(),
            'max_corner': overall_max.tolist(),
            'num_views': len(bbox_list),
            'all_bboxes': bbox_list
        }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / '3d_bboxes.json'
    with open(output_file, 'w') as f:
        json.dump(final_bboxes, f, indent=2)
    
    logger.info(f"Saved 3D bboxes to {output_file}")
    logger.info(f"Found {len(final_bboxes)} instances:")
    for query_text, bbox_info in final_bboxes.items():
        logger.info(f"  {query_text}: center={bbox_info['center']}, size={bbox_info['size']}")
    
    return final_bboxes


if __name__ == "__main__":
    import time
    
    parser = ArgumentParser(description="ScanNet++ 3D BBox Extraction")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--index", type=str, default="0")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--semantic_queries", type=str, nargs='+', required=True,
                       help="List of semantic class names to extract (e.g., 'chair' 'table' 'sofa')")
    
    args = get_combined_args(parser)
    args.ckpt_paths = [
        os.path.join(args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}")
        for level in [1, 2, 3]
    ]
    args.output_path = os.path.join(args.output_dir, args.dataset_name + f"_{args.index}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.output_path, f'{timestamp}.log')
    logger = get_logger(f'{args.dataset_name}', log_file=log_file, log_level=logging.INFO)
    
    safe_state(args.quiet)
    logger.info(f"Arguments: {args}")
    
    with torch.no_grad():
        extract_3d_bboxes_for_scene(
            model.extract(args),
            pipeline.extract(args),
            args,
            args.semantic_queries,
            Path(args.output_path)
        )
    
    logger.info("Extraction complete!")

