#
# Direct 3D BBox Extraction from Gaussian Splatting Model
# Extracts 3D bounding boxes directly in 3D space without 2D projection
#

import numpy as np
import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from gaussian_renderer import GaussianModel
from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from eval.openclip_encoder import OpenCLIPNetwork
from utils.general_utils import safe_state
from utils.vq_utils import get_weights_and_indices

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, DBSCAN clustering will not work")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available, HDBSCAN clustering will not work")


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


def extract_gaussian_features(
    gaussians: GaussianModel,
    topk: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features from Gaussian model:
    - Gaussian point positions (xyz)
    - Top-K indices (sparse coefficient indices)
    - Top-K weights (sparse coefficient weights)
    - Global codebook
    
    Args:
        gaussians: GaussianModel instance
        topk: Number of top-K indices to extract
    
    Returns:
        xyz: [N, 3] Gaussian point positions
        topk_indices: [N, topk] Top-K indices for each point
        topk_weights: [N, topk] Top-K weights for each point
        codebooks: [L, K, 512] Global codebooks (L layers, K codebook size)
    """
    # Extract Gaussian positions
    xyz = gaussians.get_xyz  # [N, 3]
    
    # Extract language feature logits
    logits = gaussians._language_feature_logits  # [N, L*K]
    
    # Get codebooks
    codebooks = gaussians._language_feature_codebooks  # [L, K, 512]
    
    # Get Top-K weights and indices for each layer
    layer_num, codebook_size, feature_dim = codebooks.shape
    
    all_topk_weights = []
    all_topk_indices = []
    
    for layer_idx in range(layer_num):
        # Get logits for this layer
        layer_logits = logits[:, layer_idx * codebook_size:(layer_idx + 1) * codebook_size]  # [N, K]
        
        # Get Top-K weights and indices
        weights, indices = get_weights_and_indices(layer_logits, topk)  # [N, topk]
        
        # Adjust indices to global codebook index
        global_indices = indices.long() + layer_idx * codebook_size
        
        all_topk_weights.append(weights)
        all_topk_indices.append(global_indices)
    
    # Stack all layers
    # For multi-layer, we can either:
    # 1. Concatenate all layers (total topk * layer_num)
    # 2. Use only the last layer
    # Here we use concatenation to preserve all information
    topk_weights = torch.cat(all_topk_weights, dim=1)  # [N, topk * L]
    topk_indices = torch.cat(all_topk_indices, dim=1)  # [N, topk * L]
    
    # Flatten codebooks for easier indexing
    flat_codebooks = codebooks.view(-1, feature_dim)  # [L*K, 512]
    
    return xyz, topk_indices, topk_weights, flat_codebooks


def compute_point_features(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    codebooks: torch.Tensor,
    layer_num: int = 3
) -> torch.Tensor:
    """
    Compute semantic features for each Gaussian point using sparse coefficients
    Uses residual connection across layers (levels)
    
    Args:
        topk_indices: [N, topk * L] Top-K indices (L is number of levels)
        topk_weights: [N, topk * L] Top-K weights
        codebooks: [total_codebook_size, 512] Combined codebooks from all levels
        layer_num: Number of levels/layers
    
    Returns:
        features: [N, 512] Semantic features for each point
    """
    device = topk_indices.device
    N = topk_indices.shape[0]
    feature_dim = codebooks.shape[1]
    
    # Initialize features
    features = torch.zeros(N, feature_dim, device=device)
    
    # For each layer/level, accumulate features with residual connection
    topk_per_layer = topk_indices.shape[1] // layer_num
    
    for layer_idx in range(layer_num):
        # Get indices and weights for this layer
        start_idx = layer_idx * topk_per_layer
        end_idx = (layer_idx + 1) * topk_per_layer
        layer_indices = topk_indices[:, start_idx:end_idx]  # [N, topk]
        layer_weights = topk_weights[:, start_idx:end_idx]  # [N, topk]
        
        # Get codebook vectors
        layer_indices_long = layer_indices.long()  # [N, topk]
        # Clamp indices to valid range
        layer_indices_long = torch.clamp(layer_indices_long, 0, codebooks.shape[0] - 1)
        codebook_vectors = codebooks[layer_indices_long]  # [N, topk, 512]
        
        # Weighted sum
        layer_features = (codebook_vectors * layer_weights.unsqueeze(-1)).sum(dim=1)  # [N, 512]
        
        # Accumulate with residual connection
        if layer_idx == 0:
            features = layer_features
        else:
            features = features + layer_features
    
    # Normalize
    features = features / (features.norm(dim=1, keepdim=True) + 1e-10)
    
    return features


def compute_semantic_similarity(
    point_features: torch.Tensor,
    text_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between point features and text embedding
    
    Args:
        point_features: [N, 512] Point semantic features
        text_embedding: [512] Text embedding
    
    Returns:
        similarities: [N] Cosine similarities
    """
    # Normalize text embedding
    text_embedding = text_embedding / (text_embedding.norm() + 1e-10)
    
    # Compute cosine similarity
    similarities = torch.mm(point_features, text_embedding.unsqueeze(1)).squeeze(1)  # [N]
    
    return similarities


def filter_by_threshold(
    xyz: np.ndarray,
    similarities: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Filter points by similarity threshold
    
    Args:
        xyz: [N, 3] Point positions
        similarities: [N] Similarities
        threshold: Similarity threshold
    
    Returns:
        filtered_xyz: [M, 3] Filtered point positions (M <= N)
    """
    mask = similarities >= threshold
    filtered_xyz = xyz[mask]
    
    return filtered_xyz


def cluster_points(
    points: np.ndarray,
    method: str = 'dbscan',
    eps: float = 0.1,
    min_samples: int = 10,
    min_cluster_size: int = 50
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Cluster 3D points using DBSCAN or HDBSCAN
    
    Args:
        points: [N, 3] Point positions
        method: 'dbscan' or 'hdbscan'
        eps: DBSCAN eps parameter (for DBSCAN)
        min_samples: Minimum samples for DBSCAN
        min_cluster_size: Minimum cluster size for HDBSCAN
    
    Returns:
        labels: [N] Cluster labels (-1 for noise)
        clusters: List of point arrays for each cluster
    """
    if len(points) == 0:
        return np.array([]), []
    
    if method == 'dbscan':
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for DBSCAN clustering")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(points)
    elif method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required for HDBSCAN clustering")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(points)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Extract clusters
    unique_labels = np.unique(labels)
    clusters = []
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    
    return labels, clusters


def compute_aabb(points: np.ndarray) -> Dict:
    """
    Compute Axis-Aligned Bounding Box (AABB)
    
    Args:
        points: [N, 3] Point positions
    
    Returns:
        bbox_dict: Dictionary with AABB parameters
    """
    if len(points) == 0:
        return {
            'center': [0, 0, 0],
            'size': [0, 0, 0],
            'min_corner': [0, 0, 0],
            'max_corner': [0, 0, 0]
        }
    
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    
    return {
        'center': center.tolist(),
        'size': size.tolist(),
        'min_corner': min_corner.tolist(),
        'max_corner': max_corner.tolist()
    }


def compute_obb(points: np.ndarray) -> Dict:
    """
    Compute Oriented Bounding Box (OBB) using PCA
    
    Args:
        points: [N, 3] Point positions
    
    Returns:
        bbox_dict: Dictionary with OBB parameters
    """
    if len(points) < 3:
        # Fallback to AABB if too few points
        return compute_aabb(points)
    
    # Center points
    center = points.mean(axis=0)
    centered_points = points - center
    
    # Compute covariance matrix
    cov = np.cov(centered_points.T)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project points to principal axes
    projected = centered_points @ eigenvectors
    
    # Compute AABB in principal space
    min_proj = projected.min(axis=0)
    max_proj = projected.max(axis=0)
    size = max_proj - min_proj
    
    # Transform back to world space
    # The eigenvectors form the rotation matrix
    rotation = eigenvectors.T  # [3, 3]
    
    # Compute 8 corners of OBB in principal space
    corners_proj = np.array([
        [min_proj[0], min_proj[1], min_proj[2]],
        [max_proj[0], min_proj[1], min_proj[2]],
        [max_proj[0], max_proj[1], min_proj[2]],
        [min_proj[0], max_proj[1], min_proj[2]],
        [min_proj[0], min_proj[1], max_proj[2]],
        [max_proj[0], min_proj[1], max_proj[2]],
        [max_proj[0], max_proj[1], max_proj[2]],
        [min_proj[0], max_proj[1], max_proj[2]],
    ])
    
    # Transform corners to world space
    corners_world = (corners_proj @ rotation) + center
    
    return {
        'center': center.tolist(),
        'size': size.tolist(),
        'rotation': rotation.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'corners': corners_world.tolist(),
        'min_corner': corners_world.min(axis=0).tolist(),
        'max_corner': corners_world.max(axis=0).tolist()
    }


def extract_3d_bboxes_direct(
    dataset: ModelParams,
    args,
    semantic_queries: List[str],
    output_dir: Path,
    similarity_threshold: float = 0.3,
    clustering_method: str = 'dbscan',
    clustering_params: Dict = None
):
    """
    Extract 3D bounding boxes directly in 3D space
    
    Args:
        dataset: Model parameters
        args: Additional arguments
        semantic_queries: List of semantic class names
        output_dir: Output directory
        similarity_threshold: Similarity threshold for filtering
        clustering_method: 'dbscan' or 'hdbscan'
        clustering_params: Parameters for clustering
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = OpenCLIPNetwork(device)
    
    # Load scene to get structure
    dataset.model_path = args.ckpt_paths[0]
    temp_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, temp_gaussians, shuffle=False, load_iteration=args.checkpoint)
    
    # Load and combine checkpoints from all levels
    logger.info("Loading checkpoints from all levels...")
    all_codebooks = []
    all_xyz = None
    all_topk_indices_list = []
    all_topk_weights_list = []
    
    for level_idx in range(3):
        gaussians = GaussianModel(dataset.sh_degree)
        checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        # Extract features
        xyz, topk_indices, topk_weights, codebooks = extract_gaussian_features(gaussians, topk=args.topk)
        
        # Store xyz (should be same across levels)
        if all_xyz is None:
            all_xyz = xyz
        
        # Store codebooks for each level
        all_codebooks.append(codebooks)
        
        # Store indices and weights for each level
        all_topk_indices_list.append(topk_indices)
        all_topk_weights_list.append(topk_weights)
    
    # Combine codebooks from all levels
    # Stack codebooks: [L1*K, 512], [L2*K, 512], [L3*K, 512] -> [L1*K+L2*K+L3*K, 512]
    combined_codebooks = torch.cat(all_codebooks, dim=0)  # [total_codebook_size, 512]
    
    # Adjust indices to global codebook indices
    codebook_size_per_level = all_codebooks[0].shape[0]
    adjusted_indices_list = []
    adjusted_weights_list = []
    
    for level_idx in range(3):
        # Adjust indices to point to combined codebook
        adjusted_indices = all_topk_indices_list[level_idx] + level_idx * codebook_size_per_level
        adjusted_indices_list.append(adjusted_indices)
        adjusted_weights_list.append(all_topk_weights_list[level_idx])
    
    # Concatenate indices and weights from all levels
    all_topk_indices = torch.cat(adjusted_indices_list, dim=1)  # [N, topk * 3]
    all_topk_weights = torch.cat(adjusted_weights_list, dim=1)  # [N, topk * 3]
    
    language_feature_codebooks = combined_codebooks
    
    # Convert to numpy for clustering
    xyz_np = all_xyz.detach().cpu().numpy()
    
    # Get layer number (3 levels, each with potentially multiple layers)
    # For simplicity, we treat each level as a separate "layer" in the residual connection
    layer_num = 3  # 3 levels
    
    logger.info(f"Extracted {len(xyz_np)} Gaussian points")
    logger.info(f"Codebook shape: {language_feature_codebooks.shape}")
    logger.info(f"Top-K indices shape: {all_topk_indices.shape}")
    logger.info(f"Top-K weights shape: {all_topk_weights.shape}")
    
    # Compute point features
    logger.info("Computing semantic features for each point...")
    point_features = compute_point_features(
        all_topk_indices,
        all_topk_weights,
        language_feature_codebooks,
        layer_num=layer_num
    )  # [N, 512]
    
    # Process each semantic query
    all_results = {}
    
    for query_text in tqdm(semantic_queries, desc="Processing semantic queries"):
        logger.info(f"Processing query: {query_text}")
        
        # Encode text
        text_embedding = clip_model.encode_text([query_text], device)[0]  # [512]
        text_embedding = text_embedding / (text_embedding.norm() + 1e-10)
        
        # Compute similarities
        similarities = compute_semantic_similarity(point_features, text_embedding)  # [N]
        similarities_np = similarities.detach().cpu().numpy()
        
        # Filter by threshold
        filtered_points = filter_by_threshold(xyz_np, similarities_np, similarity_threshold)
        
        logger.info(f"  Filtered to {len(filtered_points)} points (threshold={similarity_threshold})")
        
        if len(filtered_points) < 10:
            logger.warning(f"  Too few points for {query_text}, skipping")
            continue
        
        # Cluster points
        logger.info(f"  Clustering points using {clustering_method}...")
        default_clustering_params = {
            'dbscan': {'eps': 0.1, 'min_samples': 10},
            'hdbscan': {'min_cluster_size': 50}
        }
        
        if clustering_params is None:
            params = default_clustering_params.get(clustering_method, {})
        else:
            params = clustering_params
        
        try:
            if clustering_method == 'dbscan':
                labels, clusters = cluster_points(
                    filtered_points,
                    method='dbscan',
                    eps=params.get('eps', 0.1),
                    min_samples=params.get('min_samples', 10)
                )
            else:  # hdbscan
                labels, clusters = cluster_points(
                    filtered_points,
                    method='hdbscan',
                    min_cluster_size=params.get('min_cluster_size', 50)
                )
            
            logger.info(f"  Found {len(clusters)} clusters")
            
            # Compute bboxes for each cluster
            instance_bboxes = []
            for cluster_idx, cluster_points in enumerate(clusters):
                if len(cluster_points) < 3:
                    continue
                
                # Compute AABB
                aabb = compute_aabb(cluster_points)
                
                # Compute OBB
                obb = compute_obb(cluster_points)
                
                instance_bboxes.append({
                    'cluster_id': cluster_idx,
                    'num_points': len(cluster_points),
                    'aabb': aabb,
                    'obb': obb
                })
            
            if len(instance_bboxes) == 0:
                logger.warning(f"  No valid clusters found for {query_text}")
                continue
            
            all_results[query_text] = {
                'num_instances': len(instance_bboxes),
                'total_points': len(filtered_points),
                'instances': instance_bboxes
            }
            
            logger.info(f"  Extracted {len(instance_bboxes)} instances for {query_text}")
            
        except Exception as e:
            logger.error(f"  Error clustering points for {query_text}: {e}")
            continue
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / '3d_bboxes_direct.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")
    logger.info(f"Summary:")
    for query_text, result in all_results.items():
        logger.info(f"  {query_text}: {result['num_instances']} instances, {result['total_points']} points")
    
    return all_results


if __name__ == "__main__":
    import time
    
    parser = ArgumentParser(description="Direct 3D BBox Extraction from Gaussian Model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--index", type=str, default="0")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--semantic_queries", type=str, nargs='+', required=True,
                       help="List of semantic class names")
    parser.add_argument("--similarity_threshold", type=float, default=0.3,
                       help="Similarity threshold for filtering points")
    parser.add_argument("--clustering_method", type=str, default='dbscan',
                       choices=['dbscan', 'hdbscan'],
                       help="Clustering method")
    parser.add_argument("--dbscan_eps", type=float, default=0.1,
                       help="DBSCAN eps parameter")
    parser.add_argument("--dbscan_min_samples", type=int, default=10,
                       help="DBSCAN min_samples parameter")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=50,
                       help="HDBSCAN min_cluster_size parameter")
    
    args = get_combined_args(parser)
    args.ckpt_paths = [
        os.path.join(args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}")
        for level in [1, 2, 3]
    ]
    args.output_path = os.path.join(args.output_dir, args.dataset_name + f"_{args.index}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.output_path, f'direct_3d_bbox_{timestamp}.log')
    logger = get_logger(f'{args.dataset_name}', log_file=log_file, log_level=logging.INFO)
    
    safe_state(args.quiet)
    logger.info(f"Arguments: {args}")
    
    # Prepare clustering parameters
    clustering_params = {}
    if args.clustering_method == 'dbscan':
        clustering_params = {
            'eps': args.dbscan_eps,
            'min_samples': args.dbscan_min_samples
        }
    else:
        clustering_params = {
            'min_cluster_size': args.hdbscan_min_cluster_size
        }
    
    with torch.no_grad():
        extract_3d_bboxes_direct(
            model.extract(args),
            args,
            args.semantic_queries,
            Path(args.output_path),
            similarity_threshold=args.similarity_threshold,
            clustering_method=args.clustering_method,
            clustering_params=clustering_params
        )
    
    logger.info("Extraction complete!")

