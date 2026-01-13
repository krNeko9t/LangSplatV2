#!/usr/bin/env python
"""
GS级别的语言特征可视化/检索脚本（不依赖渲染weight-map）。

功能：
- 先计算 per-GS 的 512-d language feature（全局共享codebook + 每GS top-k 权重/索引）
- 导出 PLY 点云（使用GS中心点）：
  1) Query relevance：与文本query的相似度（cosine/dot）作为标量，用colormap上色
  2) PCA embedding：对512-d做PCA到3维，映射到RGB显示embedding结构

用法示例：
python visualize_gs_wise.py \
  -s ../../data/lerf_ovs/teatime \
  --dataset_name teatime --index 0 \
  --ckpt_root_path output --output_dir gs_vis \
  --checkpoint 10000 --topk 4 \
  --query "teapot" --export_query_ply --max_points 30000

python visualize_gs_wise.py \
  -s ../../data/lerf_ovs/teatime \
  --dataset_name teatime --index 0 \
  --ckpt_root_path output --output_dir gs_vis \
  --checkpoint 10000 --topk 4 \
  --export_pca_ply --max_points 50000 --pca_fit_points 200000
"""

import os
import random
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/huggingface_cache"

from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from eval.openclip_encoder import OpenCLIPNetwork
from utils.general_utils import safe_state
from utils.vq_utils import get_weights_and_indices


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def _write_xyz_rgb_ply(xyz: np.ndarray, rgb_u8: np.ndarray, out_path: Path):
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb_u8.ndim == 2 and rgb_u8.shape[1] == 3
    assert xyz.shape[0] == rgb_u8.shape[0]

    verts = np.empty(
        xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"] = xyz[:, 0].astype(np.float32)
    verts["y"] = xyz[:, 1].astype(np.float32)
    verts["z"] = xyz[:, 2].astype(np.float32)
    verts["red"] = rgb_u8[:, 0].astype(np.uint8)
    verts["green"] = rgb_u8[:, 1].astype(np.uint8)
    verts["blue"] = rgb_u8[:, 2].astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=True).write(str(out_path))


def build_combined_gaussians_from_3levels(args) -> GaussianModel:
    """
    构建一个“组合gaussians”：
    - xyz/opacity/scale/rot等来自 level-1 checkpoint（ckpt_paths[0]）
    - language codebooks 来自3个level堆叠成 [3, 64, 512]
    - language topk weights/indices 也按3个level拼接（indices为扁平索引 0..3*64-1）
    """
    combined = GaussianModel(args.sh_degree)

    # 基础几何（取level 1）
    ckpt0 = os.path.join(args.ckpt_paths[0], f"chkpnt{args.checkpoint}.pth")
    (model_params, _) = torch.load(ckpt0)
    combined.restore(model_params, args, mode="test")

    language_feature_weights = []
    language_feature_indices = []
    language_feature_codebooks = []

    for level_idx in range(3):
        g = GaussianModel(args.sh_degree)
        ckpt = os.path.join(args.ckpt_paths[level_idx], f"chkpnt{args.checkpoint}.pth")
        (mp, _) = torch.load(ckpt)
        g.restore(mp, args, mode="test")

        # codebook: [64, 512]（每个level一个）
        language_feature_codebooks.append(g._language_feature_codebooks.view(-1, 512))

        # topk per-GS: weights/indices（indices是层内0..63）
        w, idx = get_weights_and_indices(g._language_feature_logits, args.topk)
        language_feature_weights.append(w)  # [P, topk]
        language_feature_indices.append(idx + int(level_idx * g._language_feature_codebooks.shape[1]))  # 扁平索引

    # 堆叠为 [3, 64, 512]
    combined._language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
    # 拼接为 [P, 3*topk]
    combined._language_feature_weights = torch.cat(language_feature_weights, dim=1)
    combined._language_feature_indices = torch.from_numpy(
        torch.cat(language_feature_indices, dim=1).detach().cpu().numpy()
    ).to(combined._language_feature_weights.device)

    return combined


@torch.no_grad()
def compute_per_gs_feature(combined: GaussianModel) -> torch.Tensor:
    """
    返回 [P, 512]，L2归一化。
    """
    return combined.compute_per_gaussian_language_features_from_topk(
        combined._language_feature_weights,
        combined._language_feature_indices,
        normalize=True,
    )


@torch.no_grad()
def export_query_relevance_ply(
    combined: GaussianModel,
    per_gs_feat: torch.Tensor,
    query: str,
    out_path: Path,
    max_points: int,
    device: torch.device,
):
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives([query])
    text_feat = clip_model.pos_embeds[0].to(per_gs_feat.dtype).to(per_gs_feat.device)  # [512], normalized
    scores = per_gs_feat @ text_feat  # [P], in [-1, 1]

    xyz = combined.get_xyz.detach()
    if max_points and scores.numel() > max_points:
        sel = torch.topk(scores, k=max_points, largest=True).indices
        scores = scores[sel]
        xyz = xyz[sel]

    s_min = float(scores.min().item())
    s_max = float(scores.max().item())
    scores01 = (scores - s_min) / (s_max - s_min + 1e-10)
    colors = (plt.cm.turbo(scores01.detach().cpu().numpy())[..., :3] * 255.0).astype(np.uint8)

    _write_xyz_rgb_ply(
        xyz.detach().cpu().numpy().astype(np.float32),
        colors,
        out_path,
    )
    return {"score_min": s_min, "score_max": s_max, "num_points": int(xyz.shape[0])}


def export_pca_embedding_ply(
    combined: GaussianModel,
    per_gs_feat: torch.Tensor,
    out_path: Path,
    max_points: int,
    pca_fit_points: int,
    seed: int,
):
    """
    PCA(512->3) 后做RGB可视化。注意：PCA只在CPU上做（sklearn）。
    """
    feat = per_gs_feat.detach().cpu().numpy().astype(np.float32)  # [P, 512]
    xyz = combined.get_xyz.detach().cpu().numpy().astype(np.float32)  # [P, 3]

    P = feat.shape[0]
    rng = np.random.RandomState(seed)
    fit_n = min(P, int(pca_fit_points) if pca_fit_points and pca_fit_points > 0 else P)
    if fit_n < P:
        fit_idx = rng.choice(P, size=fit_n, replace=False)
        feat_fit = feat[fit_idx]
    else:
        feat_fit = feat

    pca = PCA(n_components=3, random_state=seed)
    pca.fit(feat_fit)
    emb3 = pca.transform(feat)  # [P, 3]

    # 映射到[0,1]再转RGB
    emb_min = emb3.min(axis=0, keepdims=True)
    emb_max = emb3.max(axis=0, keepdims=True)
    emb01 = (emb3 - emb_min) / (emb_max - emb_min + 1e-10)
    rgb = np.clip(emb01 * 255.0, 0, 255).astype(np.uint8)

    if max_points and P > max_points:
        sel = rng.choice(P, size=int(max_points), replace=False)
        xyz = xyz[sel]
        rgb = rgb[sel]

    _write_xyz_rgb_ply(xyz, rgb, out_path)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "num_points": int(xyz.shape[0]),
    }


def main():
    seed_everything(42)

    parser = ArgumentParser(description="GS-wise language feature visualization")
    model = ModelParams(parser, sentinel=True)

    parser.add_argument("--ckpt_root_path", default="output", type=str)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="gs_vis")
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=4)

    # 导出选项
    parser.add_argument("--export_query_ply", action="store_true", help="导出按query相似度上色的PLY（turbo）")
    parser.add_argument("--query", type=str, default=None, help="文本查询（用于query ply）")
    parser.add_argument("--export_pca_ply", action="store_true", help="导出PCA(512->3) embedding 的RGB PLY")

    parser.add_argument("--max_points", type=int, default=50000, help="导出PLY最多点数（query为topN，pca为随机采样）")
    parser.add_argument("--pca_fit_points", type=int, default=200000, help="PCA拟合时的采样点数（过大可能慢）")
    parser.add_argument("--seed", type=int, default=42)

    args = get_combined_args(parser)
    safe_state(False)

    if not args.export_query_ply and not args.export_pca_ply:
        raise ValueError("请至少指定一个导出：--export_query_ply 或 --export_pca_ply")
    if args.export_query_ply and not args.query:
        raise ValueError("使用 --export_query_ply 时必须提供 --query")

    # ckpt paths（3个level）
    args.ckpt_paths = [
        os.path.join(args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}") for level in [1, 2, 3]
    ]

    out_root = Path(args.output_dir) / (args.dataset_name + f"_{args.index}") / f"chkpnt{args.checkpoint}"
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("配置：")
    print(f"  ckpt_paths: {args.ckpt_paths}")
    print(f"  output: {out_root}")
    print(f"  topk: {args.topk}")
    print(f"  device: {device}")

    # 组合gaussians并计算per-GS feature
    combined = build_combined_gaussians_from_3levels(args)
    per_gs_feat = compute_per_gs_feature(combined)

    if args.export_query_ply:
        out_path = out_root / f"gs_query_{args.query}_top{args.max_points}.ply"
        info = export_query_relevance_ply(
            combined=combined,
            per_gs_feat=per_gs_feat,
            query=args.query,
            out_path=out_path,
            max_points=args.max_points,
            device=device,
        )
        print(f"[query ply] {out_path}  points={info['num_points']} score∈[{info['score_min']:.4f},{info['score_max']:.4f}]")

    if args.export_pca_ply:
        out_path = out_root / f"gs_pca_rgb_max{args.max_points}_fit{args.pca_fit_points}.ply"
        info = export_pca_embedding_ply(
            combined=combined,
            per_gs_feat=per_gs_feat,
            out_path=out_path,
            max_points=args.max_points,
            pca_fit_points=args.pca_fit_points,
            seed=args.seed,
        )
        evr = info["explained_variance_ratio"]
        print(f"[pca ply] {out_path}  points={info['num_points']}  EVR={evr}")


if __name__ == "__main__":
    main()

