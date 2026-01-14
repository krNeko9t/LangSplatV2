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
from typing import Optional

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
from utils.sh_utils import RGB2SH


def _slice_gaussians_cpu(gaussians: GaussianModel, sel: Optional[np.ndarray]):
    """
    在CPU上对Gaussian字段做子集切片，返回一个“轻量对象”，仅用于写PLY。
    这样避免各种cuda/nn.Parameter副作用，也避免依赖 Scene / dataset。
    """
    class _G:
        pass

    g = _G()
    g.max_sh_degree = gaussians.max_sh_degree
    g.construct_list_of_attributes = gaussians.construct_list_of_attributes

    def _maybe_slice(t: torch.Tensor):
        t_cpu = t.detach().cpu()
        if sel is None:
            return t_cpu
        return t_cpu[sel]

    g._xyz = _maybe_slice(gaussians._xyz)
    g._features_dc = _maybe_slice(gaussians._features_dc)
    g._features_rest = _maybe_slice(gaussians._features_rest)
    g._opacity = _maybe_slice(gaussians._opacity)
    g._scaling = _maybe_slice(gaussians._scaling)
    g._rotation = _maybe_slice(gaussians._rotation)
    return g


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


def _write_xyz_rgb_ply(xyz: np.ndarray, rgb_u8: np.ndarray, out_path: Path, *, text: bool = False):
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
    PlyData([PlyElement.describe(verts, "vertex")], text=text).write(str(out_path))


def _write_gaussian_rgb_ply(
    gaussians: GaussianModel,
    rgb_float01: np.ndarray,
    out_path: Path,
    *,
    text: bool = False,
):
    """
    写“3DGS / Supersplat”兼容的 PLY：包含 x/y/z, nx/ny/nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*。
    我们把可视化颜色写进 f_dc_0..2（SH DC），并把 f_rest 清零，保证颜色不随视角变化。
    """
    if rgb_float01.ndim != 2 or rgb_float01.shape[1] != 3:
        raise ValueError(f"rgb_float01 must be [P,3], got {rgb_float01.shape}")

    xyz = gaussians._xyz.detach().cpu().numpy().astype(np.float32)
    P = xyz.shape[0]
    if rgb_float01.shape[0] != P:
        raise ValueError(f"rgb_float01 P mismatch: {rgb_float01.shape[0]} vs {P}")

    normals = np.zeros_like(xyz, dtype=np.float32)

    # f_dc: store SH DC coefficients corresponding to desired RGB
    rgb_t = torch.from_numpy(rgb_float01.astype(np.float32))
    f_dc = RGB2SH(rgb_t).numpy().astype(np.float32)  # [P,3]

    # f_rest: zeros
    # gaussian_model.save_ply flattens _features_rest to [P, N]
    rest_dim = int(gaussians._features_rest.shape[1] * gaussians._features_rest.shape[2])
    f_rest = np.zeros((P, rest_dim), dtype=np.float32)

    opacities = gaussians._opacity.detach().cpu().numpy().astype(np.float32)  # [P,1]
    scale = gaussians._scaling.detach().cpu().numpy().astype(np.float32)      # [P,3] (log-scales in this repo)
    rotation = gaussians._rotation.detach().cpu().numpy().astype(np.float32)  # [P,4]

    # dtype schema must match attribute order used by gaussian_model.save_ply
    dtype_full = [(attribute, "f4") for attribute in gaussians.construct_list_of_attributes()]
    elements = np.empty(P, dtype=dtype_full)

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")], text=text).write(str(out_path))


def _write_gaussian_ply_with_extra(
    gaussians: GaussianModel,
    extra_f4: dict,
    out_path: Path,
    *,
    text: bool = False,
):
    """
    写“正常3DGS / Supersplat”兼容的 PLY（保留checkpoint原始SH颜色/外观），
    并在 vertex 属性末尾追加自定义 extra 字段（通常会被渲染器忽略，但便于后处理）。

    extra_f4:
      - key: 属性名（建议以 extra_ 前缀避免冲突）
      - value: np.ndarray，shape 为 [P] 或 [P,1] 或 [P,C]，dtype 可转换为 float32
    """
    xyz = gaussians._xyz.detach().cpu().numpy().astype(np.float32)
    P = int(xyz.shape[0])
    normals = np.zeros_like(xyz, dtype=np.float32)

    f_dc = (
        gaussians._features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    f_rest = (
        gaussians._features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    opacities = gaussians._opacity.detach().cpu().numpy().astype(np.float32)
    scale = gaussians._scaling.detach().cpu().numpy().astype(np.float32)
    rotation = gaussians._rotation.detach().cpu().numpy().astype(np.float32)

    attributes_base = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    dtype_base = [(attribute, "f4") for attribute in gaussians.construct_list_of_attributes()]

    extra_names = sorted(list(extra_f4.keys()))
    extra_cols = []
    dtype_extra = []
    for name in extra_names:
        v = extra_f4[name]
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if v.shape[0] != P:
            raise ValueError(f"extra[{name}] P mismatch: {v.shape[0]} vs {P}")
        v = v.astype(np.float32)
        extra_cols.append(v)
        for c in range(int(v.shape[1])):
            # 当用户传入 [P,C] 且 C>1 时，按 name_0/name_1... 展开
            prop_name = name if v.shape[1] == 1 else f"{name}_{c}"
            dtype_extra.append((prop_name, "f4"))

    dtype_full = dtype_base + dtype_extra
    elements = np.empty(P, dtype=dtype_full)

    if len(extra_cols) > 0:
        extras_concat = np.concatenate(extra_cols, axis=1)
        attributes_full = np.concatenate((attributes_base, extras_concat), axis=1)
    else:
        attributes_full = attributes_base

    elements[:] = list(map(tuple, attributes_full))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")], text=text).write(str(out_path))


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
    ply_format: str,
):
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives([query])
    text_feat = clip_model.pos_embeds[0].to(per_gs_feat.dtype).to(per_gs_feat.device)  # [512], normalized
    scores = per_gs_feat @ text_feat  # [P], in [-1, 1]

    sel_t = None
    if max_points and scores.numel() > max_points:
        sel_t = torch.topk(scores, k=max_points, largest=True).indices
        scores = scores[sel_t]

    s_min = float(scores.min().item())
    s_max = float(scores.max().item())
    scores01 = (scores - s_min) / (s_max - s_min + 1e-10)
    colors = (plt.cm.turbo(scores01.detach().cpu().numpy())[..., :3] * 255.0).astype(np.uint8)
    colors01 = (colors.astype(np.float32) / 255.0)

    if ply_format == "pointcloud":
        xyz = combined.get_xyz.detach()
        if sel_t is not None:
            xyz = xyz[sel_t]
        _write_xyz_rgb_ply(
            xyz.detach().cpu().numpy().astype(np.float32),
            colors,
            out_path,
            text=False,
        )
    elif ply_format == "gaussian":
        sel_np = sel_t.detach().cpu().numpy() if sel_t is not None else None
        g = _slice_gaussians_cpu(combined, sel_np)
        _write_gaussian_rgb_ply(g, colors01, out_path, text=False)
    elif ply_format == "gaussian_extra":
        sel_np = sel_t.detach().cpu().numpy() if sel_t is not None else None
        g = _slice_gaussians_cpu(combined, sel_np)
        score_raw = scores.detach().cpu().numpy().astype(np.float32)
        score_01 = scores01.detach().cpu().numpy().astype(np.float32)
        extra = {
            "extra_query_r": colors01[:, 0].astype(np.float32),
            "extra_query_g": colors01[:, 1].astype(np.float32),
            "extra_query_b": colors01[:, 2].astype(np.float32),
            "extra_query_score": score_raw,
            "extra_query_score01": score_01,
        }
        _write_gaussian_ply_with_extra(g, extra, out_path, text=False)
    else:
        raise ValueError(f"Unknown ply_format={ply_format}")
    num_points = int(scores.shape[0])
    return {"score_min": s_min, "score_max": s_max, "num_points": num_points}


def export_pca_embedding_ply(
    combined: GaussianModel,
    per_gs_feat: torch.Tensor,
    out_path: Path,
    max_points: int,
    pca_fit_points: int,
    seed: int,
    ply_format: str,
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
    rgb01 = (rgb.astype(np.float32) / 255.0)

    sel = None
    if max_points and P > max_points:
        sel = rng.choice(P, size=int(max_points), replace=False)
        xyz = xyz[sel]
        rgb = rgb[sel]
        rgb01 = rgb01[sel]

    if ply_format == "pointcloud":
        _write_xyz_rgb_ply(xyz, rgb, out_path, text=False)
    elif ply_format == "gaussian":
        g = _slice_gaussians_cpu(combined, sel)
        _write_gaussian_rgb_ply(g, rgb01, out_path, text=False)
    elif ply_format == "gaussian_extra":
        g = _slice_gaussians_cpu(combined, sel)
        extra = {
            "extra_pca_r": rgb01[:, 0].astype(np.float32),
            "extra_pca_g": rgb01[:, 1].astype(np.float32),
            "extra_pca_b": rgb01[:, 2].astype(np.float32),
        }
        _write_gaussian_ply_with_extra(g, extra, out_path, text=False)
    else:
        raise ValueError(f"Unknown ply_format={ply_format}")
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
    parser.add_argument(
        "--ply_format",
        type=str,
        default="gaussian",
        choices=["gaussian", "gaussian_extra", "pointcloud"],
        help=(
            "导出PLY格式："
            "gaussian=3DGS/Supersplat兼容（用可视化色覆盖f_dc，颜色不随视角变化）；"
            "gaussian_extra=保留checkpoint原始3DGS外观，并把可视化结果写到 extra_* 额外属性；"
            "pointcloud=仅xyz+rgb"
        ),
    )

    parser.add_argument("--max_points", type=int, default=50000, help="导出PLY最多点数（query为topN，pca为随机采样）")
    parser.add_argument("--pca_fit_points", type=int, default=200000, help="PCA拟合时的采样点数（过大可能慢）")
    parser.add_argument("--seed", type=int, default=42)

    args = get_combined_args(parser)
    # 重要：get_combined_args 会把“默认值为None”的字段从Namespace里丢掉（只合并 v!=None 的项），
    # 但 ModelParams.extract 会把缺失/None 的字段用默认值补齐。
    model_params = model.extract(args)
    # 后续逻辑里我们只需要 sh_degree；为了兼容既有写法，这里回填到 args 上。
    args.sh_degree = model_params.sh_degree
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
            ply_format=args.ply_format,
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
            ply_format=args.ply_format,
        )
        evr = info["explained_variance_ratio"]
        print(f"[pca ply] {out_path}  points={info['num_points']}  EVR={evr}")


if __name__ == "__main__":
    main()

