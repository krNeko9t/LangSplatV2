#!/usr/bin/env python
"""
导出 LangSplatV2 的“最终产物”（不依赖渲染 weight-map）：

1) PLY（每点）：
   - 默认导出“3DGS / Supersplat 兼容”字段：x/y/z, nx/ny/nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*
   - 并追加 extra 属性：weight_0..weight_63: float32（来自每个 GS 的 language logits -> softmax；可选 top-k 稀疏化）

2) Codebook：
   - 二进制 .bin（Float32Array），长度 64*512（row-major）

3) Query 列表：
   - JSON: {"queries":[{"name":"elephant","vector":[...512 floats...]}]}
   - text embedding 使用 OpenCLIPNetwork（与仓库其它可视化脚本一致，默认归一化）

本脚本只保留“导出真正需要”的参数；不再携带 eval/可视化相关的占位参数（如 gt_json/threshold 等）。
"""

import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/huggingface_cache"
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from eval.openclip_encoder import OpenCLIPNetwork


def _resolve_ckpt_dir(ckpt_root: str, ckpt_prefix: str, level: int) -> Path:
    root = Path(ckpt_root)
    # 最常见：<ckpt_root>/<ckpt_prefix>_<level>/
    cand1 = root / f"{ckpt_prefix}_{level}"
    if cand1.exists():
        return cand1

    # 兼容：用户直接把 prefix 写成带 level 的目录名
    cand2 = root / ckpt_prefix
    if cand2.exists():
        return cand2

    # 再兜底：在 root 下找一个以 prefix 开头、以 _{level} 结尾的目录
    pattern = f"{ckpt_prefix}_*"
    matches = sorted([p for p in root.glob(pattern) if p.is_dir() and str(p.name).endswith(f"_{level}")])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"找到多个候选 checkpoint 目录（请检查 --ckpt_prefix 是否更精确）：{[str(m) for m in matches[:10]]}"
        )
    raise FileNotFoundError(f"找不到 checkpoint 目录：{cand1} 或 {cand2}（root={root}）")


def _infer_scene_id_from_ckpt_prefix(ckpt_prefix: str) -> str:
    """
    常见 ckpt_prefix 形如:
      - <scene>_<index>      e.g. figurines_0, 0a7cc12c0e_0
    此时 scene_id 默认取最后一个 '_' 之前的部分。
    若不含 '_'，则直接返回 ckpt_prefix。
    """
    s = str(ckpt_prefix)
    if "_" not in s:
        return s
    return s.rsplit("_", 1)[0]


def _construct_3dgs_attribute_names(
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> List[str]:
    """
    复刻 scene/gaussian_model.py:GaussianModel.construct_list_of_attributes 的命名规则，
    以保证导出的 PLY 与本仓库 3DGS PLY schema 兼容。
    """
    l = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(int(features_dc.shape[1] * features_dc.shape[2])):
        l.append(f"f_dc_{i}")
    for i in range(int(features_rest.shape[1] * features_rest.shape[2])):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    for i in range(int(scaling.shape[1])):
        l.append(f"scale_{i}")
    for i in range(int(rotation.shape[1])):
        l.append(f"rot_{i}")
    return l


def _write_gaussian_ply_with_extra_from_tensors(
    *,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    extra_f4: dict,
    out_path: Path,
    text: bool = False,
):
    """
    导出“正常3DGS / Supersplat”兼容的 PLY（保留 checkpoint 原始外观参数），
    并在 vertex 属性末尾追加自定义 extra 字段（float32）。

    extra_f4:
      - key: 属性名
      - value: np.ndarray / torch.Tensor，shape 为 [P] / [P,1] / [P,C]
    """
    xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
    P = int(xyz_np.shape[0])
    normals = np.zeros_like(xyz_np, dtype=np.float32)

    f_dc = (
        features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    f_rest = (
        features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    opacities = opacity.detach().cpu().numpy().astype(np.float32)
    scale = scaling.detach().cpu().numpy().astype(np.float32)
    rot = rotation.detach().cpu().numpy().astype(np.float32)

    attributes_base = np.concatenate((xyz_np, normals, f_dc, f_rest, opacities, scale, rot), axis=1)
    names_base = _construct_3dgs_attribute_names(features_dc, features_rest, scaling, rotation)
    dtype_base = [(name, "f4") for name in names_base]

    extra_names = sorted(list(extra_f4.keys()))
    extra_cols = []
    dtype_extra = []
    for name in extra_names:
        v = extra_f4[name]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if v.shape[0] != P:
            raise ValueError(f"extra[{name}] P mismatch: {v.shape[0]} vs {P}")
        v = v.astype(np.float32)
        extra_cols.append(v)
        for c in range(int(v.shape[1])):
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


def _load_language_from_ckpt(ckpt_path: Path) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    返回:
      - xyz: [P,3] float32 (numpy)
      - logits: [P, L*K] float32 (torch)
      - codebooks: [L, K, 512] float32 (torch)
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not (isinstance(ckpt, (tuple, list)) and len(ckpt) == 2):
        raise ValueError(f"未知 checkpoint 格式：{type(ckpt)}")
    model_params, _ = ckpt
    if not (isinstance(model_params, (tuple, list)) and len(model_params) == 14):
        raise ValueError(
            f"checkpoint 里没有 language feature（期望 model_params 长度=14），实际 len={len(model_params) if isinstance(model_params,(tuple,list)) else 'N/A'}"
        )

    # capture(include_feature=True) 的顺序见 scene/gaussian_model.py:GaussianModel.capture
    xyz = model_params[1]
    logits = model_params[7]
    codebooks = model_params[8]

    xyz = torch.as_tensor(xyz).detach().cpu().to(torch.float32).numpy()
    logits = torch.as_tensor(logits).detach().cpu().to(torch.float32)
    codebooks = torch.as_tensor(codebooks).detach().cpu().to(torch.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz shape 异常：{xyz.shape}")
    if logits.ndim != 2:
        raise ValueError(f"logits shape 异常：{logits.shape}")
    if codebooks.ndim != 3 or codebooks.shape[-1] != 512:
        raise ValueError(f"codebooks shape 异常：{codebooks.shape}")

    return xyz.astype(np.float32), logits, codebooks


def _compute_weights_64(
    logits: torch.Tensor,
    *,
    topk: int,
    require_single_level: bool = True,
) -> torch.Tensor:
    """
    logits: [P, L*K]
    返回:
      - weights: [P,64] float32
    """
    P, LK = logits.shape
    K = 64
    if LK % K != 0:
        raise ValueError(f"logits 维度不是 64 的整数倍：LK={LK}")
    L = LK // K
    if require_single_level and L != 1:
        raise ValueError(f"当前 checkpoint 的 level 数 L={L}，但你要求导出固定 64 维权重。请指定单 level 的 ckpt。")

    layer_logits = logits[:, 0:K]  # 只导出第 0 层（通常 L=1）
    if topk is not None and int(topk) > 0 and int(topk) < K:
        # top-k 稀疏化：非 top-k 置 0，并重新归一化
        from utils.vq_utils import softmax_to_topk_soft_code

        w = softmax_to_topk_soft_code(layer_logits, int(topk))
    else:
        w = layer_logits.softmax(dim=1)
    return w.to(torch.float32)


def _write_codebook_bin(codebook_64x512: np.ndarray, out_path: Path):
    if codebook_64x512.shape != (64, 512):
        raise ValueError(f"codebook 必须是 [64,512]，实际 {codebook_64x512.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_64x512.astype("<f4").reshape(-1).tofile(str(out_path))


def _write_queries_json(queries: List[str], out_path: Path, device: torch.device):
    if len(queries) == 0:
        raise ValueError("queries 不能为空。请用 --queries 传入文本列表。")

    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(list(queries))
    embeds = clip_model.pos_embeds.detach().to(torch.float32).cpu().numpy()  # [N,512], 已归一化

    payload = {"queries": []}
    for name, vec in zip(queries, embeds):
        payload["queries"].append({"name": str(name), "vector": vec.astype(np.float32).tolist()})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _parse_levels(args) -> List[int]:
    if args.levels is not None and len(args.levels) > 0:
        return [int(x) for x in args.levels]
    return [int(args.feature_level)]


def main():
    # 与其它脚本一致：离线 HuggingFace（如果用户在环境里配置了缓存路径）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    parser = ArgumentParser(description="Export LangSplatV2 final products (PLY + codebook.bin + queries.json)")

    # 输出路径相关
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="输出子目录名；默认从 --ckpt_prefix 推断（如 figurines_0 -> figurines）",
    )
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--ckpt_prefix", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--topk", type=int, default=4, help="导出 weights 时的 top-k 稀疏化（0/>=64 表示不稀疏）")
    parser.add_argument("--output_dir", type=str, required=True)

    # 额外：选择导出 level（每个 level 单独一套 64 权重 + 64x512 codebook）
    parser.add_argument("--feature_level", type=int, default=1, help="单个导出 level（默认 1）")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="一次导出多个 level，如 --levels 1 2 3")

    # query
    parser.add_argument("--queries", type=str, nargs="+", required=True, help="需要导出的 query 文本列表，如 --queries elephant camera")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="生成 query embedding 的设备")

    args = parser.parse_args()

    levels = _parse_levels(args)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    scene_id = str(args.scene_id) if args.scene_id is not None else _infer_scene_id_from_ckpt_prefix(args.ckpt_prefix)
    out_root = Path(args.output_dir) / scene_id / f"chkpnt{int(args.checkpoint)}"
    out_root.mkdir(parents=True, exist_ok=True)

    # queries.json（同一 scene/checkpoint 只写一次）
    queries_out = out_root / "queries.json"
    _write_queries_json(args.queries, queries_out, device=device)

    # 每个 level 导出一套 PLY + codebook
    for level in levels:
        ckpt_dir = _resolve_ckpt_dir(args.ckpt_root, args.ckpt_prefix, level=int(level))
        ckpt_path = ckpt_dir / f"chkpnt{int(args.checkpoint)}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint 文件：{ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if not (isinstance(ckpt, (tuple, list)) and len(ckpt) == 2):
            raise ValueError(f"未知 checkpoint 格式：{type(ckpt)}")
        model_params, _ = ckpt
        if not (isinstance(model_params, (tuple, list)) and len(model_params) == 14):
            raise ValueError(
                f"checkpoint 里没有 language feature（期望 model_params 长度=14），实际 len={len(model_params) if isinstance(model_params,(tuple,list)) else 'N/A'}"
            )

        # capture(include_feature=True) 的顺序见 scene/gaussian_model.py:GaussianModel.capture
        active_sh_degree = int(model_params[0])
        xyz_t = torch.as_tensor(model_params[1]).detach().to(torch.float32)
        f_dc_t = torch.as_tensor(model_params[2]).detach().to(torch.float32)
        f_rest_t = torch.as_tensor(model_params[3]).detach().to(torch.float32)
        scaling_t = torch.as_tensor(model_params[4]).detach().to(torch.float32)
        rotation_t = torch.as_tensor(model_params[5]).detach().to(torch.float32)
        opacity_t = torch.as_tensor(model_params[6]).detach().to(torch.float32)
        logits = torch.as_tensor(model_params[7]).detach().to(torch.float32)
        codebooks = torch.as_tensor(model_params[8]).detach().to(torch.float32)

        # codebooks: [L,K,512]，此导出格式要求 K=64 且 L=1
        L, K, D = codebooks.shape
        if K != 64 or D != 512:
            raise ValueError(f"当前 checkpoint codebook 维度不是 [L,64,512]：{codebooks.shape}")
        if L != 1:
            raise ValueError(
                f"当前 checkpoint 的 codebook 有 {L} 个 level；本导出格式要求单 level（64*512）。"
            )

        weights64 = _compute_weights_64(logits, topk=int(args.topk), require_single_level=True).cpu().numpy()
        codebook = codebooks[0].cpu().numpy()

        lvl_dir = out_root / f"L{int(level)}"
        ply_out = lvl_dir / "gaussians_with_weights64.ply"
        bin_out = lvl_dir / "codebook_64x512.bin"

        extra = {f"weight_{i}": weights64[:, i] for i in range(64)}
        _write_gaussian_ply_with_extra_from_tensors(
            xyz=xyz_t,
            features_dc=f_dc_t,
            features_rest=f_rest_t,
            opacity=opacity_t,
            scaling=scaling_t,
            rotation=rotation_t,
            extra_f4=extra,
            out_path=ply_out,
            text=False,
        )
        _write_codebook_bin(codebook, bin_out)

        print(f"[OK] level={level}")
        print(f"  ckpt: {ckpt_path}")
        print(f"  ply:  {ply_out}   (P={int(xyz_t.shape[0])}, sh_degree={active_sh_degree})")
        print(f"  bin:  {bin_out}   (len={64*512})")

    print(f"[DONE] outputs in: {out_root}")


if __name__ == "__main__":
    main()

