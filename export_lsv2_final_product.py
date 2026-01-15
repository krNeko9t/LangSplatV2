#!/usr/bin/env python
"""
导出 LangSplatV2 的“最终产物”（不依赖渲染 weight-map）：

1) PLY（每点）：
   - x/y/z: float32
   - weight_0..weight_63: float32   （来自每个 GS 的 language logits -> softmax；可选 top-k 稀疏化）

2) Codebook：
   - 二进制 .bin（Float32Array），长度 64*512（row-major）

3) Query 列表：
   - JSON: {"queries":[{"name":"elephant","vector":[...512 floats...]}]}
   - text embedding 使用 OpenCLIPNetwork（与仓库其它可视化脚本一致，默认归一化）

参数风格对齐 .vscode/launch.json(156-165)：--scene_id, -s, --gt_json, --ckpt_root, --ckpt_prefix,
--checkpoint, --threshold, --topk, --output_dir, --save_visuals。
"""

import json
import os
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


def _write_ply_xyz_weights64(xyz: np.ndarray, weights: np.ndarray, out_path: Path):
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert weights.ndim == 2 and weights.shape[1] == 64
    assert xyz.shape[0] == weights.shape[0]

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [(f"weight_{i}", "f4") for i in range(64)]
    verts = np.empty(xyz.shape[0], dtype=dtype)
    verts["x"] = xyz[:, 0].astype(np.float32)
    verts["y"] = xyz[:, 1].astype(np.float32)
    verts["z"] = xyz[:, 2].astype(np.float32)
    for i in range(64):
        verts[f"weight_{i}"] = weights[:, i].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out_path))


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

    # 对齐 launch.json(156-165)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("-s", "--source_path", type=str, default=None)
    parser.add_argument("--gt_json", type=str, default=None)
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--ckpt_prefix", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--topk", type=int, default=4, help="导出 weights 时的 top-k 稀疏化（0/>=64 表示不稀疏）")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_visuals", action="store_true")

    # 额外：选择导出 level（每个 level 单独一套 64 权重 + 64x512 codebook）
    parser.add_argument("--feature_level", type=int, default=1, help="单个导出 level（默认 1）")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="一次导出多个 level，如 --levels 1 2 3")

    # query
    parser.add_argument("--queries", type=str, nargs="+", required=True, help="需要导出的 query 文本列表，如 --queries elephant camera")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="生成 query embedding 的设备")

    args = parser.parse_args()

    levels = _parse_levels(args)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    out_root = Path(args.output_dir) / str(args.scene_id) / f"chkpnt{int(args.checkpoint)}"
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

        xyz, logits, codebooks = _load_language_from_ckpt(ckpt_path)

        # codebooks: [L,K,512]，此导出格式要求 K=64 且 L=1
        L, K, D = codebooks.shape
        if K != 64 or D != 512:
            raise ValueError(f"当前 checkpoint codebook 维度不是 [L,64,512]：{codebooks.shape}")
        if L != 1:
            raise ValueError(
                f"当前 checkpoint 的 codebook 有 {L} 个 level；本导出格式要求单 level（64*512）。"
            )

        weights = _compute_weights_64(logits, topk=int(args.topk), require_single_level=True).cpu().numpy()
        codebook = codebooks[0].cpu().numpy()

        lvl_dir = out_root / f"L{int(level)}"
        ply_out = lvl_dir / "weights64.ply"
        bin_out = lvl_dir / "codebook_64x512.bin"

        _write_ply_xyz_weights64(xyz, weights, ply_out)
        _write_codebook_bin(codebook, bin_out)

        print(f"[OK] level={level}")
        print(f"  ckpt: {ckpt_path}")
        print(f"  ply:  {ply_out}   (P={xyz.shape[0]})")
        print(f"  bin:  {bin_out}   (len={64*512})")

    print(f"[DONE] outputs in: {out_root}")


if __name__ == "__main__":
    main()

