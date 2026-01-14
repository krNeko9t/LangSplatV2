import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/huggingface_cache"
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

NAMES = ["default", "s", "m", "l"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def colorize_ids(id_map: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    id_map: (H,W) int32, -1为背景
    return: (H,W,3) uint8 随机着色图
    """
    assert id_map.ndim == 2
    H, W = id_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    m = id_map >= 0
    if not np.any(m):
        return out

    ids = id_map[m]
    max_id = int(ids.max())

    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(max_id + 1, 3), dtype=np.uint8)
    out[m] = colors[id_map[m]]
    return out

def make_overlay(img_rgb: np.ndarray, colored: np.ndarray, alpha: float) -> np.ndarray:
    if img_rgb is None:
        return colored
    overlay = (img_rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)

def process_one(seg_path: str, images_dir: str, out_dir: str, alpha: float, seed: int):
    seg = np.load(seg_path)  # (4,H,W)
    if seg.ndim != 3 or seg.shape[0] != 4:
        raise ValueError(f"seg_maps should be (4,H,W), got {seg.shape} from {seg_path}")

    base = os.path.basename(seg_path)
    # xxx_s.npy -> xxx
    prefix = base[:-6] if base.endswith("_s.npy") else os.path.splitext(base)[0]

    img_rgb = None
    if images_dir:
        # 尝试匹配常见后缀
        cand = None
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
            p = os.path.join(images_dir, prefix + ext)
            if os.path.exists(p):
                cand = p
                break
        if cand is not None:
            bgr = cv2.imread(cand, cv2.IMREAD_COLOR)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    for i, name in enumerate(NAMES):
        id_map = seg[i].astype(np.int32)
        colored = colorize_ids(id_map, seed=seed + i)

        ensure_dir(out_dir)
        out_color = os.path.join(out_dir, f"{prefix}_{name}_color.png")
        cv2.imwrite(out_color, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

        if img_rgb is not None:
            # 尺寸不一致时，按seg_maps尺寸缩放原图
            H, W = id_map.shape
            if (img_rgb.shape[0], img_rgb.shape[1]) != (H, W):
                img_resized = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img_rgb

            overlay = make_overlay(img_resized, colored, alpha=alpha)
            out_overlay = os.path.join(out_dir, f"{prefix}_{name}_overlay.png")
            cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse/0b031f3119",
        help="你的数据集根目录（包含 images/ 和 language_features）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./segmaps_viz",
        help="输出目录，默认: dataset_path/segmaps_viz",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="叠加透明度（0~1）",
    )
    ap.add_argument("--seed", type=int, default=1234, help="随机着色种子")
    ap.add_argument("--limit", type=int, default=10, help="只处理前N个，-1为全部")
    ap.add_argument("--no_image", action="store_true", help="不叠加原图，只输出color图")
    args = ap.parse_args()

    dataset_path = args.dataset_path
    seg_dir = os.path.join(dataset_path, "language_features")
    img_dir = "" if args.no_image else os.path.join(dataset_path, "images")
    out_dir = args.out_dir if args.out_dir else os.path.join(dataset_path, "segmaps_viz")

    seg_paths = sorted(glob(os.path.join(seg_dir, "*_s.npy")))
    if len(seg_paths) == 0:
        raise FileNotFoundError(f"no *_s.npy found in {seg_dir}")

    if args.limit is not None and args.limit > 0:
        seg_paths = seg_paths[: args.limit]

    for p in tqdm(seg_paths, desc="visualizing seg_maps"):
        process_one(p, img_dir, out_dir, alpha=args.alpha, seed=args.seed)

    print(f"Done. Saved to: {out_dir}")

if __name__ == "__main__":
    main()