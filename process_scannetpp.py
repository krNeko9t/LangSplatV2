import os
import subprocess
import sys
from pathlib import Path

# ================= 配置区域 =================

root = Path("/home/bingxing2/ailab/liuyifei/lyj/Dataset/scannetpp/scannetpp")

# 1. 待训练的场景列表 (支持相对路径或绝对路径)
SCENE_LIST = [
    dir for dir in root.iterdir()
]

# 2. 训练迭代次数
ITERATIONS = 30_000


def prepare_scannetpp_structure(scene_path: Path):
    """
    针对 ScanNet++ 数据集结构进行适配：
    1. 创建 sparse/0
    2. 链接 dslr/colmap 下的 txt 文件
    3. 链接 dslr/images 到根目录 images
    """
    print(f"  [检查结构] {scene_path.name} ...")

    # 检查源目录是否存在
    colmap_src = scene_path / "dslr" / "colmap"
    images_src = scene_path / "dslr" / "resized_undistorted_images"  # 注意：如果你的图片在 resized_images，请修改这里

    if not colmap_src.exists():
        print(f"  [跳过适配] 未检测到 dslr/colmap，假设该数据已经是标准 3DGS 格式。")
        return

    # 1. 创建 sparse/0
    sparse_dir = scene_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 2. 建立 txt 文件的软链接
    files_to_link = ["cameras.txt", "images.txt", "points3D.txt"]
    for filename in files_to_link:
        src_file = colmap_src / filename
        dst_file = sparse_dir / filename
        
        if src_file.exists():
            if not dst_file.exists():
                try:
                    os.symlink(src_file, dst_file)
                    print(f"    -> Link created: {filename}")
                except OSError as e:
                    print(f"    [Warn] 无法创建链接 {filename}: {e}")
        else:
            print(f"    [Warn] 源文件缺失: {src_file}")

    # 3. 建立 images 文件夹链接
    dst_images = scene_path / "images"
    if images_src.exists() and not dst_images.exists():
        try:
            os.symlink(images_src, dst_images)
            print(f"    -> Link created: images folder")
        except OSError as e:
            print(f"    [Warn] 无法链接 images: {e}")

if __name__ == "__main__":
    for dir in SCENE_LIST:
        prepare_scannetpp_structure(dir)