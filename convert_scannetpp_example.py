#!/usr/bin/env python
"""
转换ScanNet++示例数据到LangSplatV2格式
针对 scanet_example 目录结构
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def find_colmap_sparse(colmap_dir):
    """查找COLMAP sparse重建结果"""
    colmap_path = Path(colmap_dir)
    
    # 可能的路径
    possible_paths = [
        colmap_path / "sparse" / "0",
        colmap_path / "0",
        colmap_path,
    ]
    
    for path in possible_paths:
        if path.exists():
            # 检查是否有COLMAP文件
            required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
            if all((path / f).exists() for f in required_files):
                return path
    
    # 也检查文本格式
    for path in possible_paths:
        if path.exists():
            required_files_txt = ['cameras.txt', 'images.txt', 'points3D.txt']
            if all((path / f).exists() for f in required_files_txt):
                return path
    
    return None


def find_images_dir(scene_dir):
    """查找图像目录"""
    scene_path = Path(scene_dir)
    
    # 可能的图像目录（更全面的搜索）
    possible_dirs = [
        scene_path / "dslr" / "resized_undistorted_images",
        scene_path / "dslr" / "resized_images",
        scene_path / "dslr" / "images",
        scene_path / "dslr" / "resized_anon_masks",  # 可能包含图像
        scene_path / "iphone" / "images",
        scene_path / "images",  # 直接在根目录
    ]
    
    # 递归搜索所有可能的图像目录
    all_possible_dirs = []
    for base_dir in [scene_path / "dslr", scene_path / "iphone", scene_path]:
        if base_dir.exists():
            # 查找所有包含图像的目录
            for subdir in base_dir.rglob("*"):
                if subdir.is_dir():
                    images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + \
                             list(subdir.glob("*.JPG")) + list(subdir.glob("*.PNG"))
                    if len(images) > 0:
                        all_possible_dirs.append(subdir)
    
    # 优先使用已知的目录
    for img_dir in possible_dirs:
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + \
                     list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.PNG"))
            if len(images) > 0:
                return img_dir
    
    # 如果已知目录都不存在，使用找到的第一个包含图像的目录
    if all_possible_dirs:
        return all_possible_dirs[0]
    
    return None


def convert_scannetpp_example(
    input_dir: str,
    output_dir: str,
    scene_name: str = "scanet_example",
    use_dslr: bool = True,
    images_dir: str = None
):
    """
    转换ScanNet++示例数据到LangSplatV2格式
    
    Args:
        input_dir: 输入的scanet_example目录
        output_dir: 输出目录
        scene_name: 场景名称
        use_dslr: 是否使用DSLR数据（否则使用iPhone数据）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在转换ScanNet++示例数据...")
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    
    # 1. 查找并复制图像
    print("\n[1/3] 处理图像...")
    
    # 如果手动指定了图像目录，直接使用
    if images_dir:
        images_source = Path(images_dir)
        # 如果是相对路径，尝试相对于输入目录或当前工作目录解析
        if not images_source.is_absolute() and not images_source.exists():
            # 尝试相对于输入目录
            try_path1 = input_path / images_source
            # 尝试相对于当前工作目录
            try_path2 = Path.cwd() / images_source
            if try_path1.exists():
                images_source = try_path1
                print(f"使用相对于输入目录的路径: {images_source}")
            elif try_path2.exists():
                images_source = try_path2
                print(f"使用相对于当前目录的路径: {images_source}")
        
        if not images_source.exists():
            print(f"错误: 指定的图像目录不存在: {images_source}")
            print(f"尝试的路径:")
            print(f"  - {images_source}")
            if not Path(images_dir).is_absolute():
                print(f"  - {input_path / images_dir}")
                print(f"  - {Path.cwd() / images_dir}")
            return False
    else:
        # 自动查找
        if use_dslr:
            source_dir = input_path / "dslr"
            images_source = find_images_dir(input_path / "dslr")
        else:
            source_dir = input_path / "iphone"
            images_source = find_images_dir(input_path / "iphone")
    
    if images_source is None:
        print(f"错误: 无法找到图像目录!")
        print("\n已搜索以下位置:")
        print("  - dslr/resized_undistorted_images")
        print("  - dslr/resized_images")
        print("  - dslr/images")
        print("  - iphone/images")
        print("\n正在尝试在整个目录结构中查找图像...")
        
        # 列出所有可能的图像目录
        scene_path = Path(input_dir)
        found_dirs = []
        for base_dir in [scene_path / "dslr", scene_path / "iphone", scene_path]:
            if base_dir.exists():
                for subdir in base_dir.rglob("*"):
                    if subdir.is_dir():
                        images = (list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) +
                                 list(subdir.glob("*.JPG")) + list(subdir.glob("*.PNG")) +
                                 list(subdir.glob("*.jpeg")) + list(subdir.glob("*.JPEG")))
                        if len(images) > 0:
                            found_dirs.append((subdir, len(images)))
        
        if found_dirs:
            print(f"\n找到 {len(found_dirs)} 个包含图像的目录:")
            for img_dir, count in found_dirs[:10]:  # 只显示前10个
                rel_path = img_dir.relative_to(scene_path)
                print(f"  - {rel_path} ({count} 张图像)")
            if len(found_dirs) > 10:
                print(f"  ... 还有 {len(found_dirs) - 10} 个目录")
            print("\n请手动指定正确的图像目录，或检查目录结构")
            print("使用 --images_dir 参数指定图像目录")
        else:
            print("\n未找到包含图像的目录。请检查:")
            print("  1. 输入目录路径是否正确")
            print("  2. 图像文件是否存在 (jpg, png)")
            print("  3. 图像文件是否在子目录中")
        
        return False
    
    print(f"在以下目录找到图像: {images_source}")
    images_dst = output_path / "images"
    
    if images_dst.exists():
        shutil.rmtree(images_dst)
    images_dst.mkdir(parents=True, exist_ok=True)
    
    # 复制图像 - 支持更多格式
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_source.glob(ext)))
    
    image_files = sorted(set(image_files))  # 去重并排序
    print(f"找到 {len(image_files)} 张图像...")
    
    if len(image_files) == 0:
        print(f"错误: 在 {images_source} 中未找到图像文件!")
        print(f"请检查图像文件是否存在，支持的格式: jpg, jpeg, png")
        # 列出目录内容帮助调试
        all_files = list(images_source.iterdir())
        if all_files:
            print(f"\n目录中的文件/文件夹 (前10个):")
            for item in all_files[:10]:
                print(f"  - {item.name} ({'目录' if item.is_dir() else '文件'})")
        return False
    
    print(f"正在复制 {len(image_files)} 张图像...")
    for img_file in tqdm(image_files, desc="复制图像"):
        # 保持原始文件名
        dst_file = images_dst / img_file.name
        shutil.copy2(img_file, dst_file)
    
    print(f"✓ 图像已复制到 {images_dst}")
    
    # 2. 查找并复制COLMAP sparse重建结果
    print("\n[2/3] 处理COLMAP稀疏重建结果...")
    
    if use_dslr:
        colmap_source_dir = input_path / "dslr" / "colmap"
    else:
        colmap_source_dir = input_path / "iphone" / "colmap"
    
    sparse_src = find_colmap_sparse(colmap_source_dir)
    
    if sparse_src is None:
        print(f"错误: 无法找到COLMAP稀疏重建结果!")
        print(f"已搜索: {colmap_source_dir}")
        print("\n请确保COLMAP重建结果存在，或运行以下命令:")
        print("  colmap feature_extractor --database_path database.db --image_path images")
        print("  colmap exhaustive_matcher --database_path database.db")
        print("  colmap mapper --database_path database.db --image_path images --output_path sparse")
        return False
    
    print(f"在以下位置找到COLMAP稀疏重建: {sparse_src}")
    
    sparse_dst = output_path / "sparse" / "0"
    sparse_dst.parent.mkdir(parents=True, exist_ok=True)
    
    if sparse_dst.exists():
        shutil.rmtree(sparse_dst)
    
    # 复制COLMAP文件
    print("正在复制COLMAP文件...")
    
        # 检查是二进制还是文本格式
        if (sparse_src / "cameras.bin").exists():
            # 二进制格式
            required_files = ["cameras.bin", "images.bin", "points3D.bin"]
            for f in required_files:
                src_file = sparse_src / f
                if src_file.exists():
                    shutil.copy2(src_file, sparse_dst / f)
                    print(f"  ✓ 已复制 {f}")
                else:
                    print(f"  ✗ 缺少 {f}")
    elif (sparse_src / "cameras.txt").exists():
        # 文本格式，需要转换为二进制（或直接使用文本格式）
        print("  发现文本格式的COLMAP文件")
        print("  注意: LangSplatV2更推荐二进制格式")
        print("  尝试转换为二进制格式...")
        
        # 使用COLMAP转换工具
        import subprocess
        import shutil as shutil_module
        
        # 先检查colmap命令是否可用
        colmap_available = False
        try:
            result = subprocess.run(
                ["colmap", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                colmap_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            colmap_available = False
        
        if colmap_available:
            try:
                print("  使用COLMAP转换工具...")
                result = subprocess.run([
                    "colmap", "model_converter",
                    "--input_path", str(sparse_src),
                    "--output_path", str(sparse_dst),
                    "--output_type", "BIN"
                ], check=True, capture_output=True, text=True, timeout=30)
                print("  ✓ 成功转换为二进制格式")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ COLMAP转换失败: {e}")
                print("  错误输出:", e.stderr if hasattr(e, 'stderr') else 'N/A')
                print("  将直接复制文本文件...")
                if sparse_dst.exists():
                    shutil_module.rmtree(sparse_dst)
                shutil_module.copytree(sparse_src, sparse_dst)
                print("  ✓ 已复制文本格式文件")
                print("  注意: 您可能需要手动转换为二进制格式:")
                print(f"    colmap model_converter --input_path {sparse_src} --output_path {sparse_dst} --output_type BIN")
            except (FileNotFoundError, PermissionError) as e:
                print(f"  ⚠ 无法执行COLMAP命令: {e}")
                print("  将直接复制文本文件...")
                if sparse_dst.exists():
                    shutil_module.rmtree(sparse_dst)
                shutil_module.copytree(sparse_src, sparse_dst)
                print("  ✓ 已复制文本格式文件")
        else:
            print("  ⚠ COLMAP命令不可用，直接复制文本文件")
            if sparse_dst.exists():
                shutil_module.rmtree(sparse_dst)
            shutil_module.copytree(sparse_src, sparse_dst)
            print("  ✓ 已复制文本格式文件")
            print("  注意: LangSplatV2可能需要二进制格式，您可以稍后手动转换:")
            print(f"    colmap model_converter --input_path {sparse_src} --output_path {sparse_dst} --output_type BIN")
        else:
            print(f"  ✗ 在 {sparse_src} 中未找到有效的COLMAP文件")
            return False
    
    print(f"✓ COLMAP稀疏重建已复制到 {sparse_dst}")
    
    # 3. 验证结果
    print("\n[3/3] 验证输出...")
    
    # 检查必需文件
    required = {
        "images": images_dst.exists() and len(list(images_dst.glob("*.jpg")) + list(images_dst.glob("*.png"))) > 0,
        "cameras.bin": (sparse_dst / "cameras.bin").exists(),
        "images.bin": (sparse_dst / "images.bin").exists(),
        "points3D.bin": (sparse_dst / "points3D.bin").exists(),
    }
    
    all_ok = all(required.values())
    
    for item, status in required.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {item}")
    
    if all_ok:
        print("\n✓ 转换成功!")
        print(f"\n输出目录: {output_path}")
        print("\n下一步:")
        print("1. 验证数据:")
        print(f"   python check_scannetpp_data.py --data_root {output_dir} --scene {scene_name}")
        print("2. 预处理（生成语言特征）:")
        print(f"   python preprocess.py --dataset_path {output_path}")
        print("3. 训练:")
        print(f"   bash train_scannetpp.sh {scene_name} 0")
        return True
    else:
        print("\n✗ 转换未完成，请检查上面的错误信息")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ScanNet++ example data to LangSplatV2 format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory (scanet_example)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Manually specify images directory (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/scannetpp",
        help="Output directory for converted data"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="scanet_example",
        help="Scene name in output directory"
    )
    parser.add_argument(
        "--use_iphone",
        action="store_true",
        help="Use iPhone data instead of DSLR data"
    )
    
    args = parser.parse_args()
    
    success = convert_scannetpp_example(
        args.input_dir,
        args.output_dir,
        args.scene_name,
        use_dslr=not args.use_iphone,
        images_dir=args.images_dir
    )
    
    exit(0 if success else 1)

