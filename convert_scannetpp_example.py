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
    
    print(f"Converting ScanNet++ example data...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # 1. 查找并复制图像
    print("\n[1/3] Processing images...")
    
    # 如果手动指定了图像目录，直接使用
    if images_dir:
        images_source = Path(images_dir)
        if not images_source.exists():
            print(f"ERROR: Specified images directory does not exist: {images_source}")
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
        print(f"ERROR: Could not find images directory!")
        print("\nSearched in:")
        print("  - dslr/resized_undistorted_images")
        print("  - dslr/resized_images")
        print("  - dslr/images")
        print("  - iphone/images")
        print("\nTrying to find images in the directory structure...")
        
        # 列出所有可能的图像目录
        scene_path = Path(input_dir)
        found_dirs = []
        for base_dir in [scene_path / "dslr", scene_path / "iphone", scene_path]:
            if base_dir.exists():
                for subdir in base_dir.rglob("*"):
                    if subdir.is_dir():
                        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + \
                                 list(subdir.glob("*.JPG")) + list(subdir.glob("*.PNG"))
                        if len(images) > 0:
                            found_dirs.append((subdir, len(images)))
        
        if found_dirs:
            print(f"\nFound {len(found_dirs)} directories with images:")
            for img_dir, count in found_dirs[:10]:  # 只显示前10个
                rel_path = img_dir.relative_to(scene_path)
                print(f"  - {rel_path} ({count} images)")
            if len(found_dirs) > 10:
                print(f"  ... and {len(found_dirs) - 10} more")
            print("\nPlease specify the correct images directory manually or check the directory structure.")
        else:
            print("\nNo directories with images found. Please check:")
            print("  1. The input directory path is correct")
            print("  2. Image files exist (jpg, png)")
            print("  3. Image files are in a subdirectory")
        
        return False
    
    print(f"Found images in: {images_source}")
    images_dst = output_path / "images"
    
    if images_dst.exists():
        shutil.rmtree(images_dst)
    images_dst.mkdir(parents=True, exist_ok=True)
    
    # 复制图像
    image_files = sorted(list(images_source.glob("*.jpg")) + list(images_source.glob("*.png")))
    print(f"Copying {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Copying images"):
        # 保持原始文件名或重命名为frame_XXXXXX格式
        dst_file = images_dst / img_file.name
        shutil.copy2(img_file, dst_file)
    
    print(f"✓ Images copied to {images_dst}")
    
    # 2. 查找并复制COLMAP sparse重建结果
    print("\n[2/3] Processing COLMAP sparse reconstruction...")
    
    if use_dslr:
        colmap_source_dir = input_path / "dslr" / "colmap"
    else:
        colmap_source_dir = input_path / "iphone" / "colmap"
    
    sparse_src = find_colmap_sparse(colmap_source_dir)
    
    if sparse_src is None:
        print(f"ERROR: Could not find COLMAP sparse reconstruction!")
        print(f"Searched in: {colmap_source_dir}")
        print("\nPlease ensure COLMAP reconstruction exists, or run:")
        print("  colmap feature_extractor --database_path database.db --image_path images")
        print("  colmap exhaustive_matcher --database_path database.db")
        print("  colmap mapper --database_path database.db --image_path images --output_path sparse")
        return False
    
    print(f"Found COLMAP sparse in: {sparse_src}")
    
    sparse_dst = output_path / "sparse" / "0"
    sparse_dst.parent.mkdir(parents=True, exist_ok=True)
    
    if sparse_dst.exists():
        shutil.rmtree(sparse_dst)
    
    # 复制COLMAP文件
    print("Copying COLMAP files...")
    
    # 检查是二进制还是文本格式
    if (sparse_src / "cameras.bin").exists():
        # 二进制格式
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        for f in required_files:
            src_file = sparse_src / f
            if src_file.exists():
                shutil.copy2(src_file, sparse_dst / f)
                print(f"  ✓ Copied {f}")
            else:
                print(f"  ✗ Missing {f}")
    elif (sparse_src / "cameras.txt").exists():
        # 文本格式，需要转换为二进制（或直接使用文本格式）
        print("  Found text format COLMAP files")
        print("  Note: LangSplatV2 prefers binary format")
        print("  Converting text to binary...")
        
        # 使用COLMAP转换工具
        import subprocess
        try:
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", str(sparse_src),
                "--output_path", str(sparse_dst),
                "--output_type", "BIN"
            ], check=True)
            print("  ✓ Converted to binary format")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠ Could not convert to binary, copying text files")
            print("  Note: You may need to convert manually using:")
            print("    colmap model_converter --input_path <src> --output_path <dst> --output_type BIN")
            # 直接复制文本文件
            shutil.copytree(sparse_src, sparse_dst)
    else:
        print(f"  ✗ No valid COLMAP files found in {sparse_src}")
        return False
    
    print(f"✓ COLMAP sparse copied to {sparse_dst}")
    
    # 3. 验证结果
    print("\n[3/3] Verifying output...")
    
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
        print("\n✓ Conversion successful!")
        print(f"\nOutput directory: {output_path}")
        print("\nNext steps:")
        print("1. Verify data:")
        print(f"   python check_scannetpp_data.py --data_root {output_dir} --scene {scene_name}")
        print("2. Preprocess (generate language features):")
        print(f"   python preprocess.py --dataset_path {output_path}")
        print("3. Train:")
        print(f"   bash train_scannetpp.sh {scene_name} 0")
        return True
    else:
        print("\n✗ Conversion incomplete. Please check the errors above.")
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

