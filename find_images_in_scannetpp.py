#!/usr/bin/env python
"""
诊断脚本：查找ScanNet++数据中的图像目录
"""
import sys
from pathlib import Path

def find_images(input_dir):
    """查找所有包含图像的目录"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"ERROR: Directory does not exist: {input_path}")
        print(f"Absolute path: {input_path.resolve()}")
        return
    
    print(f"Searching for images in: {input_path.resolve()}\n")
    
    # 检查主要目录
    print("Checking directory structure:")
    for subdir in ['dslr', 'iphone', 'scans']:
        path = input_path / subdir
        if path.exists():
            print(f"  ✓ {subdir}/ exists")
            # 列出直接子目录
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"    Subdirectories: {', '.join(subdirs[:10])}")
        else:
            print(f"  ✗ {subdir}/ does not exist")
    
    # 查找所有图像文件
    print("\nSearching for image directories...")
    image_dirs = {}
    
    for base_dir in [input_path / "dslr", input_path / "iphone", input_path]:
        if base_dir.exists():
            for subdir in base_dir.rglob("*"):
                if subdir.is_dir():
                    images = (list(subdir.glob("*.jpg")) + 
                             list(subdir.glob("*.png")) +
                             list(subdir.glob("*.JPG")) + 
                             list(subdir.glob("*.PNG")))
                    if len(images) > 0:
                        rel_path = subdir.relative_to(input_path)
                        image_dirs[str(rel_path)] = len(images)
    
    if image_dirs:
        print(f"\n✓ Found {len(image_dirs)} directories with images:\n")
        # 按图像数量排序
        sorted_dirs = sorted(image_dirs.items(), key=lambda x: x[1], reverse=True)
        for dir_path, count in sorted_dirs:
            abs_path = (input_path / dir_path).resolve()
            print(f"  {dir_path}")
            print(f"    → {count} images")
            print(f"    → Full path: {abs_path}\n")
        
        # 推荐使用
        best_dir = sorted_dirs[0][0]
        print(f"\nRecommended: Use '{best_dir}'")
        print(f"\nRun conversion with:")
        print(f"  python convert_scannetpp_example.py \\")
        print(f"    --input_dir {input_path.resolve()} \\")
        print(f"    --output_dir ./data/scannetpp \\")
        print(f"    --scene_name scanet_example \\")
        print(f"    --images_dir {input_path / best_dir}")
    else:
        print("\n✗ No images found!")
        print("\nPlease check:")
        print("  1. The input directory path is correct")
        print("  2. Image files exist (jpg, png)")
        print("  3. You have read permissions")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_images_in_scannetpp.py <input_dir>")
        print("Example: python find_images_in_scannetpp.py ../Dataset/scanet_example")
        sys.exit(1)
    
    find_images(sys.argv[1])

