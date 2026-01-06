#!/usr/bin/env python
"""
SAM模型checkpoint下载脚本

用法:
    python download_sam.py              # 下载默认的ViT-H模型
    python download_sam.py --model vit_l # 下载ViT-L模型
    python download_sam.py --model vit_b # 下载ViT-B模型
    python download_sam.py --all         # 下载所有模型
"""

import os
import sys
import urllib.request
from pathlib import Path
import argparse


def download_file(url, filepath, desc=None):
    """下载文件并显示进度"""
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, int(count * block_size * 100 / total_size))
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            size_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f'\r[{bar}] {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()
    
    try:
        if desc:
            print(f"正在下载: {desc}")
        print(f"URL: {url}")
        print(f"保存到: {filepath}")
        
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\n✓ 下载完成: {filepath}")
        file_size = filepath.stat().st_size / (1024**3)
        print(f"  文件大小: {file_size:.2f} GB\n")
        return True
    except KeyboardInterrupt:
        print("\n\n下载被用户中断")
        if filepath.exists():
            filepath.unlink()
        return False
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_sam_checkpoint(model_type='vit_h'):
    """下载SAM模型checkpoint"""
    
    # 模型URL映射
    model_urls = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    }
    
    # 文件名映射
    filenames = {
        'vit_h': 'sam_vit_h_4b8939.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_b': 'sam_vit_b_01ec64.pth',
    }
    
    # 模型描述
    model_descs = {
        'vit_h': 'SAM ViT-H (最大模型，性能最好)',
        'vit_l': 'SAM ViT-L (中等模型，平衡性能和速度)',
        'vit_b': 'SAM ViT-B (小模型，快速推理)',
    }
    
    if model_type not in model_urls:
        print(f"错误: 未知的模型类型 '{model_type}'")
        print(f"可用的模型类型: {', '.join(model_urls.keys())}")
        return False
    
    # 创建目录
    ckpt_dir = Path('ckpts')
    ckpt_dir.mkdir(exist_ok=True)
    
    url = model_urls[model_type]
    filename = filenames[model_type]
    filepath = ckpt_dir / filename
    
    # 如果文件已存在，询问是否覆盖
    if filepath.exists():
        file_size = filepath.stat().st_size / (1024**2)
        print(f"文件已存在: {filepath}")
        print(f"文件大小: {file_size:.2f} MB")
        response = input("是否重新下载? (y/N): ").strip().lower()
        if response != 'y':
            print("跳过下载\n")
            return True
        filepath.unlink()
    
    # 下载文件
    desc = model_descs[model_type]
    return download_file(url, filepath, desc)


def main():
    parser = argparse.ArgumentParser(
        description='下载SAM模型checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_sam.py              # 下载ViT-H模型（默认）
  python download_sam.py --model vit_l # 下载ViT-L模型
  python download_sam.py --model vit_b # 下载ViT-B模型
  python download_sam.py --all         # 下载所有模型
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['vit_h', 'vit_l', 'vit_b'],
        default='vit_h',
        help='要下载的模型类型 (默认: vit_h)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='下载所有模型'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM模型checkpoint下载工具")
    print("=" * 60)
    print()
    
    if args.all:
        # 下载所有模型
        models = ['vit_h', 'vit_l', 'vit_b']
        print(f"将下载 {len(models)} 个模型\n")
        success_count = 0
        for model_type in models:
            if download_sam_checkpoint(model_type):
                success_count += 1
            print("-" * 60)
        
        print(f"\n完成: 成功下载 {success_count}/{len(models)} 个模型")
    else:
        # 下载指定模型
        if download_sam_checkpoint(args.model):
            print("下载成功！")
            print(f"\n使用方法:")
            print(f"  python preprocess.py --dataset_path <数据集路径>")
            print(f"    或")
            print(f"  python preprocess.py --dataset_path <数据集路径> \\")
            print(f"    --sam_ckpt_path ckpts/{args.model.replace('_', '_')}.pth")
        else:
            print("下载失败，请检查网络连接或手动下载")
            sys.exit(1)


if __name__ == '__main__':
    main()
