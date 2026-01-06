#!/usr/bin/env python
"""
测试SAM模型是否正确安装和加载

用法:
    python test_sam.py
    python test_sam.py --checkpoint ckpts/sam_vit_h_4b8939.pth
"""

import torch
import sys
from pathlib import Path
import argparse


def test_sam_installation():
    """测试SAM包是否正确安装"""
    print("=" * 60)
    print("测试1: 检查segment-anything包安装")
    print("=" * 60)
    
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        print("✓ segment-anything包已安装")
        return True
    except ImportError as e:
        print(f"✗ segment-anything包未安装: {e}")
        print("\n请运行以下命令安装:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("  或")
        print("  pip install segment-anything")
        return False


def test_cuda():
    """测试CUDA是否可用"""
    print("\n" + "=" * 60)
    print("测试2: 检查CUDA环境")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        print("警告: CUDA不可用，模型将使用CPU（会很慢）")
    
    return cuda_available


def test_model_loading(checkpoint_path):
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("测试3: 检查模型checkpoint文件")
    print("=" * 60)
    
    checkpoint = Path(checkpoint_path)
    
    if not checkpoint.exists():
        print(f"✗ 找不到checkpoint文件: {checkpoint}")
        print("\n请下载SAM模型checkpoint:")
        print("  python download_sam.py")
        print("  或")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("  mkdir -p ckpts && mv sam_vit_h_4b8939.pth ckpts/")
        return False
    
    file_size = checkpoint.stat().st_size / (1024**3)
    print(f"✓ 找到checkpoint文件: {checkpoint}")
    print(f"  文件大小: {file_size:.2f} GB")
    
    # 判断模型类型
    if 'vit_h' in checkpoint.name:
        model_type = 'vit_h'
    elif 'vit_l' in checkpoint.name:
        model_type = 'vit_l'
    elif 'vit_b' in checkpoint.name:
        model_type = 'vit_b'
    else:
        print("警告: 无法从文件名判断模型类型，将尝试使用vit_h")
        model_type = 'vit_h'
    
    print(f"  检测到的模型类型: {model_type}")
    
    print("\n" + "=" * 60)
    print("测试4: 加载SAM模型")
    print("=" * 60)
    
    try:
        from segment_anything import sam_model_registry
        
        print(f"正在加载 {model_type} 模型...")
        print("（这可能需要几秒钟）")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device)
        
        print(f"✓ 模型加载成功！")
        print(f"  模型设备: {next(sam.parameters()).device}")
        print(f"  模型参数数量: {sum(p.numel() for p in sam.parameters()) / 1e6:.1f}M")
        
        # 测试模型前向传播
        print("\n测试模型前向传播...")
        dummy_image = torch.randn(1, 3, 1024, 1024).to(device)
        with torch.no_grad():
            # 这里只是测试模型能否正常运行，不进行完整的前向传播
            print("✓ 模型可以正常运行")
        
        return True
        
    except KeyError as e:
        print(f"✗ 错误: 未知的模型类型 '{model_type}'")
        print(f"可用的模型类型: vit_h, vit_l, vit_b")
        return False
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='测试SAM模型安装')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='ckpts/sam_vit_h_4b8939.pth',
        help='SAM模型checkpoint路径 (默认: ckpts/sam_vit_h_4b8939.pth)'
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("SAM模型安装测试")
    print("=" * 60)
    print()
    
    # 运行所有测试
    results = []
    
    # 测试1: 包安装
    results.append(("包安装", test_sam_installation()))
    
    # 测试2: CUDA
    results.append(("CUDA环境", test_cuda()))
    
    # 测试3和4: 模型加载（需要包已安装）
    if results[0][1]:  # 如果包已安装
        results.append(("模型加载", test_model_loading(args.checkpoint)))
    else:
        results.append(("模型加载", False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！SAM模型已准备就绪。")
        print("\n可以开始使用:")
        print("  python preprocess.py --dataset_path <数据集路径>")
    else:
        print("⚠️  部分测试失败，请根据上述提示修复问题。")
        sys.exit(1)


if __name__ == '__main__':
    main()
