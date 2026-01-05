#!/usr/bin/env python
"""
检查ScanNet++数据准备是否完整
"""
import os
from pathlib import Path
import struct

def read_colmap_bin_header(filepath):
    """读取COLMAP二进制文件头"""
    try:
        with open(filepath, 'rb') as f:
            if 'cameras' in filepath.name:
                num_cameras = struct.unpack('Q', f.read(8))[0]
                return {'num_cameras': num_cameras}
            elif 'images' in filepath.name:
                num_images = struct.unpack('Q', f.read(8))[0]
                return {'num_images': num_images}
            elif 'points3D' in filepath.name:
                num_points = struct.unpack('Q', f.read(8))[0]
                return {'num_points': num_points}
    except Exception as e:
        return {'error': str(e)}
    return {}

def check_scene(scene_path):
    """检查单个场景的数据完整性"""
    scene_path = Path(scene_path)
    if not scene_path.exists():
        return {'valid': False, 'errors': [f'Scene path does not exist: {scene_path}']}
    
    errors = []
    warnings = []
    info = {}
    
    # 检查images文件夹
    images_dir = scene_path / 'images'
    if not images_dir.exists():
        errors.append('Missing images/ folder')
    else:
        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        if len(images) == 0:
            errors.append('No images found in images/ folder')
        else:
            info['num_images'] = len(images)
            # 检查图像命名
            sample_names = [img.name for img in images[:5]]
            info['sample_image_names'] = sample_names
    
    # 检查sparse文件夹
    sparse_dir = scene_path / 'sparse' / '0'
    if not sparse_dir.exists():
        errors.append('Missing sparse/0/ folder')
    else:
        required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        for f in required_files:
            filepath = sparse_dir / f
            if not filepath.exists():
                errors.append(f'Missing {f}')
            else:
                # 读取文件头信息
                header = read_colmap_bin_header(filepath)
                if 'error' in header:
                    errors.append(f'Cannot read {f}: {header["error"]}')
                else:
                    info.update(header)
    
    # 检查语言特征（可选，训练前需要）
    lf_dir = scene_path / 'language_features'
    if not lf_dir.exists():
        warnings.append('language_features/ folder not found (will be generated during preprocessing)')
    else:
        lf_files = list(lf_dir.glob('*_f.npy'))
        info['num_language_features'] = len(lf_files)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'info': info
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check ScanNet++ data preparation')
    parser.add_argument('--data_root', type=str, default='./data/scannetpp',
                       help='Root directory of prepared data')
    parser.add_argument('--scene', type=str, default=None,
                       help='Check single scene (optional)')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if args.scene:
        scenes = [data_root / args.scene]
    else:
        scenes = [d for d in data_root.iterdir() if d.is_dir()]
    
    print(f"Checking {len(scenes)} scene(s)...\n")
    
    all_valid = True
    for scene_path in scenes:
        result = check_scene(scene_path)
        scene_name = scene_path.name
        
        if result['valid']:
            print(f"✓ {scene_name}: OK")
            if result['info']:
                for key, value in result['info'].items():
                    print(f"    {key}: {value}")
        else:
            print(f"✗ {scene_name}: FAILED")
            for error in result['errors']:
                print(f"    ERROR: {error}")
            all_valid = False
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"    WARNING: {warning}")
        
        print()
    
    if all_valid:
        print("All scenes are ready for training!")
    else:
        print("Some scenes have errors. Please fix them before training.")
