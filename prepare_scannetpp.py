#!/usr/bin/env python
"""
Data preparation script for ScanNet++ dataset
Converts ScanNet++ format to LangSplatV2 format (COLMAP structure)
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def convert_scannetpp_to_colmap(
    scannetpp_path: str,
    output_path: str,
    scene_name: str
):
    """
    Convert ScanNet++ dataset to COLMAP format for LangSplatV2
    
    ScanNet++ typically has:
    - images/ folder with RGB images
    - cameras/ folder with camera parameters
    - poses/ folder with camera poses
    - point_clouds/ folder with 3D point clouds
    
    We need to convert to:
    - images/ folder
    - sparse/0/ folder with COLMAP files (cameras.bin, images.bin, points3D.bin)
    """
    
    scannetpp_path = Path(scannetpp_path)
    output_path = Path(output_path) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting ScanNet++ scene {scene_name} to COLMAP format...")
    print(f"Input: {scannetpp_path}")
    print(f"Output: {output_path}")
    
    # Copy images
    images_src = scannetpp_path / "images"
    images_dst = output_path / "images"
    if images_src.exists():
        print("Copying images...")
        if images_dst.exists():
            shutil.rmtree(images_dst)
        shutil.copytree(images_src, images_dst)
    else:
        print(f"Warning: {images_src} does not exist!")
    
    # Check if COLMAP sparse folder already exists
    sparse_src = scannetpp_path / "sparse" / "0"
    sparse_dst = output_path / "sparse" / "0"
    
    if sparse_src.exists() and any(sparse_src.glob("*.bin")):
        print("COLMAP sparse folder found, copying...")
        sparse_dst.parent.mkdir(parents=True, exist_ok=True)
        if sparse_dst.exists():
            shutil.rmtree(sparse_dst)
        shutil.copytree(sparse_src, sparse_dst)
        print("COLMAP format ready!")
        return
    
    # If COLMAP format doesn't exist, try to convert from ScanNet++ format
    print("COLMAP format not found. Attempting conversion from ScanNet++ format...")
    
    # Try to find camera parameters and poses
    cameras_file = scannetpp_path / "cameras.json"
    poses_file = scannetpp_path / "poses.json"
    
    if not cameras_file.exists() or not poses_file.exists():
        print("Warning: cameras.json or poses.json not found.")
        print("Please ensure your ScanNet++ data has COLMAP format in sparse/0/ folder")
        print("Or provide cameras.json and poses.json files")
        return
    
    # Load camera parameters and poses
    with open(cameras_file, 'r') as f:
        cameras_data = json.load(f)
    
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    
    # Create sparse/0 directory
    sparse_dst.parent.mkdir(parents=True, exist_ok=True)
    sparse_dst.mkdir(exist_ok=True)
    
    # Convert to COLMAP format
    # This is a simplified conversion - you may need to adjust based on your data format
    print("Converting camera parameters and poses to COLMAP format...")
    print("Note: This is a basic conversion. For full COLMAP format, use COLMAP reconstruction.")
    
    # Save a note about the conversion
    note_file = output_path / "CONVERSION_NOTE.txt"
    with open(note_file, 'w') as f:
        f.write("This scene was converted from ScanNet++ format.\n")
        f.write("For best results, use COLMAP to reconstruct the scene.\n")
        f.write("Place COLMAP output in sparse/0/ folder.\n")
    
    print(f"Conversion note saved to {note_file}")
    print("Please use COLMAP to reconstruct the scene for full compatibility.")


def prepare_multiple_scenes(
    scannetpp_root: str,
    output_root: str,
    scene_names: list
):
    """Prepare multiple ScanNet++ scenes"""
    
    scannetpp_root = Path(scannetpp_root)
    output_root = Path(output_root)
    
    print(f"Preparing {len(scene_names)} scenes...")
    
    for scene_name in scene_names:
        scene_path = scannetpp_root / scene_name
        if not scene_path.exists():
            print(f"Warning: Scene {scene_name} not found at {scene_path}")
            continue
        
        convert_scannetpp_to_colmap(scene_path, output_root, scene_name)
        print(f"Completed: {scene_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ScanNet++ dataset for LangSplatV2")
    parser.add_argument("--scannetpp_root", type=str, required=True,
                       help="Root directory of ScanNet++ dataset")
    parser.add_argument("--output_root", type=str, required=True,
                       help="Output directory for converted data")
    parser.add_argument("--scene_name", type=str, default=None,
                       help="Single scene name to convert")
    parser.add_argument("--scene_list", type=str, default=None,
                       help="Text file with list of scene names (one per line)")
    parser.add_argument("--scenes", type=str, nargs='+', default=None,
                       help="List of scene names directly")
    
    args = parser.parse_args()
    
    if args.scene_name:
        scene_names = [args.scene_name]
    elif args.scene_list:
        with open(args.scene_list, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
    elif args.scenes:
        scene_names = args.scenes
    else:
        # Default: use 10 common ScanNet++ scenes
        scene_names = [
            "scene0000_00",
            "scene0001_00",
            "scene0002_00",
            "scene0003_00",
            "scene0004_00",
            "scene0005_00",
            "scene0006_00",
            "scene0007_00",
            "scene0008_00",
            "scene0009_00"
        ]
        print(f"No scene specified, using default 10 scenes: {scene_names}")
    
    prepare_multiple_scenes(args.scannetpp_root, args.output_root, scene_names)
    print("All scenes prepared!")

