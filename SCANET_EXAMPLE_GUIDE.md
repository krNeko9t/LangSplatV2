# ScanNet++ 示例数据转换指南

针对您的 `scanet_example` 目录结构，本指南说明如何转换为LangSplatV2格式。

## 您的目录结构

```
scanet_example/
├── dslr/
│   ├── colmap/              # COLMAP重建结果（可能在这里）
│   ├── resized_images/     # 图像文件
│   ├── resized_undistorted_images/  # 去畸变图像（推荐使用）
│   └── ...
├── iphone/
│   ├── colmap/              # COLMAP重建结果（可能在这里）
│   ├── pose_intrinsic_imu.json  # 相机参数和位姿
│   └── ...
└── scans/
    └── ...                  # 语义标注（可选）
```

## 快速转换

### 方法1：使用DSLR数据（推荐）

```bash
python convert_scannetpp_example.py \
    --input_dir /path/to/scanet_example \
    --output_dir ./data/scannetpp \
    --scene_name scanet_example
```

### 方法2：使用iPhone数据

```bash
python convert_scannetpp_example.py \
    --input_dir /path/to/scanet_example \
    --output_dir ./data/scannetpp \
    --scene_name scanet_example \
    --use_iphone
```

## 转换步骤详解

### 步骤1：检查COLMAP重建结果

脚本会自动查找COLMAP重建结果，优先查找：
- `dslr/colmap/sparse/0/` 或 `dslr/colmap/0/`
- `iphone/colmap/sparse/0/` 或 `iphone/colmap/0/`

**如果COLMAP重建结果不存在**，需要先运行COLMAP重建：

```bash
# 进入dslr目录
cd scanet_example/dslr

# 使用resized_undistorted_images（推荐）或resized_images
colmap feature_extractor \
    --database_path database.db \
    --image_path resized_undistorted_images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

colmap exhaustive_matcher \
    --database_path database.db

colmap mapper \
    --database_path database.db \
    --image_path resized_undistorted_images \
    --output_path colmap/sparse
```

### 步骤2：运行转换脚本

```bash
# 使用DSLR数据（推荐，质量更好）
python convert_scannetpp_example.py \
    --input_dir /path/to/scanet_example \
    --output_dir ./data/scannetpp \
    --scene_name scanet_example
```

脚本会：
1. ✅ 查找并复制图像（从 `resized_undistorted_images` 或 `resized_images`）
2. ✅ 查找并复制COLMAP sparse重建结果
3. ✅ 验证输出格式

### 步骤3：验证转换结果

```bash
python check_scannetpp_data.py \
    --data_root ./data/scannetpp \
    --scene scanet_example
```

应该看到：
```
✓ scanet_example: OK
    num_images: 150
    num_cameras: 1
    num_images_colmap: 150
    num_points: 123456
```

## 输出目录结构

转换后会生成：

```
data/scannetpp/
└── scanet_example/
    ├── images/              # 从dslr或iphone复制的图像
    │   ├── frame_000000.jpg
    │   ├── frame_000001.jpg
    │   └── ...
    └── sparse/0/            # COLMAP重建结果
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## 如果COLMAP重建结果不存在

### 选项1：使用COLMAP重建（推荐）

```bash
cd scanet_example/dslr

# 创建colmap目录
mkdir -p colmap/sparse

# 运行COLMAP重建
colmap feature_extractor \
    --database_path colmap/database.db \
    --image_path resized_undistorted_images

colmap exhaustive_matcher \
    --database_path colmap/database.db

colmap mapper \
    --database_path colmap/database.db \
    --image_path resized_undistorted_images \
    --output_path colmap/sparse
```

### 选项2：从iPhone的pose_intrinsic_imu.json转换

如果iPhone数据有 `pose_intrinsic_imu.json`，可以转换为COLMAP格式（需要额外脚本）。

## 常见问题

### Q1: 找不到COLMAP重建结果

**解决**：
1. 检查 `dslr/colmap/` 或 `iphone/colmap/` 目录
2. 如果不存在，运行COLMAP重建（见上面步骤）
3. 确保重建结果在 `sparse/0/` 目录下

### Q2: 找不到图像文件

**解决**：
脚本会按以下顺序查找：
1. `dslr/resized_undistorted_images/` （推荐）
2. `dslr/resized_images/`
3. `dslr/images/`
4. `iphone/images/`

如果都不存在，请检查图像文件的实际位置。

### Q3: 图像文件名不匹配

**解决**：
- 脚本会保持原始文件名
- 如果COLMAP中的图像名称与文件不匹配，需要手动调整
- 可以检查 `sparse/0/images.txt` 或 `sparse/0/images.bin` 中的图像名称

### Q4: 使用哪个数据源？

- **DSLR数据**：通常质量更好，推荐使用
- **iPhone数据**：如果DSLR数据不可用，可以使用

## 完整工作流程

```bash
# 1. 转换数据
python convert_scannetpp_example.py \
    --input_dir /path/to/scanet_example \
    --output_dir ./data/scannetpp \
    --scene_name scanet_example

# 2. 验证数据
python check_scannetpp_data.py \
    --data_root ./data/scannetpp \
    --scene scanet_example

# 3. 预处理（生成语言特征）
python preprocess.py \
    --dataset_path ./data/scannetpp/scanet_example \
    --resolution -1 \
    --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth

# 4. 训练
bash train_scannetpp.sh scanet_example 0
```

## 检查清单

转换前检查：
- [ ] `dslr/resized_undistorted_images/` 或 `dslr/resized_images/` 存在
- [ ] `dslr/colmap/` 或 `iphone/colmap/` 存在
- [ ] COLMAP重建结果包含 `cameras.bin`, `images.bin`, `points3D.bin`

转换后检查：
- [ ] `data/scannetpp/scanet_example/images/` 包含图像
- [ ] `data/scannetpp/scanet_example/sparse/0/` 包含COLMAP文件
- [ ] 运行 `check_scannetpp_data.py` 验证通过

## 下一步

数据转换完成后，继续：
1. 预处理：生成语言特征
2. 训练：训练LangSplatV2模型
3. 评估：提取3D边界框

参考 `SCANNETPP_README.md` 了解详细流程。

