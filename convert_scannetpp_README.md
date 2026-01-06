# ScanNet++转换脚本使用说明

## 脚本功能

`convert_scannetpp.py` 脚本用于将ScanNet++数据集转换为LangSplatV2所需的格式，包含以下4个步骤：

1. **Step 1**: 复制DSLR图像到`images/`文件夹
2. **Step 2**: 将ScanNet++相机参数转换为COLMAP格式（二进制文件）
3. **Step 3**: 处理训练好的3DGS场景，复制并重命名文件到正确位置
4. **Step 4**: 验证数据结构完整性

## 使用方法

### 基本用法

```bash
python convert_scannetpp.py \
    --scannetpp_path /path/to/scannetpp \
    --scene_name scene_0000_00 \
    --output_path /path/to/output \
    --gs_scene_path /path/to/3dgs/scene
```

### 参数说明

- `--scannetpp_path`: ScanNet++数据集根路径（包含所有场景的目录）
- `--scene_name`: 场景名称（例如：`scene_0000_00`）
- `--output_path`: 输出路径（转换后的数据将保存在此路径下）
- `--gs_scene_path`: 训练好的3DGS场景路径（包含`app_model/`, `point_cloud/`等文件夹）

### 可选参数

- `--skip_step1`: 跳过Step 1（如果图像已复制）
- `--skip_step2`: 跳过Step 2（如果COLMAP文件已生成）
- `--skip_step3`: 跳过Step 3（如果3DGS场景已处理）
- `--skip_step4`: 跳过Step 4（如果不需要验证）

### 示例

```bash
# 完整转换流程
python convert_scannetpp.py \
    --scannetpp_path /data/scannetpp \
    --scene_name scene_0000_00 \
    --output_path /data/langsplatv2 \
    --gs_scene_path /data/3dgs_output/scene_0000_00

# 只执行Step 2和Step 3（假设图像已复制）
python convert_scannetpp.py \
    --scannetpp_path /data/scannetpp \
    --scene_name scene_0000_00 \
    --output_path /data/langsplatv2 \
    --gs_scene_path /data/3dgs_output/scene_0000_00 \
    --skip_step1
```

## 输入数据结构要求

### ScanNet++数据集结构

脚本期望的ScanNet++数据结构：

```
<scannetpp_path>/
└── <scene_name>/
    └── dslr/
        ├── resized_images/      # DSLR图像文件夹（已调整大小）
        │   ├── image_0000.jpg
        │   ├── image_0001.jpg
        │   └── ...
        └── colmap/              # COLMAP重建结果（文本格式）
            ├── cameras.txt      # 相机内参（COLMAP文本格式）
            ├── images.txt       # 相机外参和图像信息（COLMAP文本格式）
            ├── points3D.txt     # 稀疏3D点云（COLMAP文本格式）
            └── database.db      # COLMAP数据库（可选，脚本不使用）
```

**注意**: ScanNet++已经提供了COLMAP格式的文件（文本格式），脚本只需要将其转换为二进制格式。

### COLMAP文件格式

ScanNet++已经提供了COLMAP格式的文件（文本格式），脚本会直接读取这些文件：

#### cameras.txt 格式

COLMAP标准格式，每行包含一个相机的内参：

```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 3840 2160 3000.0 3000.0 1920.0 1080.0
```

#### images.txt 格式

COLMAP标准格式，每两行表示一个图像：

第一行：
```
IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
```

第二行：
```
POINTS2D[] as (X, Y, POINT3D_ID)
```

示例：
```
1 0.5 0.5 0.5 0.5 0.0 0.0 0.0 1 image_0000.jpg
100.0 200.0 1 150.0 250.0 2 ...
```

#### points3D.txt 格式

COLMAP标准格式，每行包含一个3D点：

```
POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
```

示例：
```
1 1.0 2.0 3.0 255 128 64 0.5 1 0 2 1
```

**注意**: 脚本使用项目中的`read_intrinsics_text()`和`read_extrinsics_text()`函数来读取这些文件。

### 3DGS场景结构

训练好的3DGS场景应包含以下结构：

```
<gs_scene_path>/
├── app_model/
│   └── iteration_30000/
│       └── app.pth
├── cameras.json
├── cfg_args
├── input.ply
├── multi_view.json
└── point_cloud/
    └── iteration_30000/
        └── point_cloud.ply
```

## 输出数据结构

转换后的数据结构：

```
<output_path>/
└── <scene_name>/
    ├── images/                    # DSLR图像（Step 1）
    │   ├── image_0000.jpg
    │   └── ...
    ├── sparse/                    # COLMAP稀疏重建（Step 2）
    │   └── 0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── output/                    # 3DGS场景（Step 3）
        └── <scene_name>/
            ├── point_cloud/
            │   └── iteration_30000/
            │       └── point_cloud.ply
            ├── cameras.json
            ├── cfg_args
            ├── chkpnt30000.pth    # 从app.pth重命名
            └── input.ply
```

## 文件映射说明

### Step 3中的文件映射

| 源文件 | 目标文件 | 说明 |
|--------|----------|------|
| `app_model/iteration_30000/app.pth` | `output/<scene_name>/chkpnt30000.pth` | checkpoint文件重命名 |
| `point_cloud/iteration_30000/point_cloud.ply` | `output/<scene_name>/point_cloud/iteration_30000/point_cloud.ply` | 点云文件复制 |
| `cameras.json` | `output/<scene_name>/cameras.json` | 相机JSON文件复制 |
| `cfg_args` | `output/<scene_name>/cfg_args` | 配置文件复制 |
| `input.ply` | `output/<scene_name>/input.ply` | 输入点云复制 |

## 常见问题

### 1. 相机参数文件格式不匹配

**问题**: 如果ScanNet++的相机参数文件格式与脚本期望的不同

**解决方案**: 修改`load_scannetpp_cameras()`函数以匹配实际格式

```python
def load_scannetpp_cameras(intrinsics_path: str, extrinsics_path: str):
    # 根据实际格式修改此函数
    ...
```

### 2. 找不到相机参数文件

**问题**: 脚本提示找不到`intrinsics.txt`或`extrinsics.txt`

**解决方案**: 
- 检查ScanNet++数据集的目录结构
- 确认相机参数文件的实际位置
- 如果路径不同，修改脚本中的路径查找逻辑

### 3. COLMAP格式转换错误

**问题**: 生成的COLMAP文件无法被项目读取

**解决方案**:
- 检查相机参数是否正确
- 确认旋转矩阵的格式（COLMAP使用转置的旋转矩阵）
- 验证四元数转换是否正确

### 4. 3DGS场景文件缺失

**问题**: Step 3中某些文件不存在

**解决方案**:
- 检查3DGS训练输出是否完整
- 确认文件路径正确
- 脚本会显示警告，但不会中断执行

## 验证输出

转换完成后，可以使用以下命令验证：

```bash
# 检查图像数量
ls -1 <output_path>/<scene_name>/images/*.jpg | wc -l

# 检查COLMAP文件
ls -lh <output_path>/<scene_name>/sparse/0/

# 检查3DGS文件
ls -lh <output_path>/<scene_name>/output/<scene_name>/
```

## 下一步

转换完成后，可以按照LangSplatV2的标准流程运行：

1. **预处理**:
   ```bash
   python preprocess.py --dataset_path <output_path>/<scene_name>
   ```

2. **训练**:
   ```bash
   bash train.sh <output_path> <scene_name> 0
   ```

3. **评估**:
   ```bash
   # 需要创建对应的评估脚本
   bash eval_scannetpp.sh <scene_name> 0 10000
   ```

## 注意事项

1. **相机参数格式**: 脚本中的`load_scannetpp_cameras()`函数是基于假设的格式编写的。如果ScanNet++的实际格式不同，需要修改此函数。

2. **空点云**: Step 2生成的`points3D.bin`是空的，因为ScanNet++可能不提供稀疏点云数据。这不影响LangSplatV2的运行，因为项目会使用3DGS的点云。

3. **文件权限**: 确保有足够的权限读取ScanNet++数据和写入输出路径。

4. **磁盘空间**: 确保输出路径有足够的磁盘空间（特别是图像文件）。

## 调试

如果遇到问题，可以：

1. 使用`--skip_step*`参数逐步执行，定位问题
2. 检查脚本输出的错误信息
3. 验证输入数据的格式和路径
4. 查看Step 4的验证结果

## 联系与支持

如果遇到问题或需要修改脚本以适应特定的数据格式，请：
1. 检查ScanNet++数据集的文档
2. 参考COLMAP格式规范
3. 查看LangSplatV2项目的代码和文档
