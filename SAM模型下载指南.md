# SAM (Segment Anything Model) 模型下载和使用指南

## 一、SAM模型加载原理

### 代码解析

```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 加载SAM模型
sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
```

**工作原理：**
1. `sam_model_registry` 是一个字典，包含不同SAM模型类型的构建函数
2. `"vit_h"` 表示使用 ViT-H (Vision Transformer Huge) 模型，这是最大的模型，性能最好
3. `checkpoint=sam_ckpt_path` 指定预训练权重文件的路径
4. `.to('cuda')` 将模型加载到GPU上

### 可用的模型类型

SAM提供了3种模型大小：

| 模型类型 | 参数量 | 性能 | 推荐场景 |
|---------|--------|------|---------|
| `vit_h` | 636M | 最高 | 高质量分割（默认） |
| `vit_l` | 308M | 较高 | 平衡性能和速度 |
| `vit_b` | 91M | 较快 | 快速推理，显存受限 |

## 二、安装segment-anything包

### 方法1: 使用项目子模块（推荐）

项目已经包含了 `segment-anything-langsplat` 子模块，在安装环境时会自动安装：

```bash
# 如果还没有克隆子模块
git submodule update --init --recursive

# 安装环境（会自动安装子模块）
conda env create --file environment.yml
conda activate langsplat_v2
```

### 方法2: 直接安装segment-anything

```bash
# 激活环境
conda activate langsplat_v2

# 安装segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

或者安装特定版本：

```bash
pip install segment-anything
```

## 三、下载SAM模型权重

### 方法1: 手动下载（推荐）

#### 步骤1: 创建checkpoint目录

```bash
mkdir -p ckpts
cd ckpts
```

#### 步骤2: 下载权重文件

**ViT-H模型（推荐，代码默认使用）：**
```bash
# 使用wget下载
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 或使用curl下载
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**ViT-L模型（备选）：**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

**ViT-B模型（备选，显存受限时使用）：**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

#### 步骤3: 验证文件

```bash
# 检查文件大小（ViT-H应该约2.4GB）
ls -lh sam_vit_h_4b8939.pth
```

### 方法2: 使用Python脚本下载

创建一个下载脚本 `download_sam.py`：

```python
import os
import urllib.request
from pathlib import Path

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
    
    # 创建目录
    ckpt_dir = Path('ckpts')
    ckpt_dir.mkdir(exist_ok=True)
    
    url = model_urls[model_type]
    filename = filenames[model_type]
    filepath = ckpt_dir / filename
    
    # 如果文件已存在，跳过下载
    if filepath.exists():
        print(f"文件已存在: {filepath}")
        return str(filepath)
    
    print(f"正在下载 {model_type} 模型...")
    print(f"URL: {url}")
    print(f"保存到: {filepath}")
    
    # 下载文件
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r进度: {percent}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\n下载完成: {filepath}")
        print(f"文件大小: {filepath.stat().st_size / (1024**3):.2f} GB")
        return str(filepath)
    except Exception as e:
        print(f"\n下载失败: {e}")
        return None

if __name__ == '__main__':
    # 下载ViT-H模型（默认）
    download_sam_checkpoint('vit_h')
    
    # 如果需要其他模型，取消注释：
    # download_sam_checkpoint('vit_l')
    # download_sam_checkpoint('vit_b')
```

运行脚本：

```bash
python download_sam.py
```

## 四、使用SAM模型

### 在preprocess.py中使用

代码默认路径是 `ckpts/sam_vit_h_4b8939.pth`，如果文件在这个位置，直接运行：

```bash
python preprocess.py --dataset_path <数据集路径>
```

### 指定自定义路径

如果checkpoint在其他位置，使用 `--sam_ckpt_path` 参数：

```bash
python preprocess.py \
    --dataset_path <数据集路径> \
    --sam_ckpt_path /path/to/your/sam_vit_h_4b8939.pth
```

### 使用不同的模型类型

如果要使用ViT-L或ViT-B模型（显存更小），需要修改代码：

```python
# 在preprocess.py中，将第362行改为：
sam = sam_model_registry["vit_l"](checkpoint="ckpts/sam_vit_l_0b3195.pth").to('cuda')
# 或
sam = sam_model_registry["vit_b"](checkpoint="ckpts/sam_vit_b_01ec64.pth").to('cuda')
```

## 五、验证安装

创建一个测试脚本 `test_sam.py`：

```python
import torch
from segment_anything import sam_model_registry

# 测试加载模型
print("测试SAM模型加载...")

# 检查CUDA
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")

# 尝试加载模型
try:
    sam = sam_model_registry["vit_h"](checkpoint="ckpts/sam_vit_h_4b8939.pth")
    print("✓ SAM模型加载成功！")
    print(f"模型设备: {next(sam.parameters()).device}")
except FileNotFoundError:
    print("✗ 错误: 找不到checkpoint文件")
    print("请确保文件存在于: ckpts/sam_vit_h_4b8939.pth")
except Exception as e:
    print(f"✗ 错误: {e}")
```

运行测试：

```bash
python test_sam.py
```

## 六、常见问题

### 1. 文件下载失败

**问题**: 网络连接问题导致下载失败

**解决方案**:
- 使用代理或VPN
- 使用国内镜像（如果有）
- 手动从浏览器下载：访问 https://github.com/facebookresearch/segment-anything#model-checkpoints

### 2. 显存不足

**问题**: ViT-H模型需要较大显存（约4GB）

**解决方案**:
- 使用ViT-L模型（约2.4GB显存）
- 使用ViT-B模型（约1.2GB显存）
- 修改代码使用CPU（不推荐，会很慢）

### 3. 找不到checkpoint文件

**问题**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决方案**:
```bash
# 检查文件是否存在
ls -lh ckpts/sam_vit_h_4b8939.pth

# 如果不存在，确保路径正确
python preprocess.py --dataset_path <路径> --sam_ckpt_path <完整路径>
```

### 4. 模型加载很慢

**问题**: 首次加载模型需要时间

**说明**: 这是正常的，SAM模型较大，加载需要几秒钟。后续使用会更快。

## 七、文件结构

下载后的目录结构应该是：

```
LangSplatV2/
├── ckpts/
│   └── sam_vit_h_4b8939.pth  (约2.4GB)
├── preprocess.py
└── ...
```

## 八、官方资源链接

- **SAM GitHub**: https://github.com/facebookresearch/segment-anything
- **模型checkpoint下载页面**: https://github.com/facebookresearch/segment-anything#model-checkpoints
- **SAM论文**: https://arxiv.org/abs/2304.02643

## 九、快速开始命令总结

```bash
# 1. 创建checkpoint目录
mkdir -p ckpts
cd ckpts

# 2. 下载ViT-H模型（推荐）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 3. 返回项目根目录
cd ..

# 4. 验证安装
python test_sam.py

# 5. 使用SAM进行预处理
python preprocess.py --dataset_path <你的数据集路径>
```

## 十、模型文件大小参考

| 模型 | 文件大小 | 显存需求（推理） | 显存需求（训练） |
|------|---------|----------------|----------------|
| sam_vit_h_4b8939.pth | ~2.4 GB | ~4 GB | ~8 GB |
| sam_vit_l_0b3195.pth | ~1.2 GB | ~2.4 GB | ~5 GB |
| sam_vit_b_01ec64.pth | ~375 MB | ~1.2 GB | ~3 GB |

**注意**: 预处理阶段只使用推理模式，所以显存需求相对较小。
