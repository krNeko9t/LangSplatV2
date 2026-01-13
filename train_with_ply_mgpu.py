import subprocess
import sys
import os
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 配置区域 ====================

# 1. 数据根目录
DATA_ROOT = Path("/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse")

# 2. 场景列表
SCENE_LIST = [
    # "027cd6ea0f",
    # "09d6e808b4",
    "0a7cc12c0e",
    # "0b031f3119",
    # "0d8ead0038",
    # "116456116b",
    # "17a5e7d36c",
    # "1cefb55d50",
    # "20871b98f3"
]

# 3. 训练参数配置
INDEX = 0
TOPK = 4
ITERATIONS = 10000
LEVELS = [1, 2, 3]  # 需要跑的 level 顺序

# 4. 可用 GPU 设置 (0-7)
NUM_GPUS = 4
GPU_IDS = list(range(NUM_GPUS))

# ================================================

def run_scene_training(task_args):
    """
    单个场景的完整训练流程（包含 Level 1, 2, 3）
    """
    scene_name, gpu_queue = task_args
    
    # 1. 获取 GPU
    gpu_id = gpu_queue.get()
    
    # 构造一些基础路径
    scene_path = DATA_ROOT / scene_name
    # 你的 start_ply 路径逻辑
    start_ply_path = scene_path / "output" / scene_name / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    
    # 输出目录
    output_model_dir = Path(f"output_pgsr/{scene_name}_{INDEX}")

    # 设置环境变量，只对当前子进程生效
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] 启动场景: {scene_name}")

    try:
        # 检查 start_ply 是否存在，不存在则直接报错跳过，避免浪费时间
        if not start_ply_path.exists():
            raise FileNotFoundError(f"找不到 ply 文件: {start_ply_path}")

        # 2. 顺序执行 Level 1 -> 2 -> 3
        for level in LEVELS:
            print(f"[GPU {gpu_id}] [{scene_name}] 正在运行 Level {level} ...")
            
            cmd = [
                sys.executable, "train.py",
                "-s", str(scene_path),
                "-m", str(output_model_dir),
                "--start_ply", str(start_ply_path),
                "--feature_level", str(level),
                "--vq_layer_num", "1",
                "--codebook_size", "64",
                "--cos_loss",
                "--topk", str(TOPK),
                "--iterations", str(ITERATIONS),
                "-r", "2"
            ]

            # 执行命令
            subprocess.run(cmd, check=True, env=env)
        
        print(f"[GPU {gpu_id}] [完成] 场景 {scene_name} 所有 Level 训练结束。")

    except subprocess.CalledProcessError:
        print(f"[GPU {gpu_id}] [失败] {scene_name} 在 Level {level} 训练报错退出！")
    except Exception as e:
        print(f"[GPU {gpu_id}] [错误] {scene_name}: {e}")
    finally:
        # 3. 无论成功失败，归还 GPU
        gpu_queue.put(gpu_id)

def main():
    # 检查 train.py 是否存在
    if not Path("train.py").exists():
        print("错误：当前目录下找不到 train.py")
        sys.exit(1)

    print(f"数据根目录: {DATA_ROOT}")
    print(f"可用 GPU: {GPU_IDS}")
    print(f"待处理场景: {len(SCENE_LIST)}")
    print(f"每个场景将执行 Levels: {LEVELS}\n")

    # 初始化 GPU 队列
    gpu_queue = queue.Queue()
    for gid in GPU_IDS:
        gpu_queue.put(gid)

    # 准备任务
    tasks = [(scene, gpu_queue) for scene in SCENE_LIST]

    # 并发执行
    # 只要有空闲 GPU，就会开启新线程处理下一个场景
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        futures = [executor.submit(run_scene_training, task) for task in tasks]
        
        for future in as_completed(futures):
            pass # 这里的异常已经在 run_scene_training 内部捕获打印了

    print("\n所有训练任务结束。")

if __name__ == "__main__":
    main()