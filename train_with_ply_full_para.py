import subprocess
import sys
import os
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 配置区域 ====================

DATA_ROOT = Path("/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse")

SCENE_LIST = [
    # "027cd6ea0f",
    # "09d6e808b4",
    # "0a7cc12c0e",
    "0b031f3119",
    # "0d8ead0038",
    "116456116b",
    # "17a5e7d36c",
    # "1cefb55d50",
    # "20871b98f3"
]

# 训练参数
INDEX = 0
TOPK = 4
ITERATIONS = 30000
LEVELS = [1, 2, 3]  # 所有需要跑的 level

# 显卡设置
NUM_GPUS = 4
GPU_IDS = list(range(NUM_GPUS))
# GPU_IDS = [2,3,4,5]

# ================================================

def run_single_task(task_args):
    """
    最小任务单元：跑一个场景的一个 Level
    """
    scene_name, level, gpu_queue = task_args
    
    # 1. 申请 GPU
    gpu_id = gpu_queue.get()
    
    # 构造路径
    scene_path = DATA_ROOT / scene_name
    start_ply_path = scene_path / "output" / scene_name / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    
    # 注意：如果多个 Level 同时往同一个文件夹写 checkpoint/log，可能会有冲突。
    # 如果你的代码内部没有处理子文件夹，建议在这里把 output_dir 区分开，比如加后缀
    # output_model_dir = Path(f"output_pgsr/{scene_name}_{INDEX}_L{level}") 
    output_model_dir = Path(f"output_pgsr/{scene_name}_{INDEX}")

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    task_id = f"[{scene_name}|L{level}]"
    print(f"[GPU {gpu_id}] 启动任务 {task_id}")

    cmd = [
        sys.executable, "train_nogui.py",
        "-s", str(scene_path),
        "-m", str(output_model_dir),
        "--start_ply", str(start_ply_path),
        "--feature_level", str(level),
        "--vq_layer_num", "1",
        "--codebook_size", "64",
        "--cos_loss",
        "--topk", str(TOPK),
        "--iterations", str(ITERATIONS),
        # "-r", "2"
    ]

    start_checkpoint_path = Path(f"output_pgsr/{scene_name}_{INDEX}_{level}") / "chkpnt10000.pth"
    print(f"[GPU {gpu_id}] [检查点文件] {start_checkpoint_path}")
    if start_checkpoint_path.exists():
        print(f"[GPU {gpu_id}] [使用] {task_id} 检查点文件")
        cmd.append("--start_checkpoint")
        cmd.append(str(start_checkpoint_path))

    try:
        # 检查 ply
        if not start_ply_path.exists():
            print(f"[GPU {gpu_id}] [跳过] {task_id} 找不到 ply 文件")
            return

        subprocess.run(cmd, check=True, env=env)
        print(f"[GPU {gpu_id}] [完成] {task_id}")

    except subprocess.CalledProcessError:
        print(f"[GPU {gpu_id}] [失败] {task_id} 训练报错退出！")
    except Exception as e:
        print(f"[GPU {gpu_id}] [错误] {task_id}: {e}")
    finally:
        # 2. 必须归还 GPU
        gpu_queue.put(gpu_id)

def main():
    if not Path("train.py").exists():
        print("错误：找不到 train.py")
        sys.exit(1)

    # 初始化 GPU 队列
    gpu_queue = queue.Queue()
    for gid in GPU_IDS:
        gpu_queue.put(gid)

    # 3. 展开所有任务 (Flatten)
    # 任务列表：[(场景A, L1), (场景A, L2), (场景A, L3), (场景B, L1)...]
    all_tasks = []
    for scene in SCENE_LIST:
        for level in LEVELS:
            all_tasks.append((scene, level, gpu_queue))

    print(f"总任务数: {len(all_tasks)} (Scenes: {len(SCENE_LIST)} * Levels: {len(LEVELS)})")
    print(f"并发数: {len(GPU_IDS)}")
    print("开始执行...\n")

    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        futures = [executor.submit(run_single_task, task) for task in all_tasks]
        
        for future in as_completed(futures):
            pass

    print("\n所有训练任务结束。")

if __name__ == "__main__":
    main()