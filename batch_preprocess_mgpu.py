import subprocess
import sys
import os
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================

DATA_ROOT = Path("/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse")

SCENE_LIST = [
    "027cd6ea0f",
    "09d6e808b4",
    "0a7cc12c0e",
    "0b031f3119",
    "0d8ead0038",
    "116456116b",
    "17a5e7d36c",
    "1cefb55d50",
    "20871b98f3"
]

# 设置可用的 GPU 数量 (这里设为 8)
NUM_GPUS = 8
# 如果你想指定具体的卡号，比如只用后4张，可以改成: [4, 5, 6, 7]
GPU_IDS = list(range(NUM_GPUS)) 

# ===========================================

def process_single_scene(scene_data):
    """
    单个场景的处理函数，由线程池调用
    """
    scene_rel_path, gpu_queue, root_abs = scene_data
    
    # 1. 申请 GPU (如果没有空闲的，这里会阻塞等待)
    gpu_id = gpu_queue.get()
    
    try:
        full_dataset_path = root_abs / scene_rel_path
        
        # 打印信息时加上 GPU ID，方便查看
        print(f"[GPU {gpu_id}] 开始处理: {scene_rel_path}")

        if not full_dataset_path.exists():
            print(f"[GPU {gpu_id}] [跳过] 路径不存在: {full_dataset_path}")
            return

        # 2. 设置环境变量，只让子进程看到申请到的这块 GPU
        current_env = os.environ.copy()
        current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 构造命令
        cmd = [
            sys.executable, "preprocess.py",
            "--dataset_path", str(full_dataset_path),
            # "-r", "2"
        ]

        # 3. 执行命令
        # capture_output=False 让子进程的输出直接打印到屏幕，
        # 如果嫌输出太乱，可以设为 True 并记录到日志文件
        subprocess.run(cmd, check=True, env=current_env)
        print(f"[GPU {gpu_id}] [成功] {scene_rel_path} 处理完毕。")

    except subprocess.CalledProcessError:
        print(f"[GPU {gpu_id}] [失败] {scene_rel_path} 处理出错！")
    except Exception as e:
        print(f"[GPU {gpu_id}] [异常] {scene_rel_path}: {e}")
    finally:
        # 4. 关键：无论成功失败，必须归还 GPU，否则后面的任务会死锁
        gpu_queue.put(gpu_id)

def run_preprocess_parallel():
    program = Path("preprocess.py")
    if not program.exists():
        print(f"错误：找不到 {program}")
        sys.exit(1)

    root_abs = DATA_ROOT.resolve()
    print(f"数据根目录: {root_abs}")
    print(f"可用 GPU 列表: {GPU_IDS}")
    print(f"待处理场景数: {len(SCENE_LIST)}\n")

    # 初始化 GPU 队列
    gpu_queue = queue.Queue()
    for gid in GPU_IDS:
        gpu_queue.put(gid)

    # 准备任务参数
    tasks = []
    for sub_path in SCENE_LIST:
        tasks.append((sub_path, gpu_queue, root_abs))

    # 开启线程池
    # max_workers 设为 GPU 数量，保证同一时间最多有 NUM_GPUS 个任务在跑
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_scene, task) for task in tasks]
        
        # 等待所有任务完成
        for future in as_completed(futures):
            # 这里可以处理任务的返回值或异常，目前逻辑都在 process_single_scene 里处理了
            pass

    print("\n所有并发任务结束。")

if __name__ == "__main__":
    run_preprocess_parallel()