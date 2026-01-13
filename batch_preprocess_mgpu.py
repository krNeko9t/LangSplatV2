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
    "027cd6ea0f", # doing
    "09d6e808b4",
    # "0a7cc12c0e", # done
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

# 失败重试配置
# - MAX_RETRY: 单个 scene 最多尝试次数（包含第一次）
# - RETRY_SLEEP_SEC: 每次失败后的等待时间（秒）；可结合 RETRY_BACKOFF 做指数退避
# - RETRY_BACKOFF: 退避倍率；=1 表示固定间隔
MAX_RETRY = 5
RETRY_SLEEP_SEC = 30
RETRY_BACKOFF = 1.5

# 日志配置：每个 scene 一个 log 文件（包含子进程 stdout/stderr）
LOG_DIR = Path("./logs/preprocess_mgpu")

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

        # 为每个 scene 准备独立日志文件（追加写入，支持多次重试）
        # 用 scene_rel_path 作为文件名时，先做简单的安全替换，避免路径分隔符
        safe_scene_name = str(scene_rel_path).replace("/", "__")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = (LOG_DIR / f"{safe_scene_name}.log").resolve()

        if not full_dataset_path.exists():
            print(f"[GPU {gpu_id}] [跳过] 路径不存在: {full_dataset_path}")
            try:
                with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                    logf.write(f"[SKIP] path not found: {full_dataset_path}\n")
            except Exception:
                pass
            return {"scene": scene_rel_path, "status": "skipped", "attempts": 0}

        # 2. 设置环境变量，只让子进程看到申请到的这块 GPU
        current_env = os.environ.copy()
        current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 构造命令
        cmd = [
            sys.executable, "preprocess.py",
            "--dataset_path", str(full_dataset_path),
            # "--max_images", "2",
            # "--sam_autocast", "bf16"
            # "-r", "2"
        ]

        # 3. 执行命令（失败自动重试，直到 MAX_RETRY）
        # capture_output=False 让子进程的输出直接打印到屏幕，
        # 如果嫌输出太乱，可以设为 True 并记录到日志文件
        sleep_s = float(RETRY_SLEEP_SEC)
        last_err = None
        with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
            logf.write(f"\n===== START scene={scene_rel_path} gpu={gpu_id} time={time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            logf.write(f"cmd: {' '.join(cmd)}\n")

            for attempt in range(1, int(MAX_RETRY) + 1):
                try:
                    print(f"[GPU {gpu_id}] 尝试 {attempt}/{MAX_RETRY}: {scene_rel_path} -> log: {log_path}")
                    logf.write(f"\n--- attempt {attempt}/{MAX_RETRY} time={time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    # 子进程输出全部写入该 scene 的 log 文件
                    subprocess.run(
                        cmd,
                        check=True,
                        env=current_env,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    logf.write(f"\n[OK] attempt {attempt}/{MAX_RETRY}\n")
                    print(f"[GPU {gpu_id}] [成功] {scene_rel_path} 处理完毕。")
                    return {"scene": scene_rel_path, "status": "success", "attempts": attempt, "log": str(log_path)}
                except subprocess.CalledProcessError as e:
                    last_err = e
                    logf.write(f"\n[FAIL] CalledProcessError: returncode={getattr(e, 'returncode', None)}\n")
                    if attempt >= int(MAX_RETRY):
                        break
                    print(
                        f"[GPU {gpu_id}] [失败] {scene_rel_path} (attempt {attempt}/{MAX_RETRY})，"
                        f"{int(sleep_s)}s 后重试... -> log: {log_path}"
                    )
                    logf.write(f"[RETRY] sleep {sleep_s}s\n")
                    time.sleep(sleep_s)
                    sleep_s *= float(RETRY_BACKOFF)
                except Exception as e:
                    last_err = e
                    logf.write(f"\n[EXCEPTION] {repr(e)}\n")
                    if attempt >= int(MAX_RETRY):
                        break
                    print(
                        f"[GPU {gpu_id}] [异常] {scene_rel_path}: {e} (attempt {attempt}/{MAX_RETRY})，"
                        f"{int(sleep_s)}s 后重试... -> log: {log_path}"
                    )
                    logf.write(f"[RETRY] sleep {sleep_s}s\n")
                    time.sleep(sleep_s)
                    sleep_s *= float(RETRY_BACKOFF)

            logf.write(f"\n===== END scene={scene_rel_path} status=FAILED time={time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

        # 用尽重试次数
        print(f"[GPU {gpu_id}] [最终失败] {scene_rel_path}，已达最大重试次数 MAX_RETRY={MAX_RETRY}")
        return {
            "scene": scene_rel_path,
            "status": "failed",
            "attempts": int(MAX_RETRY),
            "error": repr(last_err),
            "log": str(log_path),
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] [异常] {scene_rel_path}: {e}")
        return {"scene": scene_rel_path, "status": "failed", "attempts": 0, "error": repr(e)}
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
    print(f"日志目录: {LOG_DIR.resolve()}\n")

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
        results = []
        for future in as_completed(futures):
            # 这里可以处理任务的返回值或异常，目前逻辑都在 process_single_scene 里处理了
            try:
                results.append(future.result())
            except Exception as e:
                # 理论上 process_single_scene 已经吞掉大多数异常并返回 dict，
                # 这里兜底避免线程池直接报错导致统计缺失
                results.append({"scene": "<unknown>", "status": "failed", "attempts": 0, "error": repr(e)})

    # 汇总
    success = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    skipped = [r for r in results if isinstance(r, dict) and r.get("status") == "skipped"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]

    print("\n所有并发任务结束。")
    print(f"成功: {len(success)}，跳过: {len(skipped)}，失败: {len(failed)}")
    if failed:
        print("最终失败的场景列表（达到 MAX_RETRY 仍失败）：")
        for r in failed:
            print(
                f"  - {r.get('scene')} (attempts={r.get('attempts')}) "
                f"err={r.get('error')} log={r.get('log')}"
            )
        # 让上层调度/脚本更容易感知失败
        sys.exit(2)

if __name__ == "__main__":
    run_preprocess_parallel()