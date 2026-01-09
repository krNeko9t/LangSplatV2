import subprocess
import sys
from pathlib import Path

# ================= 配置区域 =================

# 1. 定义数据的公共根目录 (Root)
# 它可以是绝对路径，也可以是相对于当前脚本的路径 (比如 "..")
# 举例：如果你所有的比如 lerf_ovs, scannetpp 都在上一级目录
DATA_ROOT = Path("/home/bingxing2/ailab/liuyifei/lyj/Dataset/scannetpp/scannetpp/") 

# 2. 定义场景列表 (Scenes)
# 这里只需要填相对于 DATA_ROOT 的路径
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

# ===========================================

def run_preprocess():
    # 检查核心脚本是否存在
    program = Path("preprocess.py")
    if not program.exists():
        print(f"错误：找不到 {program}，请确保脚本在正确的位置。")
        sys.exit(1)

    # 确保根目录是绝对路径，防止 subprocess 混淆
    root_abs = DATA_ROOT.resolve()

    print(f"当前数据根目录设定为: {root_abs}\n")

    for i, sub_path in enumerate(SCENE_LIST):
        # 拼接完整路径: Root + Scene
        full_dataset_path = root_abs / sub_path
        
        print(f"{'='*60}")
        print(f"任务 [{i+1}/{len(SCENE_LIST)}] : 预处理 {sub_path}")
        print(f"完整路径: {full_dataset_path}")
        print(f"{'='*60}")

        if not full_dataset_path.exists():
            print(f"[跳过] 路径不存在: {full_dataset_path}")
            continue

        # 构造命令
        # 对应 args: ["--dataset_path", full_path]
        cmd = [
            sys.executable, "preprocess.py",
            "--dataset_path", str(full_dataset_path)
        ]

        try:
            # check=True 表示如果脚本报错(非0退出)，这里会抛出异常
            subprocess.run(cmd, check=True)
            print(f"\n[成功] {sub_path} 处理完毕。")
        except subprocess.CalledProcessError:
            print(f"\n[失败] {sub_path} 处理过程中发生错误！")
            # 如果希望出错后继续跑下一个，这里保留 continue
            # 如果希望出错直接停止，换成 sys.exit(1)
            continue
        except KeyboardInterrupt:
            print("\n用户手动停止。")
            sys.exit(0)

    print("\n所有预处理任务结束。")

if __name__ == "__main__":
    run_preprocess()