# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn as nn
from random import randint, Random
from utils.loss_utils import l1_loss, ssim, cos_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vq_utils import load_2d_language_feature, ResidualVectorQuantizationWithClustering
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt

# 每个元素的绝对值表示使用的轴（1=x,2=y,3=z），符号表示是否翻转
START_PLY_TRANSFORM = [2, 1, -3]
START_PLY_TRANSFORM = [2,1,-3]

def select_train_cameras(scene: Scene, args):
    """
    选择用于训练的相机子集，用于过拟合/调试。
    - 优先使用 --train_subset_first_n（>0）
    - 否则若 --train_subset_ratio < 1.0，则随机抽取固定比例（可用 --train_subset_seed 复现）
    - 否则返回全部训练相机
    """
    train_cams_all = scene.getTrainCameras().copy()
    n_all = len(train_cams_all)
    if n_all == 0:
        return train_cams_all

    first_n = getattr(args, "train_subset_first_n", -1)
    ratio = getattr(args, "train_subset_ratio", 1.0)
    seed = getattr(args, "train_subset_seed", 0)

    if first_n is not None and int(first_n) > 0:
        n = min(int(first_n), n_all)
        subset = train_cams_all[:n]
        print(f"[TrainSubset] using first {n}/{n_all} train cameras")
        return subset

    if ratio is not None and float(ratio) < 1.0:
        r = max(0.0, float(ratio))
        k = int(n_all * r)
        k = max(1, min(k, n_all))
        rng = Random(int(seed))
        idxs = rng.sample(range(n_all), k=k)
        subset = [train_cams_all[i] for i in idxs]
        print(f"[TrainSubset] using random {k}/{n_all} train cameras (ratio={r}, seed={seed})")
        return subset

    print(f"[TrainSubset] using all {n_all} train cameras")
    return train_cams_all

def parse_subset_schedule(schedule_str: str):
    """
    Parse schedule string like: "0:0.1,5000:0.2,10000:0.5,20000:1.0"
    Returns sorted list of (start_iter, ratio).
    """
    if not schedule_str:
        return None
    items = []
    for part in schedule_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid schedule item '{part}', expected 'iter:ratio'")
        it_s, ratio_s = part.split(":", 1)
        it = int(it_s)
        ratio = float(ratio_s)
        items.append((it, ratio))
    items.sort(key=lambda x: x[0])
    return items

def get_ratio_for_iter(schedule, iteration: int, default_ratio: float):
    if not schedule:
        return default_ratio
    ratio = default_ratio
    for it, r in schedule:
        if iteration >= it:
            ratio = r
        else:
            break
    return ratio

def apply_start_ply_transform(gaussians):
    transform = START_PLY_TRANSFORM
    if not transform :
        return
    coords = gaussians._xyz
    perm = [abs(idx) - 1 for idx in transform]
    signs = torch.tensor([1.0 if idx > 0 else -1.0 for idx in transform], device=coords.device)
    # Build the 3x3 axis transform matrix A such that: x' = A x
    A = torch.zeros((3, 3), device=coords.device, dtype=coords.dtype)
    for i in range(3):
        A[i, perm[i]] = signs[i]

    def _mat3_to_quat_wxyz(M: torch.Tensor) -> torch.Tensor:
        """Convert a single 3x3 rotation matrix to a quaternion in (w,x,y,z). Assumes det(M)=+1."""
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            S = torch.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (M[2, 1] - M[1, 2]) / S
            y = (M[0, 2] - M[2, 0]) / S
            z = (M[1, 0] - M[0, 1]) / S
        elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
            S = torch.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
            w = (M[2, 1] - M[1, 2]) / S
            x = 0.25 * S
            y = (M[0, 1] + M[1, 0]) / S
            z = (M[0, 2] + M[2, 0]) / S
        elif M[1, 1] > M[2, 2]:
            S = torch.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
            w = (M[0, 2] - M[2, 0]) / S
            x = (M[0, 1] + M[1, 0]) / S
            y = 0.25 * S
            z = (M[1, 2] + M[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
            w = (M[1, 0] - M[0, 1]) / S
            x = (M[0, 2] + M[2, 0]) / S
            y = (M[1, 2] + M[2, 1]) / S
            z = 0.25 * S
        q = torch.stack([w, x, y, z])
        q = q / (q.norm() + 1e-12)
        return q

    def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product for quaternions in (w,x,y,z). Supports broadcasting."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)

    with torch.no_grad():
        transformed = coords[:, perm] * signs
        gaussians._xyz.data.copy_(transformed)
        # Apply the same axis transform to gaussian orientations:
        # If x' = A x, then rotation matrices should update as R' = A R, i.e. q' = qA ⊗ q
        qA = _mat3_to_quat_wxyz(A).to(device=coords.device, dtype=coords.dtype)
        q = gaussians._rotation.data
        q_new = _quat_mul_wxyz(qA.view(1, 4), q)
        q_new = q_new / (q_new.norm(dim=-1, keepdim=True) + 1e-12)
        gaussians._rotation.data.copy_(q_new)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, start_ply, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    schedule = parse_subset_schedule(getattr(args, "train_subset_schedule", None))
    schedule_ratio = get_ratio_for_iter(schedule, 0, getattr(args, "train_subset_ratio", 1.0))
    args.train_subset_ratio = schedule_ratio
    train_cameras = select_train_cameras(scene, args)
    start_from_ply = False

    if start_ply and not checkpoint:
        # 从 PLY 文件加载基础 3DGS 参数
        print(f"Loading from PLY file: {start_ply}")
        gaussians.load_ply(start_ply, scale_mode=args.start_ply_scale_mode, quat_order=args.start_ply_quat_order)
        # 设置 spatial_lr_scale（从 scene 获取 cameras_extent）
        gaussians.spatial_lr_scale = scene.cameras_extent
        # 初始化 max_radii2D（训练语言特征时不需要，但为了完整性设置）
        if gaussians.max_radii2D.shape[0] != gaussians.get_xyz.shape[0]:
            gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        apply_start_ply_transform(gaussians)
        gaussians.training_setup(opt)
        start_from_ply = True
    else:
        gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint and not start_from_ply:
            raise ValueError("checkpoint or start_ply missing!!!!!")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
    elif start_from_ply:
        first_iter = 0
    
    # Initialize language feature codebooks
    if opt.include_feature and first_iter == 0:
        device = torch.device("cuda")
        features = load_2d_language_feature(dataset.lf_path, device)
        rvq = ResidualVectorQuantizationWithClustering(opt.vq_layer_num, opt.codebook_size, features.shape[1], device).to(device)
        rvq.fit_quantizers(features)
        codebooks = torch.stack(rvq.quantizers, dim=0).to(device)
        with torch.no_grad():
            gaussians._language_feature_codebooks.data.copy_(codebooks)

        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_record = []
    iter_record = []
    smooth_loss = None
    last_schedule_ratio = schedule_ratio
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if schedule:
            schedule_ratio = get_ratio_for_iter(schedule, iteration, args.train_subset_ratio)
            if schedule_ratio != last_schedule_ratio:
                args.train_subset_ratio = schedule_ratio
                train_cameras = select_train_cameras(scene, args)
                viewpoint_stack = None
                last_schedule_ratio = schedule_ratio

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        opt.topk = args.topk
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        image, language_feature_weight_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["language_feature_weight_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        if opt.include_feature:
            # gt_language_feature [512 H W]
            gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level)
            # In this paper, we select layer_num = 1
            layer_num, _, _ = gaussians.get_language_feature_codebooks.shape
            layer_idx = min(int(iteration / 10000 * layer_num), layer_num - 1)
            language_feature = gaussians.compute_layer_feature_map(language_feature_weight_map, layer_idx)
            if args.normalize:
                language_feature = language_feature / (language_feature.norm(dim=0, keepdim=True) + 1e-10)
            loss = 0
            if args.cos_loss:
                cosloss = cos_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)
                loss += cosloss
            if args.l1_loss:
                Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)   
                loss += Ll1

        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()
        
        iter_record.append(iteration)
        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = smooth_loss * 0.99 + loss.item() * 0.01
        loss_record.append(smooth_loss)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if (iteration < opt.iterations) and (iteration % args.accum_iter == 0):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                # if iteration == 10000:
                #     return
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55557)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 6000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100, 6000, 10_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_ply", type=str, default = None)
    parser.add_argument(
        "--start_ply_scale_mode",
        type=str,
        default="log",
        choices=["log", "linear", "auto"],
        help="How to interpret scale_* in --start_ply. log: assume log-scales (default, for in-repo PLY). "
             "linear: assume linear scales and convert to log. auto: heuristic detect and convert when likely linear."
    )
    parser.add_argument(
        "--start_ply_quat_order",
        type=str,
        default="wxyz",
        choices=["wxyz", "xyzw"],
        help="Quaternion component order stored in --start_ply rot_*. wxyz matches this repo; some exporters use xyzw."
    )
    parser.add_argument('--cos_loss', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument(
        "--train_subset_ratio",
        type=float,
        default=1.0,
        help="训练时仅使用训练视角的一个比例子集（<1.0 时生效，如 0.1 表示随机抽取10%）。默认 1.0 表示全部。"
    )
    parser.add_argument(
        "--train_subset_first_n",
        type=int,
        default=-1,
        help="训练时仅使用训练集前 N 个视角（>0 时生效，优先级高于 --train_subset_ratio）。例如 10 表示只用前10个视角。"
    )
    parser.add_argument(
        "--train_subset_seed",
        type=int,
        default=0,
        help="当使用 --train_subset_ratio 随机抽样时的随机种子（保证可复现）。"
    )
    parser.add_argument(
        "--train_subset_schedule",
        type=str,
        default=None,
        help="训练视角比例的阶段调度，格式: '0:0.1,5000:0.2,10000:0.5,20000:1.0'。"
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_ply, args.debug_from, args)
    # All done
    print("\nTraining complete.")
