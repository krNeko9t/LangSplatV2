#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import re
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.vq_utils import softmax_to_topk_soft_code

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._language_feature_logits = None
        self._language_feature_codebooks = None
        self._language_feature_weights = None
        self._language_feature_indices = None
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, include_feature=False):
        if include_feature:
            assert self._language_feature_logits is not None, "language feature logits is None"
            assert self._language_feature_codebooks is not None, "language feature codebooks is None"
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature_logits,
                self._language_feature_codebooks,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )            
    
    def restore(self, model_args, training_args, mode='train'):
        if len(model_args) == 14: # for language feature
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._language_feature_logits,
            self._language_feature_codebooks,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif len(model_args) == 12:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            if not training_args.include_feature:
                self.optimizer.load_state_dict(opt_dict)
        
        if mode == 'train':
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
        

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature_logits(self):
        if self._language_feature_logits is not None:
            return self._language_feature_logits
        else:
            raise ValueError('language feature logits is None')
    
    @property
    def get_language_feature_codebooks(self):
        if self._language_feature_codebooks is not None:
            return self._language_feature_codebooks
        else:
            raise ValueError('language feature codebooks is None')
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # language_feature = torch.zeros((fused_point_cloud.shape[0], 512), device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
        # 在从pointcloud初始化的时候是再训练原始gs的时候，这个时候不需要进行feature的初始化
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if training_args.include_feature:
            if self._language_feature_logits is None or self._language_feature_logits.shape[0] != self._xyz.shape[0]:
                # initialize language feature logits and codebooks
                language_feature_logits = torch.zeros((self._xyz.shape[0], training_args.vq_layer_num * training_args.codebook_size), device="cuda")
                language_feature_codebooks = torch.randn((training_args.vq_layer_num, training_args.codebook_size, 512), device="cuda")
                self._language_feature_logits = nn.Parameter(language_feature_logits.requires_grad_(True))
                self._language_feature_codebooks = nn.Parameter(language_feature_codebooks.requires_grad_(True))
                
            l = [
                {'params': [self._language_feature_logits, self._language_feature_codebooks], 
                 'lr': training_args.language_feature_lr, "name": "language_feature"},
            ]
            self._xyz.requires_grad_(False)
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._opacity.requires_grad_(False)
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]
            assert self._language_feature_logits is None and self._language_feature_codebooks is None

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # l.append('language_feature')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, scale_mode: str = "log", quat_order: str = "wxyz"):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Be tolerant to common naming variants: scale_0..2 or scale0..2
        prop_names = [p.name for p in plydata.elements[0].properties]
        scale_names = [n for n in prop_names if re.fullmatch(r"scale_?\d+", n)]
        scale_names = sorted(scale_names, key=lambda x: int(re.findall(r"\d+", x)[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # NOTE:
        # - This codebase stores internal scaling as log-scales (activated via exp in `get_scaling`).
        # - Some external 3DGS exporters store linear scales in PLY. If we load linear scales as log,
        #   exp() will explode and gaussians can look "needle-like"/stretched along one direction.
        # - To avoid affecting in-repo PLYs, scale conversion is controlled by `scale_mode`:
        #   - "log": assume input is already log-scales (default; backward compatible)
        #   - "linear": assume input is linear scales -> convert to log
        #   - "auto": heuristic detect and convert when likely linear
        # - Also be strict about quaternion fields: accept only rot_0..rot_3 to avoid accidentally
        #   picking up unrelated properties starting with "rot".
        if len(scale_names) != 3:
            raise ValueError(f"Expected 3 scale_* properties, got {len(scale_names)}: {scale_names}")

        if scale_mode not in {"log", "linear", "auto"}:
            raise ValueError(f"Invalid scale_mode={scale_mode}. Expected one of: log, linear, auto.")

        if scale_mode in {"linear", "auto"}:
            finite_scales = scales[np.isfinite(scales)]
            median_scale = float(np.median(finite_scales)) if finite_scales.size > 0 else float("nan")
            should_convert = (scale_mode == "linear") or (np.isfinite(median_scale) and median_scale > 0.0)
            if should_convert:
                scales = np.log(np.clip(scales, 1e-12, None))
                if scale_mode == "linear":
                    print("[load_ply] scale_mode=linear: converted linear scales to log-scales.")
                else:
                    print(f"[load_ply] scale_mode=auto: detected likely linear scales (median={median_scale:.4g}); converted to log-scales.")

        rot_names = [n for n in prop_names if re.fullmatch(r"rot_?\d+", n)]
        rot_names = sorted(rot_names, key=lambda x: int(re.findall(r"\d+", x)[-1]))
        if len(rot_names) != 4:
            raise ValueError(f"Expected 4 rot_* properties (rot_0..rot_3), got {len(rot_names)}: {rot_names}")
        rots = np.zeros((xyz.shape[0], 4))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Quaternion order compatibility:
        # - This codebase expects (w, x, y, z) a.k.a. "wxyz"
        # - Some exporters store (x, y, z, w) a.k.a. "xyzw"
        if quat_order not in {"wxyz", "xyzw"}:
            raise ValueError(f"Invalid quat_order={quat_order}. Expected one of: wxyz, xyzw.")
        if quat_order == "xyzw":
            rots = rots[:, [3, 0, 1, 2]]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        print(self._xyz.shape[0])
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        # "language_feature": new_language_feature,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        self._xyz = optimizable_tensors["xyz"]
        # print(self._xyz.shape[0])
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # new_language_feature = self._language_feature[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # new_language_feature = self._language_feature[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def get_render_weights(self, k):
        logits = self._language_feature_logits
        layer_num, codebook_size, _ = self._language_feature_codebooks.shape
        weights = []
        for i in range(layer_num):
            soft_code = softmax_to_topk_soft_code(logits[:, i*codebook_size:(i+1)*codebook_size], k)
            weights.append(soft_code)
        return torch.cat(weights, dim=-1).float()

    @torch.no_grad()
    def get_topk_weights_and_indices(self, k: int):
        """
        从每个Gaussian的logits里取每层top-k的(权重, 索引)。

        - logits: [P, L*K]
        - codebooks: [L, K, D]  (这里D通常是512)
        返回:
        - weights: [P, L, k]   (float32)
        - indices: [P, L, k]   (int64, 每层内0..K-1)
        """
        if self._language_feature_logits is None or self._language_feature_codebooks is None:
            raise ValueError("language feature logits/codebooks is None")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        logits = self._language_feature_logits
        layer_num, codebook_size, _ = self._language_feature_codebooks.shape
        P = logits.shape[0]

        weights_out = []
        indices_out = []
        for l in range(layer_num):
            layer_logits = logits[:, l * codebook_size:(l + 1) * codebook_size]  # [P, K]
            probs = layer_logits.softmax(dim=1)
            w, idx = torch.topk(probs, k, dim=1)  # [P, k], [P, k]
            w = w / (w.sum(dim=1, keepdim=True) + 1e-10)
            weights_out.append(w)
            indices_out.append(idx.to(torch.int64))

        weights = torch.stack(weights_out, dim=1).to(torch.float32)  # [P, L, k]
        indices = torch.stack(indices_out, dim=1)  # [P, L, k]
        assert weights.shape[:2] == (P, layer_num)
        return weights, indices

    def compute_per_gaussian_language_features(self, k: int = 4, accumulate_levels: bool = True, normalize: bool = True):
        """
        计算每个Gaussian自己的语言feature向量（per-GS feature）。

        关键点:
        - 这一步完全在GS维度上完成：feature_i = Σ_l Σ_j w_{i,l,j} * C_{l, idx_{i,l,j}}
        - 如果你未来想“每个GS有自己的codebook”，这一步就是你要的“per-GS feature”定义；
          只不过当前实现里 C 仍然是全局共享的（每层一个codebook）。

        参数:
        - k: 每层top-k（稀疏系数个数）
        - accumulate_levels: True则按层累加（类似 residual VQ 的累加），返回 [P, D]
                             False则返回每层feature，形状 [P, L, D]
        - normalize: 是否做L2归一化（推荐，便于与CLIP做cosine）
        """
        if self._language_feature_codebooks is None:
            raise ValueError("language feature codebooks is None")
        weights, indices = self.get_topk_weights_and_indices(k)  # [P, L, k], [P, L, k]
        codebooks = self._language_feature_codebooks  # [L, K, D]

        # gather每层topk codewords
        # codebooks[l][indices[:,l,:]] -> [P, k, D]
        feats_per_level = []
        for l in range(codebooks.shape[0]):
            C_l = codebooks[l]  # [K, D]
            idx_l = indices[:, l, :]  # [P, k]
            w_l = weights[:, l, :].unsqueeze(-1)  # [P, k, 1]
            codewords = C_l[idx_l]  # [P, k, D]
            feat_l = (w_l * codewords).sum(dim=1)  # [P, D]
            feats_per_level.append(feat_l)

        feats_per_level = torch.stack(feats_per_level, dim=1)  # [P, L, D]
        if accumulate_levels:
            feat = feats_per_level.sum(dim=1)  # [P, D]
            if normalize:
                feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-10)
            return feat
        else:
            if normalize:
                feats_per_level = feats_per_level / (feats_per_level.norm(dim=2, keepdim=True) + 1e-10)
            return feats_per_level

    def compute_per_gaussian_language_features_from_topk(
        self,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        从“已算好的top-k (weights, indices)”直接计算 per-GS 的语言feature。

        适用场景：
        - 推理/可视化阶段你可能只保留了 top-k 权重与索引（比如 `quick_render` 用的那套），
          不想/不能再访问 logits。

        约定：
        - self._language_feature_codebooks: [L, K, D]
        - topk_weights: [P, L*k] 或 [P, k*L]
        - topk_indices: [P, L*k]，索引可以是：
            - “扁平索引” 0..(L*K-1)（推荐，当前 `visualize_lerf.py` 就是这么做的：level_offset + local_idx）
            - 或者 “层内索引” 0..(K-1)（此时你需要自己保证每层不混）

        返回：
        - per_gs_feat: [P, D]
        """
        if self._language_feature_codebooks is None:
            raise ValueError("language feature codebooks is None")
        if topk_weights.ndim != 2 or topk_indices.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got weights={topk_weights.shape}, indices={topk_indices.shape}")
        if topk_weights.shape != topk_indices.shape:
            raise ValueError(f"weights/indices shape mismatch: {topk_weights.shape} vs {topk_indices.shape}")

        codebooks = self._language_feature_codebooks  # [L, K, D]
        L, K, D = codebooks.shape
        codebook_flat = codebooks.reshape(L * K, D)  # [L*K, D]

        # indices 可能是 float（历史代码里有 float），这里统一转 long；若是浮点则四舍五入
        if not torch.is_floating_point(topk_indices):
            idx = topk_indices.to(torch.int64)
        else:
            idx = torch.round(topk_indices).to(torch.int64)

        # 安全裁剪，避免越界（不改变排序语义）
        idx = idx.clamp(min=0, max=L * K - 1)

        # gather codewords: [P, LK, D]
        codewords = codebook_flat[idx]  # [P, LK, D]
        w = topk_weights.to(codewords.dtype).unsqueeze(-1)  # [P, LK, 1]
        feat = (w * codewords).sum(dim=1)  # [P, D]

        if normalize:
            feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-10)
        return feat
    
    def compute_feature_maps(self, language_feature_weight_map):
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(D, -1)
        language_features = []
        layer_num, codebook_size, _ = self._language_feature_codebooks.shape
        for i in range(layer_num):
            language_feature = self.get_language_feature_codebooks[i].T @ language_feature_weight_map[i * codebook_size:(i+1)*codebook_size]
            language_feature = language_feature.view(512, H, W)
            if i > 0:
                language_feature += language_features[-1].detach()
            language_features.append(language_feature)
        return torch.stack(language_features, dim=1)

    def compute_layer_feature_map(self, language_feature_weight_map, layer_idx):
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(D, -1)
        layer_num, codebook_size, _ = self._language_feature_codebooks.shape
        for i in range(layer_idx + 1):
            language_feature = self.get_language_feature_codebooks[i].T @ language_feature_weight_map[i * codebook_size:(i+1)*codebook_size]
            language_feature = language_feature.view(512, H, W)
            if i > 0:
                language_feature += language_feature_before.detach()
            language_feature_before = language_feature
        return language_feature
    
    def compute_final_feature_map(self, language_feature_weight_map):
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(D, -1) 
        language_feature = self.get_language_feature_codebooks.view(-1, 512).T @ language_feature_weight_map
        language_feature = language_feature.view(512, H, W)
        return language_feature