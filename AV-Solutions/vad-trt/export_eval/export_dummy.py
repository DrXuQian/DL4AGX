# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Export VAD model to ONNX using dummy/fake data (no nuScenes dataset required)

import sys
sys.path.append('')
import numpy as np
import argparse
import os
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings("ignore")

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export VAD to ONNX with dummy data')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-dir', default='scratch', help='output directory for ONNX files')
    args = parser.parse_args()
    return args


def create_dummy_img_metas(batch_size=1, num_cams=6):
    """Create dummy img_metas with required fields."""
    # lidar2img transformation matrices (4x4) for 6 cameras
    # Each camera needs a 4x4 matrix combining rotation, translation and projection
    lidar2img = []
    for i in range(num_cams):
        # Create a proper projection matrix
        mat = np.array([
            [1000.0, 0.0, 320.0, 0.0],
            [0.0, 1000.0, 180.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        lidar2img.append(mat)

    img_metas = []
    for b in range(batch_size):
        meta = {
            'scene_token': 'dummy_scene',
            'can_bus': np.zeros(18, dtype=np.float32),  # [x, y, z, ..., angle]
            'lidar2img': lidar2img,
            'img_shape': [(360, 640, 3)] * num_cams,
            'ori_shape': [(900, 1600, 3)] * num_cams,
            'pad_shape': [(384, 640, 3)] * num_cams,
            'scale_factor': 0.4,
            'flip': False,
            'pcd_horizontal_flip': False,
            'pcd_vertical_flip': False,
            'box_mode_3d': 1,  # LiDARInstance3DBoxes
            'box_type_3d': LiDARInstance3DBoxes,
            'img_norm_cfg': {
                'mean': np.array([123.675, 116.28, 103.53]),
                'std': np.array([58.395, 57.12, 57.375]),
                'to_rgb': True
            },
            'sample_idx': 'dummy_sample',
            'pts_filename': 'dummy.bin',
            'img_filename': ['dummy.jpg'] * num_cams,
        }
        img_metas.append(meta)
    return img_metas


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Import plugin modules
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"Loading plugin: {_module_path}")
            plg_lib = importlib.import_module(_module_path)

    # Build model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    model = model.eval().cuda().float()

    # Create dummy inputs
    # Image: [B, N, C, H, W] where N=6 cameras, after 0.4x scale: 360x640
    batch_size = 1
    num_cams = 6
    img_h, img_w = 360, 640  # After 0.4x scaling from 900x1600
    # Pad to multiple of 32: 384x640
    pad_h, pad_w = 384, 640

    dummy_img = torch.randn(batch_size, num_cams, 3, pad_h, pad_w).cuda()
    dummy_img_metas = create_dummy_img_metas(batch_size, num_cams)

    print(f"Dummy image shape: {dummy_img.shape}")

    # Setup hooks for export
    from bev_deploy.hook import HookHelper, Hook
    from bev_deploy.patch.inspect import AutoInspectHelper
    from bev_deploy.patch.bevformer.patch import (
        patch_spatial_cross_attn_forward,
        patch_point_sampling,
        patch_bevformer_encoder_forward,
        fn_lidar2img,
        fn_canbus
    )
    from patch.patch_head import (
        patch_VADHead_select_and_pad_query,
        patch_VADPerceptionTransformer_get_bev_features,
        patch_VADHead_select_and_pad_pred_map,
        patch_VADHead_forward
    )

    ch = HookHelper()
    ch.attach_hook(model, "vadv1")

    # Prepare img_metas for model
    # Keep lidar2img as list of numpy arrays (expected by encoder.py point_sampling)
    # It will be converted to tensor inside the model

    # Wrap in list for forward_test format
    img_metas_wrapped = [dummy_img_metas]
    img_wrapped = [dummy_img]

    # Dummy ego trajectory inputs
    ego_his_trajs = torch.zeros(batch_size, 1, 2, 2).cuda()  # history trajectory
    ego_fut_trajs = torch.zeros(batch_size, 6, 2).cuda()  # future trajectory
    ego_fut_cmd = torch.zeros(batch_size, 1).long().cuda()  # command
    ego_lcf_feat = torch.zeros(batch_size, 1, 256).cuda()  # LCF features

    # Dummy gt for test mode
    gt_bboxes_3d = [[LiDARInstance3DBoxes(torch.zeros(0, 9).cuda())]]
    gt_labels_3d = [[torch.zeros(0).long().cuda()]]
    gt_attr_labels = [[torch.zeros(0).long().cuda()]]
    fut_valid_flag = [[True]]  # Future valid flag

    print("Running model inference to capture hooks...")

    # Set up capture
    Hook.cache._capture["vadv1.extract_img_feat"] = []
    Hook.cache._capture["vadv1.pts_bbox_head.forward"] = []

    with torch.no_grad():
        with Hook.capture() as _:
            try:
                # Reset prev_frame_info for clean start
                model.prev_frame_info = {
                    'prev_bev': None,
                    'scene_token': None,
                    'prev_pos': 0,
                    'prev_angle': 0,
                }

                result = model.forward_test(
                    img_metas=img_metas_wrapped,
                    img=img_wrapped,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    ego_his_trajs=ego_his_trajs.unsqueeze(0),
                    ego_fut_trajs=ego_fut_trajs.unsqueeze(0),
                    ego_fut_cmd=ego_fut_cmd.unsqueeze(0),
                    ego_lcf_feat=ego_lcf_feat.unsqueeze(0),
                    gt_attr_labels=gt_attr_labels,
                    fut_valid_flag=fut_valid_flag,
                )
                print("Model inference successful!")
            except Exception as e:
                print(f"Error during inference (non-fatal for export): {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with ONNX export since hooks should be captured...")

    # Apply patches for ONNX export
    print("Applying patches for ONNX export...")

    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.layers.0.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.layers.1.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.layers.2.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.point_sampling"]._patch(patch_point_sampling)
    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.forward"]._patch(patch_bevformer_encoder_forward)

    ch.hooks["vadv1.pts_bbox_head.select_and_pad_query"]._patch(patch_VADHead_select_and_pad_query)
    ch.hooks["vadv1.pts_bbox_head.transformer.get_bev_features"]._patch(patch_VADPerceptionTransformer_get_bev_features)
    ch.hooks["vadv1.pts_bbox_head.select_and_pad_pred_map"]._patch(patch_VADHead_select_and_pad_pred_map)
    ch.hooks["vadv1.pts_bbox_head.forward"]._patch(patch_VADHead_forward)

    def fn_fwd(args, kwargs):
        m = args[1]  # img_metas
        m = fn_lidar2img(m)
        m = fn_canbus(args[0][0], m, 100, 100, [0.6, 0.3])
        return (args[0], m), kwargs

    # Export ONNX
    print("Exporting ONNX models...")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Export pts_bbox_head.forward
        ah = AutoInspectHelper(ch.hooks["vadv1.pts_bbox_head.forward"], [fn_fwd])
        ah.export()
        print(f"Exported: vadv1.pts_bbox_head.forward")

        # Export extract_img_feat
        ah = AutoInspectHelper(ch.hooks["vadv1.extract_img_feat"], [])
        ah.export()
        print(f"Exported: vadv1.extract_img_feat")

        print("\nONNX export completed!")
        print(f"Output files are in: {args.output_dir}/")

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
