# VAD ONNX Export Guide

This directory contains scripts for exporting VAD (Vectorized Autonomous Driving) model components to ONNX format for TensorRT deployment.

## Prerequisites

1. **Docker Container**: Use the `vad-trt` docker container with the VAD environment set up
2. **VAD Repository**: Clone VAD repo to `/workspace/VAD_repo` inside the container
3. **Model Checkpoint**: Download `VAD_tiny.pth` to `/workspace/VAD_repo/ckpts/`
4. **nuScenes Dataset**: Mount nuScenes dataset (v1.0-mini or full) to `/data`

## Export Scripts

### Option 1: Export with Real Dataset (Recommended for Accuracy Validation)

These scripts use the actual nuScenes dataset for export:

```bash
cd /workspace/DL4AGX/AV-Solutions/vad-trt/export_eval
export PYTHONPATH=.:/workspace/VAD_repo

# Export first frame models (image encoder + detection head without prev_bev)
python export_no_prev.py /workspace/VAD_repo/projects/configs/VAD/VAD_tiny_stage_2.py \
    /workspace/VAD_repo/ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

# Export subsequent frame model (detection head with prev_bev)
python export_prev.py /workspace/VAD_repo/projects/configs/VAD/VAD_tiny_stage_2.py \
    /workspace/VAD_repo/ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp
```

### Option 2: Export with Dummy Data (For Quick Testing)

This script uses synthetic data, useful when nuScenes dataset is not available:

```bash
cd /workspace/DL4AGX/AV-Solutions/vad-trt/export_eval
export PYTHONPATH=.:/workspace/VAD_repo

python export_dummy.py /workspace/VAD_repo/projects/configs/VAD/VAD_tiny_stage_2.py \
    /workspace/VAD_repo/ckpts/VAD_tiny.pth --output-dir scratch
```

## Output Files

After successful export, you will find the following in `scratch/`:

```
scratch/
├── vadv1.extract_img_feat/
│   ├── sim_vadv1.extract_img_feat.onnx    # Simplified ONNX (use this for TensorRT)
│   ├── vadv1.extract_img_feat.onnx        # Original ONNX
│   └── *.bin                               # Input/output tensors for validation
├── vadv1.pts_bbox_head.forward/
│   ├── sim_vadv1.pts_bbox_head.forward.onnx
│   ├── vadv1.pts_bbox_head.forward.onnx
│   └── *.bin
└── vadv1_prev.pts_bbox_head.forward/
    ├── sim_vadv1_prev.pts_bbox_head.forward.onnx
    ├── vadv1_prev.pts_bbox_head.forward.onnx
    └── *.bin
```

### Model Descriptions

| Model | Description | Input | Output |
|-------|-------------|-------|--------|
| `vadv1.extract_img_feat` | Image feature encoder (ResNet backbone + FPN) | 6 camera images | Multi-level features |
| `vadv1.pts_bbox_head.forward` | Detection head (first frame, no temporal context) | Features + metadata | BEV embed, predictions |
| `vadv1_prev.pts_bbox_head.forward` | Detection head (subsequent frames with prev_bev) | Features + prev_bev + metadata | BEV embed, predictions |

## Building TensorRT Engines

After ONNX export, build TensorRT engines using `trtexec`:

```bash
# Set TensorRT path
export TRT_ROOT=<path to your tensorrt dir>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_ROOT/lib
export PATH=$PATH:$TRT_ROOT/bin

# Create output directories
mkdir -p vadv1.extract_img_feat vadv1.pts_bbox_head.forward vadv1_prev.pts_bbox_head.forward

# Build image encoder (FP16)
trtexec --onnx=scratch/vadv1.extract_img_feat/sim_vadv1.extract_img_feat.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --fp16 \
        --saveEngine=vadv1.extract_img_feat/vadv1.extract_img_feat.fp16.engine

# Build detection head (first frame)
trtexec --onnx=scratch/vadv1.pts_bbox_head.forward/sim_vadv1.pts_bbox_head.forward.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --saveEngine=vadv1.pts_bbox_head.forward/vadv1.pts_bbox_head.forward.engine

# Build detection head (with prev_bev)
trtexec --onnx=scratch/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --saveEngine=vadv1_prev.pts_bbox_head.forward/vadv1_prev.pts_bbox_head.forward.engine
```

## Running Benchmark

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_ROOT/lib
python test_tensorrt.py /workspace/VAD_repo/projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp
```

## Troubleshooting

### 1. Missing map expansion files
If you see errors about missing JSON files in `maps/expansion/`, download the nuScenes map expansion:
```bash
# Download from nuScenes website and extract to data/nuscenes/maps/
```

### 2. mmdet3d import error
Apply this fix in `/workspace/VAD_repo/projects/mmdet3d_plugin/core/bbox/structures/lidar_box3d.py`:
```python
# from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet3d.ops import points_in_boxes_all as points_in_boxes_gpu
```

### 3. img_norm_cfg mismatch
Update the config file with correct normalization:
```python
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
```

## References

- [VAD Official Repository](https://github.com/hustvl/VAD)
- [VAD-TensorRT README](../README.md)
