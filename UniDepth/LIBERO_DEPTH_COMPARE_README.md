# LIBERO-10 Depth Comparison (UniDepthV2 vs Simulator GT)

This guide shows a new user how to:

1. install prerequisites,
2. create and activate a virtual environment,
3. prepare LIBERO rollout videos and GT depth files,
4. run UniDepthV2 depth estimation and compare against GT.

The comparison script is:

- `scripts/libero_video_depth_compare.py`

It expects RGB rollout videos and matching `*_depth.npz` files.

## 1) Prerequisites

- Linux (tested in this repo with Python 3.10+)
- NVIDIA GPU + CUDA-enabled PyTorch is recommended (CPU also works, but slower)
- `ffmpeg` (optional, useful for re-encoding videos)

Install system dependencies (Ubuntu/WSL example):

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg
```

## 2) Clone and create virtual environment

From the repo root:

```bash
cd /path/to/UniDepth
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

Set `PYTHONPATH` (recommended for running scripts directly):

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## 3) Data download and expected layout

You need LIBERO rollout RGB videos and their matching depth files.

The script expects each task to have:

- `rollout_<task>_success.mp4` (agent view)
- `rollout_<task>_success_wrist.mp4` (wrist view, optional but recommended)
- `rollout_<task>_success_depth.npz` (GT depth arrays)

Place them in one directory (default used by the script is `unidepth/videos/`):

```text
unidepth/videos/
  rollout_xxx_success.mp4
  rollout_xxx_success_wrist.mp4
  rollout_xxx_success_depth.npz
  ...
```

`*_depth.npz` should contain keys like:

- `agentview_depth` with shape `(T, H, W, 1)` or `(T, H, W)`
- `wrist_depth` with shape `(T, H, W, 1)` or `(T, H, W)`

## 4) Model download behavior

No manual model download is required in most cases.

On first run, UniDepthV2 weights are loaded from Hugging Face:

- `lpiccinelli/unidepth-v2-vits14`
- `lpiccinelli/unidepth-v2-vitb14`
- `lpiccinelli/unidepth-v2-vitl14` (default backbone)

Weights are cached under `~/.cache/huggingface/hub/` for later runs.

## 5) Run a quick single-task test

```bash
cd /path/to/UniDepth
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"

python scripts/libero_video_depth_compare.py \
  --videos-dir unidepth/videos \
  --one rollout_put_both_moka_pots_on_the_stove_success.mp4 \
  --views agentview wrist \
  --out-dir unidepth/videos/depth_compare_test
```

## 6) Run full comparison on LIBERO-10 rollouts

```bash
cd /path/to/UniDepth
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"

python scripts/libero_video_depth_compare.py \
  --videos-dir unidepth/videos \
  --views agentview wrist \
  --out-dir unidepth/videos/depth_compare_raw_intrinsics_mujoco
```

## 7) Output files and how to read them

For each task and view, outputs include:

- `comparison_rel.avi` (RGB / prediction / GT / relative error)
- `comparison_mae.avi` (RGB / prediction / GT / absolute error)
- `metrics.json`

Top-level summary:

- `metrics_overall.json`

Example:

```text
unidepth/videos/depth_compare_raw_intrinsics_mujoco/
  metrics_overall.json
  rollout_<task_a>/agentview/metrics.json
  rollout_<task_a>/wrist/metrics.json
  rollout_<task_b>/...
```

## 8) Useful flags

- `--backbone {s,b,l}`: model size (default `l`)
- `--no-scale-align`: disable median scale alignment before scoring
- `--no-camera-intrinsics`: run inference without explicit intrinsics
- `--video-format {mjpg_avi,mp4_mp4v}`: output video container/codec style
- `--one <task or mp4 name>`: run one rollout only

## 9) Common issues

- **`ModuleNotFoundError: unidepth`**
  - Run from repo root and set:
    - `export PYTHONPATH="$(pwd):$PYTHONPATH"`
  - Ensure editable install completed:
    - `pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118`

- **Model download fails on first run**
  - Check internet access to Hugging Face.
  - Re-run once connection is available.

- **xFormers / memory efficient attention runtime error**
  - If you see errors like `No operator found for memory_efficient_attention_forward`,
    uninstall xFormers and rerun:
    - `pip uninstall -y xformers`
  - UniDepthV2 will then fall back to PyTorch attention kernels.

- **Very slow runtime**
  - Verify CUDA is available:
    - `python -c "import torch; print(torch.cuda.is_available())"`
  - If `False`, inference is running on CPU.

- **Video not playing in some players**
  - Use `.avi` output (`mjpg_avi`, default), or re-encode with ffmpeg:
    - `ffmpeg -i comparison_rel.avi -c:v libx264 comparison_rel_h264.mp4`

