export CUDA_VISIBLE_DEVICES=2
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python scripts/replay_depth_align_from_actions.py \
  --action-path data/libero/failure/10/vanilla_task8/episode_3/actions.npy \
  --task-suite-name libero_10 \
  --task-id 8 \
  --initial-state-idx 3 \
  --trace-out-dir data/libero/depth_align_trace/10_task8_episode3_replay \