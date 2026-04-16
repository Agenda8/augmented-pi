# Simple depth-alignment evaluation on LIBERO task 8, episode 3 (initial_state_idx=2)
export CUDA_VISIBLE_DEVICES=3
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python examples/libero/main.py \
  --args.port 8050 \
  --args.task_suite_name libero_10 \
  --args.task_id 8 \
  --args.num_trials_per_task 1 \
  --args.fixed_initial_state_idx 2 \
  --args.replan_steps 5 \
  --args.enable-depth-align \
  --args.save-frame \
  --args.frame-out-path data/libero/frame/10/depth_align_episode3 \
  --args.save-depth-align-trace \
  --args.depth-align-trace-out-path data/libero/depth_align_trace/10_task8_episode3 \
  --args.no-save-video \
  --args.video-out-path data/libero/videos/10/depth_align_episode3 \
  --args.results-path data/libero/eval_results/10/depth_align_episode3.json \
  --args.no-save-failure \
  --args.failure-path data/libero/failure/10/depth_align_episode3 \
  --args.no-save-depth \
  --args.depth-out-path data/libero/depth/10/depth_align_episode3 \
  --args.no-action-quant \
  --args.no-vla_skip \
  --args.no-action_aware_chunk

# Kill server started by eval2.sh
SERVER_PIDS=$(lsof -t -i:8050 || true)
if [ -n "$SERVER_PIDS" ]; then
	kill $SERVER_PIDS
fi
