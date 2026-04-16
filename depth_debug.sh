source examples/libero/.venv/bin/activate
python scripts/debug_depth_align.py \
	--depth-path data/libero/depth/10/vanilla_task8/episode_00003/wrist_depth.npy \
	--output-dir data/libero/depth_debug/episode_3_wrist \
	--stride 5 \
	--max-frames 40 