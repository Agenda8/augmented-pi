import csv
import dataclasses
import json
import pathlib
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tyro

# Allow running this file directly via "python scripts/debug_depth_align.py".
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from augmented.depth_guided_align import DepthGuidedAlignConfig, DepthGuidedAligner


@dataclasses.dataclass
class Args:
    depth_path: str
    output_dir: str = "data/libero/depth_debug"
    frame_start: int = 0
    frame_end: Optional[int] = None
    stride: int = 1
    max_frames: Optional[int] = None
    save_visualizations: bool = True


def _load_depth(path: pathlib.Path) -> np.ndarray:
    depth = np.load(path)
    if depth.ndim not in (3, 4):
        raise ValueError(f"Expected depth data with shape [T,H,W] or [T,H,W,1], got {depth.shape}")
    return depth


def _iter_frame_indices(total: int, start: int, end: Optional[int], stride: int, max_frames: Optional[int]) -> List[int]:
    end_idx = total if end is None else min(end, total)
    if not (0 <= start < end_idx):
        raise ValueError(f"Invalid frame range: start={start}, end={end_idx}, total={total}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    indices = list(range(start, end_idx, stride))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _save_frame_visualization(
    frame: np.ndarray,
    near_mask: Optional[np.ndarray],
    gripper_mask: Optional[np.ndarray],
    anchor: Tuple[float, float],
    target_xy: Optional[Tuple[float, float]],
    out_path: pathlib.Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(frame, cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth")

    if near_mask is not None and np.any(near_mask):
        overlay = np.zeros((near_mask.shape[0], near_mask.shape[1], 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = near_mask.astype(np.float32) * 0.35
        ax.imshow(overlay, origin="upper")

    if gripper_mask is not None and np.any(gripper_mask):
        overlay = np.zeros((gripper_mask.shape[0], gripper_mask.shape[1], 4), dtype=np.float32)
        overlay[..., 2] = 1.0
        overlay[..., 3] = gripper_mask.astype(np.float32) * 0.28
        ax.imshow(overlay, origin="upper")

    ax.scatter([anchor[0]], [anchor[1]], c="cyan", s=70, marker="x", label="gripper anchor")
    if target_xy is not None:
        ax.scatter([target_xy[0]], [target_xy[1]], c="red", s=40, label="detected target")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(args: Args) -> None:
    depth_path = pathlib.Path(args.depth_path)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_data = _load_depth(depth_path)
    frame_indices = _iter_frame_indices(
        total=depth_data.shape[0],
        start=args.frame_start,
        end=args.frame_end,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    config = DepthGuidedAlignConfig(enabled=True)
    aligner = DepthGuidedAligner(config)

    rows = []
    trigger_count = 0
    found_count = 0

    vis_dir = output_dir / "frames"
    if args.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for idx in frame_indices:
        frame = np.asarray(depth_data[idx], dtype=np.float32)
        if frame.ndim == 3 and frame.shape[-1] >= 1:
            frame = frame[..., 0]

        analysis = aligner.analyze_depth_frame(frame, include_mask=True)

        found = bool(analysis.target_found)
        triggered = bool(analysis.should_trigger)
        found_count += int(found)
        trigger_count += int(triggered)

        rows.append(
            {
                "frame_idx": idx,
                "target_found": int(found),
                "should_trigger": int(triggered),
                "pixel_count": int(analysis.pixel_count),
                "target_x": None if analysis.target_cx is None else float(analysis.target_cx),
                "target_y": None if analysis.target_cy is None else float(analysis.target_cy),
                "anchor_x": float(analysis.target_anchor[0]),
                "anchor_y": float(analysis.target_anchor[1]),
                "center_dist": None if analysis.center_dist is None else float(analysis.center_dist),
                "target_depth": None if analysis.target_depth is None else float(analysis.target_depth),
                "depth_error": None if analysis.depth_error is None else float(analysis.depth_error),
                "depth_gap_abs": None if analysis.depth_gap_abs is None else float(analysis.depth_gap_abs),
            }
        )

        if args.save_visualizations:
            title = (
                f"frame {idx} | found={int(found)} trigger={int(triggered)} "
                f"dist={analysis.center_dist if analysis.center_dist is not None else -1:.2f}"
            )
            target_xy = None
            if analysis.target_cx is not None and analysis.target_cy is not None:
                target_xy = (analysis.target_cx, analysis.target_cy)

            _save_frame_visualization(
                frame=frame,
                near_mask=analysis.near_mask,
                gripper_mask=analysis.gripper_mask,
                anchor=analysis.target_anchor,
                target_xy=target_xy,
                out_path=vis_dir / f"frame_{idx:05d}.png",
                title=title,
            )

    csv_path = output_dir / "frame_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "target_found",
                "should_trigger",
                "pixel_count",
                "target_x",
                "target_y",
                "anchor_x",
                "anchor_y",
                "center_dist",
                "target_depth",
                "depth_error",
                "depth_gap_abs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    dists = [r["center_dist"] for r in rows if r["center_dist"] is not None]
    depth_gap_abs_vals = [r["depth_gap_abs"] for r in rows if r["depth_gap_abs"] is not None]
    depth_error_vals = [r["depth_error"] for r in rows if r["depth_error"] is not None]

    summary = {
        "depth_path": str(depth_path),
        "frames_analyzed": len(rows),
        "target_found_ratio": (found_count / len(rows)) if rows else 0.0,
        "trigger_ratio": (trigger_count / len(rows)) if rows else 0.0,
        "center_dist_mean": float(np.mean(dists)) if dists else None,
        "center_dist_std": float(np.std(dists)) if dists else None,
        "depth_gap_abs_mean": float(np.mean(depth_gap_abs_vals)) if depth_gap_abs_vals else None,
        "depth_gap_abs_std": float(np.std(depth_gap_abs_vals)) if depth_gap_abs_vals else None,
        "depth_error_mean": float(np.mean(depth_error_vals)) if depth_error_vals else None,
        "depth_error_std": float(np.std(depth_error_vals)) if depth_error_vals else None,
        "csv_path": str(csv_path),
        "vis_dir": str(vis_dir) if args.save_visualizations else None,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
