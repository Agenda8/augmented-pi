#!/usr/bin/env python3
"""
Pick one wrist rollout video, sample several pixel locations, and print
predicted depth (UniDepthV2) vs simulator GT depth at those points.

Also writes a wide PNG: **RGB (raw)** | **predicted depth** | **GT depth**,
same resolution per column, same numbered points on all three panels.

Usage::

  python scripts/wrist_depth_points_inspect.py \\
    --wrist-video /path/to/rollout_..._success_wrist.mp4

If ``--npz`` is omitted, it uses ``<task_stem>_depth.npz`` next to the video
(same convention as ``libero_video_depth_compare.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from unidepth.models import UniDepthV2
from unidepth.utils import colorize


def _task_stem_from_wrist_path(video_path: Path) -> str:
    stem = video_path.stem
    if not stem.endswith("_wrist"):
        raise ValueError(
            f"Expected a wrist rollout path ending with '_wrist.mp4', got: {video_path.name}"
        )
    return stem[: -len("_wrist")]


def _resolve_pretrained_source(model_name: str) -> str:
    repo_id = f"lpiccinelli/{model_name}"
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--lpiccinelli--{model_name}"
        / "snapshots"
    )
    if not cache_root.is_dir():
        return repo_id
    snapshots = sorted(p for p in cache_root.iterdir() if p.is_dir())
    if not snapshots:
        return repo_id
    return str(snapshots[-1])


def _load_gt_frame(npz_path: Path, frame_idx: int) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    key = "wrist_depth"
    if key not in data:
        raise KeyError(f"Key {key!r} not in {npz_path} (available: {data.files})")
    d = np.asarray(data[key])
    if d.ndim == 4:
        d = d[..., 0]
    if frame_idx < 0 or frame_idx >= d.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {d.shape[0]})")
    return d[frame_idx].astype(np.float32)


@torch.inference_mode()
def _predict_depth(model: UniDepthV2, rgb_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    out = model.infer(x, camera=None)
    return out["depth"].squeeze().float().cpu().numpy()


def _read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ret, bgr = cap.read()
    if not ret:
        return None
    return bgr


def _sample_points(h: int, w: int, n: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


BAR_H = 36


def _title_bar_rgb(w: int, title: str) -> np.ndarray:
    bar = np.full((BAR_H, w, 3), 40, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bar


def _draw_numbered_points_rgb(img_rgb: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    out = img_rgb.copy()
    for i, (x, y) in enumerate(points, start=1):
        cv2.circle(out, (x, y), 6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _column_rgb(title: str, content_rgb: np.ndarray) -> np.ndarray:
    h, w = content_rgb.shape[:2]
    bar = _title_bar_rgb(w, title)
    return np.vstack([bar, content_rgb])


def _rgb_raw_column(bgr: np.ndarray, points: list[tuple[int, int]], title: str) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    marked = _draw_numbered_points_rgb(rgb, points)
    return _column_rgb(title, marked)


def _depth_column(
    depth: np.ndarray,
    points: list[tuple[int, int]],
    vmin: float,
    vmax: float,
    title: str,
) -> np.ndarray:
    dep_rgb = colorize(depth, vmin=vmin, vmax=vmax, cmap="magma_r")
    marked = _draw_numbered_points_rgb(dep_rgb, points)
    return _column_rgb(title, marked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wrist-video",
        type=Path,
        required=True,
        help="Path to rollout_*_success_wrist.mp4",
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to matching *_depth.npz (default: sibling of task stem)",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index (default: middle frame)",
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=10,
        help="Number of random pixel samples",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for point sampling",
    )
    p.add_argument(
        "--backbone",
        choices=["s", "b", "l"],
        default="l",
        help="UniDepthV2 ViT size",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (RGB | pred | GT)",
    )
    args = p.parse_args()

    wrist_path = args.wrist_video.resolve()
    if not wrist_path.is_file():
        print(f"Not a file: {wrist_path}", file=sys.stderr)
        raise SystemExit(1)

    task_stem = _task_stem_from_wrist_path(wrist_path)
    npz_path = args.npz
    if npz_path is None:
        npz_path = wrist_path.parent / f"{task_stem}_depth.npz"
    else:
        npz_path = npz_path.resolve()
    if not npz_path.is_file():
        print(f"Missing npz: {npz_path}", file=sys.stderr)
        raise SystemExit(1)

    cap = cv2.VideoCapture(str(wrist_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release()
        print("Could not read frame count from video", file=sys.stderr)
        raise SystemExit(1)

    frame_idx = args.frame if args.frame >= 0 else n // 2
    bgr = _read_frame(cap, frame_idx)
    cap.release()
    if bgr is None:
        print(f"Failed to read frame {frame_idx}", file=sys.stderr)
        raise SystemExit(1)

    h, w = bgr.shape[:2]
    d_gt_full = _load_gt_frame(npz_path, frame_idx)
    d_gt = cv2.resize(d_gt_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"unidepth-v2-vit{args.backbone}14"
    source = _resolve_pretrained_source(name)
    model = UniDepthV2.from_pretrained(source)
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    d_pred = _predict_depth(model, bgr, device)
    if d_pred.shape[0] != h or d_pred.shape[1] != w:
        d_pred = cv2.resize(d_pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    points = _sample_points(h, w, args.num_points, args.seed)
    lo = float(min(np.min(d_pred), np.min(d_gt)))
    hi = float(max(np.max(d_pred), np.max(d_gt)))
    if hi <= lo + 1e-8:
        lo, hi = 0.0, 1.0

    print(f"wrist_video: {wrist_path}")
    print(f"npz: {npz_path}")
    print(f"frame: {frame_idx} / {n}")
    print(f"resolution: W={w} H={h}")
    print()
    print(f"{'#':>3}  {'x':>5}  {'y':>5}  {'pred_depth':>14}  {'gt_depth':>14}")
    print("-" * 52)
    for i, (x, y) in enumerate(points, start=1):
        pv = float(d_pred[y, x])
        gv = float(d_gt[y, x])
        print(f"{i:3d}  {x:5d}  {y:5d}  {pv:14.6f}  {gv:14.6f}")

    col_rgb = _rgb_raw_column(bgr, points, "RGB (raw)")
    col_pred = _depth_column(d_pred, points, lo, hi, "Predicted depth")
    col_gt = _depth_column(d_gt, points, lo, hi, "GT depth (sim)")
    mh = max(col_rgb.shape[0], col_pred.shape[0], col_gt.shape[0])

    def _pad_col(col: np.ndarray, target_h: int) -> np.ndarray:
        if col.shape[0] >= target_h:
            return col
        pad = np.zeros((target_h - col.shape[0], col.shape[1], 3), dtype=np.uint8)
        return np.vstack([col, pad])

    col_rgb = _pad_col(col_rgb, mh)
    col_pred = _pad_col(col_pred, mh)
    col_gt = _pad_col(col_gt, mh)
    combined = np.concatenate([col_rgb, col_pred, col_gt], axis=1)

    out_path = args.out
    if out_path is None:
        out_path = wrist_path.parent / f"{wrist_path.stem}_pred_gt_points_f{frame_idx}.png"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print()
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
