#!/usr/bin/env python3
"""
Encode a wrist rollout video as three horizontal panels per frame:

  **Predicted depth (colorized)** | **GT depth (colorized)** | **RGB (raw)**

Uses UniDepthV2 on each frame and ``wrist_depth`` from the matching ``*_depth.npz``.
Samples several pixel locations (once, from the first frame geometry) and draws
markers plus short ``P=.. G=..`` text at each point on all three panels.

Example::

  python scripts/wrist_depth_triple_video.py \\
    --wrist-video unidepth/videos/rollout_..._success_wrist.mp4 \\
    --out unidepth/videos/rollout_..._wrist_triple.avi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole
from unidepth.utils import colorize


def _task_stem_from_wrist_path(video_path: Path) -> str:
    stem = video_path.stem
    if not stem.endswith("_wrist"):
        raise ValueError(
            f"Expected *_wrist.mp4, got: {video_path.name}"
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


def _load_gt_stack(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    key = "wrist_depth"
    if key not in data:
        raise KeyError(f"Key {key!r} not in {npz_path} (available: {data.files})")
    d = np.asarray(data[key])
    if d.ndim == 4:
        d = d[..., 0]
    return d.astype(np.float32)


def _median_scale(gt: np.ndarray, pred: np.ndarray, min_gt: float) -> float:
    m = (gt > min_gt) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not np.any(m):
        return 1.0
    r = (gt / np.clip(pred, 1e-8, None))[m]
    return float(np.median(r))


@torch.inference_mode()
def _predict_depth(
    model: UniDepthV2,
    rgb_bgr: np.ndarray,
    device: torch.device,
    intrinsics: np.ndarray | None,
) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    camera = None
    if intrinsics is not None:
        k = torch.from_numpy(intrinsics.astype(np.float32))
        camera = Pinhole(K=k.unsqueeze(0))
    out = model.infer(x, camera=camera)
    return out["depth"].squeeze().float().cpu().numpy()


def _sample_points(h: int, w: int, n: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
) -> None:
    x, y = org
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_points_with_depths(
    img_rgb: np.ndarray,
    points: list[tuple[int, int]],
    preds: list[float],
    gts: list[float],
    label_mode: str,
) -> np.ndarray:
    out = img_rgb.copy()
    fs = 0.32
    for i, ((x, y), pv, gv) in enumerate(zip(points, preds, gts), start=1):
        cv2.circle(out, (x, y), 5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        _put_text_outline(out, str(i), (x + 6, y - 6), fs + 0.08, (255, 255, 255))
        if label_mode == "pred":
            t = f"P{pv:.3f}"
        elif label_mode == "gt":
            t = f"G{gv:.3f}"
        else:
            t = f"P{pv:.2f} G{gv:.2f}"
        _put_text_outline(out, t, (x + 6, y + 10), fs, (255, 255, 200))
    return out


def _open_writer(out_path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, dict[str, Any]]:
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    meta = {"path": str(out_path), "fourcc": "MJPG", "format": "mjpg_avi"}
    return writer, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrist-video", type=Path, required=True)
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--start", type=int, default=0, help="First frame index")
    p.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Last frame index inclusive (-1 = last)",
    )
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames in range")
    p.add_argument("--num-points", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", choices=["s", "b", "l"], default="l")
    p.add_argument("--fx", type=float, default=166.812848)
    p.add_argument("--fy", type=float, default=166.812848)
    p.add_argument("--cx", type=float, default=128.0)
    p.add_argument("--cy", type=float, default=128.0)
    p.add_argument(
        "--no-camera-intrinsics",
        action="store_true",
        help="Infer with camera=None instead of passing K matrix.",
    )
    p.add_argument(
        "--colormap-align",
        action="store_true",
        help="Per-frame median-scale pred to GT before depth colormap (visual only).",
    )
    p.add_argument(
        "--min-gt",
        type=float,
        default=1e-4,
        help="Min GT for median scale (if --colormap-align).",
    )
    p.add_argument(
        "--bar-h",
        type=int,
        default=28,
        help="Top label bar height in pixels.",
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

    gt_stack = _load_gt_stack(npz_path)
    t_gt = int(gt_stack.shape[0])

    cap = cv2.VideoCapture(str(wrist_path))
    n_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224

    end = args.end if args.end >= 0 else min(n_cap, t_gt) - 1
    start = max(0, args.start)
    end = min(end, n_cap - 1 if n_cap > 0 else end, t_gt - 1)
    if end < start:
        cap.release()
        print("Invalid frame range", file=sys.stderr)
        raise SystemExit(1)

    total = end - start + 1
    if args.max_frames > 0:
        total = min(total, args.max_frames)
        end = start + total - 1

    points = _sample_points(h, w, args.num_points, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"unidepth-v2-vit{args.backbone}14"
    model = UniDepthV2.from_pretrained(_resolve_pretrained_source(name))
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()
    intrinsics = None
    if not args.no_camera_intrinsics:
        intrinsics = np.array(
            [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    bar_h = max(0, args.bar_h)
    tri_w, tri_h = w * 3, h + bar_h
    out_path = args.out
    if out_path is None:
        out_path = wrist_path.parent / f"{wrist_path.stem}_triple.avi"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer, wmeta = _open_writer(out_path, fps, (tri_w, tri_h))
    if not writer.isOpened():
        cap.release()
        print(f"Failed to open VideoWriter for {out_path}", file=sys.stderr)
        raise SystemExit(1)

    for k in range(total):
        fi = start + k
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        d_gt = cv2.resize(gt_stack[fi], (w, h), interpolation=cv2.INTER_NEAREST).astype(
            np.float32
        )
        d_pred = _predict_depth(model, bgr, device, intrinsics)
        if d_pred.shape[0] != h or d_pred.shape[1] != w:
            d_pred = cv2.resize(
                d_pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )

        d_pred_vis = d_pred.astype(np.float32)
        if args.colormap_align:
            s = _median_scale(d_gt, d_pred_vis, args.min_gt)
            d_pred_vis = d_pred_vis * s

        lo = min(float(np.min(d_pred_vis)), float(np.min(d_gt)))
        hi = max(float(np.max(d_pred_vis)), float(np.max(d_gt)))
        if hi <= lo + 1e-8:
            lo, hi = 0.0, 1.0

        pred_rgb = colorize(d_pred_vis, vmin=lo, vmax=hi, cmap="magma_r")
        gt_rgb = colorize(d_gt, vmin=lo, vmax=hi, cmap="magma_r")
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        preds = [float(d_pred[y, x]) for x, y in points]
        gts = [float(d_gt[y, x]) for x, y in points]

        pred_m = _draw_points_with_depths(pred_rgb, points, preds, gts, "pred")
        gt_m = _draw_points_with_depths(gt_rgb, points, preds, gts, "gt")
        raw_m = _draw_points_with_depths(raw_rgb, points, preds, gts, "both")

        row = np.concatenate([pred_m, gt_m, raw_m], axis=1)
        bar = np.full((bar_h, tri_w, 3), 30, dtype=np.uint8)
        labels = [
            "Predicted depth",
            "GT depth (sim)",
            "RGB (raw)",
        ]
        for j, lab in enumerate(labels):
            x0 = j * w + 8
            cv2.putText(
                bar,
                lab,
                (x0, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        frame_rgb = np.vstack([bar, row])
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if k % 50 == 0 or k == total - 1:
            print(f"frame {fi} ({k + 1}/{total}) -> {wmeta['path']}", flush=True)

    cap.release()
    writer.release()
    print(f"done: {out_path} ({total} frames, {tri_w}x{tri_h} @ {fps:.2f} fps)")


if __name__ == "__main__":
    main()
