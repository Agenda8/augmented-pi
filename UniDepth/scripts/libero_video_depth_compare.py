#!/usr/bin/env python3
"""
Run UniDepthV2 on each RGB rollout in ``unidepth/videos/`` and compare to LIBERO
simulator depths from matching ``*_depth.npz`` (e.g. from pi0 / LeRobot rollouts).

Expected layout (under ``--videos-dir``)::

  rollout_..._success.mp4
  rollout_..._success_depth.npz   # contains ``agentview_depth`` and/or ``wrist_depth``

The ``.npz`` is loaded with keys (typical)::

  agentview_depth: (T, H_gt, W_gt, 1) float32
  wrist_depth:     (T, H_gt, W_gt, 1) float32

RGB is read from the video (often 224x224). Ground-truth depth is resized to
match each frame. UniDepth outputs metric depth; LIBERO z-buffer can be a narrow
[0,1] range, so this script **optionally** applies a per-frame **median
scale** ``s = median(gt / pred)`` so the comparison and error map are on a
common scale (set ``--no-scale-align`` to skip).

Outputs under ``--out-dir`` (default: ``<videos-dir>/depth_compare``)::

  <task_stem>/
    agentview/comparison_rel.avi
    agentview/comparison_mae.avi
    agentview/metrics.json
    wrist/comparison_rel.avi
    wrist/comparison_mae.avi
    wrist/metrics.json
    metrics.json              # per-task summary for both views
  metrics_overall.json        # aggregate metrics per view over all tasks

**Note:** ``comparison.mp4`` (``mp4v``) is optional; it is *not* H.264, and some
media players (and editors like Cursor) may not play it. Use **VLC**, re-encode
with ``ffmpeg -c:v libx264``, or use the default ``comparison.avi``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole
from unidepth.utils import colorize, image_grid

NEAR = 0.010609816
FAR = 530.490780727


def _resolve_pretrained_source(model_name: str) -> str:
    """Prefer online hub id, but fall back to local HF snapshot cache."""
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
    snapshots = sorted([p for p in cache_root.iterdir() if p.is_dir()])
    if not snapshots:
        return repo_id
    return str(snapshots[-1])


def _view_from_video_name(video_name: str) -> str | None:
    if not video_name.endswith(".mp4") or "depth_preview" in video_name:
        return None
    return "wrist" if video_name.endswith("_wrist.mp4") else "agentview"


def _task_stem_from_video_name(video_name: str) -> str:
    stem = Path(video_name).stem
    if stem.endswith("_wrist"):
        return stem[: -len("_wrist")]
    return stem


def _list_rollout_pairs(videos_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for p in sorted(videos_dir.glob("*.mp4")):
        view = _view_from_video_name(p.name)
        if view is None:
            continue
        task_stem = _task_stem_from_video_name(p.name)
        depth_npz = videos_dir / f"{task_stem}_depth.npz"
        if not depth_npz.is_file():
            print(
                f"[skip] no depth file for {p.name} (expected {depth_npz.name})",
                file=sys.stderr,
            )
            continue
        item = grouped.setdefault(
            task_stem,
            {
                "task_stem": task_stem,
                "npz_path": depth_npz,
                "videos": {},
            },
        )
        item["videos"][view] = p
    return [grouped[k] for k in sorted(grouped)]


def _load_gt_stack(npz_path: Path, key: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key {key!r} not in {npz_path} (available: {data.files})")
    d = data[key]
    d = np.asarray(d)
    if d.ndim == 4:
        d = d[..., 0]
    return d.astype(np.float32)


@torch.inference_mode()
def _predict_depth(
    model: UniDepthV2,
    rgb_bgr: np.ndarray,
    device: torch.device,
    intrinsics: np.ndarray | None,
) -> np.ndarray:
    """rgb_bgr: (H, W, 3) uint8, BGR from OpenCV."""
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    camera = None
    if intrinsics is not None:
        k = torch.from_numpy(intrinsics.astype(np.float32))
        camera = Pinhole(K=k.unsqueeze(0))
    out = model.infer(x, camera=camera)
    depth = out["depth"].squeeze().float().cpu().numpy()
    return depth


def _median_scale(gt: np.ndarray, pred: np.ndarray, min_gt: float) -> float:
    m = (gt > min_gt) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not np.any(m):
        return 1.0
    r = (gt / np.clip(pred, 1e-8, None))[m]
    return float(np.median(r))


def _metric_to_sim_depth(z_m: np.ndarray) -> np.ndarray:
    z_m = np.asarray(z_m, dtype=np.float32)
    d = (1.0 - NEAR / np.clip(z_m, 1e-8, None)) / (1.0 - NEAR / FAR)
    return d.astype(np.float32)


def _open_video_writer(
    out_path: Path, fps: float, size: tuple[int, int], fmt: str
) -> tuple[cv2.VideoWriter, dict[str, Any]]:
    w, h = size
    meta: dict[str, Any] = {"path": str(out_path), "format": fmt}
    if fmt == "mp4_mp4v":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        meta["fourcc"] = "mp4v"
    elif fmt == "mjpg_avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        meta["fourcc"] = "MJPG"
    else:
        raise ValueError(f"Unknown video format: {fmt!r}")
    return writer, meta


def process_video(
    rgb_path: Path,
    npz_path: Path,
    model: UniDepthV2,
    device: torch.device,
    gt_key: str,
    scale_align: bool,
    min_gt: float,
    out_subdir: Path,
    err_vmax: float,
    mae_vmax: float | None,
    depth_vmin: float | None,
    depth_vmax: float | None,
    video_format: str,
    intrinsics: np.ndarray | None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(rgb_path))
    n_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224

    gt_stack = _load_gt_stack(npz_path, gt_key)
    n_gt = int(gt_stack.shape[0])
    t = n_gt
    if n_cap > 0:
        t = min(n_cap, n_gt)
    if t == 0:
        cap.release()
        raise RuntimeError(
            f"No frames: video frames={n_cap}, gt T={n_gt} for {rgb_path}"
        )

    out_subdir.mkdir(parents=True, exist_ok=True)
    ext = ".avi" if video_format == "mjpg_avi" else ".mp4"
    out_video_rel = out_subdir / f"comparison_rel{ext}"
    out_video_mae = out_subdir / f"comparison_mae{ext}"
    cell = max(h, w)
    grid_w, grid_h = cell * 2, cell * 2
    writer_rel, wmeta_rel = _open_video_writer(
        out_video_rel, fps, (grid_w, grid_h), video_format
    )
    writer_mae, wmeta_mae = _open_video_writer(
        out_video_mae, fps, (grid_w, grid_h), video_format
    )
    if not writer_rel.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for {out_video_rel}")
    if not writer_mae.isOpened():
        cap.release()
        writer_rel.release()
        raise RuntimeError(f"Failed to open VideoWriter for {out_video_mae}")

    mae_list: list[float] = []
    arel_list: list[float] = []
    scale_list: list[float] = []
    n_written = 0

    for i in range(t):
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        d_gt = cv2.resize(
            gt_stack[i], (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

        pred_metric = _predict_depth(model, bgr, device, intrinsics)
        pred = _metric_to_sim_depth(pred_metric)

        s = 1.0
        if scale_align:
            s = _median_scale(d_gt, pred, min_gt)
        pred_s = pred * s
        m = d_gt > min_gt
        abs_err = np.abs(pred_s - d_gt)
        if np.any(m):
            mae_list.append(float(np.mean(abs_err[m])))
            arel_list.append(
                float(np.mean((abs_err[m] / np.clip(d_gt[m], 1e-8, None))))
            )
        scale_list.append(s)

        vmin_d = depth_vmin
        vmax_d = depth_vmax
        if vmin_d is None:
            lo = min(float(np.min(pred_s)), float(np.min(d_gt)))
        else:
            lo = vmin_d
        if vmax_d is None:
            hi = max(float(np.max(pred_s)), float(np.max(d_gt)))
        else:
            hi = vmax_d
        if hi <= lo + 1e-8:
            lo, hi = 0.0, 1.0

        rgb_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        c_pred = colorize(pred_s, vmin=lo, vmax=hi, cmap="magma_r")
        c_gt = colorize(d_gt, vmin=lo, vmax=hi, cmap="magma_r")
        rel_err_map = abs_err / np.clip(d_gt, 1e-6, None)
        rel_err_map[~m] = 0.0
        c_err_rel = colorize(
            rel_err_map, vmin=0.0, vmax=err_vmax, cmap="coolwarm"
        )
        vmax_mae = mae_vmax if mae_vmax is not None else max(hi - lo, 1e-6)
        c_err_mae = colorize(abs_err, vmin=0.0, vmax=vmax_mae, cmap="coolwarm")

        tiles_rel = [rgb_rgb, c_pred, c_gt, c_err_rel]
        tiles_mae = [rgb_rgb, c_pred, c_gt, c_err_mae]
        h0, w0 = rgb_rgb.shape[:2]
        for j, im in enumerate(tiles_rel):
            if im.shape[0] != h0 or im.shape[1] != w0:
                tiles_rel[j] = cv2.resize(
                    im, (w0, h0), interpolation=cv2.INTER_NEAREST
                )
        for j, im in enumerate(tiles_mae):
            if im.shape[0] != h0 or im.shape[1] != w0:
                tiles_mae[j] = cv2.resize(
                    im, (w0, h0), interpolation=cv2.INTER_NEAREST
                )
        arr_rel = image_grid(tiles_rel, 2, 2)
        arr_mae = image_grid(tiles_mae, 2, 2)
        frame_rel_bgr = cv2.cvtColor(arr_rel, cv2.COLOR_RGB2BGR)
        frame_mae_bgr = cv2.cvtColor(arr_mae, cv2.COLOR_RGB2BGR)
        if frame_rel_bgr.shape[0] != grid_h or frame_rel_bgr.shape[1] != grid_w:
            frame_rel_bgr = cv2.resize(
                frame_rel_bgr, (grid_w, grid_h), interpolation=cv2.INTER_AREA
            )
        if frame_mae_bgr.shape[0] != grid_h or frame_mae_bgr.shape[1] != grid_w:
            frame_mae_bgr = cv2.resize(
                frame_mae_bgr, (grid_w, grid_h), interpolation=cv2.INTER_AREA
            )
        writer_rel.write(frame_rel_bgr)
        writer_mae.write(frame_mae_bgr)
        n_written += 1

    cap.release()
    writer_rel.release()
    writer_mae.release()

    summary = {
        "rgb_path": str(rgb_path.resolve()),
        "npz_path": str(npz_path.resolve()),
        "pred_depth_space": "mujoco_0_1_from_metric",
        "pred_depth_conversion": {"near": NEAR, "far": FAR},
        "comparison_video_rel": wmeta_rel,
        "comparison_video_mae": wmeta_mae,
        "gt_key": gt_key,
        "num_frames": n_written,
        "num_metric_frames": len(mae_list),
        "scale_align": scale_align,
        "median_median_scale": float(np.median(scale_list)) if scale_list else None,
        "mean_mae": float(np.mean(mae_list)) if mae_list else None,
        "mean_arel": float(np.mean(arel_list)) if arel_list else None,
        "mean_arel_percent": (float(np.mean(arel_list) * 100) if arel_list else None),
    }
    with (out_subdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _aggregate_view_metrics(view_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    maes: list[float] = []
    arels: list[float] = []
    scales: list[float] = []
    weighted_mae_num = 0.0
    weighted_arel_num = 0.0
    weighted_den = 0.0
    total_frames = 0
    total_metric_frames = 0

    for sm in view_summaries:
        n_frames = int(sm.get("num_frames") or 0)
        n_metric = int(sm.get("num_metric_frames") or 0)
        total_frames += n_frames
        total_metric_frames += n_metric

        mae = sm.get("mean_mae")
        arel = sm.get("mean_arel")
        scale = sm.get("median_median_scale")
        if mae is not None:
            maes.append(float(mae))
        if arel is not None:
            arels.append(float(arel))
        if scale is not None:
            scales.append(float(scale))

        if mae is not None and n_metric > 0:
            weighted_mae_num += float(mae) * float(n_metric)
            weighted_den += float(n_metric)
        if arel is not None and n_metric > 0:
            weighted_arel_num += float(arel) * float(n_metric)

    mean_mae = float(np.mean(maes)) if maes else None
    mean_arel = float(np.mean(arels)) if arels else None
    weighted_mean_mae = (weighted_mae_num / weighted_den) if weighted_den > 0 else None
    weighted_mean_arel = (weighted_arel_num / weighted_den) if weighted_den > 0 else None

    return {
        "num_tasks": len(view_summaries),
        "total_frames": total_frames,
        "total_metric_frames": total_metric_frames,
        "median_median_scale": float(np.median(scales)) if scales else None,
        "mean_mae": mean_mae,
        "mean_arel": mean_arel,
        "mean_arel_percent": (mean_arel * 100.0) if mean_arel is not None else None,
        "weighted_mean_mae": weighted_mean_mae,
        "weighted_mean_arel": weighted_mean_arel,
        "weighted_mean_arel_percent": (
            weighted_mean_arel * 100.0 if weighted_mean_arel is not None else None
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "unidepth" / "videos",
        help="Folder with rollout RGB mp4 and matching *_depth.npz",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <videos-dir>/depth_compare",
    )
    p.add_argument(
        "--backbone",
        choices=["s", "b", "l"],
        default="l",
        help="UniDepthV2 ViT size (s / b / l).",
    )
    p.add_argument(
        "--no-scale-align",
        action="store_true",
        help="Do not apply per-frame median scale; compare raw metric pred to sim GT.",
    )
    p.add_argument(
        "--min-gt",
        type=float,
        default=1e-4,
        help="Ignore GT pixels below this (and mask error visualization).",
    )
    p.add_argument(
        "--err-vmax",
        type=float,
        default=0.25,
        help="Max for coloring relative error (AbsRel) map in bottom-right panel.",
    )
    p.add_argument(
        "--mae-vmax",
        type=float,
        default=None,
        help="Max for coloring MAE map in bottom-right panel; default uses per-frame depth span.",
    )
    p.add_argument(
        "--depth-vmin",
        type=float,
        default=None,
        help="Optional fixed min for pred/GT colormap; default = min(pred,gt).",
    )
    p.add_argument(
        "--depth-vmax",
        type=float,
        default=None,
        help="Optional fixed max for pred/GT colormap; default = max(pred,gt).",
    )
    p.add_argument(
        "--one",
        type=str,
        default=None,
        help="Process only one task stem or one rollout mp4 name.",
    )
    p.add_argument(
        "--video-format",
        choices=["mjpg_avi", "mp4_mp4v"],
        default="mjpg_avi",
        help="mjpg_avi: comparison.avi (MJPEG, widely playable). "
        "mp4_mp4v: comparison.mp4 (may not play in all apps; not H.264).",
    )
    p.add_argument(
        "--views",
        nargs="+",
        choices=["agentview", "wrist"],
        default=["agentview", "wrist"],
        help="Views to process for each task (default: both).",
    )
    p.add_argument("--agent-fx", type=float, default=309.019336)
    p.add_argument("--agent-fy", type=float, default=309.019336)
    p.add_argument("--agent-cx", type=float, default=128.0)
    p.add_argument("--agent-cy", type=float, default=128.0)
    p.add_argument("--wrist-fx", type=float, default=166.812848)
    p.add_argument("--wrist-fy", type=float, default=166.812848)
    p.add_argument("--wrist-cx", type=float, default=128.0)
    p.add_argument("--wrist-cy", type=float, default=128.0)
    p.add_argument(
        "--no-camera-intrinsics",
        action="store_true",
        help="Ignore camera intrinsics and infer with camera=None.",
    )
    args = p.parse_args()

    videos_dir = args.videos_dir
    if not videos_dir.is_dir():
        print(f"Not a directory: {videos_dir}", file=sys.stderr)
        raise SystemExit(1)

    out_dir = args.out_dir or (videos_dir / "depth_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"unidepth-v2-vit{args.backbone}14"
    source = _resolve_pretrained_source(name)
    try:
        model = UniDepthV2.from_pretrained(source)
    except Exception as e:
        if "/" not in source:
            raise
        fallback = _resolve_pretrained_source(name)
        if fallback == source:
            raise
        print(
            f"[warn] failed loading {source!r}: {e}; retrying with local cache {fallback!r}",
            file=sys.stderr,
        )
        model = UniDepthV2.from_pretrained(fallback)
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    scale_align = not args.no_scale_align
    intrinsics_by_view: dict[str, np.ndarray | None]
    if args.no_camera_intrinsics:
        intrinsics_by_view = {"agentview": None, "wrist": None}
    else:
        intrinsics_by_view = {
            "agentview": np.array(
                [
                    [args.agent_fx, 0.0, args.agent_cx],
                    [0.0, args.agent_fy, args.agent_cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "wrist": np.array(
                [
                    [args.wrist_fx, 0.0, args.wrist_cx],
                    [0.0, args.wrist_fy, args.wrist_cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        }

    tasks = _list_rollout_pairs(videos_dir)
    if args.one:
        target = args.one
        target_stem = _task_stem_from_video_name(target)
        tasks = [t for t in tasks if t["task_stem"] == target_stem]
        if not tasks:
            print(f"No video matching --one {args.one!r}", file=sys.stderr)
            raise SystemExit(1)

    overall_by_view: dict[str, list[dict[str, Any]]] = {v: [] for v in args.views}

    for task in tasks:
        task_stem: str = task["task_stem"]
        npz_path: Path = task["npz_path"]
        videos_by_view: dict[str, Path] = task["videos"]
        task_out_dir = out_dir / task_stem
        task_out_dir.mkdir(parents=True, exist_ok=True)
        task_summary: dict[str, Any] = {
            "task_stem": task_stem,
            "npz_path": str(npz_path.resolve()),
            "views": {},
        }

        for view in args.views:
            vp = videos_by_view.get(view)
            if vp is None:
                print(f"[skip] {task_stem}: missing {view} video", file=sys.stderr)
                continue
            gt_key = f"{view}_depth"
            osub = task_out_dir / view
            try:
                sm = process_video(
                    rgb_path=vp,
                    npz_path=npz_path,
                    model=model,
                    device=device,
                    gt_key=gt_key,
                    scale_align=scale_align,
                    min_gt=args.min_gt,
                    out_subdir=osub,
                    err_vmax=args.err_vmax,
                    mae_vmax=args.mae_vmax,
                    depth_vmin=args.depth_vmin,
                    depth_vmax=args.depth_vmax,
                    video_format=args.video_format,
                    intrinsics=intrinsics_by_view.get(view),
                )
            except Exception as e:
                print(f"[error] {vp.name} ({view}): {e}", file=sys.stderr)
                continue
            task_summary["views"][view] = sm
            overall_by_view.setdefault(view, []).append(sm)
            mae = sm.get("mean_mae")
            arel = sm.get("mean_arel_percent")
            vpath = (sm.get("comparison_video_rel") or {}).get("path", "")
            print(
                f"OK {task_stem} [{view}] -> {vpath or osub} | mean_mae={mae} mean_arel%={arel}"
            )

        with (task_out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(task_summary, f, indent=2, ensure_ascii=False)

        if not task_summary["views"]:
            continue

    overall = {
        "videos_dir": str(videos_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "views_requested": args.views,
        "num_tasks_scanned": len(tasks),
        "scale_align": scale_align,
        "per_view": {},
    }
    for view in args.views:
        overall["per_view"][view] = _aggregate_view_metrics(overall_by_view.get(view, []))

    overall_path = out_dir / "metrics_overall.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote overall metrics -> {overall_path}")


if __name__ == "__main__":
    main()
