#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from unidepth.models import UniDepthV2
from unidepth.utils import colorize


def resolve_pretrained_source(model_name: str) -> str:
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


@torch.inference_mode()
def predict_depth(model: UniDepthV2, image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous()
    pred = model.infer(x, camera=None)
    depth = pred["depth"].squeeze().float().cpu().numpy()
    return depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UniDepthV2 on one image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "unidepth" / "images" / "9986.jpg",
        help="Input RGB image path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "unidepth" / "images" / "depth_outputs",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--backbone",
        choices=["s", "b", "l"],
        default="l",
        help="UniDepthV2 ViT size.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional depth colormap min; default uses depth min.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional depth colormap max; default uses depth max.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"unidepth-v2-vit{args.backbone}14"
    source = resolve_pretrained_source(model_name)
    model = UniDepthV2.from_pretrained(source)
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    depth = predict_depth(model, image_bgr)
    vmin = args.vmin if args.vmin is not None else float(np.min(depth))
    vmax = args.vmax if args.vmax is not None else float(np.max(depth))
    if vmax <= vmin + 1e-8:
        vmin, vmax = 0.0, 1.0
    depth_color = colorize(depth, vmin=vmin, vmax=vmax, cmap="magma_r")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    color_path = args.out_dir / f"{stem}_depth_color.png"
    npy_path = args.out_dir / f"{stem}_depth.npy"

    cv2.imwrite(str(color_path), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))
    np.save(npy_path, depth.astype(np.float32))

    print(f"input: {args.image.resolve()}")
    print(f"depth_npy: {npy_path.resolve()}")
    print(f"depth_color: {color_path.resolve()}")
    print(f"depth_min: {float(np.min(depth)):.6f}")
    print(f"depth_max: {float(np.max(depth)):.6f}")


if __name__ == "__main__":
    main()

