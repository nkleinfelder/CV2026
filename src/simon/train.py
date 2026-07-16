#!/usr/bin/env python3
"""Detached model-comparison run for bee-orientation regression.

Trains every architecture in the config to completion, writes checkpoints,
per-epoch histories, a comparison table, and diagnostic plots to
`--checkpoint-dir`, and survives losing the interactive session (submit via
`train.slurm`). The tiny-subset overfit gate lives only in the notebook — it is
a pre-flight check, not part of the full run.

All work happens inside `main()` under the `if __name__ == "__main__"` guard:
with `num_workers > 0`, DataLoader workers re-import this module (spawn/forkserver
on Python 3.14), and unguarded top-level training code would re-run in every
worker.

Example:
    python train.py --data-dir /scratch/cvcdt011/data \
        --checkpoint-dir /scratch/<you>/checkpoints --num-workers 4
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

# Headless backend: must be selected before pyplot is imported (via orientation).
import matplotlib

matplotlib.use("Agg")

import torch

import orientation as ori
from orientation import Config


def parse_args() -> argparse.Namespace:
    d = Config()  # defaults come straight from the dataclass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # paths
    p.add_argument("--data-dir", type=Path, default=d.data_dir,
                   help="Root holding crops/ and the trajectories/video.")
    p.add_argument("--checkpoint-dir", type=Path, default=d.checkpoint_dir,
                   help="Writable output dir for checkpoints, tables and plots (use scratch).")
    p.add_argument("--index-cache", type=Path, default=d.index_cache_path,
                   help="Sample-index CSV cache path (stores converted angles).")
    p.add_argument("--no-index-cache", action="store_true",
                   help="Ignore/skip the index cache and rebuild from the trajectory files.")

    # data / split
    p.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Source frame width height for the border-crop filter; read from the video if omitted.")
    p.add_argument("--split-strategy", choices=("trajectory", "random"), default=d.split_strategy)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--test-fraction", type=float, default=d.test_fraction)
    p.add_argument("--axial-labels", action="store_true",
                   help="Treat labels as axial (head/tail ambiguous): regress sin/cos of 2*theta.")

    # input pipeline
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--rotation-augment", action="store_true",
                   help="Enable label-consistent rotation augmentation (off by default).")

    # models / training
    p.add_argument("--models", nargs="+", default=list(d.model_names),
                   help="timm model names to compare.")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--no-pretrained", action="store_true", help="Train from scratch (no ImageNet weights).")
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    p.add_argument("--seed", type=int, default=d.seed)

    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return replace(
        Config(),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        index_cache_path=args.index_cache,
        use_index_cache=not args.no_index_cache,
        frame_size=tuple(args.frame_size) if args.frame_size is not None else None,
        split_strategy=args.split_strategy,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        axial_labels=args.axial_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rotation_augment=args.rotation_augment,
        model_names=tuple(args.models),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp,
        seed=args.seed,
    )


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)

    # Fixed-size inputs -> let cuDNN pick the fastest kernels.
    torch.backends.cudnn.benchmark = True
    # Progress bars only when attached to a terminal; under SLURM they'd spam the log.
    progress = sys.stderr.isatty()

    print(f"Device: {ori.DEVICE} | CUDA devices: {torch.cuda.device_count()}", flush=True)
    print(f"Models: {list(cfg.model_names)} | epochs: {cfg.epochs} | batch: {cfg.batch_size} "
          f"| image_size: {cfg.image_size} | amp: {cfg.use_amp} | augment: {cfg.rotation_augment}", flush=True)
    print(f"Data dir: {cfg.data_dir}", flush=True)

    # Fail fast if the output dir is not writable, before spending time on indexing.
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {cfg.checkpoint_dir}", flush=True)

    ori.seed_everything(cfg.seed)

    samples = ori.load_or_build_index(cfg)
    train_samples, val_samples, test_samples = ori.split_samples(samples, cfg)
    for split_name, split in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        n_traj = len({s.traj_id for s in split})
        print(f"{split_name}: {len(split)} samples from {n_traj} trajectories", flush=True)

    # A saved record that the label/orientation convention is correct.
    ori.plot_label_check(train_samples, cfg, n=24, save_path=cfg.checkpoint_dir / "label_check.png")

    train_loader, val_loader, test_loader = ori.make_loaders(
        train_samples, val_samples, test_samples, cfg
    )

    results: dict[str, ori.RunResult] = {}
    for name in cfg.model_names:
        try:
            results[name] = ori.train_and_evaluate(
                name, cfg, train_loader, val_loader, test_loader, progress=progress
            )
        except Exception:  # keep the surviving models even if one OOMs/errors
            print(f"!!! model {name} failed:\n{traceback.format_exc()}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        print("All models failed — nothing to report.", flush=True)
        return 1

    # Comparison table (models + baselines) -> CSV, and echoed to the log.
    comparison = ori.build_comparison_table(results, train_samples, test_samples, cfg)
    comparison.round(2).to_csv(cfg.checkpoint_dir / "comparison.csv")
    print("\n=== comparison (test set) ===", flush=True)
    print(comparison.round(2).to_string(), flush=True)

    # Diagnostic plots.
    ori.plot_training_curves(results, save_path=cfg.checkpoint_dir / "training_curves.png")
    ori.plot_error_diagnostics(results, cfg, save_path=cfg.checkpoint_dir / "error_diagnostics.png")

    # Best model (lowest test MAE) -> qualitative prediction arrows.
    best_name = min(results, key=lambda n: results[n].test_metrics["mae_deg"])
    best_res = results[best_name]
    print(f"\nBest model: {best_name} (test MAE {best_res.test_metrics['mae_deg']:.2f}°)", flush=True)

    best_model = ori.create_model(best_name, cfg).to(ori.DEVICE)
    best_model.load_state_dict(
        torch.load(best_res.checkpoint_path, map_location=ori.DEVICE, weights_only=True)
    )
    best_model.eval()
    ori.plot_predictions(
        test_samples, best_model, cfg, n=8,
        save_path=cfg.checkpoint_dir / f"predictions_{best_name}.png",
    )

    summary = {
        "best_model": best_name,
        "test_metrics": {name: res.test_metrics for name, res in results.items()},
        "failed_models": [name for name in cfg.model_names if name not in results],
        "config": {
            "models": list(cfg.model_names),
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "image_size": cfg.image_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "pretrained": cfg.pretrained,
            "use_amp": cfg.use_amp,
            "rotation_augment": cfg.rotation_augment,
            "axial_labels": cfg.axial_labels,
            "split_strategy": cfg.split_strategy,
            "seed": cfg.seed,
            "data_dir": str(cfg.data_dir),
        },
    }
    (cfg.checkpoint_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote run_summary.json to {cfg.checkpoint_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
