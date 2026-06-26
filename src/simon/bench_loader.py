#!/usr/bin/env python3
"""Used this to test dataloader speed, since GPUs were poorly utilized when using multiple.
Just reverted to using 1 GPU after this, so this can be ignored.

Times two things independently:
  raw:    single-process open+decode+resize per image (one worker's capacity)
  loader: full DataLoader throughput with no model attached
and prints both so they can be compared to the img/s the training run reports.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from PIL import Image

from simon import orientation as ori
from simon.config import Config


def bench_raw_reads(samples, n: int) -> None:
    picks = random.sample(samples, min(n, len(samples)))
    times_ms = []
    for s in picks:
        t0 = time.perf_counter()
        with Image.open(s.path) as img:
            img.convert("RGB").resize((224, 224))
        times_ms.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times_ms)
    p90 = statistics.quantiles(times_ms, n=10)[-1]
    print(
        f"raw open+decode+resize: median {med:.1f} ms/img, p90 {p90:.1f} ms/img "
        f"({len(picks)} files, single process)",
        flush=True,
    )
    print(
        f"  -> one worker can do ~{1000 / med:.0f} img/s; "
        f"pipeline ceiling with N workers ~= N * that",
        flush=True,
    )


def bench_loader(loader, n_batches: int, label: str) -> None:
    it = iter(loader)
    t0 = time.perf_counter()
    images, _ = next(it)
    first = time.perf_counter() - t0
    print(
        f"{label}: first batch after {first:.1f}s (worker startup + prefetch)",
        flush=True,
    )

    window_start, window_samples = time.perf_counter(), 0
    n_samples = 0
    for batch_idx in range(1, n_batches):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        n_samples += images.size(0)
        if batch_idx % 20 == 0:
            now = time.perf_counter()
            rate = (n_samples - window_samples) / (now - window_start)
            print(f"{label}: batch {batch_idx} | {rate:.0f} img/s", flush=True)
            window_start, window_samples = now, n_samples


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    d = Config()
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--index-cache", type=Path, default=d.index_cache_path)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--batches", type=int, default=101, help="loader batches to time")
    p.add_argument(
        "--raw-files", type=int, default=300, help="files for the raw-read test"
    )
    args = p.parse_args()

    cfg = replace(
        Config(),
        data_dir=args.data_dir,
        index_cache_path=args.index_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    samples = ori.load_or_build_index(cfg)
    train_samples, _, _ = ori.split_samples(samples, cfg)
    print(
        f"{len(train_samples)} train samples | batch {cfg.batch_size} "
        f"| workers {cfg.num_workers}",
        flush=True,
    )

    bench_raw_reads(train_samples, args.raw_files)

    train_loader, _, _ = ori.make_loaders(train_samples, [], [], cfg)
    bench_loader(train_loader, args.batches, label=f"loader({cfg.num_workers}w)")


if __name__ == "__main__":
    main()
