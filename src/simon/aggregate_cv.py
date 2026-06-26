#!/usr/bin/env python3
"""Aggregate the crop-scale cross-validation runs from train_cv.slurm.

Reads run_summary.json from each run directory into one table with a
(crop set, model) row per training, so the effect of crop size is easy to read.
Runs still in progress (no run_summary.json yet) are skipped with a note, so
it's safe to run while the array job is only partially done.
See run-commands.md for usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="Checkpoint dirs of the individual runs (each holding run_summary.json).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional CSV path for the aggregated table.")
    args = p.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            print(f"skipping {run_dir}: no run_summary.json (run not finished?)", file=sys.stderr)
            continue
        summary = json.loads(summary_path.read_text())
        crop_set = summary["config"].get("crops_subdir", run_dir.name)
        crop_size = summary["config"].get("expected_crop_size")
        for model, metrics in summary["test_metrics"].items():
            rows.append({"crop_set": crop_set, "crop_size_px": crop_size, "model": model, **metrics})
        for model in summary.get("failed_models", []):
            print(f"note: {crop_set}/{model} failed during training", file=sys.stderr)

    if not rows:
        print("No finished runs found.", file=sys.stderr)
        return 1

    table = (
        pd.DataFrame(rows)
        .set_index(["crop_set", "model"])
        .sort_values("mae_deg")
    )
    print(table.round(2).to_string())

    best = table.index[0]
    print(f"\nBest overall: crop_set={best[0]}, model={best[1]} "
          f"(test MAE {table.iloc[0]['mae_deg']:.2f}°)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.out)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
