# Repository overview

Computer-vision project: predicting the **head orientation of individual bees**
from a top-down recording. A tracker gives us per-bee trajectory files (frame,
position, orientation angle); from those we cut a small fixed-size crop around
each bee in each frame and train CNNs to regress the heading. Orientation is
predicted as `[sin θ, cos θ]` and decoded back to an angle, and models are scored
with circular angular error (MAE in degrees, accuracy within 15/30/45°).

## Layout

```
.
├── overview.md              # this file
├── README.md                # setup notes (HPC venv symlink, half_size)
├── run-commands.md          # exact commands to preprocess and train
├── experiment_overview.csv  # aggregated metrics for every experiment we ran
├── pyproject.toml / uv.lock # dependencies (managed with uv)
│
├── src/
│   ├── preprocessing.py     # crop extraction from the video + trajectory files
│   ├── preprocessing.ipynb  # same pipeline as a notebook, plus outlier-inspection widgets
│   │
│   ├── common/
│   │   └── bee_dataset.py   # shared: sample indexing, train/val/test split, Dataset
│   │
│   ├── simon/               # the full training pipeline (produces the reported results); a package
│   │   ├── __init__.py      #   marks simon as an importable package
│   │   ├── config.py        #   the run Config dataclass (every training knob)
│   │   ├── orientation.py   #   core: models, losses/metrics, train & eval loops
│   │   ├── train.py         #   entry point — trains and compares several architectures
│   │   ├── train.slurm      #   SLURM job for a single detached run
│   │   ├── train_cv.slurm   #   SLURM array: cross-validation over crop sizes
│   │   ├── aggregate_cv.py  #   collect the cross-validation runs into one table
│   │   ├── bench_loader.py  #   profile the input pipeline (find data-loading bottlenecks)
│   │   └── checkpoints/     #   example outputs of a run (weights, metrics, plots)
│   │
│   ├── ronny/               # data-cleaning helpers only (not the training pipeline)
│   │   ├── classify.py      #   run a trained model over crops and score against ground truth
│   │   └── train.ipynb      #   exploratory notebook — NOT used to generate the results
│   │
│   ├── convnext/            # only a training-history figure is tracked (DDP script is gitignored)
│   │
│   └── data/                # a cleaned version of the dataset made with a different approach;
│                            # rec1_trajectories/ is tracked, the video itself is not
│
├── output/                  # results kept in the repo
│   ├── training/summaries/  #   run_summary.json per experiment (01…05)
│   ├── training/copy-files.sh  # fetch those summaries off the HPC
│   ├── preprocessing/       #   example crop / outlier figures
│   └── loss.pg.png
│
└── sample_index_crops_not_cleaned/   # cached sample-index CSVs (80/100/140 px crops)
```

## How the results are produced

1. **Preprocess** (`src/preprocessing.py`): stream the video once, drop trajectory
   frames whose position or orientation jumps (z-score outlier filter), and write
   one PNG crop per bee per frame to `<data-dir>/<crops-subdir>/<traj_id>/`.
   The crop side length is `2 * half_size`, which is how the differently sized
   crop sets (80/100/140 px) are made.
2. **Train** (`src/simon/train.py`): index the crops via
   `common/bee_dataset.py`, split by *trajectory* so consecutive frames of the
   same bee can't leak between splits, then train and compare the models. Each run
   writes checkpoints, per-epoch history, a comparison table and diagnostic plots
   to its checkpoint dir (see `src/simon/checkpoints/` for an example). On the
   HPC this is submitted with `train.slurm` / `train_cv.slurm`.
3. **Aggregate**: `aggregate_cv.py` merges the cross-validation runs, and the
   per-experiment `run_summary.json` files under `output/training/summaries/` are
   collected into `experiment_overview.csv`.

The experiments in `experiment_overview.csv` (`01-initial-training` through
`05-oversampling-100`) track the effect of data cleaning, crop size, the loss
function, and orientation oversampling.

## Running it

All commands (preprocessing, single training runs, cross-validation) with their
flags are in **`run-commands.md`**. Dependencies are managed with `uv`; on the
HPC use `uv run --no-sync` so the compute node doesn't touch the network.
