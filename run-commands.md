# Run commands

All commands run from the repo root via `uv`. On the HPC add `--no-sync` to
`uv run` so no network/index access happens from a compute node (the `.venv`
must already be synced once via `uv sync` on the login node).

The project installs as an editable package on `uv sync` (see `pyproject.toml`),
exposing `common` and `simon` as importable packages — that is what lets the
scripts import each other without any `sys.path` fiddling.

## Crop generation

`src/preprocessing.py` (script version of `src/preprocessing.ipynb`) reads the
video + trajectory files from `--data-dir` and writes crops to
`<data-dir>/<crops-subdir>/<traj_id>/frame_XXXXXX.png`. The crop side length is
`2 * --half-size`. It is resumable — existing crops are skipped. Every flag is
optional; omitted flags fall back to the notebook's hardcoded values
(data dir `/scratch/cvcdt011/data` or local `src/data`, output `crops`, half-size 50).

```bash
# Notebook defaults: 100px crops into <data-dir>/crops
uv run --no-sync python src/preprocessing.py

# Larger crops (150px) into <data-dir>/crops_large
uv run --no-sync python src/preprocessing.py \
    --data-dir /scratch/cvcdt011/data \
    --crops-subdir crops_large \
    --half-size 75

# Smaller crops (50px) into <data-dir>/crops_small
uv run --no-sync python src/preprocessing.py \
    --data-dir /scratch/cvcdt011/data \
    --crops-subdir crops_small \
    --half-size 25

-------

# 80px crops
uv run --no-sync python src/preprocessing.py \
    --data-dir /scratch/cvcdt011/data \
    --trajectories-subdir rec1_trajectories \
    --crops-subdir crops_not_cleaned/80 \
    --half-size 40

# 100px crops
uv run --no-sync python src/preprocessing.py \
    --data-dir /scratch/cvcdt011/data \
    --trajectories-subdir rec1_trajectories \
    --crops-subdir crops_not_cleaned/100 \
    --half-size 50

# 140px crops
uv run --no-sync python src/preprocessing.py \
    --data-dir /scratch/cvcdt011/data \
    --trajectories-subdir rec1_trajectories \
    --crops-subdir crops_not_cleaned/140 \
    --half-size 70

srun --partition=interactive --gpus=a30:1 --cpus-per-task=20 --mem=100G --time=3-00:00:00 --pty \
  uv run --no-sync python src/simon/train.py \
#    --loss-fn cbrt_cos \
#    --oversample-orientation \
    --data-dir /scratch/cvcdt011/data \
    --crops-subdir crops_after_manual_removal/100 \
    --checkpoint-dir /scratch/cvcdt011/checkpoints/crops_after_manual_removal_100_err_cbrt_oversampling \
    --index-cache /scratch/cvcdt011/checkpoints/crops_after_manual_removal_100_err_cbrt_oversampling/sample_index.csv \
    --num-workers 12 \
    --models resnet18 resnet50 convnext_tiny mobilenetv4_conv_medium
```

Useful extra flags: `--video-name`, `--trajectories-subdir` (input names under
`--data-dir`), `--z-threshold` (outlier filter), `--num-writers` (PNG writer threads).

## Training — single run (one crops directory)

Direct (e.g. inside an interactive `srun` session with a GPU):

```bash
# Default crop set (crops/), default models (resnet18 resnet50 mobilenetv4_conv_medium)
uv run --no-sync python src/simon/train.py \
    --data-dir /scratch/cvcdt011/data \
    --checkpoint-dir /scratch/cvcdt004/checkpoints \
    --num-workers 12

# Specific crop set and specific models
uv run --no-sync python src/simon/train.py \
    --data-dir /scratch/cvcdt011/data \
    --crops-subdir crops_not_cleaned/140 \
    --checkpoint-dir /scratch/cvcdt004/checkpoints/cv_crops_not_cleaned_140 \
    --models convnext_tiny resnet18 resnet50 mobilenetv4_conv_medium \
    --num-workers 12
```

Notes:

- The crop size is auto-detected from the first PNG (override: `--expected-crop-size N`).
- The sample-index cache defaults to `sample_index.csv` for `crops/` and
  `sample_index_<subdir>.csv` otherwise; delete it (or pass `--no-index-cache`)
  after regenerating crops.
- Other common flags: `--epochs`, `--batch-size`, `--lr`, `--image-size`,
  `--rotation-augment`, `--no-pretrained`, `--no-progress`.

## Training — cross validation over crop scales

Run one interactive `train.py` invocation per crop size
(`crops_after_manual_removal/{80,100,140}`), each in its own checkpoint dir:

```bash
srun --partition=interactive --gpus=a30:1 --cpus-per-task=20 --mem=100G --time=3-00:00:00 --pty \
  uv run --no-sync python src/simon/train.py \
    --data-dir /scratch/cvcdt011/data \
    --crops-subdir crops_after_manual_removal/80 \
    --checkpoint-dir /scratch/cvcdt004/checkpoints/cv_crops_after_manual_removal_80 \
    --num-workers 12

# repeat with --crops-subdir crops_after_manual_removal/100 and /140,
# each with its own --checkpoint-dir
```

Use the same seed and trajectory files across the three runs so the
train/val/test split is identical — results isolate the effect of crop size.

Aggregate the finished runs into one comparison table:

```bash
uv run --no-sync python src/simon/aggregate_cv.py /scratch/cvcdt004/checkpoints/cv_*
uv run --no-sync python src/simon/aggregate_cv.py /scratch/cvcdt004/checkpoints/cv_* --out cv_comparison.csv
```
