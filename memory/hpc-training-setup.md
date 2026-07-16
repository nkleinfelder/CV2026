---
name: hpc-training-setup
description: RAMSES HPC cluster setup and the measured single-GPU vs DataParallel throughput for the bee-orientation project
metadata:
  type: project
---

Bee-orientation training runs on the RAMSES cluster (user `cvcdt004`, repo at `~/CV2026`). Data lives on another account's scratch: `/scratch/cvcdt011/data` (~1M crop files, 710k train samples); outputs go to `/scratch/cvcdt004/checkpoints`.

Key finding (2026-07-16, diagnosed with `bench_loader.py`): the input pipeline is fast (raw reads 2.7 ms/img; 12-worker loader delivers 3000–10000 img/s), but `nn.DataParallel` over 4x A30 collapsed training to ~255 img/s for resnet18-class models. One A30 (`-G a30:1`, no DataParallel) trains at ~2250 img/s — ~5 min/epoch. Don't recommend multi-GPU for these small models; the fix history is in the repo.

User launches interactive runs from tmux on a login node (`srun ... -p interactive -G a30:1 --pty bash`), pipes output through `tee` (which disables tqdm — plain-print heartbeats in `train_one_epoch` cover that), and has `train.slurm` for detached runs.
