#!/bin/bash
# Submit with:  sbatch train.slurm
# Detached full model-comparison run for bee-orientation regression.
# Unlike an interactive `srun --pty bash`, a batch job has no terminal attached,
# so it keeps running after you disconnect. Watch progress with:
#   tail -f /scratch/cvcdt004/checkpoints/logs/bee-orient-<jobid>.out
#
# Resources mirror the interactive session you already use:
#   srun --time=3-00:00:00 --mem=100G --cpus-per-task=20 -p interactive -G a30:4 --pty bash
#SBATCH --job-name=bee-orient
#SBATCH --partition=interactive
# #SBATCH --account=<ACCOUNT>           # uncomment + set ONLY if sbatch reports it needs an account
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # one process; DataParallel spans the GPUs itself
#SBATCH --gpus=a30:4                     # 4x A30, same as `-G a30:4`
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/cvcdt004/checkpoints/logs/bee-orient-%j.out
#SBATCH --error=/scratch/cvcdt004/checkpoints/logs/bee-orient-%j.err

set -euo pipefail

# The log dir in --output/--error above must exist BEFORE the job starts:
#   mkdir -p /scratch/cvcdt004/checkpoints/logs

DATA_DIR=/scratch/cvcdt011/data
CKPT_DIR=/scratch/cvcdt004/checkpoints     # writable output dir (checkpoints, tables, plots)

cd "$SLURM_SUBMIT_DIR"

echo "Host: $(hostname) | Job: ${SLURM_JOB_ID:-?} | GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

# Run via uv against the existing (already-synced) .venv. --no-sync avoids any
# network/index access from the compute node. `exec` replaces the shell so
# signals (e.g. SLURM time-limit) reach Python directly. Keep this the LAST line.
# Alternative without uv:  source .venv/bin/activate && exec python train.py ...
exec uv run --no-sync python train.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --num-workers 12
    # add --frame-size W H if the video isn't readable from the compute node
