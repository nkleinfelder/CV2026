#!/usr/bin/env python
# coding: utf-8

# # Training ConvNeXt-Tiny for Bee Orientation Regression
# 
# This notebook implements a complete PyTorch pipeline to train a **ConvNeXt-Tiny** model to regress the orientation angle of cropped bee images.
# 
# ### Key Approach:
# 
# - **Circular Targets:** Rather than regressing raw angles directly (which introduces discontinuity at 0/360 degrees), we project angles to the unit circle: $\mathbf{y} = [\sin(\alpha), \cos(\alpha)]$.
# - **Evaluation:** Predictions are mapped back to degrees using `atan2` and validated with a wrap-around adjusted Mean Absolute Error (MAE).
# 
# ### HPC / Multi-GPU Strategy:
# 
# - **DistributedDataParallel (DDP):** Used when launched with `torchrun` (multiple processes, one per GPU). Scales linearly and is the recommended approach for HPC clusters.
# - **DataParallel fallback:** When running interactively with multiple GPUs but without `torchrun`.
# - **Mixed Precision (AMP):** `torch.amp.autocast` + `GradScaler` for ~2× throughput on Tensor Core GPUs.
# - **`torch.compile(mode="reduce-overhead")`:** Kernel fusion without the `max_autotune_gemm` pass (safe for GPUs with few SMs).
# - **Fast DataLoader:** `multiprocessing_context='fork'` (avoids Python 3.14 forkserver auth issue), persistent workers, and prefetch factor tuned for HPC I/O.
# - **Linear LR Scaling:** Learning rate is scaled linearly with the effective batch size (number of GPUs × per-GPU batch size).
# 

# ### 1. Imports
# 
# Load core machine learning frameworks, data processing utilities, model architectures, and plotting tools.

# In[3]:


import os
import math
import warnings
import logging
from pathlib import Path

# ── OMP_NUM_THREADS must be set BEFORE any OpenMP-backed library (numpy, torch)
# is imported. torchrun defaults this to 1 (safe but slow). We set it to the
# number of CPUs available per process so OpenMP parallelism is fully utilised.
# On SLURM, SLURM_CPUS_PER_TASK is the authoritative source; fall back to
# cpu_count() when running interactively.
if "OMP_NUM_THREADS" not in os.environ:
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    os.environ["OMP_NUM_THREADS"] = str(_cpus)

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Suppress expected / harmless warnings ────────────────────────────────────
# 1. torch.compile ignores profiler record_function() hooks by design.
warnings.filterwarnings("ignore", message=".*record_function.*")
# 2. Silence the noisy torch inductor / logging at WARNING level.
#    Errors are still shown (level=ERROR keeps those).
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._logging").setLevel(logging.ERROR)

print("Imports OK")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU count       : {torch.cuda.device_count()}")
print(f"OMP_NUM_THREADS : {os.environ['OMP_NUM_THREADS']}")


# ### 2. Distributed / Device Setup
# 
# Detect whether we are running under `torchrun` (DDP) or interactively.
# 
# - **DDP mode** (`torchrun --nproc_per_node=N train.py`): each process gets one GPU via `LOCAL_RANK`.
# - **Interactive / single-node multi-GPU**: falls back to `DataParallel` if >1 GPU present.
# - **Single GPU / CPU**: plain training.
# 
# > **Tip:** To launch DDP from the command line on SLURM, use:
# > ```bash
# > torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train_ddp.py
# > ```

# In[ ]:


# ── Distributed initialisation ──────────────────────────────────────────────
USE_DDP = "LOCAL_RANK" in os.environ  # set by torchrun

if USE_DDP:
    # Read LOCAL_RANK and set the device BEFORE init_process_group so we can
    # pass device_id= and silence the barrier() device-context warning.
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    IS_MAIN = global_rank == 0  # only rank-0 prints / saves checkpoints
else:
    local_rank  = 0
    global_rank = 0
    world_size  = 1
    IS_MAIN     = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if IS_MAIN:
    print(f"Mode      : {'DDP' if USE_DDP else 'Single-process'}")
    print(f"World size: {world_size}")
    print(f"Device    : {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir       = Path("/scratch/cvcdt011/data")
crops_dir      = data_dir / "crops"
trajectory_dir = data_dir / "rec1_trajectories"
CHECKPOINT_PATH = Path.cwd() / "convnext/best_convnext_tiny_bee_orientation.pth"


# In[16]:


# Check if it exists and is actually a folder (directory)
if CHECKPOINT_PATH.is_dir():
    print(f"Yes, the folder exists: {CHECKPOINT_PATH}")
else:
    print(f"No, folder does not exist at: {CHECKPOINT_PATH}")


# ### 3. Parse Trajectory Ground Truths
# 
# Load and read trajectory `.txt` files to construct a mapping from unique `(trajectory_id, frame_id)` combinations to their ground truth orientation angle (stored in degrees in the 5th column).

# In[ ]:


if IS_MAIN:
    print("Loading trajectories and building orientation angle lookup...")

angle_lookup = {}

for txt_file in sorted(list(trajectory_dir.glob("*.txt"))):
    tid = txt_file.stem
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                frame_id = int(parts[0])
                angle = float(parts[4])
                angle_lookup[(tid, frame_id)] = angle

if IS_MAIN:
    print(f"Loaded {len(angle_lookup)} coordinates from trajectory files.")


# ### 4. Custom Dataset Definition
# 
# Define a custom PyTorch `Dataset` class that maps image crop paths, reads files, and calculates the sine and cosine targets for orientation.

# In[ ]:


class BeeOrientationDataset(Dataset):
    def __init__(self, split_dir, angle_lookup, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.angle_lookup = angle_lookup
        self.samples = []

        # Iterate directory structure: crops/<split>/<trajectory_id>/frame_xxxxxx.png
        for traj_dir in self.split_dir.iterdir():
            if traj_dir.is_dir():
                tid = traj_dir.name
                for img_path in traj_dir.glob("*.png"):
                    try:
                        frame_id = int(img_path.stem.split("_")[1])
                        if (tid, frame_id) in self.angle_lookup:
                            self.samples.append((img_path, tid, frame_id))
                    except (IndexError, ValueError):
                        continue

        if IS_MAIN:
            print(f"Loaded {len(self.samples)} samples from {self.split_dir.name} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tid, frame_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # Transform degrees -> radians -> sin/cos targets
        angle_deg = self.angle_lookup[(tid, frame_id)]
        angle_rad = np.deg2rad(angle_deg)
        target = torch.tensor(
            [np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, target


# ### 5. Augmentations and DataLoaders
# 
# Prepare PyTorch normalization and resizing transforms and set up the corresponding DataLoaders.
# 
# **HPC tuning notes:**
# - `multiprocessing_context='fork'` — **critical on Python 3.14+**: the new default start method (`forkserver`) requires an auth handshake that breaks `persistent_workers`. `fork` is safe and fastest on Linux.
# - `num_workers` reads `SLURM_CPUS_PER_TASK` when available, otherwise `min(8, cpu_count())`.
# - `persistent_workers=True` keeps worker processes alive between epochs (avoids respawn overhead).
# - `prefetch_factor=4` pre-loads 4 batches per worker into memory ahead of time.
# - `pin_memory=True` enables faster CPU→GPU transfers via pinned (page-locked) memory.
# - `DistributedSampler` ensures each GPU sees a non-overlapping shard of data in DDP mode.
# - **Effective batch size** = `batch_size_per_gpu × world_size`; the LR is scaled accordingly.

# In[ ]:


# ── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE_PER_GPU = 64          # per-GPU batch size; increase if VRAM allows
NUM_EPOCHS         = 10
BASE_LR            = 1e-4        # LR for a single GPU with batch_size=64
WEIGHT_DECAY       = 1e-2
# Divide available CPUs by world_size so total workers across all GPU processes
# does not exceed the system CPU count (avoids DataLoader oversubscription warning).
_cpus_total = 20
NUM_WORKERS = max(1, _cpus_total // world_size)
USE_AMP            = torch.cuda.is_available()  # mixed precision
USE_COMPILE        = torch.__version__ >= "2.0"  # torch.compile (PyTorch >= 2.0)

# Effective LR scales linearly with the effective batch size (linear scaling rule)
effective_batch_size = BATCH_SIZE_PER_GPU * world_size
scaled_lr = BASE_LR * (effective_batch_size / 64)

if IS_MAIN:
    print(f"Batch per GPU      : {BATCH_SIZE_PER_GPU}")
    print(f"Effective batch    : {effective_batch_size}")
    print(f"Scaled LR          : {scaled_lr:.2e}")
    print(f"AMP enabled        : {USE_AMP}")
    print(f"torch.compile      : {USE_COMPILE}")
    print(f"DataLoader workers : {NUM_WORKERS}")

# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset = BeeOrientationDataset(crops_dir / "train", angle_lookup, transform=train_transform)
val_dataset   = BeeOrientationDataset(crops_dir / "val",   angle_lookup, transform=val_transform)
test_dataset  = BeeOrientationDataset(crops_dir / "test",  angle_lookup, transform=val_transform)

# ── Samplers (DDP shards data across GPUs) ───────────────────────────────────
train_sampler = DistributedSampler(train_dataset, shuffle=True)  if USE_DDP else None
val_sampler   = DistributedSampler(val_dataset,   shuffle=False) if USE_DDP else None
test_sampler  = DistributedSampler(test_dataset,  shuffle=False) if USE_DDP else None

# ── DataLoader kwargs ────────────────────────────────────────────────────────
# IMPORTANT: Python 3.14 changed the default multiprocessing start method from
# 'fork' to 'forkserver'. The forkserver method requires an authentication
# handshake that fails when combined with persistent_workers (ConnectionResetError
# errno 104). Forcing 'fork' is safe on Linux and is the fastest option.
_loader_kwargs = dict(
    batch_size              = BATCH_SIZE_PER_GPU,
    num_workers             = NUM_WORKERS,
    pin_memory              = True,
    multiprocessing_context = "fork" if NUM_WORKERS > 0 else None,
    persistent_workers      = NUM_WORKERS > 0,
    prefetch_factor         = 4 if NUM_WORKERS > 0 else None,
)

train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=(train_sampler is None), **_loader_kwargs)
val_loader   = DataLoader(val_dataset,   sampler=val_sampler,   shuffle=False,                  **_loader_kwargs)
test_loader  = DataLoader(test_dataset,  sampler=test_sampler,  shuffle=False,                  **_loader_kwargs)

if IS_MAIN:
    print(f"Train batches/GPU: {len(train_loader)} | Val batches/GPU: {len(val_loader)}")


# ### 6. Model Initialization
# 
# Load ConvNeXt-Tiny with ImageNet weights and swap out its final classifier to output 2 values (sin, cos).
# 
# **Multi-GPU wrapping:**
# - `gradient_as_bucket_view=True` in DDP makes gradient tensors share memory with DDP communication buckets, which fixes the grad-strides mismatch warning that appears when combining DDP with `torch.compile`.
# - `torch.compile(mode="reduce-overhead")` enables kernel fusion and CUDA graph capture without triggering the `max_autotune_gemm` pass (which requires many Streaming Multiprocessors and emits a warning on smaller GPUs).

# In[ ]:


if IS_MAIN:
    print("Initializing ConvNeXt-Tiny model...")

model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# Replace classifier head: 768 -> 2 (sin, cos)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, 2)

model = model.to(device)

# ── Wrap for multi-GPU ────────────────────────────────────────────────────────
if USE_DDP:
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        # gradient_as_bucket_view=True: gradients share memory with DDP's
        # all-reduce buckets. This eliminates the "Grad strides do not match
        # bucket view strides" warning that occurs when DDP is combined with
        # torch.compile, and also reduces peak memory usage.
        gradient_as_bucket_view=True,
    )
    if IS_MAIN:
        print(f"DDP: wrapped model across {world_size} GPUs")
elif torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    if IS_MAIN:
        print(f"DataParallel: using {torch.cuda.device_count()} GPUs")
else:
    if IS_MAIN:
        print(f"Single device: {device}")

# ── torch.compile (PyTorch >= 2.0 with CUDA) ──────────────────────────────────
# mode="reduce-overhead": uses CUDA graphs to reduce kernel launch overhead.
# Avoids the max_autotune_gemm pass (which needs many SMs and warns otherwise).
# Use mode="max-autotune" only on high-end GPUs (A100/H100) with many SMs.
if USE_COMPILE and torch.cuda.is_available():
    model = torch.compile(model, mode="reduce-overhead")
    if IS_MAIN:
        print("torch.compile(mode='reduce-overhead') enabled")


# ### 7. Loss, Optimizer, Scaler, and Scheduler
# 
# - **MSE loss** on the (sin, cos) vector.
# - **AdamW** with the linearly-scaled LR.
# - **GradScaler** for AMP (mixed precision) — prevents gradient underflow in fp16.
# - **CosineAnnealingLR** scheduler.

# In[ ]:


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# GradScaler for automatic mixed precision -- no-op when USE_AMP=False
scaler = torch.amp.GradScaler(enabled=USE_AMP)


# ### 8. Custom Angular Metric
# 
# Mean Absolute Error (MAE) adjusted for circular wrapping to accurately score prediction error in degrees.

# In[ ]:


def compute_angular_error_deg(pred, target):
    """Computes Mean Absolute Error (MAE) in degrees using circular wrapping."""
    pred_rad   = torch.atan2(pred[:, 0],   pred[:, 1])
    target_rad = torch.atan2(target[:, 0], target[:, 1])

    diff_rad = pred_rad - target_rad
    diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))

    diff_deg = torch.abs(torch.rad2deg(diff_rad))
    return diff_deg.mean().item()


# ### 9. Training and Validation Step Helpers
# 
# Loop functions to run training and evaluation epochs with:
# - **AMP autocast** for mixed-precision forward passes.
# - **GradScaler** for safe fp16 backward pass.
# - **DDP sampler epoch reset** to re-shuffle data correctly each epoch.

# In[ ]:


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()

    # DDP: update sampler seed so each GPU sees a different shuffle each epoch
    if USE_DDP and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    running_loss = 0.0
    running_mae  = 0.0

    pbar = tqdm(loader, desc=f"Train E{epoch+1}", leave=False, disable=not IS_MAIN)
    for images, targets in pbar:
        images  = images.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # faster than zeroing

        # ── Mixed precision forward ──────────────────────────────────────────
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                enabled=USE_AMP):
            outputs = model(images)
            loss    = criterion(outputs, targets)

        # ── Scaled backward + step ───────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_mae  += compute_angular_error_deg(outputs.float(), targets.float()) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae  = running_mae  / len(loader.dataset)
    return epoch_loss, epoch_mae


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae  = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False, disable=not IS_MAIN)
        for images, targets in pbar:
            images  = images.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                    enabled=USE_AMP):
                outputs = model(images)
                loss    = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            running_mae  += compute_angular_error_deg(outputs.float(), targets.float()) * images.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_mae  = running_mae  / len(loader.dataset)
    return val_loss, val_mae


# ### 10. Main Training Loop
# 
# Execute training across epochs, update validation performance logs, and cache the best checkpoint on disk.
# 
# > **Checkpoint note:** Only `rank 0` saves checkpoints to avoid concurrent writes from multiple processes.

# In[ ]:


best_val_loss = float("inf")
history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}

if IS_MAIN:
    print("Starting training...")
    print(f"  GPUs        : {world_size}")
    print(f"  Epochs      : {NUM_EPOCHS}")
    print(f"  LR (scaled) : {scaled_lr:.2e}")
    print(f"  AMP         : {USE_AMP}")

for epoch in range(NUM_EPOCHS):
    train_loss, train_mae = train_epoch(
        model, train_loader, criterion, optimizer, scaler, device, epoch
    )
    val_loss, val_mae = validate(model, val_loader, criterion, device)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_mae"].append(train_mae)
    history["val_loss"].append(val_loss)
    history["val_mae"].append(val_mae)

    if IS_MAIN:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
        print(f"  Train Loss (MSE): {train_loss:.4f} | Train MAE: {train_mae:.2f}\u00b0")
        print(f"  Val   Loss (MSE): {val_loss:.4f}   | Val   MAE: {val_mae:.2f}\u00b0")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Unwrap DDP/DataParallel to save only the raw model weights
            raw_model = model
            if isinstance(model, (DDP, nn.DataParallel)):
                raw_model = model.module
            # Strip torch.compile wrapper if present
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                CHECKPOINT_PATH,
            )
            print("  --> Saved best model checkpoint!")

    # Synchronise processes at end of each epoch
    if USE_DDP:
        dist.barrier()

if USE_DDP:
    dist.destroy_process_group()


# ### 11. Plot Training History
# 
# Plot trends for training/validation losses and angular MAEs to evaluate model optimisation and trace convergence.

# In[ ]:


if IS_MAIN:
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"],   label="Val Loss")
    plt.title("MSE Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_mae"], label="Train MAE (\u00b0)")
    plt.plot(epochs_range, history["val_mae"],   label="Val MAE (\u00b0)")
    plt.title("Mean Absolute Angular Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (Degrees)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()


# ### 12. Evaluate on Test Split
# 
# Load the saved model parameters that achieved the lowest validation loss, run evaluation on the test DataLoader, and report final metrics.
# 
# > The checkpoint stores a dictionary; we load only `model_state_dict` here.

# In[ ]:


if IS_MAIN:
    # Reload a clean model for evaluation
    eval_model = convnext_tiny()
    in_features = eval_model.classifier[2].in_features
    eval_model.classifier[2] = nn.Linear(in_features, 2)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model = eval_model.to(device)

    test_loss, test_mae = validate(eval_model, test_loader, criterion, device)
    print(f"\nFinal Test Results (checkpoint from epoch {checkpoint['epoch']}):")
    print(f"  Test Loss (MSE): {test_loss:.4f}")
    print(f"  Test MAE       : {test_mae:.2f}\u00b0")


# ### 13. SLURM Submission Example
# 
# To run this as a **DDP job** on a SLURM cluster, convert this notebook to a `.py` script first:
# 
# ```bash
# jupyter nbconvert --to script train.ipynb --output train_ddp
# ```
# 
# Then submit with a SLURM batch script like:
# 
# ```bash
# #!/bin/bash
# #SBATCH --job-name=bee_orientation
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4          # one task per GPU
# #SBATCH --gres=gpu:4                 # request 4 GPUs
# #SBATCH --cpus-per-task=8            # NUM_WORKERS per GPU (= OMP_NUM_THREADS)
# #SBATCH --mem=64G
# #SBATCH --time=04:00:00
# #SBATCH --output=slurm-%j.out
# 
# module load python cuda
# source .venv/bin/activate
# 
# # Set OMP_NUM_THREADS to match CPUs allocated per task.
# # This suppresses the torchrun W0715 warning and enables full OpenMP parallelism.
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# 
# torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train_ddp.py
# ```
