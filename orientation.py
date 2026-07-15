"""Shared logic for bee-orientation model training and evaluation.

This is the single source of truth used by both `orientation_training.ipynb`
(interactive exploration + plots) and `train.py` (detached HPC runs). The dataset
classes live in `bee_dataset.py` so DataLoader workers can import them; everything
else — config, indexing, splitting, models, loss/metrics, and the train/eval
loops — lives here.

Design notes carried over from the notebook:
- Orientation is regressed as `[sin(k*theta), cos(k*theta)]` (k=2 for axial
  labels, else 1) and decoded with `atan2(sin, cos) / k`. This order is assumed
  everywhere.
- The loss is a bounded cosine loss `1 - cos(k*delta_theta)`, robust to the
  occasional head/tail-flipped label.
- Plot helpers take a `save_path`: pass a path to write a PNG (headless scripts)
  or leave it None to `plt.show()` inline (notebook).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from bee_dataset import BeeOrientationDataset, Sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- data ---
    data_dir: Path = Path("/scratch/cvcdt011/data")
    crops_subdir: str = "crops"
    trajectories_subdir: str = "rec1_trajectories"
    video_name: str = "rec1.mp4"
    # Label convention, confirmed visually via the gate cell: orientation_angle
    # is a compass bearing in degrees (0 = up, 90 = right, clockwise on screen).
    angles_in_degrees: bool = True
    # True if orientation_angle is axial (head/tail ambiguous, e.g. from an ellipse
    # fit): the model then regresses [sin(2*theta), cos(2*theta)] and all primary
    # error metrics are computed mod 180 degrees.
    axial_labels: bool = False
    # Maps orientation_angle into display coordinates (x right, y down), where all
    # downstream code assumes theta = atan2(dy, dx). See ANGLE_REMAPS:
    #   "compass"     0=up, 90=right, clockwise   -> theta - pi/2   (our data)
    #   "ydown"       already display convention (identity)
    #   "yup"         y-up / math convention      -> -theta
    #   "transposed"  axes swapped                -> pi/2 - theta
    angle_convention: str = "compass"
    expected_crop_size: int = 100     # 2 * half_size from preprocessing.py
    filter_partial_crops: bool = True # drop border crops smaller than expected_crop_size
    # Source frame (width, height) for the border-crop check; read from the video
    # if None. Set manually if the video is not accessible from this node.
    frame_size: tuple[int, int] | None = None
    index_cache_path: Path = Path("sample_index.csv")
    use_index_cache: bool = True      # delete the cache file after changing data settings

    # --- split ---
    split_strategy: str = "trajectory"  # "trajectory" (no temporal leakage) or "random"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    # --- input pipeline ---
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    # Rotation augmentation rotates the patch and adjusts theta accordingly.
    # It assumes theta is measured as atan2(dy, dx) in image coordinates (y pointing
    # down). Verify the label convention before enabling, otherwise labels get corrupted.
    rotation_augment: bool = False

    # --- models / training ---
    model_names: tuple[str, ...] = ("resnet18", "resnet50", "mobilenetv4_conv_medium")
    pretrained: bool = True
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    # Model weights (~50-100 MB each) belong on scratch, not in the quota-limited
    # home directory / repo.
    checkpoint_dir: Path = Path("/scratch/cvcdt004/checkpoints")

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / self.crops_subdir

    @property
    def trajectories_dir(self) -> Path:
        return self.data_dir / self.trajectories_subdir

    @property
    def video_path(self) -> Path:
        return self.data_dir / self.video_name

    @property
    def angle_multiplier(self) -> float:
        """2.0 for axial labels (period 180°), 1.0 for directional labels."""
        return 2.0 if self.axial_labels else 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Sample index
# --------------------------------------------------------------------------- #
# Remaps a raw label angle into display convention (see Config.angle_convention).
# "compass" is our data: bearing 0=up, 90=right (clockwise), i.e. the on-screen
# direction is (sin t, -cos t), which equals display angle t - pi/2.
ANGLE_REMAPS: dict[str, Callable[[float], float]] = {
    "compass": lambda t: t - math.pi / 2,
    "ydown": lambda t: t,
    "yup": lambda t: -t,
    "transposed": lambda t: math.pi / 2 - t,
}


def get_frame_size(cfg: Config) -> tuple[int, int]:
    """Source frame (width, height), from Config or read once from the video."""
    if cfg.frame_size is not None:
        return cfg.frame_size
    import cv2

    cap = cv2.VideoCapture(str(cfg.video_path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"Could not read frame size from {cfg.video_path} — set Config.frame_size manually."
        )
    return w, h


def load_samples(cfg: Config) -> tuple[list[Sample], int]:
    """Build the (crop path, orientation) index from the trajectory files.

    Returns (samples, n_partial_dropped). Avoids per-file I/O: crop existence
    is checked against one directory listing per trajectory, and border crops
    are detected from the bee position (same clipping rule as preprocessing.py).
    """
    half = cfg.expected_crop_size // 2
    frame_w, frame_h = get_frame_size(cfg) if cfg.filter_partial_crops else (0, 0)
    remap = ANGLE_REMAPS[cfg.angle_convention]

    samples: list[Sample] = []
    n_partial = 0
    for traj_file in sorted(cfg.trajectories_dir.glob("*.txt")):
        traj_id = traj_file.stem
        crop_dir = cfg.crops_dir / traj_id
        if not crop_dir.exists():
            continue
        existing = {entry.name for entry in os.scandir(crop_dir)}
        for line in traj_file.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            frame = int(parts[0])
            filename = f"frame_{frame:06d}.png"
            if filename not in existing:
                continue
            if cfg.filter_partial_crops:
                # preprocessing.py indexes crop *rows* with position_x and
                # *columns* with position_y, so x is clipped against the frame
                # height and y against the width.
                x = int(round(float(parts[1])))
                y = int(round(float(parts[2])))
                if x - half < 0 or x + half > frame_h or y - half < 0 or y + half > frame_w:
                    n_partial += 1
                    continue
            theta = float(parts[4])
            if cfg.angles_in_degrees:
                theta = math.radians(theta)
            theta = remap(theta)
            samples.append(Sample(path=crop_dir / filename, theta=theta, traj_id=traj_id))
    return samples, n_partial


def spot_check_crop_sizes(samples: list[Sample], cfg: Config, n: int = 50) -> None:
    """Verify on a few random kept crops that the geometric filter matches reality."""
    for s in random.Random(cfg.seed).sample(samples, min(n, len(samples))):
        with Image.open(s.path) as img:
            assert img.size == (cfg.expected_crop_size, cfg.expected_crop_size), (
                f"{s.path} has size {img.size}, expected full "
                f"{cfg.expected_crop_size}px crop — geometric border filter "
                "disagrees with the actual crops; check frame_size."
            )


def load_or_build_index(cfg: Config) -> list[Sample]:
    if cfg.use_index_cache and cfg.index_cache_path.exists():
        df = pd.read_csv(cfg.index_cache_path)
        samples = [
            Sample(path=Path(p), theta=float(t), traj_id=str(tid))
            for p, t, tid in zip(df["path"], df["theta"], df["traj_id"])
        ]
        print(f"Loaded {len(samples)} samples from cache {cfg.index_cache_path}")
        return samples

    samples, n_partial = load_samples(cfg)
    print(f"Indexed {len(samples)} samples from {len({s.traj_id for s in samples})} trajectories")
    if cfg.filter_partial_crops:
        print(f"Dropped {n_partial} partial border crops (geometric check)")
        spot_check_crop_sizes(samples, cfg)
    if cfg.use_index_cache:
        pd.DataFrame(
            {
                "path": [str(s.path) for s in samples],
                "theta": [s.theta for s in samples],
                "traj_id": [s.traj_id for s in samples],
            }
        ).to_csv(cfg.index_cache_path, index=False)
        print(f"Cached index to {cfg.index_cache_path}")
    return samples


# --------------------------------------------------------------------------- #
# Train / val / test split
# --------------------------------------------------------------------------- #
def split_samples(
    samples: list[Sample], cfg: Config
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(cfg.seed)
    if cfg.split_strategy == "trajectory":
        traj_ids = sorted({s.traj_id for s in samples})
        rng.shuffle(traj_ids)
        n = len(traj_ids)
        n_test = max(1, round(n * cfg.test_fraction))
        n_val = max(1, round(n * cfg.val_fraction))
        test_ids = set(traj_ids[:n_test])
        val_ids = set(traj_ids[n_test : n_test + n_val])
        train = [s for s in samples if s.traj_id not in test_ids | val_ids]
        val = [s for s in samples if s.traj_id in val_ids]
        test = [s for s in samples if s.traj_id in test_ids]
    elif cfg.split_strategy == "random":
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = round(n * cfg.test_fraction)
        n_val = round(n * cfg.val_fraction)
        test = shuffled[:n_test]
        val = shuffled[n_test : n_test + n_val]
        train = shuffled[n_test + n_val :]
    else:
        raise ValueError(f"Unknown split strategy: {cfg.split_strategy!r}")

    assert train and val and test, (
        f"Empty split (train={len(train)}, val={len(val)}, test={len(test)}) — "
        "too few trajectories for the requested fractions."
    )
    return train, val, test


# --------------------------------------------------------------------------- #
# Dataset & dataloaders
# --------------------------------------------------------------------------- #
def make_loaders(
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    cfg: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
        )

    return (
        loader(BeeOrientationDataset(train_samples, cfg, augment=True), shuffle=True),
        loader(BeeOrientationDataset(val_samples, cfg), shuffle=False),
        loader(BeeOrientationDataset(test_samples, cfg), shuffle=False),
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def create_model(name: str, cfg: Config) -> nn.Module:
    return timm.create_model(name, pretrained=cfg.pretrained, num_classes=2)


def unwrap(model: nn.Module) -> nn.Module:
    """The underlying module, whether or not `model` is a DataParallel wrapper.

    DataParallel prefixes state_dict keys with `module.`; always checkpointing the
    unwrapped module keeps the weights loadable into a plain (single-GPU) model.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


# --------------------------------------------------------------------------- #
# Loss & metrics
# --------------------------------------------------------------------------- #
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cosine_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1 - cos(k * delta_theta): the raw (N, 2) output is L2-normalized onto the
    unit circle and compared to the unit target, so the loss depends only on
    the angular error — ~ (k*delta)^2 / 2 near zero, flattening towards the
    period boundary (flipped-label outliers contribute a bounded loss of 2).
    """
    return (1.0 - F.cosine_similarity(outputs, targets, dim=1)).mean()


def angles_from_sincos(output: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Decode theta from a (N, 2) [sin, cos] prediction."""
    return torch.atan2(output[:, 0], output[:, 1]) / cfg.angle_multiplier


def signed_angular_error(theta_true: torch.Tensor, theta_pred: torch.Tensor) -> torch.Tensor:
    """Wrap-around-safe directional error in (-pi, pi]."""
    diff = theta_true - theta_pred
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def signed_axial_error(theta_true: torch.Tensor, theta_pred: torch.Tensor) -> torch.Tensor:
    """Error mod 180 degrees, in (-pi/2, pi/2] — ignores head/tail flips."""
    diff = theta_true - theta_pred
    return torch.atan2(torch.sin(2.0 * diff), torch.cos(2.0 * diff)) / 2.0


def angular_metrics(theta_true: np.ndarray, theta_pred: np.ndarray, cfg: Config) -> dict[str, float]:
    t_true = torch.from_numpy(theta_true)
    t_pred = torch.from_numpy(theta_pred)
    axial_err = signed_axial_error(t_true, t_pred)
    # In axial mode the directional error is meaningless — the axial error is primary.
    primary_err = axial_err if cfg.axial_labels else signed_angular_error(t_true, t_pred)
    abs_deg = torch.rad2deg(primary_err.abs())
    return {
        "mae_deg": abs_deg.mean().item(),
        "median_ae_deg": abs_deg.median().item(),
        "rmse_deg": torch.sqrt(torch.rad2deg(primary_err).pow(2).mean()).item(),
        "axial_mae_deg": torch.rad2deg(axial_err.abs()).mean().item(),
        "acc15_deg": (abs_deg <= 15).float().mean().item() * 100,
        "acc30_deg": (abs_deg <= 30).float().mean().item() * 100,
        "acc45_deg": (abs_deg <= 45).float().mean().item() * 100,
    }


def metrics_for_predictions(
    theta_true: np.ndarray, theta_pred: np.ndarray, cfg: Config
) -> dict[str, float]:
    """Full metric set (incl. cosine loss) for arbitrary angle predictions."""
    k = cfg.angle_multiplier
    # For unit vectors the cosine loss reduces to 1 - cos(k * delta_theta).
    loss = float(np.mean(1.0 - np.cos(k * (theta_true - theta_pred))))
    return {"loss": loss, **angular_metrics(theta_true, theta_pred, cfg)}


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def circular_mean(theta: np.ndarray, cfg: Config) -> float:
    k = cfg.angle_multiplier
    return float(np.arctan2(np.sin(k * theta).mean(), np.cos(k * theta).mean()) / k)


def baseline_metrics(
    train_samples: list[Sample], test_samples: list[Sample], cfg: Config
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    train_thetas = np.array([s.theta for s in train_samples])
    test_thetas = np.array([s.theta for s in test_samples])

    mean_pred = np.full_like(test_thetas, circular_mean(train_thetas, cfg))
    random_pred = rng.uniform(-math.pi, math.pi, size=len(test_thetas))

    return {
        "baseline: circular mean": metrics_for_predictions(test_thetas, mean_pred, cfg),
        "baseline: uniform random": metrics_for_predictions(test_thetas, random_pred, cfg),
    }


# --------------------------------------------------------------------------- #
# Training & evaluation loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: LossFn,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    desc: str = "train",
    progress: bool = True,
) -> float:
    model.train()
    total_loss, n_samples = 0.0, 0
    batches = tqdm(loader, desc=desc, unit="batch", leave=False, disable=not progress)
    for images, targets in batches:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        batches.set_postfix(loss=f"{total_loss / n_samples:.4f}")
    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: LossFn,
    cfg: Config,
    desc: str = "eval",
    progress: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Returns (metrics, theta_true, theta_pred) over the whole loader."""
    model.eval()
    total_loss, n_samples = 0.0, 0
    trues: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    for images, targets in tqdm(loader, desc=desc, unit="batch", leave=False, disable=not progress):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        trues.append(angles_from_sincos(targets.float(), cfg).cpu())
        preds.append(angles_from_sincos(outputs.float(), cfg).cpu())
    theta_true = torch.cat(trues).numpy()
    theta_pred = torch.cat(preds).numpy()
    metrics = {"loss": total_loss / n_samples, **angular_metrics(theta_true, theta_pred, cfg)}
    return metrics, theta_true, theta_pred


@dataclass
class RunResult:
    model_name: str
    history: pd.DataFrame
    test_metrics: dict[str, float]
    theta_true: np.ndarray
    theta_pred: np.ndarray
    checkpoint_path: Path


def train_and_evaluate(
    name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    progress: bool = True,
) -> RunResult:
    print(f"\n=== {name} ===", flush=True)
    seed_everything(cfg.seed)

    # 1. Create the model and send to the primary device
    model = create_model(name, cfg).to(DEVICE)

    # 2. Replicate across GPUs if available (outputs gather onto the primary GPU)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!", flush=True)
        model = nn.DataParallel(model)

    criterion = cosine_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=cfg.use_amp and DEVICE.type == "cuda")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint the *unwrapped* module so weights load into a plain model later.
    checkpoint_path = cfg.checkpoint_dir / f"{name}_best.pt"
    history_path = cfg.checkpoint_dir / f"{name}_history.csv"

    history: list[dict[str, float]] = []
    best_val_mae = float("inf")

    epochs = tqdm(range(1, cfg.epochs + 1), desc=name, unit="epoch", disable=not progress)
    for epoch in epochs:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, cfg,
            desc=f"epoch {epoch}/{cfg.epochs} · train", progress=progress,
        )
        val_metrics, _, _ = evaluate(
            model, val_loader, criterion, cfg,
            desc=f"epoch {epoch}/{cfg.epochs} · val", progress=progress,
        )
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        pd.DataFrame(history).to_csv(history_path, index=False)

        marker = ""
        if val_metrics["mae_deg"] < best_val_mae:
            best_val_mae = val_metrics["mae_deg"]
            torch.save(unwrap(model).state_dict(), checkpoint_path)
            marker = "  *"
        epochs.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_mae=f"{val_metrics['mae_deg']:.2f}°",
            best=f"{best_val_mae:.2f}°",
        )
        # One clean line per epoch — survives non-TTY (SLURM) logs where the bars
        # above are disabled.
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val MAE {val_metrics['mae_deg']:6.2f}°{marker}",
            flush=True,
        )

    unwrap(model).load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    test_metrics, theta_true, theta_pred = evaluate(
        model, test_loader, criterion, cfg, desc=f"{name} · test", progress=progress
    )
    (cfg.checkpoint_dir / f"{name}_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(
        f"test: loss {test_metrics['loss']:.4f} | MAE {test_metrics['mae_deg']:.2f}° | "
        f"median {test_metrics['median_ae_deg']:.2f}°",
        flush=True,
    )

    return RunResult(
        model_name=name,
        history=pd.DataFrame(history),
        test_metrics=test_metrics,
        theta_true=theta_true,
        theta_pred=theta_pred,
        checkpoint_path=checkpoint_path,
    )


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
METRIC_ORDER = [
    "mae_deg", "median_ae_deg", "rmse_deg", "axial_mae_deg",
    "acc15_deg", "acc30_deg", "acc45_deg", "loss",
]


def build_comparison_table(
    results: dict[str, RunResult],
    train_samples: list[Sample],
    test_samples: list[Sample],
    cfg: Config,
) -> pd.DataFrame:
    comparison_rows = {name: res.test_metrics for name, res in results.items()}
    comparison_rows.update(baseline_metrics(train_samples, test_samples, cfg))
    return pd.DataFrame(comparison_rows).T.rename_axis("model")[METRIC_ORDER]


# --------------------------------------------------------------------------- #
# Plot helpers (save_path=None -> plt.show() inline; else write a PNG)
# --------------------------------------------------------------------------- #
def _finish(fig: plt.Figure, save_path: Path | None) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}", flush=True)
    else:
        plt.show()


def draw_orientation(ax: plt.Axes, img: Image.Image, theta: float, color: str = "red") -> None:
    """Draw an orientation arrow from the patch center.

    theta is in display convention: theta = atan2(dy, dx), y pointing down,
    as stored in the sample index. For our compass-bearing labels this is
    equivalent to (dx, dy) = (sin(bearing), -cos(bearing)).
    """
    w, h = img.size
    cx, cy = w / 2, h / 2
    r = 0.4 * min(w, h)
    ax.arrow(
        cx, cy, r * math.cos(theta), r * math.sin(theta),
        color=color, width=1.0, head_width=5.0, length_includes_head=True,
    )


def compass_degrees(theta: float) -> float:
    """Display-convention radians back to the raw label bearing (0°=up, clockwise)."""
    return math.degrees(theta + math.pi / 2) % 360


def plot_label_check(
    samples: list[Sample], cfg: Config, n: int = 24, per_row: int = 8, save_path: Path | None = None
) -> None:
    """Draw orientation arrows on random patches to sanity-check the label convention."""
    picks = random.Random(cfg.seed).sample(samples, min(n, len(samples)))
    n_rows = math.ceil(len(picks) / per_row)
    fig, axes = plt.subplots(
        n_rows, per_row, figsize=(2.2 * per_row, 2.6 * n_rows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, s in zip(axes.flat, picks):
        img = Image.open(s.path).convert("RGB")
        ax.imshow(img)
        draw_orientation(ax, img, s.theta, color="red")
        ax.set_title(f"{compass_degrees(s.theta):.1f}°", fontsize=10)
    _finish(fig, save_path)


def plot_training_curves(results: dict[str, RunResult], save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, res in results.items():
        axes[0].plot(res.history["epoch"], res.history["train_loss"], label=f"{name} (train)")
        axes[0].plot(res.history["epoch"], res.history["val_loss"], "--", label=f"{name} (val)")
        axes[1].plot(res.history["epoch"], res.history["val_mae_deg"], label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("1 − cos(Δθ)")
    axes[0].set_title("cosine loss")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val MAE [°]")
    axes[1].set_title("Validation angular error")
    axes[1].legend(fontsize=8)
    _finish(fig, save_path)


def primary_error_deg(res: RunResult, cfg: Config) -> np.ndarray:
    t_true = torch.from_numpy(res.theta_true)
    t_pred = torch.from_numpy(res.theta_pred)
    err = signed_axial_error(t_true, t_pred) if cfg.axial_labels else signed_angular_error(t_true, t_pred)
    return np.degrees(err.numpy())


def plot_error_diagnostics(
    results: dict[str, RunResult], cfg: Config, save_path: Path | None = None
) -> None:
    err_range = 90.0 if cfg.axial_labels else 180.0
    fig, axes = plt.subplots(2, len(results), figsize=(4.5 * len(results), 7), squeeze=False)
    for col, (name, res) in enumerate(results.items()):
        err_deg = primary_error_deg(res, cfg)
        axes[0][col].hist(err_deg, bins=72, range=(-err_range, err_range))
        axes[0][col].set_title(f"{name}\nMAE {res.test_metrics['mae_deg']:.1f}°")
        axes[0][col].set_xlabel("signed error [°]")
        axes[1][col].scatter(np.degrees(res.theta_true), err_deg, s=2, alpha=0.3)
        axes[1][col].axhline(0.0, color="black", lw=0.5)
        axes[1][col].set_xlabel("true angle [°]")
        axes[1][col].set_ylabel("signed error [°]")
    _finish(fig, save_path)


@torch.no_grad()
def plot_predictions(
    samples: list[Sample], model: nn.Module, cfg: Config, n: int = 8, save_path: Path | None = None
) -> None:
    """Red arrow: ground truth. Blue arrow: model prediction."""
    picks = random.sample(samples, min(n, len(samples)))
    ds = BeeOrientationDataset(picks, cfg)
    images = torch.stack([ds[i][0] for i in range(len(picks))]).to(DEVICE)
    theta_pred = angles_from_sincos(model(images).float(), cfg).cpu().numpy()

    fig, axes = plt.subplots(1, len(picks), figsize=(2.2 * len(picks), 2.6))
    for ax, s, tp in zip(np.atleast_1d(axes), picks, theta_pred):
        img = Image.open(s.path).convert("RGB")
        ax.imshow(img)
        draw_orientation(ax, img, s.theta, color="red")
        draw_orientation(ax, img, float(tp), color="deepskyblue")
        err = math.degrees(math.atan2(math.sin(s.theta - tp), math.cos(s.theta - tp)))
        ax.set_title(f"err {err:+.1f}°", fontsize=9)
        ax.axis("off")
    _finish(fig, save_path)
