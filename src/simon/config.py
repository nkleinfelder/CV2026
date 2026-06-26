"""Run configuration for bee-orientation training.

The Config dataclass carries every knob (data paths, split, input pipeline,
model list, training hyperparameters). Kept in its own module so config and the
train/eval logic in orientation.py stay separable; train.py builds a Config from
CLI args via dataclasses.replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # --- data ---
    data_dir: Path = Path("/scratch/cvcdt011/data")
    crops_subdir: str = "crops"
    trajectories_subdir: str = "rec1_trajectories"
    video_name: str = "rec1.mp4"
    angles_in_degrees: bool = True
    # Axial labels are head/tail ambiguous: regress [sin(2t), cos(2t)] and score mod 180°.
    axial_labels: bool = False
    # How orientation_angle maps into display coords (theta = atan2(dy, dx), y down).
    # See ANGLE_REMAPS in common/bee_dataset.py; our data is "compass" (0=up, cw).
    angle_convention: str = "compass"
    expected_crop_size: int = 100  # 2 * half_size from preprocessing.py
    filter_partial_crops: bool = (
        True  # drop border crops smaller than expected_crop_size
    )
    frame_size: tuple[int, int] | None = (
        None  # (w, h) for the border check; read from video if None
    )
    index_cache_path: Path = Path("sample_index.csv")
    use_index_cache: bool = True  # delete the cache after changing data settings

    # --- split ---
    split_strategy: str = "trajectory"  # "trajectory" (no temporal leakage) or "random"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    # --- input pipeline ---
    image_size: int = 224
    # Sized for A30 (24 GB) with AMP; lr below was tuned at batch 64.
    batch_size: int = 256
    num_workers: int = 4
    # Rotate the patch and adjust theta with it. Assumes theta = atan2(dy, dx), y down.
    rotation_augment: bool = False
    # Even out the heading distribution by appending rotated/mirrored crops (train
    # split only, see make_loaders). Redundant with rotation_augment.
    oversample_orientations: bool = False
    oversample_bins: int = 36

    # --- models / training ---
    model_names: tuple[str, ...] = ("resnet18", "resnet50", "mobilenetv4_conv_medium")
    pretrained: bool = True
    loss_fn: str = "cosine"  # key into LOSS_FNS ("cosine" or "cbrt_cos")
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    # Checkpoints (~50-100 MB each) belong on scratch, not the home quota.
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
