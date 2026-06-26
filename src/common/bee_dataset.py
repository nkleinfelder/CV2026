"""Shared data loading for bee orientation training.

Turns the preprocessed crops (crops/<traj_id>/frame_XXXXXX.png) plus the
trajectory .txt files into training samples, splits them without temporal
leakage, and serves them as a PyTorch Dataset. Every train script imports from
here so the label convention and input pipeline stay identical.

Kept as a real importable module (not notebook cells) because DataLoader workers
re-import the defining module under spawn/forkserver (Python 3.14). The repo is
an editable-installed package (see pyproject.toml), so this imports as
common.bee_dataset from anywhere without sys.path tricks.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Callable, NamedTuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Sample(NamedTuple):
    path: Path
    theta: float  # radians, display convention: theta = atan2(dy, dx), y down
    traj_id: str
    # Oversampling bakes a transform in here; theta above is the heading after it.
    # Replayed on the pixels in BeeOrientationDataset.__getitem__ (mirror, then rotate).
    rotate_deg: float = 0.0  # extra PIL rotation (degrees, CCW on screen)
    mirror: bool = False  # horizontal (left-right) flip


# Remap a raw label angle (radians) into display convention, where downstream
# code assumes theta = atan2(dy, dx) (y down). Our data is "compass".
#   compass: 0=up, cw -> theta - pi/2      ydown: identity
#   yup: -theta                            transposed: pi/2 - theta
ANGLE_REMAPS: dict[str, Callable[[float], float]] = {
    "compass": lambda t: t - math.pi / 2,
    "ydown": lambda t: t,
    "yup": lambda t: -t,
    "transposed": lambda t: math.pi / 2 - t,
}


def index_crops(
    crops_dir: Path,
    trajectories_dir: Path,
    *,
    angles_in_degrees: bool = True,
    angle_convention: str = "compass",
    expected_crop_size: int = 100,
    frame_size: tuple[int, int] | None = None,
) -> tuple[list[Sample], int]:
    """Build the (crop path, orientation) index from the trajectory files.

    Returns (samples, n_partial_dropped). Crop existence is checked against one
    directory listing per trajectory rather than a stat per frame. If frame_size
    (width, height) is given, border crops that preprocessing had to clip are
    dropped with the same rule as preprocessing; pass None to keep them.
    """
    half = expected_crop_size // 2
    remap = ANGLE_REMAPS[angle_convention]
    frame_w, frame_h = frame_size if frame_size is not None else (0, 0)

    samples: list[Sample] = []
    n_partial = 0
    for traj_file in sorted(Path(trajectories_dir).glob("*.txt")):
        traj_id = traj_file.stem
        crop_dir = Path(crops_dir) / traj_id
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
            if frame_size is not None:
                # preprocessing indexes crop rows with position_x, columns with
                # position_y, so x clips against frame height, y against width.
                x = int(round(float(parts[1])))
                y = int(round(float(parts[2])))
                if x - half < 0 or x + half > frame_h or y - half < 0 or y + half > frame_w:
                    n_partial += 1
                    continue
            theta = float(parts[4])
            if angles_in_degrees:
                theta = math.radians(theta)
            theta = remap(theta)
            samples.append(Sample(path=crop_dir / filename, theta=theta, traj_id=traj_id))
    return samples, n_partial


def split_by_trajectory(
    samples: list[Sample],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Train/val/test split keeping whole trajectories together, so nearly
    identical consecutive frames of the same bee cannot leak across splits."""
    rng = random.Random(seed)
    traj_ids = sorted({s.traj_id for s in samples})
    rng.shuffle(traj_ids)
    n = len(traj_ids)
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    test_ids = set(traj_ids[:n_test])
    val_ids = set(traj_ids[n_test : n_test + n_val])
    train = [s for s in samples if s.traj_id not in test_ids | val_ids]
    val = [s for s in samples if s.traj_id in val_ids]
    test = [s for s in samples if s.traj_id in test_ids]
    assert train and val and test, (
        f"Empty split (train={len(train)}, val={len(val)}, test={len(test)}) — "
        "too few trajectories for the requested fractions."
    )
    return train, val, test


def balance_by_orientation(
    samples: list[Sample],
    *,
    n_bins: int = 36,
    target_multiple: float = 1.0,
    mirror: bool = True,
    seed: int = 42,
) -> list[Sample]:
    """Oversample under-represented headings by rotating/mirroring real crops.

    Bee headings aren't uniform, so a plain dataset over-teaches the common ones.
    Keeps every original sample and appends synthetic ones that reuse real crops
    rotated (and optionally flipped) into under-filled heading bins — real pixels,
    not duplicates. theta is binned into n_bins and each bin topped up to the
    busiest bin's count; the chosen rotation/mirror is baked into the Sample so
    the label stays exact.

    Apply to the train split only; redundant with rotation_augment.
    """
    if not samples:
        return list(samples)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    rng = random.Random(seed)
    two_pi = 2.0 * math.pi
    bin_width = two_pi / n_bins

    counts = [0] * n_bins
    for s in samples:
        counts[int((s.theta % two_pi) / bin_width) % n_bins] += 1

    target = round(max(counts) * target_multiple)

    balanced = list(samples)
    for b in range(n_bins):
        for _ in range(max(0, target - counts[b])):
            src = samples[rng.randrange(len(samples))]
            do_mirror = mirror and rng.random() < 0.5
            # Horizontal flip maps theta = atan2(dy, dx) -> pi - theta.
            base_theta = (math.pi - src.theta) if do_mirror else src.theta
            theta_target = (b + rng.random()) * bin_width  # random spot in bin b
            # PIL rotate by r (CCW) turns heading into base_theta - r, so pick r
            # that lands the crop on theta_target.
            balanced.append(
                Sample(
                    path=src.path,
                    theta=theta_target % two_pi,
                    traj_id=src.traj_id,
                    rotate_deg=math.degrees(base_theta - theta_target) % 360.0,
                    mirror=do_mirror,
                )
            )
    rng.shuffle(balanced)
    return balanced


class BeeOrientationDataset(Dataset):
    """Serves (image tensor, [sin(k*theta), cos(k*theta)]) pairs.

    k (angle_multiplier) is 2.0 for axial labels, else 1.0. rotation_augment
    rotates the patch and adjusts theta (assuming theta = atan2(dy, dx), y down).
    oversample_orientations appends rotated/mirrored crops via
    balance_by_orientation — enable on the train dataset only.
    """

    def __init__(
        self,
        samples: list[Sample],
        *,
        image_size: int = 224,
        angle_multiplier: float = 1.0,
        rotation_augment: bool = False,
        oversample_orientations: bool = False,
        oversample_bins: int = 36,
        oversample_seed: int = 42,
    ) -> None:
        if oversample_orientations:
            samples = balance_by_orientation(
                samples, n_bins=oversample_bins, seed=oversample_seed
            )
        self.samples = samples
        self.angle_multiplier = angle_multiplier
        self.rotation_augment = rotation_augment
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        theta = sample.theta

        # Replay the baked-in oversampling transform (theta already accounts for
        # it, so only the pixels change): mirror first, then rotate.
        if sample.mirror:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if sample.rotate_deg:
            img = img.rotate(sample.rotate_deg, resample=Image.Resampling.BILINEAR)

        if self.rotation_augment:
            angle_deg = random.uniform(0.0, 360.0)
            # PIL rotates CCW on screen; with y down that decreases theta.
            img = img.rotate(angle_deg, resample=Image.Resampling.BILINEAR)
            theta = theta - math.radians(angle_deg)

        k = self.angle_multiplier
        target = torch.tensor([math.sin(k * theta), math.cos(k * theta)], dtype=torch.float32)
        return self.transform(img), target
