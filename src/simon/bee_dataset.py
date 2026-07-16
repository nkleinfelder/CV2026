"""Dataset classes for bee orientation training.

Lives in a real module (not the notebook) because DataLoader workers with
num_workers > 0 unpickle the dataset in a fresh process: under Python 3.14's
forkserver start method the worker re-imports the defining module, and classes
defined in notebook cells (`__main__`) don't exist there. Keep this file next
to the notebook so the workers can import it.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import NamedTuple

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


class BeeOrientationDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg, augment: bool = False) -> None:
        # cfg is the notebook's Config; only image_size, angle_multiplier and
        # rotation_augment are read here (and not stored), so the dataset stays
        # picklable without this module depending on the notebook.
        self.samples = samples
        self.angle_multiplier = cfg.angle_multiplier
        self.rotation_augment = augment and cfg.rotation_augment
        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
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

        if self.rotation_augment:
            angle_deg = random.uniform(0.0, 360.0)
            # PIL rotates counter-clockwise on screen; with image coordinates
            # (y down) this decreases theta = atan2(dy, dx) by the same amount.
            img = img.rotate(angle_deg, resample=Image.Resampling.BILINEAR)
            theta = theta - math.radians(angle_deg)

        k = self.angle_multiplier
        target = torch.tensor([math.sin(k * theta), math.cos(k * theta)], dtype=torch.float32)
        return self.transform(img), target
