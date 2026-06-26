#!/usr/bin/env python3
"""
Simple bee image classification / orientation evaluation script.
Classifies bee crops, compares predictions against ground truth (if available)
using the codebase's circular angular error metrics, and saves frame-level predictions to CSV.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet18, resnet50, convnext_tiny

try:
    import timm
except ImportError:
    timm = None

# Set model, image, and (optional) ground-truth paths here.
MODELS_PATH = Path("/scratch/cvcdt004/checkpoints")
FILES_PATH = Path("/scratch/cvcdt011/data/rec2_crops")
GT_PATH = Path("/scratch/cvcdt011/data/rec2_trajectories")
PREDICTIONS_DIR = Path("/scratch/cvcdt011")

# Path validation checks
print("--- Path Validation Checks ---")
for name, path in [
    ("MODELS_PATH", MODELS_PATH),
    ("FILES_PATH", FILES_PATH),
    ("GT_PATH", GT_PATH),
    ("PREDICTIONS_DIR", PREDICTIONS_DIR),
]:
    status = "[OK]" if path.exists() else "[WARNING]"
    print(f"{status} {name} exists: {path}")
print("------------------------------\n")


def load_ground_truth(gt_dir: Path) -> dict:
    """
    Loads ground truth orientations from a directory of trajectory .txt files.
    Returns mapping: (traj_id, file_name) or file_name -> (gt_sin, gt_cos, gt_angle_deg).
    """
    gt_map = {}
    if not gt_dir or not gt_dir.is_dir():
        print(f"Warning: Ground truth directory not found at: {gt_dir}")
        return gt_map

    for txt_file in gt_dir.rglob("*.txt"):
        traj_id = txt_file.stem
        try:
            lines = [l.strip() for l in txt_file.read_text().splitlines() if l.strip()]
            for idx, line in enumerate(lines):
                parts = line.replace(",", " ").split()
                if not parts:
                    continue
                try:
                    raw_deg = float(parts[-1])
                except ValueError:
                    continue

                frame = int(parts[0]) if parts[0].isdigit() else idx
                file_name = f"frame_{frame:06d}.png"

                # Remap compass convention (0=up, 90=right, clockwise) -> display convention (x right, y down)
                theta_rad = math.radians(raw_deg) - math.pi / 2
                gt_sin, gt_cos = math.sin(theta_rad), math.cos(theta_rad)
                gt_angle_deg = math.degrees(theta_rad) % 360.0

                gt_map[(traj_id, file_name)] = (gt_sin, gt_cos, gt_angle_deg)
                gt_map[file_name] = (gt_sin, gt_cos, gt_angle_deg)
        except Exception as e:
            print(f"Warning: Failed to parse GT trajectory file {txt_file}: {e}")

    return gt_map


# Load Ground Truth if available
gt_map = load_ground_truth(GT_PATH)
if gt_map:
    print(f"Loaded ground truth annotations for {len(gt_map)} sample(s) from {GT_PATH}.")
else:
    print(f"No ground truth annotations found at: {GT_PATH}")

# Find all model checkpoints
model_files = [MODELS_PATH] if MODELS_PATH.is_file() else list(MODELS_PATH.glob("*.pt")) + list(MODELS_PATH.glob("*.pth"))

# Find all image files
img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
image_files = [FILES_PATH] if FILES_PATH.is_file() else [
    f for f in FILES_PATH.rglob("*") if f.suffix.lower() in img_extensions
]

if not model_files:
    print(f"Error: No models found at: {MODELS_PATH}")
    sys.exit(1)

if not image_files:
    print(f"Error: No images found at: {FILES_PATH}")
    sys.exit(1)

print(f"Found {len(model_files)} model(s) and {len(image_files)} image(s).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

for model_file in model_files:
    print(f"\nProcessing model: {model_file.name}")
    out_csv = PREDICTIONS_DIR / f"{model_file.stem}_predictions.csv"

    # Instantiate model based on filename
    name = model_file.name.lower()
    if "resnet18" in name:
        model = resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif "resnet50" in name:
        model = resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif "mobilenet" in name:
        if timm is None:
            raise ImportError("timm is required for MobileNet models")
        model = timm.create_model("mobilenetv4_conv_medium", num_classes=2)
    else:  # Default to convnext_tiny
        model = convnext_tiny()
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)

    # Load checkpoint weights
    checkpoint = torch.load(model_file, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    clean_state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    model.to(device).eval()

    # Classify images and compare with GT
    results = []
    for img_path in tqdm(image_files, desc=f"Classifying with {model_file.stem}"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)[0].cpu().numpy()  # outputs [sin(theta), cos(theta)]

            angle_deg = np.degrees(np.arctan2(out[0], out[1])) % 360.0
            traj_id = img_path.parent.name if img_path.parent != FILES_PATH else ""

            res_item = {
                "file_name": img_path.name,
                "trajectory_id": traj_id,
                "angle_degrees": angle_deg,
                "sin_theta": out[0],
                "cos_theta": out[1],
            }

            # Ground truth comparison using codebase circular error metric
            if gt_map:
                key = (traj_id, img_path.name) if (traj_id, img_path.name) in gt_map else (
                    img_path.name if img_path.name in gt_map else (
                        traj_id if traj_id in gt_map else None
                    )
                )
                if key is not None:
                    gt_sin, gt_cos, gt_angle = gt_map[key]
                    res_item.update({
                        "gt_angle_degrees": gt_angle,
                        "gt_sin_theta": gt_sin,
                        "gt_cos_theta": gt_cos,
                        "error_degrees": np.abs(np.degrees(np.arctan2(
                            np.sin(np.radians(angle_deg - gt_angle)),
                            np.cos(np.radians(angle_deg - gt_angle))
                        )))
                    })

            results.append(res_item)
        except Exception as e:
            print(f"Skipping {img_path.name} due to error: {e}")

    # Save predictions & metrics to CSV
    if results:
        df = pd.DataFrame(results)
        has_gt = "error_degrees" in df.columns

        cols = ["file_name", "trajectory_id", "angle_degrees", "sin_theta", "cos_theta"]
        if has_gt:
            cols += ["gt_angle_degrees", "gt_sin_theta", "gt_cos_theta", "error_degrees"]
        df = df[cols]

        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} predictions to: {out_csv}")
        print("\nSample predictions:")
        print(df.head(5).to_string(index=False))

        if has_gt:
            errs = df["error_degrees"].dropna()
            mae = errs.mean()
            median_ae = errs.median()
            rmse = np.sqrt(np.mean(errs**2))

            print(f"\n--- Ground Truth Evaluation Summary for {model_file.name} ---")
            print(f"Mean Absolute Error (MAE):     {mae:.2f}°")
            print(f"Median Absolute Error:         {median_ae:.2f}°")
            print(f"Root Mean Square Error (RMSE): {rmse:.2f}°")
            for threshold in [15, 30, 45]:
                acc = (errs <= threshold).mean() * 100
                print(f"Accuracy <= {threshold}°:                {acc:.1f}%")
