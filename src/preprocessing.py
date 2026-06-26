#!/usr/bin/env python3
"""Extract per-bee crop images from the recording video (script version of
src/preprocessing.ipynb — same logic and output layout). Every argument is
optional and falls back to the notebook's hardcoded value.

Reads the trajectory files, flags per-trajectory outlier frames (position or
orientation jumps by z-score), streams the video once, and writes one PNG per
(trajectory, frame) onto a fixed-size white canvas:

    <data-dir>/<crops-subdir>/<traj_id>/frame_XXXXXX.png

Resumable: crops already on disk are skipped, and crops of frames now flagged
as outliers are deleted. Crop side length is 2 * --half-size, which is how the
differently sized sets are made (crops=50, crops_large=75, crops_small=25).
See run-commands.md for usage.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

Z_THRESHOLD = 4.0  # Z-score threshold for outlier detection
WRITE_QUEUE_MAX = 512  # Max crops buffered in memory before the main loop blocks

# Group members must be able to read AND write the outputs
CROP_FILE_MODE = 0o664  # rw-rw-r--
CROP_DIR_MODE = 0o775  # rwxrwxr-x (x needed on dirs to enter/list them)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Input root holding the video and the trajectories dir. "
                        "Defaults to /scratch/cvcdt011/data, falling back to the "
                        "repo-local src/data when that doesn't exist (as in the notebook).")
    p.add_argument("--crops-subdir", default="crops",
                   help="Output directory name under --data-dir (e.g. crops, crops_large, crops_small).")
    p.add_argument("--half-size", type=int, default=50,
                   help="Half-width/height of each crop in pixels; crop side = 2x this.")
    p.add_argument("--video-name", default="rec1.mp4",
                   help="Video file name under --data-dir.")
    p.add_argument("--trajectories-subdir", default="rec1_trajectories",
                   help="Trajectory .txt directory name under --data-dir.")
    p.add_argument("--z-threshold", type=float, default=Z_THRESHOLD,
                   help="Z-score above which a frame counts as a trajectory outlier.")
    p.add_argument("--num-writers", type=int, default=4,
                   help="Background PNG writer threads.")
    return p.parse_args()


def read_next_row(fh):
    """Read and parse the next line from a trajectory file handle.
    Returns (frame_id, x, y, angle) or None if EOF."""
    line = fh.readline()
    if not line:
        return None
    parts = line.strip().split(",")
    return (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[4]))


def detect_trajectory_outliers(file_path: Path, z_threshold: float) -> set[int]:
    """Reads a trajectory file, computes displacement and angular change between
    consecutive steps, and returns the frame IDs flagged as outliers by z-score."""
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[4])))

    if len(rows) < 3:
        return set()

    frames = [r[0] for r in rows]
    xs = np.array([r[1] for r in rows])
    ys = np.array([r[2] for r in rows])
    thetas = np.array([r[3] for r in rows])

    dx = np.diff(xs)
    dy = np.diff(ys)
    displacements = np.sqrt(dx**2 + dy**2)

    d_theta = np.diff(thetas)
    d_theta = (d_theta + 180) % 360 - 180
    abs_d_theta = np.abs(d_theta)

    outlier_frames: set[int] = set()

    if len(displacements) > 1:
        mean_disp = np.mean(displacements)
        std_disp = np.std(displacements)
        if std_disp > 1e-6:
            z_disp = (displacements - mean_disp) / std_disp
            for idx in np.where(np.abs(z_disp) > z_threshold)[0]:
                outlier_frames.add(frames[idx + 1])

    if len(abs_d_theta) > 1:
        mean_theta = np.mean(abs_d_theta)
        std_theta = np.std(abs_d_theta)
        if std_theta > 1e-6:
            z_theta = (abs_d_theta - mean_theta) / std_theta
            for idx in np.where(np.abs(z_theta) > z_threshold)[0]:
                outlier_frames.add(frames[idx + 1])

    return outlier_frames


def make_group_writable_dir(path: Path) -> None:
    """mkdir -p, then force group rwx (mkdir's mode is masked by the umask,
    and only the owner may chmod a dir created by a groupmate)."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, CROP_DIR_MODE)
    except PermissionError:
        pass  # Dir owned by another group member — they already set its perms


def main() -> int:
    args = parse_args()

    # Same resolution order as the notebook: HPC scratch first, local data second.
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path("/scratch/cvcdt011/data")
        if not data_dir.exists():
            data_dir = Path(__file__).resolve().parent / "data"

    video_path = data_dir / args.video_name
    trajectory_dir = data_dir / args.trajectories_subdir
    crops_dir = data_dir / args.crops_subdir
    half_size = args.half_size
    crop_size = 2 * half_size

    # Notebook fallback: trajectories may live in src/data even when the video is elsewhere.
    if not trajectory_dir.exists():
        local_traj = Path(__file__).resolve().parent / "data" / args.trajectories_subdir
        if local_traj.is_dir():
            trajectory_dir = local_traj

    # Fail fast on missing inputs — cv2.VideoCapture would otherwise fail silently later.
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    txt_files = sorted(trajectory_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No trajectory .txt files in {trajectory_dir}")

    print(f"Output: {crops_dir} | crop size: {crop_size}px (half {half_size})", flush=True)
    print(f"DEBUG: Found {len(txt_files)} trajectory files.", flush=True)

    # --- Background PNG writers ---
    write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAX)
    write_errors: list[tuple[Path, str]] = []  # collected across writer threads

    def _writer_worker():
        """Background thread: pulls (path, ndarray, angle) from the queue and writes to disk."""
        from PIL import Image, PngImagePlugin

        while True:
            item = write_queue.get()
            if item is None:  # Sentinel: shut down
                write_queue.task_done()
                break
            path, img, angle = item
            try:
                # OpenCV keeps images in BGR format, convert to RGB for PIL
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("OrientationAngle", str(angle))
                # Save using PIL with zero compression to maintain fast HPC I/O
                img_pil.save(str(path), "PNG", pnginfo=metadata, compress_level=0)
                os.chmod(path, CROP_FILE_MODE)  # Group-writable regardless of umask
            except Exception as e:
                write_errors.append((path, repr(e)))  # list.append is thread-safe
                print(f"DEBUG: Write error {path}: {e}")
            write_queue.task_done()

    writer_threads = []
    for _ in range(args.num_writers):
        t = threading.Thread(target=_writer_worker, daemon=True)
        t.start()
        writer_threads.append(t)

    # --- Output dirs: one directory per bee/trajectory ---
    make_group_writable_dir(crops_dir)
    file_handles = []
    for f in txt_files:
        fh = open(f, "r")
        file_handles.append((f.stem, fh))
        make_group_writable_dir(crops_dir / f.stem)

    # --- Outliers ---
    outliers_dict: dict[str, set[int]] = {}
    total_outliers_flagged = 0
    for f in txt_files:
        outliers = detect_trajectory_outliers(f, args.z_threshold)
        outliers_dict[f.stem] = outliers
        total_outliers_flagged += len(outliers)
    print(f"DEBUG: Flagged {total_outliers_flagged} outlier frames "
          f"across {len(txt_files)} trajectories.", flush=True)

    # --- Existing crops: one directory listing per trajectory (not per-frame exists()) ---
    existing_crops: dict[str, set[int]] = {}
    for f in txt_files:
        tid = f.stem
        existing: set[int] = set()
        for p in (crops_dir / tid).iterdir():
            try:
                frame_num = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if frame_num in outliers_dict[tid]:
                p.unlink()  # Remove leftover outlier crops from previous runs
            else:
                existing.add(frame_num)
        existing_crops[tid] = existing
    print("DEBUG: Existing-crops index built.", flush=True)

    # Pre-load first row of each trajectory
    next_rows = {}
    for tid, fh in file_handles:
        row = read_next_row(fh)
        if row:
            next_rows[tid] = row

    # --- Main extraction loop ---
    # Pre-allocate white canvas in CPU numpy memory (reused across all crops)
    white_canvas = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open {video_path} — the file exists but is unreadable "
            "from this node (missing ffmpeg/codec support in this OpenCV build?)."
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)  # Increase video decoder prefetch buffer
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(
            f"OpenCV reports {total_frames} frames for {video_path} — decoder cannot read it."
        )
    print(f"DEBUG: Video opened, {total_frames} frames.", flush=True)

    frame_count = 0
    stats = {"skipped": 0, "processed": 0, "outliers_filtered": 0}
    pbar = tqdm(total=total_frames, desc="Extracting crops")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            for traj_id, fh in file_handles:
                if traj_id not in next_rows or next_rows[traj_id][0] != frame_count:
                    continue

                # Outlier: skip and do not save crop
                if frame_count in outliers_dict.get(traj_id, set()):
                    stats["outliers_filtered"] += 1

                # Already on disk: skip
                elif frame_count in existing_crops.get(traj_id, set()):
                    stats["skipped"] += 1

                # New crop: extract and enqueue for async write
                else:
                    try:
                        x, y = (
                            int(round(next_rows[traj_id][1])),
                            int(round(next_rows[traj_id][2])),
                        )

                        # Desired window bounds (position_x indexes rows,
                        # position_y indexes columns — see preprocessing.ipynb)
                        y_start = x - half_size
                        y_end = x + half_size
                        x_start = y - half_size
                        x_end = y + half_size

                        # Clamped bounds (intersection with frame)
                        ymin_f = max(0, y_start)
                        ymax_f = min(h, y_end)
                        xmin_f = max(0, x_start)
                        xmax_f = min(w, x_end)

                        # Copy white canvas and paste valid region
                        crop = white_canvas.copy()
                        crop[
                            ymin_f - y_start : ymax_f - y_start,
                            xmin_f - x_start : xmax_f - x_start,
                        ] = frame[ymin_f:ymax_f, xmin_f:xmax_f]

                        angle = float(next_rows[traj_id][3])

                        crop_filename = crops_dir / traj_id / f"frame_{frame_count:06d}.png"
                        write_queue.put((crop_filename, crop, angle))  # Non-blocking enqueue
                        stats["processed"] += 1
                    except Exception as e:
                        print(f"DEBUG: Error processing {traj_id} at frame {frame_count}: {e}")

                row = read_next_row(fh)
                if row:
                    next_rows[traj_id] = row
                else:
                    next_rows.pop(traj_id, None)

            pbar.update(1)
            if frame_count % 250 == 0:
                pbar.set_postfix(
                    {
                        "processed": stats["processed"],
                        "skipped": stats["skipped"],
                        "outliers": stats["outliers_filtered"],
                    }
                )
            frame_count += 1

    finally:
        pbar.close()
        cap.release()
        for _, fh in file_handles:
            fh.close()

    # Wait for all pending writes to finish, then shut down worker threads
    write_queue.join()
    for _ in writer_threads:
        write_queue.put(None)
    for t in writer_threads:
        t.join()

    print(
        f"DEBUG: Finished. Read {frame_count}/{total_frames} frames. "
        f"Processed: {stats['processed']}, Skipped: {stats['skipped']}, "
        f"Outliers filtered: {stats['outliers_filtered']}, Write errors: {len(write_errors)}"
    )

    # Fail loudly instead of ending a run that produced nothing / lost crops.
    if frame_count == 0:
        raise RuntimeError(f"Read 0 frames from {video_path} — decoder failed on the first frame.")
    if stats["processed"] == 0 and stats["skipped"] == 0:
        raise RuntimeError(
            "No crops written and none already on disk — trajectory frame IDs "
            f"matched none of the {frame_count} video frames read."
        )
    if write_errors:
        for path, err in write_errors[:10]:
            print(f"DEBUG: write error: {path}: {err}")
        raise RuntimeError(
            f"{len(write_errors)} of {stats['processed']} crop writes failed "
            f"(first: {write_errors[0][1]})."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
