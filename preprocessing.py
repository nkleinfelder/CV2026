from pathlib import Path

import cv2

data_dir = Path("/scratch/cvcdt011/data")


def preprocess():
    video_path = data_dir / "rec1.mp4"
    trajectory_dir = data_dir / "rec1_trajectories"

    # Open all trajectory text files and initialize generators or line pointers
    # Since each file has lines sorted by point_id (frame index), we can open them all
    # and read them incrementally.
    txt_files = sorted(list(trajectory_dir.glob("*.txt")))

    # Store active file objects and their next available line split by commas
    file_handles = []
    for f in txt_files:
        try:
            fh = open(f, "r")
            file_handles.append((f.stem, fh))
        except IOError:
            pass

    # Helper to parse a line from a trajectory file
    def read_next_row(fh):
        line = fh.readline()
        if not line:
            return None
        parts = line.strip().split(",")
        if len(parts) >= 5:
            # format: point_id, x, y, orientation
            return (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[4]))
        return None

    # We will track the next line data for each file: {traj_id: [frame_idx, x, y, orientation]}
    next_rows = {}
    for traj_id, fh in file_handles:
        row = read_next_row(fh)
        if row:
            next_rows[traj_id] = row

    # Base output directory for all crops
    base_crops_dir = data_dir / "crops"
    base_crops_dir.mkdir(parents=True, exist_ok=True)

    # Create all trajectory directories at the start
    for traj_id, _ in file_handles:
        (base_crops_dir / traj_id).mkdir(parents=True, exist_ok=True)

    # first hyper parameter
    half_size = 50

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        for _, fh in file_handles:
            fh.close()
        return

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Find all matching rows from all open files for the current frame_count
            matches = []
            for traj_id, fh in file_handles:
                # If the next row matches our current frame
                if traj_id in next_rows and next_rows[traj_id][0] == frame_count:
                    matches.append(
                        {
                            "trajectory_id": traj_id,
                            "x": next_rows[traj_id][1],
                            "y": next_rows[traj_id][2],
                            "orientation": next_rows[traj_id][3],
                        }
                    )
                    # Consume the matched row and load the next one
                    row = read_next_row(fh)
                    if row:
                        next_rows[traj_id] = row
                    else:
                        next_rows.pop(traj_id, None)

                # Crop and save if there's a match
                for match in matches:
                    traj_id = match["trajectory_id"]

                    # Nx and y need to be switched
                    x, y = int(round(match["x"])), int(round(match["y"]))

                    # Bounding box coordinates with boundary checks (using your fixed coordinates mapping)
                    ymin = max(0, x - half_size)
                    ymax = min(h, x + half_size)
                    xmin = max(0, y - half_size)
                    xmax = min(w, y + half_size)

                    crop = frame[ymin:ymax, xmin:xmax]

                    # Save crop
                    crop_filename = (
                        base_crops_dir / traj_id / f"frame_{frame_count:06d}.png"
                    )
                    cv2.imwrite(str(crop_filename), crop)
            print(f"frame count {frame_count}")
            frame_count += 1
    finally:
        cap.release()
        for _, fh in file_handles:
            fh.close()

    print(
        f"Successfully finished processing all {frame_count} frames. Crops saved to {base_crops_dir}"
    )


if __name__ == "__main__":
    preprocess()
