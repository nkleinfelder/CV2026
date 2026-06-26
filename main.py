import os
from pathlib import Path

import cv2
import pandas as pd

data_dir = Path("./data")


def main():
    video_path = data_dir / "rec1.mp4"
    trajectory_path = data_dir / "rec1_trajectories" / "000000.txt"

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    if not os.path.exists(trajectory_path):
        print(f"Error: Trajectory file not found at {trajectory_path}")
        return

    # Load the entire trajectory data once, ignoring the 4th column (index 3)
    # Set the first column ('point_id', which holds the frame index) as the index of the DataFrame
    df_trajectories = pd.read_csv(
        trajectory_path,
        header=None,
        usecols=[0, 1, 2, 4],
        names=["point_id", "x", "y", "orientation"],
        index_col="point_id",
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the corresponding trajectory row if this frame was not skipped
        # some rows in the text files are missing (probably when bees only half visible)
        if frame_count in df_trajectories.index:
            row_data = df_trajectories.loc[frame_count]
            # Print info for validation every 50 frames that are actually present
            if frame_count % 50 == 0 or frame_count == 0:
                pass
                h, w, c = frame.shape
                print(f"Frame {frame_count} ({w}x{h}):")
                print(row_data)
                print("-" * 50)
        else:
            # print(frame_count)
            # Frame was skipped in the trajectories file
            pass

        frame_count += 1

    cap.release()
    print(f"Successfully finished processing all {frame_count} frames.")


if __name__ == "__main__":
    main()
