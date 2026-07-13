from pathlib import Path
import cv2
import torch
import numpy as np

# Check whether a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: Processing on device: {device}")

data_dir = Path("/scratch/cvcdt011/data")

def preprocess():
    video_path = data_dir / "rec1.mp4"
    trajectory_dir = data_dir / "rec1_trajectories"
    base_crops_dir = data_dir / "crops"
    base_crops_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(list(trajectory_dir.glob("*.txt")))
    file_handles = []
    
    # DEBUG: Track initial file discovery
    print(f"DEBUG: Found {len(txt_files)} trajectory files.")
    for f in txt_files:
        fh = open(f, "r")
        file_handles.append((f.stem, fh))
        (base_crops_dir / f.stem).mkdir(parents=True, exist_ok=True)

    def read_next_row(fh):
        line = fh.readline()
        if not line: return None
        parts = line.strip().split(",")
        return (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[4]))

    next_rows = {}
    for tid, fh in file_handles:
        row = read_next_row(fh)
        if row: next_rows[tid] = row

    cap = cv2.VideoCapture(str(video_path))
    half_size = 50
    frame_count = 0
    stats = {"skipped": 0, "processed": 0}

    try:
        while True:
            # print(frame_count)
            ret, frame = cap.read()
            if not ret: break

            frame_tensor = torch.from_numpy(frame).to(device)
            h, w = frame.shape[:2]

            for traj_id, fh in file_handles:
                if traj_id in next_rows and next_rows[traj_id][0] == frame_count:
                    crop_filename = base_crops_dir / traj_id / f"frame_{frame_count:06d}.png"
                    
                    if crop_filename.exists():
                        # print("skipped")
                        stats["skipped"] += 1
                    else:
                        try:
                            x, y = int(round(next_rows[traj_id][1])), int(round(next_rows[traj_id][2]))
                            ymin, ymax = max(0, x - half_size), min(h, x + half_size)
                            xmin, xmax = max(0, y - half_size), min(w, y + half_size)

                            crop_tensor = frame_tensor[ymin:ymax, xmin:xmax]
                            cv2.imwrite(str(crop_filename), crop_tensor.cpu().numpy())
                            stats["processed"] += 1
                        except Exception as e:
                            print(f"DEBUG: Error processing {traj_id} at frame {frame_count}: {e}")
                    
                    row = read_next_row(fh)
                    if row: next_rows[traj_id] = row
                    else: next_rows.pop(traj_id, None)

            if frame_count % 250 == 0:
                print(f"DEBUG: Frame {frame_count} | Processed: {stats['processed']} | Skipped: {stats['skipped']}")
            frame_count += 1
            
    finally:
        cap.release()
        for _, fh in file_handles: fh.close()
    
    print(f"DEBUG: Finished. Total processed: {stats['processed']}, Total skipped: {stats['skipped']}")

if __name__ == "__main__":
    preprocess()