# sanity_check.py  — run this first before full extraction
import pandas as pd
import numpy as np

KAGGLE_DIR = r"C:\Users\524ta\Downloads\asl-signs"

def normalize_keypoints(seq):
    out = seq.copy()
    for i, frame in enumerate(seq):
        lh   = frame[:63].reshape(21, 3)
        rh   = frame[63:126].reshape(21, 3)
        pose = frame[126:].reshape(25, 3)
        for hand in [lh, rh]:
            if np.abs(hand).sum() > 0.01:
                wrist  = hand[0].copy()
                hand  -= wrist
                scale  = np.linalg.norm(hand[9] - hand[0]) + 1e-6
                hand  /= scale
        if np.abs(pose).sum() > 0.01:
            mid   = (pose[11] + pose[12]) / 2
            pose -= mid
            sw    = np.linalg.norm(pose[11] - pose[12]) + 1e-6
            pose /= sw
        out[i, :63]    = lh.flatten()
        out[i, 63:126] = rh.flatten()
        out[i, 126:]   = pose.flatten()
    return out

def add_velocity(seq):
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=-1)

def parquet_to_sequence(path, SEQ_LEN=64):
    df      = pd.read_parquet(path)
    frames  = sorted(df["frame"].unique())
    if len(frames) < 4:
        return None
    result = []
    for frame_idx in frames:
        frame_df = df[df["frame"] == frame_idx]
        lh = np.zeros(63, dtype=np.float32)
        lh_df = frame_df[frame_df["type"] == "left_hand"].sort_values("landmark_index")
        if len(lh_df) == 21:
            lh = lh_df[["x","y","z"]].values.flatten().astype(np.float32)
        rh = np.zeros(63, dtype=np.float32)
        rh_df = frame_df[frame_df["type"] == "right_hand"].sort_values("landmark_index")
        if len(rh_df) == 21:
            rh = rh_df[["x","y","z"]].values.flatten().astype(np.float32)
        pose = np.zeros(75, dtype=np.float32)
        pose_df = frame_df[frame_df["type"] == "pose"].sort_values("landmark_index")
        pose_df = pose_df[pose_df["landmark_index"] < 25]
        if len(pose_df) > 0:
            vals = pose_df[["x","y","z"]].values.astype(np.float32)
            pose[:len(vals)*3] = vals.flatten()
        result.append(np.concatenate([lh, rh, pose]))
    seq     = np.array(result, dtype=np.float32)
    indices = np.linspace(0, len(seq)-1, SEQ_LEN, dtype=int)
    seq     = seq[indices]
    seq     = normalize_keypoints(seq)
    seq     = add_velocity(seq)
    return seq.astype(np.float32)

# ── Run check ─────────────────────────────────────────────────────────
train_csv = pd.read_csv(f"{KAGGLE_DIR}/train.csv")
test_path = f"{KAGGLE_DIR}/{train_csv['path'].iloc[0]}"
print(f"Testing file: {test_path}")

seq = parquet_to_sequence(test_path)

if seq is None:
    print("FAILED — sequence returned None, file too short")
else:
    print(f"\nShape     : {seq.shape}")
    print(f"Max value : {seq.max():.4f}")
    print(f"Min value : {seq.min():.4f}")
    print(f"Mean      : {seq.mean():.4f}")
    print(f"Zeros %   : {(seq == 0).mean() * 100:.1f}%")
    print(f"\nFirst frame, first 12 values (left hand x0,y0,z0...):")
    print(seq[0, :12].round(4))

    if (seq == 0).mean() > 0.9:
        print("\nWARNING — mostly zeros, something is wrong")
    elif seq.max() < 0.001:
        print("\nWARNING — all values near zero, normalization failed")
    else:
        print("\nOK — data looks healthy, safe to run full extraction")