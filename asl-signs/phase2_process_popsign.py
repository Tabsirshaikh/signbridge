# phase2_process_popsign.py  ── FINAL version with NaN fix
import pandas as pd
import numpy as np
import json, os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────
KAGGLE_DIR = r"C:\Users\524ta\Downloads\asl-signs"
OUTPUT_DIR = "processed"
MODEL_DIR  = "model"
SEQ_LEN    = 64
# ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


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


def parquet_to_sequence(path):
    df = pd.read_parquet(path)

    # Replace all NaN with 0 immediately
    df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

    frames = sorted(df["frame"].unique())
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

    # Final NaN guard
    seq = np.nan_to_num(seq, nan=0.0)

    return seq.astype(np.float32)


def run():
    print("Loading train.csv...")
    train_csv = pd.read_csv(f"{KAGGLE_DIR}/train.csv")
    print(f"  Total samples : {len(train_csv):,}")
    print(f"  Unique signs  : {train_csv['sign'].nunique()}")

    counts    = Counter(train_csv["sign"])
    words     = [w for w, _ in counts.most_common()]
    label_map = {w: i for i, w in enumerate(words)}

    with open(f"{MODEL_DIR}/labels.json", "w") as f:
        json.dump(words, f, indent=2)
    print(f"  Saved {len(words)} labels → model/labels.json")

    print("\nExtracting keypoint sequences...")
    X_all, y_all = [], []
    errors = 0

    for _, row in tqdm(train_csv.iterrows(), total=len(train_csv)):
        pq_path = f"{KAGGLE_DIR}/{row['path']}"
        if not os.path.exists(pq_path):
            errors += 1
            continue

        seq = parquet_to_sequence(pq_path)
        if seq is None:
            errors += 1
            continue

        X_all.append(seq)
        y_all.append(label_map[row["sign"]])

    print(f"  Extracted : {len(X_all):,}")
    print(f"  Errors    : {errors:,}")

    X_arr = np.array(X_all, dtype=np.float32)
    y_arr = np.array(y_all, dtype=np.int32)

    # Final global NaN check on the whole array
    nan_count = np.isnan(X_arr).sum()
    print(f"  NaN in final array : {nan_count}")
    if nan_count > 0:
        print("  Fixing remaining NaNs...")
        X_arr = np.nan_to_num(X_arr, nan=0.0)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    for split, X, y in [("train", X_tr, y_tr),
                         ("val",   X_val, y_val),
                         ("test",  X_te,  y_te)]:
        np.save(f"{OUTPUT_DIR}/X_{split}.npy", X)
        np.save(f"{OUTPUT_DIR}/y_{split}.npy", y)
        print(f"  {split:6s}: {X.shape}  classes={np.unique(y).size}")

    print("\nDone. Run phase3_train_popsign.py next.")


if __name__ == "__main__":
    run()