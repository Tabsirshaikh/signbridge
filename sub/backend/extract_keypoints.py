"""
extract_keypoints.py
────────────────────
Reads train.csv + train_landmark_files/*.parquet
Extracts full skeleton (pose, right_hand, left_hand) for each of the 250 words.
Outputs: ../frontend/src/data/asl_keypoints.json

Run from the backend/ directory:
    python extract_keypoints.py
"""

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
# Set YASEER_DATA_DIR to the folder that contains train.csv and landmark parquet files.
_DEFAULT_YASEER = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Yaseer"))
YASEER_DIR  = os.environ.get("YASEER_DATA_DIR", _DEFAULT_YASEER)
TRAIN_CSV   = os.path.join(YASEER_DIR, "train.csv")
OUTPUT_JSON = os.path.join(
    os.path.dirname(__file__),
    r"..\frontend\src\data\asl_keypoints.json"
)

TARGET_FRAMES = 30   # frames per sign in the output JSON


def extract_full_skeleton(parquet_path: str, n_frames: int = TARGET_FRAMES):
    """
    Read a parquet file and return a list of `n_frames` dictionaries.
    Each dictionary has keys: 'pose', 'lh', 'rh', each containing a list of {x,y,z} dicts.
    Returns None if the file has insufficient data.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None

    df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

    # We need pose to be present generally. Let's see all unique frames.
    frames = sorted(df["frame"].unique())
    if len(frames) < 4:
        return None

    raw_frames = []
    
    # We will build an empty skeleton template in case hands are missing
    empty_pose = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(33)]
    empty_hand = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]

    for frame_idx in frames:
        fdf = df[df["frame"] == frame_idx]
        
        # Extractor helper
        def extract_type(t_type, req_size):
            pts = fdf[fdf["type"] == t_type].sort_values("landmark_index")[["x", "y", "z"]].values
            if len(pts) == req_size:
                return [{"x": round(float(p[0]), 4),
                         "y": round(float(p[1]), 4),
                         "z": round(float(p[2]), 4)} for p in pts]
            return None

        p_pts  = extract_type("pose", 33)
        lh_pts = extract_type("left_hand", 21)
        rh_pts = extract_type("right_hand", 21)
        
        # If no pose and no hands at all, skip frame
        if p_pts is None and lh_pts is None and rh_pts is None:
            continue
            
        raw_frames.append({
            "pose": p_pts or empty_pose,
            "lh": lh_pts or empty_hand,
            "rh": rh_pts or empty_hand
        })

    if len(raw_frames) < 4:
        return None

    # Resample to TARGET_FRAMES using linear index selection
    indices  = np.linspace(0, len(raw_frames) - 1, n_frames, dtype=int)
    resampled = [raw_frames[i] for i in indices]

    return resampled


def run():
    if not os.path.isfile(TRAIN_CSV):
        print(f"ERROR: train.csv not found at:\n  {TRAIN_CSV}\n"
              f"Set environment variable YASEER_DATA_DIR to your dataset root (folder containing train.csv).")
        return
    print("Loading train.csv …")
    train = pd.read_csv(TRAIN_CSV)
    print(f"  Total rows : {len(train):,}")
    print(f"  Unique signs: {train['sign'].nunique()}")

    # Group by sign — keep file paths for each
    sign_groups = train.groupby("sign")["path"].apply(list).to_dict()

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_JSON)), exist_ok=True)

    keypoints = {}
    errors    = []

    for sign, paths in tqdm(sign_groups.items(), desc="Extracting", ascii=True):
        sign_upper = sign.upper()
        extracted  = None

        for rel_path in paths:
            full_path = os.path.join(YASEER_DIR, rel_path.replace("/", os.sep))
            extracted = extract_full_skeleton(full_path)
            if extracted is not None:
                break

        if extracted is None:
            errors.append(sign)
        else:
            keypoints[sign_upper] = extracted

    print(f"\nExtracted : {len(keypoints)} signs")
    print(f"Failed    : {len(errors)} signs  -> {errors[:10]}")

    out_dir = os.path.dirname(os.path.abspath(OUTPUT_JSON))
    os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(keypoints, f)   # compact for fast loading

    size_mb = os.path.getsize(OUTPUT_JSON) / 1e6
    print(f"\nSaved to : {OUTPUT_JSON}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    run()
