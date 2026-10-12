# phase2b_features.py
# ─────────────────────────────────────────────────────────────────────────────
# Rebuild the keypoint features WITH LIPS (mouth morphemes) added.
#
# Produces a strict superset of the original 402-dim features:
#   original 201 position dims (hands + pose[:25])  ->  kept identical
#   + 40 lip points (x,y) = 80 new position dims     ->  the new signal
#   position 281  ->  + velocity 281  ->  FEAT_DIM = 562
#
# Design choices that matter:
#   * Memory-safe: streams every sample into an on-disk memmap, so it never
#     holds the ~13 GB float32 array in RAM (you only have ~7 GB free).
#   * Reproducible split: uses the SAME train_test_split(random_state=42,
#     stratify) as the original phase2, so the new accuracy is directly
#     comparable to the old 71.38%.
#   * Fast: vectorized per-type extraction (no per-frame pandas loop).
#
# Run from the asl-signs directory (the folder containing train.csv):
#   cd asl-signs
#   python phase2b_features.py                 # full build (~40-60 min, 1 core)
#   python phase2b_features.py --workers 6     # parallel (faster)
#   python phase2b_features.py --limit 400     # quick smoke test (~30 s)
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, argparse
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from feature_layout import (
    LIPS, LIP_CORNER_LEFT, LIP_CORNER_RIGHT,
    N_POS, FEAT_DIM,
)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = "."               # where train.csv + train_landmark_files/ live
OUTPUT_DIR = "processed_v2"
MODEL_DIR  = "model_v2"
SEQ_LEN    = 64
COPY_CHUNK = 1024              # rows per chunk when splitting (RAM control)

_LIPS_ARR = np.asarray(LIPS)
_REMAP    = np.full(468, -1, dtype=np.int64)   # face landmark idx -> lip slot 0..39
for _slot, _idx in enumerate(LIPS):
    _REMAP[_idx] = _slot


# ── Per-file extraction (vectorized) ─────────────────────────────────────────
def _grab(df, fmap, T, type_name, n_pts, dims, lip=False):
    """Scatter one landmark type into a dense (T, n_pts*dims) block."""
    out = np.zeros((T, n_pts, 3), dtype=np.float32)
    sub = df[df["type"] == type_name]
    if len(sub):
        fi   = sub["frame"].map(fmap).to_numpy()
        li   = sub["landmark_index"].to_numpy()
        vals = sub[["x", "y", "z"]].to_numpy(dtype=np.float32)
        if lip:
            keep = np.isin(li, _LIPS_ARR)
            fi, vals = fi[keep], vals[keep]
            li = _REMAP[li[keep]]                 # -> 0..39
        valid = li < n_pts                        # pose keeps only idx < 25
        out[fi[valid], li[valid]] = vals[valid]
    return out[:, :, :dims].reshape(T, n_pts * dims)


def parquet_to_sequence(path):
    """Raw parquet -> (SEQ_LEN, 562) normalized position+velocity, or None."""
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

    frames = np.sort(df["frame"].unique())
    if len(frames) < 4:
        return None
    T    = len(frames)
    fmap = {int(f): i for i, f in enumerate(frames)}

    lh   = _grab(df, fmap, T, "left_hand",  21, 3)
    rh   = _grab(df, fmap, T, "right_hand", 21, 3)
    pose = _grab(df, fmap, T, "pose",       25, 3)
    lips = _grab(df, fmap, T, "face",       40, 2, lip=True)
    seq  = np.concatenate([lh, rh, pose, lips], axis=1)      # (T, 281)

    idxs = np.linspace(0, T - 1, SEQ_LEN).astype(np.int32)   # temporal resample
    seq  = seq[idxs]
    seq  = _normalize(seq)
    vel  = np.diff(seq, axis=0, prepend=seq[:1])
    seq  = np.concatenate([seq, vel], axis=-1)               # (64, 562)
    return np.nan_to_num(seq, nan=0.0).astype(np.float32)


def _normalize(seq):
    """Per-frame normalization: hands -> wrist-relative, pose -> shoulder-mid,
    lips -> mouth-centroid, each scaled to its own reference length."""
    out = seq.copy()
    for t in range(len(out)):
        f    = out[t]
        lh   = f[0:63].reshape(21, 3)
        rh   = f[63:126].reshape(21, 3)
        pose = f[126:201].reshape(25, 3)
        lips = f[201:281].reshape(40, 2)
        for hand in (lh, rh):
            if np.abs(hand).sum() > 0.01:
                hand -= hand[0].copy()                       # wrist to origin
                hand /= np.linalg.norm(hand[9] - hand[0]) + 1e-6
        if np.abs(pose).sum() > 0.01:
            mid   = (pose[11] + pose[12]) / 2                # shoulder midpoint
            pose -= mid
            pose /= np.linalg.norm(pose[11] - pose[12]) + 1e-6
        if np.abs(lips).sum() > 0.01:
            lips -= lips.mean(0)                             # mouth centroid
            mw    = np.linalg.norm(lips[LIP_CORNER_LEFT] - lips[LIP_CORNER_RIGHT]) + 1e-6
            lips /= mw                                        # mouth width scale
    return out


# top-level worker wrapper (must be importable for multiprocessing on Windows)
def _worker(path):
    try:
        return parquet_to_sequence(path)
    except Exception:
        return None


# ── Build ────────────────────────────────────────────────────────────────────
def _stratify_or_none(y, split_name):
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    if len(nonzero) and nonzero.min() >= 2:
        return y
    print(f"  ! {split_name} split has singleton classes; using unstratified smoke-test split")
    return None


def run(workers=1, limit=None):
    t_start = time.time()
    csv_path = os.path.join(BASE_DIR, "train.csv")
    train_csv = pd.read_csv(csv_path)
    if limit:
        train_csv = train_csv.iloc[:limit].reset_index(drop=True)
    n_total = len(train_csv)
    print(f"train.csv            : {n_total:,} samples, "
          f"{train_csv['sign'].nunique()} signs")

    # Labels: reproduce the original ordering (Counter.most_common).
    words     = [w for w, _ in Counter(train_csv["sign"]).most_common()]
    label_map = {w: i for i, w in enumerate(words)}
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as fh:
        json.dump(words, fh, indent=2)
    print(f"labels               : {len(words)} -> {MODEL_DIR}/labels.json")

    paths  = [os.path.join(BASE_DIR, p) for p in train_csv["path"].tolist()]
    labels = [label_map[s] for s in train_csv["sign"].tolist()]

    # Stream into an oversized on-disk memmap; keep only valid rows at the front.
    tmp_path = os.path.join(OUTPUT_DIR, "_all_tmp.npy")
    big = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32,
        shape=(n_total, SEQ_LEN, FEAT_DIM),
    )

    y_valid, count, errors = [], 0, 0
    print(f"\nExtracting features (workers={workers}) ...")

    def _consume(seq, lbl):
        nonlocal count, errors
        if seq is None:
            errors += 1
        else:
            big[count] = seq
            y_valid.append(lbl)
            count += 1

    if workers and workers > 1:
        import multiprocessing as mp
        try:
            with mp.Pool(processes=workers) as pool:
                for seq, lbl in tqdm(
                    zip(pool.imap(_worker, paths, chunksize=16), labels),
                    total=n_total,
                ):
                    _consume(seq, lbl)
        except Exception as exc:               # fall back to single process
            print(f"  ! parallel path failed ({exc}); retrying single-process")
            big[:] = 0; y_valid.clear(); count = 0; errors = 0
            for p, lbl in tqdm(zip(paths, labels), total=n_total):
                _consume(_worker(p), lbl)
    else:
        for p, lbl in tqdm(zip(paths, labels), total=n_total):
            _consume(_worker(p), lbl)

    big.flush()
    print(f"  extracted {count:,}   skipped {errors:,}")

    y = np.asarray(y_valid, dtype=np.int32)

    # Reproduce the original two-stage stratified split, on positions.
    idx = np.arange(count)
    stratify_all = _stratify_or_none(y, "train/temp")
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.2, stratify=stratify_all, random_state=42)
    stratify_tmp = _stratify_or_none(y[idx_tmp], "val/test")
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.5, stratify=stratify_tmp, random_state=42)

    for name, sel in [("train", idx_tr), ("val", idx_val), ("test", idx_te)]:
        out = np.lib.format.open_memmap(
            os.path.join(OUTPUT_DIR, f"X_{name}.npy"), mode="w+",
            dtype=np.float32, shape=(len(sel), SEQ_LEN, FEAT_DIM),
        )
        for c0 in range(0, len(sel), COPY_CHUNK):
            chunk = sel[c0:c0 + COPY_CHUNK]
            out[c0:c0 + len(chunk)] = big[chunk]
        out.flush(); del out
        np.save(os.path.join(OUTPUT_DIR, f"y_{name}.npy"), y[sel])
        print(f"  {name:5s}: X=({len(sel)}, {SEQ_LEN}, {FEAT_DIM})  "
              f"classes={np.unique(y[sel]).size}")

    del big
    os.remove(tmp_path)

    with open(os.path.join(OUTPUT_DIR, "feature_meta.json"), "w") as fh:
        json.dump({"seq_len": SEQ_LEN, "n_pos": N_POS, "feat_dim": FEAT_DIM,
                   "num_classes": len(words), "n_valid": int(count),
                   "added": "40 lip landmarks (x,y)"}, fh, indent=2)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. "
          f"Next: python phase3b_train.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel parser processes (1 = single, robust)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N rows (smoke test)")
    args = ap.parse_args()
    run(workers=args.workers, limit=args.limit)
