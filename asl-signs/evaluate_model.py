# evaluate_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation of the trained ASL Sign Transformer on the held-out test set.
#
# Metrics computed:
#   • Top-1 Accuracy
#   • Top-5 Accuracy
#   • Per-class Precision, Recall, F1-score (classification report)
#   • Macro / Weighted F1
#   • Confusion matrix (saved as PNG)
#   • Per-class accuracy bar chart (saved as PNG)
#
# Usage:
#   cd asl-signs
#   python evaluate_model.py
#
# Dependencies: torch, numpy, scikit-learn, matplotlib, seaborn, tqdm
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "processed"
MODEL_DIR     = "model"
OUTPUT_DIR    = "eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ── Model architecture (must match training exactly) ─────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos   = torch.arange(seq_len).unsqueeze(1).float()
        dim   = torch.arange(0, d_model, 2).float()
        angle = pos / torch.pow(10000, dim / d_model)
        pe    = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(angle)
        pe[0, :, 1::2] = torch.cos(angle)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe)


class SignTransformer(nn.Module):
    def __init__(self, feat_dim, d_model, num_heads, num_layers,
                 ff_dim, dropout, seq_len, num_classes):
        super().__init__()
        self.local_motion = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=3),
            nn.GELU(),
            nn.Conv1d(feat_dim, d_model, kernel_size=1),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, src_key_padding_mask=None):
        x = x.transpose(1, 2)
        x = self.local_motion(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ── Test Dataset ─────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    """Memory-mapped, no augmentation."""
    def __init__(self):
        self.X = np.load(f"{PROCESSED_DIR}/X_test.npy", mmap_mode="r")
        self.y = np.load(f"{PROCESSED_DIR}/y_test.npy")
        print(f"  Test set: X={self.X.shape}  y={self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x    = self.X[idx].copy().astype(np.float32)
        mask = (np.abs(x[:, :201]).sum(axis=-1) < 0.01)   # True = padded frame
        return (
            torch.from_numpy(x),
            int(self.y[idx]),
            torch.from_numpy(mask),
        )


# ── Load model ───────────────────────────────────────────────────────────────
def load_model(labels):
    ckpt   = torch.load(f"{MODEL_DIR}/best_model.pt", map_location=DEVICE)
    cfg    = ckpt["config"]
    model  = SignTransformer(
        feat_dim=cfg["feat_dim"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"], dropout=cfg["dropout"],
        seq_len=cfg["seq_len"], num_classes=cfg["num_classes"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Checkpoint: epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.2f}%  "
          f"val_top5={ckpt['val_top5']:.2f}%")
    return model, cfg


# ── Inference ────────────────────────────────────────────────────────────────
def run_inference(model, loader):
    all_labels  = []
    all_preds   = []
    all_probs   = []

    t0 = time.time()
    with torch.no_grad():
        for x, labels, mask in tqdm(loader, desc="Evaluating", unit="batch"):
            x      = x.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)
            labels_np = np.array(labels)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(x, src_key_padding_mask=mask)

            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)

            all_labels.append(labels_np)
            all_preds.append(preds)
            all_probs.append(probs)

    elapsed = time.time() - t0
    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)
    return all_labels, all_preds, all_probs, elapsed


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, top_n=50):
    """
    Plot a confusion matrix for the top-N most-frequent classes.
    Full matrix saved separately as a CSV for deep inspection.
    """
    cm_full = confusion_matrix(y_true, y_pred,
                               labels=list(range(len(labels))))

    # ── save full CSV ──────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "confusion_matrix_full.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, row in enumerate(cm_full):
            writer.writerow([labels[i]] + list(row))
    print(f"  Full confusion matrix saved -> {csv_path}")

    # ── plot top-N classes ────────────────────────────────────────────────
    counts  = np.bincount(y_true, minlength=len(labels))
    top_idx = np.argsort(counts)[::-1][:top_n]
    top_lbl = [labels[i] for i in top_idx]
    cm_sub  = cm_full[np.ix_(top_idx, top_idx)]

    # Normalize per row (recall)
    row_sum = cm_sub.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_sub / row_sum

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        cm_norm, annot=False, fmt=".2f", cmap="Blues",
        xticklabels=top_lbl, yticklabels=top_lbl,
        linewidths=0.3, ax=ax, vmin=0, vmax=1
    )
    ax.set_title(f"Normalised Confusion Matrix - top {top_n} classes (row = recall)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_top{top_n}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot  -> {path}")


def plot_per_class_accuracy(y_true, y_pred, labels):
    """Bar chart of per-class accuracy (Top-1 recall per class)."""
    n = len(labels)
    per_class_correct = np.zeros(n, dtype=int)
    per_class_total   = np.zeros(n, dtype=int)

    for true, pred in zip(y_true, y_pred):
        per_class_total[true]   += 1
        if true == pred:
            per_class_correct[true] += 1

    # Only plot classes that appear in the test set
    present   = np.where(per_class_total > 0)[0]
    acc_vals  = per_class_correct[present] / per_class_total[present].clip(min=1)
    lbl_names = [labels[i] for i in present]

    # Sort by accuracy descending
    order      = np.argsort(acc_vals)[::-1]
    acc_sorted = acc_vals[order]
    lbl_sorted = [lbl_names[i] for i in order]

    # ── Full bar chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(present) * 0.22), 6))
    colors  = ["#2ecc71" if a >= 0.7 else "#e67e22" if a >= 0.4 else "#e74c3c"
               for a in acc_sorted]
    ax.bar(range(len(lbl_sorted)), acc_sorted * 100, color=colors, width=0.8)
    ax.set_xticks(range(len(lbl_sorted)))
    ax.set_xticklabels(lbl_sorted, rotation=90, fontsize=6)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 105)
    ax.axhline(70, color="gray", linestyle="--", linewidth=0.8, label="70%")
    ax.set_title("Per-class Accuracy (Top-1) on Test Set", fontsize=13)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Per-class accuracy plot -> {path}")

    # ── Bottom-20 (hardest classes) ─────────────────────────────────────
    bottom20_idx  = order[-20:][::-1]
    bottom20_acc  = acc_vals[bottom20_idx]
    bottom20_lbls = [lbl_names[i] for i in bottom20_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(bottom20_lbls)), bottom20_acc * 100,
            color="#e74c3c", height=0.6)
    ax.set_yticks(range(len(bottom20_lbls)))
    ax.set_yticklabels(bottom20_lbls, fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, 105)
    ax.set_title("20 Hardest Signs (lowest test accuracy)", fontsize=12)
    ax.set_xlabel("Accuracy (%)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "hardest_20_signs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Hardest-20 signs plot   -> {path}")

    return per_class_correct, per_class_total


def plot_top5_per_class(y_true, all_probs, labels):
    """Per-class Top-5 recall bar chart."""
    n = len(labels)
    top5_correct = np.zeros(n, dtype=int)
    totals       = np.zeros(n, dtype=int)

    for true, prob in zip(y_true, all_probs):
        top5 = np.argsort(prob)[::-1][:5]
        totals[true] += 1
        if true in top5:
            top5_correct[true] += 1

    present   = np.where(totals > 0)[0]
    acc_vals  = top5_correct[present] / totals[present].clip(min=1)
    lbl_names = [labels[i] for i in present]
    order     = np.argsort(acc_vals)[::-1]

    fig, ax = plt.subplots(figsize=(max(14, len(present) * 0.22), 6))
    ax.bar(range(len(order)), acc_vals[order] * 100, color="#3498db", width=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lbl_names[i] for i in order], rotation=90, fontsize=6)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 105)
    ax.set_title("Per-class Top-5 Accuracy on Test Set", fontsize=13)
    ax.set_ylabel("Top-5 Accuracy (%)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_top5_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Per-class Top-5 plot    -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  ASL Sign Transformer -- Evaluation Matrix")
    print("=" * 60)

    # 1. Load labels
    with open(f"{MODEL_DIR}/labels.json") as f:
        labels = json.load(f)
    num_classes = len(labels)
    print(f"\nClasses : {num_classes}")

    # 2. Load model
    print("\nLoading model checkpoint...")
    model, cfg = load_model(labels)

    # 3. Build test DataLoader
    print("\nBuilding test DataLoader...")
    test_ds = TestDataset()
    test_dl = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    # 4. Run inference
    print("\nRunning inference...")
    y_true, y_pred, y_probs, elapsed = run_inference(model, test_dl)

    n_samples = len(y_true)
    print(f"  Samples evaluated : {n_samples:,}")
    print(f"  Inference time    : {elapsed:.1f}s  "
          f"({n_samples/elapsed:.0f} samples/sec)")

    # ── 5. Core scalar metrics ──────────────────────────────────────────
    top1_acc = accuracy_score(y_true, y_pred) * 100
    top5_acc = top_k_accuracy_score(y_true, y_probs, k=5,
                                    labels=list(range(num_classes))) * 100

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",  zero_division=0
    )
    prec_wtd,  rec_wtd,  f1_wtd,  _   = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "-" * 50)
    print("  SUMMARY METRICS")
    print("-" * 50)
    print(f"  Top-1 Accuracy       : {top1_acc:6.2f}%")
    print(f"  Top-5 Accuracy       : {top5_acc:6.2f}%")
    print(f"  Macro  Precision     : {prec_macro*100:6.2f}%")
    print(f"  Macro  Recall        : {rec_macro*100:6.2f}%")
    print(f"  Macro  F1-score      : {f1_macro*100:6.2f}%")
    print(f"  Weighted Precision   : {prec_wtd*100:6.2f}%")
    print(f"  Weighted Recall      : {rec_wtd*100:6.2f}%")
    print(f"  Weighted F1-score    : {f1_wtd*100:6.2f}%")
    print("-" * 50)

    # ── 6. Full classification report ──────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        zero_division=0
    )
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ASL Sign Transformer - Classification Report\n")
        f.write(f"Test samples : {n_samples}\n")
        f.write(f"Top-1 Acc    : {top1_acc:.2f}%\n")
        f.write(f"Top-5 Acc    : {top5_acc:.2f}%\n\n")
        f.write(report)
    print(f"\n  Full report saved -> {report_path}")

    # ── 7. Summary JSON ────────────────────────────────────────────────
    summary = {
        "test_samples"        : int(n_samples),
        "top1_accuracy_pct"   : round(top1_acc, 4),
        "top5_accuracy_pct"   : round(top5_acc, 4),
        "macro_precision_pct" : round(prec_macro * 100, 4),
        "macro_recall_pct"    : round(rec_macro  * 100, 4),
        "macro_f1_pct"        : round(f1_macro   * 100, 4),
        "weighted_precision_pct": round(prec_wtd  * 100, 4),
        "weighted_recall_pct" : round(rec_wtd    * 100, 4),
        "weighted_f1_pct"     : round(f1_wtd     * 100, 4),
        "inference_seconds"   : round(elapsed, 2),
        "samples_per_second"  : round(n_samples / elapsed, 1),
    }
    json_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved  -> {json_path}")

    # ── 8. Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_confusion_matrix(y_true, y_pred, labels, top_n=50)
    plot_per_class_accuracy(y_true, y_pred, labels)
    plot_top5_per_class(y_true, y_probs, labels)

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print(f"  All outputs saved in: ./{OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
