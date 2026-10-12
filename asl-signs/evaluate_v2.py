# evaluate_v2.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluate the v2 model (model_v2/best_model.pt) on the held-out test set
# (processed_v2/X_test.npy), with and without mirror Test-Time Augmentation.
#
# Writes eval_outputs_v2/evaluation_summary.json in the SAME format as your
# original eval, so you can compare directly to the 71.38% baseline. TTA (mirror)
# usually adds another ~0.5–1%.
#
#   cd asl-signs
#   python evaluate_v2.py
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, accuracy_score,
                             top_k_accuracy_score, precision_recall_fscore_support)

from feature_layout import N_POS, FEAT_DIM, build_mirror_index

DATA_DIR   = "processed_v2"
MODEL_DIR  = "model_v2"
OUTPUT_DIR = "eval_outputs_v2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_PERM, _SIGN = build_mirror_index()
_PERM_T = torch.from_numpy(_PERM).long()
_SIGN_T = torch.from_numpy(_SIGN).float()


# ── Model (must match phase3b_train.py exactly) ──────────────────────────────
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
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(feat_dim, d_model, kernel_size=1),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes))

    def forward(self, x, src_key_padding_mask=None):
        x = x.transpose(1, 2); x = self.local_motion(x); x = x.transpose(1, 2)
        x = self.input_norm(x); x = self.pos_enc(x)
        if src_key_padding_mask is not None:
            all_pad = src_key_padding_mask.all(dim=1, keepdim=True)
            src_key_padding_mask = src_key_padding_mask & ~all_pad
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x); x = x.mean(dim=1)
        return self.head(x)


class TestDataset(Dataset):
    def __init__(self):
        self.X = np.load(f"{DATA_DIR}/X_test.npy", mmap_mode="r")
        self.y = np.load(f"{DATA_DIR}/y_test.npy")
        print(f"  Test set: X={self.X.shape}  y={self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x    = self.X[idx].copy().astype(np.float32)
        mask = (np.abs(x[:, :N_POS]).sum(axis=-1) < 0.01)
        return torch.from_numpy(x), int(self.y[idx]), torch.from_numpy(mask)


def load_model():
    ckpt = torch.load(f"{MODEL_DIR}/best_model.pt", map_location=DEVICE)
    cfg  = ckpt["config"]
    model = SignTransformer(**cfg).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    print(f"  Checkpoint: epoch={ckpt.get('epoch')}  "
          f"val_acc={ckpt.get('val_acc'):.2f}%  "
          f"weights={ckpt.get('which', '?')}")
    return model


@torch.no_grad()
def run(model, loader, tta=False):
    perm, sign = _PERM_T.to(DEVICE), _SIGN_T.to(DEVICE)
    labels_all, probs_all = [], []
    t0 = time.time()
    for x, labels, mask in loader:
        x, mask = x.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            p = F.softmax(model(x, src_key_padding_mask=mask), dim=1)
            if tta:
                xm = x[:, :, perm] * sign          # mirror: swap hands + negate x
                p  = p + F.softmax(model(xm, src_key_padding_mask=mask), dim=1)
                p  = p / 2
        labels_all.append(np.asarray(labels))
        probs_all.append(p.float().cpu().numpy())
    return (np.concatenate(labels_all), np.concatenate(probs_all),
            time.time() - t0)


def metrics(tag, y_true, y_probs, num_classes):
    y_pred = y_probs.argmax(1)
    top1 = accuracy_score(y_true, y_pred) * 100
    top5 = top_k_accuracy_score(y_true, y_probs, k=5,
                                labels=list(range(num_classes))) * 100
    pM, rM, fM, _ = precision_recall_fscore_support(y_true, y_pred,
                                                    average="macro", zero_division=0)
    pW, rW, fW, _ = precision_recall_fscore_support(y_true, y_pred,
                                                    average="weighted", zero_division=0)
    print(f"\n  [{tag}] top1={top1:.2f}%  top5={top5:.2f}%  "
          f"macroF1={fM*100:.2f}%  weightedF1={fW*100:.2f}%")
    return {"top1_accuracy_pct": round(top1, 4), "top5_accuracy_pct": round(top5, 4),
            "macro_precision_pct": round(pM*100, 4), "macro_recall_pct": round(rM*100, 4),
            "macro_f1_pct": round(fM*100, 4), "weighted_precision_pct": round(pW*100, 4),
            "weighted_recall_pct": round(rW*100, 4), "weighted_f1_pct": round(fW*100, 4)}


def main():
    print("=" * 60 + "\n  ASL Sign Transformer v2 — Evaluation\n" + "=" * 60)
    print(f"Device: {DEVICE}")
    with open(f"{MODEL_DIR}/labels.json") as fh:
        labels = json.load(fh)
    num_classes = len(labels)
    print(f"Classes: {num_classes}\n\nLoading model...")
    model = load_model()

    print("\nBuilding test loader...")
    test_dl = DataLoader(TestDataset(), batch_size=128, shuffle=False,
                         num_workers=4, pin_memory=(DEVICE == "cuda"))

    print("\nInference (no TTA)...")
    y_true, probs, elapsed = run(model, test_dl, tta=False)
    n = len(y_true)
    print(f"  {n:,} samples in {elapsed:.1f}s ({n/elapsed:.0f}/s)")
    m_plain = metrics("no-TTA", y_true, probs, num_classes)

    print("\nInference (mirror TTA)...")
    _, probs_tta, el2 = run(model, test_dl, tta=True)
    m_tta = metrics("mirror-TTA", y_true, probs_tta, num_classes)

    best = m_tta if m_tta["top1_accuracy_pct"] >= m_plain["top1_accuracy_pct"] else m_plain
    summary = {"test_samples": int(n),
               **best,
               "inference_seconds": round(elapsed, 2),
               "samples_per_second": round(n / elapsed, 1),
               "no_tta": m_plain, "mirror_tta": m_tta,
               "reported": "mirror_tta" if best is m_tta else "no_tta"}
    with open(f"{OUTPUT_DIR}/evaluation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary -> {OUTPUT_DIR}/evaluation_summary.json")

    # classification report (on the better of the two)
    y_pred = (probs_tta if best is m_tta else probs).argmax(1)
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w", encoding="utf-8") as fh:
        fh.write(f"v2 model — {summary['reported']}\n")
        fh.write(f"top1 {best['top1_accuracy_pct']:.2f}%  "
                 f"top5 {best['top5_accuracy_pct']:.2f}%\n\n")
        fh.write(classification_report(y_true, y_pred, target_names=labels,
                                       zero_division=0))

    # per-class accuracy CSV (find the hardest signs to guide next steps)
    per_c = np.zeros(num_classes); per_t = np.zeros(num_classes)
    for t, p in zip(y_true, y_pred):
        per_t[t] += 1; per_c[t] += (t == p)
    with open(f"{OUTPUT_DIR}/per_class_accuracy.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["sign", "n", "accuracy_pct"])
        for i in np.argsort(per_c / per_t.clip(min=1)):
            w.writerow([labels[i], int(per_t[i]),
                        round(per_c[i] / max(per_t[i], 1) * 100, 2)])
    print(f"  Report  -> {OUTPUT_DIR}/classification_report.txt")
    print(f"  Per-class CSV -> {OUTPUT_DIR}/per_class_accuracy.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
