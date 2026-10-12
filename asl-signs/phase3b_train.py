# phase3b_train.py
# ─────────────────────────────────────────────────────────────────────────────
# Train the v2 Sign Transformer on the lip-augmented features (processed_v2/).
#
# What's new vs phase3_training_popsign.py:
#   * FEAT_DIM 562 (adds 40 lip landmarks) — the main accuracy lever.
#   * EMA (exponential moving average of weights) — evaluated + saved. Reliable
#     +0.5–1.5% on this task.
#   * CORRECTED mirror augmentation. The old code did `x = 1.0 - x`, but after
#     per-sample normalization coordinates are wrist/mouth-centered (not in
#     [0,1]), so that was mathematically wrong. Mirror now = swap hands + negate
#     the x channels (see feature_layout.build_mirror_index).
#   * Affine augmentation (small rotation + scale) that preserves zeros, so
#     missing/padded landmarks stay missing.
#
# Fits a 6 GB GPU at batch 64 with AMP. If you hit CUDA OOM, lower BATCH_SIZE
# to 32 and set ACCUM_STEPS = 2 (same effective batch, half the memory).
#
#   cd asl-signs
#   python phase3b_train.py
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from feature_layout import (
    N_POS, FEAT_DIM, build_mirror_index, build_xy_index,
)

# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
ACCUM_STEPS = 1          # raise to 2 (+ BATCH_SIZE 32) if you hit OOM
D_MODEL     = 256
NUM_HEADS   = 8
NUM_LAYERS  = 6
FF_DIM      = 512
DROPOUT     = 0.3
LR          = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS      = 70
WARMUP_EP   = 5
SEQ_LEN     = 64
EMA_DECAY   = 0.999
PATIENCE    = 20
LABEL_SMOOTH = 0.1
DATA_DIR    = "processed_v2"
MODEL_DIR   = "model_v2"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

_MIRROR_PERM, _MIRROR_SIGN = build_mirror_index()
_XY = build_xy_index()      # (pos_x, pos_y, vel_x, vel_y)


# ── Dataset ──────────────────────────────────────────────────────────────────
class KeypointDataset(Dataset):
    def __init__(self, split, augment=False):
        self.split, self.augment = split, augment
        self.y = np.load(f"{DATA_DIR}/y_{split}.npy")
        self.X = None
        shp = np.load(f"{DATA_DIR}/X_{split}.npy", mmap_mode="r").shape
        print(f"  {split:6s}: X={shp}  y={self.y.shape}")

    def __len__(self):
        return len(self.y)

    # ── augmentations (all operate on a (T, 562) float32 array) ──────────────
    def _augment(self, x):
        # 1. mild Gaussian noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
        # 2. mirror (swap hands + negate x) — zeros map to zeros
        if np.random.random() < 0.5:
            x = x[:, _MIRROR_PERM] * _MIRROR_SIGN
        # 3. affine: rotate + scale on (x,y); no translation so zeros stay zero
        if np.random.random() < 0.5:
            x = self._affine(x)
        # 4. time masking
        if np.random.random() < 0.3:
            t0 = np.random.randint(0, SEQ_LEN)
            w  = np.random.randint(1, max(2, SEQ_LEN // 8))
            x[t0:t0 + w] = 0.0
        # 5. speed variation (temporal resample back to SEQ_LEN)
        if np.random.random() < 0.4:
            factor  = np.random.uniform(0.8, 1.2)
            new_len = int(np.clip(SEQ_LEN * factor, SEQ_LEN // 2, SEQ_LEN * 2))
            idxs    = np.linspace(0, len(x) - 1, new_len).astype(np.int32)
            x       = x[idxs]
            if len(x) >= SEQ_LEN:
                x = x[:SEQ_LEN]
            else:
                pad = np.zeros((SEQ_LEN - len(x), x.shape[1]), dtype=np.float32)
                x   = np.concatenate([x, pad])
        # 6. whole-hand dropout (pos + velocity of that hand) for robustness
        if np.random.random() < 0.15:
            x = x.copy(); x[:, 0:63] = 0.0; x[:, N_POS:N_POS + 63] = 0.0
        if np.random.random() < 0.15:
            x = x.copy(); x[:, 63:126] = 0.0; x[:, N_POS + 63:N_POS + 126] = 0.0
        return np.ascontiguousarray(x, dtype=np.float32)

    def _affine(self, x):
        px, py, vx, vy = _XY
        th = np.random.uniform(-0.22, 0.22)          # ~ ±13°
        s  = np.random.uniform(0.9, 1.1)
        c, si = np.cos(th), np.sin(th)
        x = x.copy()
        for xi, yi in ((px, py), (vx, vy)):          # same rotation on pos & vel
            ax, ay = x[:, xi].copy(), x[:, yi].copy()
            x[:, xi] = s * (c * ax - si * ay)
            x[:, yi] = s * (si * ax + c * ay)
        return x

    def __getitem__(self, idx):
        if self.X is None:
            self.X = np.load(f"{DATA_DIR}/X_{self.split}.npy", mmap_mode="r")
        x    = self.X[idx].copy()
        mask = (np.abs(x[:, :N_POS]).sum(axis=-1) < 0.01)   # True = empty frame
        if self.augment:
            x = self._augment(x)
        return (torch.from_numpy(x).float(),
                int(self.y[idx]),
                torch.from_numpy(mask))


# ── Model (identical to evaluate_v2.py) ──────────────────────────────────────
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
        x = x.transpose(1, 2)
        x = self.local_motion(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        # guard: a fully-empty sequence would make attention produce NaNs
        if src_key_padding_mask is not None:
            all_pad = src_key_padding_mask.all(dim=1, keepdim=True)
            src_key_padding_mask = src_key_padding_mask & ~all_pad
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing, self.num_classes = smoothing, num_classes

    def forward(self, pred, target):
        smooth  = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(pred, smooth)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()


# ── EMA ──────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay):
        self.decay  = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = top5 = total = 0
    for x, labels, mask in loader:
        x, labels, mask = (x.to(DEVICE, non_blocking=True),
                           labels.to(DEVICE, non_blocking=True),
                           mask.to(DEVICE, non_blocking=True))
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(x, src_key_padding_mask=mask)
        correct += (logits.argmax(1) == labels).sum().item()
        top5    += (logits.topk(5, 1).indices == labels.unsqueeze(1)).any(1).sum().item()
        total   += labels.size(0)
    return correct / total * 100, top5 / total * 100


# ── Train ────────────────────────────────────────────────────────────────────
def train():
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    with open(f"{MODEL_DIR}/labels.json") as fh:
        labels = json.load(fh)
    num_classes = len(labels)
    print(f"Classes: {num_classes}   FEAT_DIM: {FEAT_DIM}\n")

    train_ds = KeypointDataset("train", augment=True)
    val_ds   = KeypointDataset("val",   augment=False)

    counts  = np.bincount(train_ds.y, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts[train_ds.y] + 1e-6)
    sampler = WeightedRandomSampler(torch.from_numpy(weights).float(),
                                    num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True, persistent_workers=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    model = SignTransformer(FEAT_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
                            FF_DIM, DROPOUT, SEQ_LEN, num_classes).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ema        = EMA(model, EMA_DECAY)
    ema_model  = copy.deepcopy(model)
    criterion  = LabelSmoothingCE(num_classes, LABEL_SMOOTH)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scaler     = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    steps_per_ep = len(train_dl)
    total_steps  = EPOCHS * steps_per_ep
    warmup_steps = WARMUP_EP * steps_per_ep

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc, best_top5, best_epoch, patience = 0.0, 0.0, 0, 0

    cfg = dict(feat_dim=FEAT_DIM, d_model=D_MODEL, num_heads=NUM_HEADS,
               num_layers=NUM_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
               seq_len=SEQ_LEN, num_classes=num_classes)

    print(f"\n{'Ep':>3} {'loss':>7} {'trn%':>6} {'raw%':>6} "
          f"{'ema%':>6} {'ema5%':>6} {'lr':>9}  time")
    print("─" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = correct = total = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for i, (x, labels, mask) in enumerate(train_dl):
            x, labels, mask = (x.to(DEVICE, non_blocking=True),
                               labels.to(DEVICE, non_blocking=True),
                               mask.to(DEVICE, non_blocking=True))
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(x, src_key_padding_mask=mask)
                loss   = criterion(logits, labels) / ACCUM_STEPS
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

            run_loss += loss.item() * ACCUM_STEPS
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_acc = correct / total * 100
        ema_model.load_state_dict(ema.shadow)
        raw_acc, _        = evaluate(model, val_dl)
        ema_acc, ema_top5 = evaluate(ema_model, val_dl)

        print(f"{epoch:3d} {run_loss/steps_per_ep:7.3f} {train_acc:6.1f} "
              f"{raw_acc:6.1f} {ema_acc:6.1f} {ema_top5:6.1f} "
              f"{optimizer.param_groups[0]['lr']:9.2e}  {time.time()-t0:.0f}s")

        score = max(ema_acc, raw_acc)
        if score > best_acc:
            best_acc, best_top5, best_epoch, patience = score, ema_top5, epoch, 0
            state = ema.shadow if ema_acc >= raw_acc else model.state_dict()
            torch.save({"epoch": epoch, "state_dict": state,
                        "val_acc": score, "val_top5": ema_top5,
                        "which": "ema" if ema_acc >= raw_acc else "raw",
                        "config": cfg},
                       f"{MODEL_DIR}/best_model.pt")
            print(f"      ✓ best {best_acc:.2f}% "
                  f"({'ema' if ema_acc >= raw_acc else 'raw'})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stop — no gain for {PATIENCE} epochs")
                break

    print(f"\nBest val accuracy: {best_acc:.2f}%  (epoch {best_epoch})")
    with open(f"{MODEL_DIR}/training_summary.json", "w") as fh:
        json.dump({"best_val_accuracy": best_acc, "best_val_top5": best_top5,
                   "best_epoch": best_epoch, "num_classes": num_classes,
                   "feat_dim": FEAT_DIM, "d_model": D_MODEL,
                   "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN}, fh, indent=2)
    print("Next: python evaluate_v2.py")


if __name__ == "__main__":
    train()
