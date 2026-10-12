# phase3_train_popsign.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import json, os, time
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────
# FEAT_DIM is 402 because we have 201 keypoints + 201 velocity features
# NUM_LAYERS is 6 meaning the transformer has 6 stacked encoder blocks
# each block learns increasingly abstract motion patterns:
#   layers 1-2 → raw joint positions
#   layers 3-4 → hand shape relationships
#   layers 5-6 → full sign motion pattern
BATCH_SIZE  = 64
D_MODEL     = 256
NUM_HEADS   = 8
NUM_LAYERS  = 6      # 6 stacked transformer encoder blocks
FF_DIM      = 512
DROPOUT     = 0.4
LR          = 1e-4
EPOCHS      = 80
SEQ_LEN     = 64
FEAT_DIM    = 402    # 201 keypoints + 201 velocity
DATA_DIR    = "processed"
MODEL_DIR   = "model"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────
class KeypointDataset(Dataset):
    def __init__(self, split, augment=False):
        self.split   = split
        self.augment = augment
        self.y       = np.load(f"{DATA_DIR}/y_{split}.npy")
        self.length  = len(self.y)
        self.X       = None
        
        # Check shape temporarily for print
        dummy_X = np.load(f"{DATA_DIR}/X_{split}.npy", mmap_mode='r')
        print(f"  {split:6s}: X={dummy_X.shape}  y={self.y.shape}")

    def augment_sequence(self, x):
        # Gaussian noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)

        # Mirror — swap left/right hand + flip x coords
        if np.random.random() < 0.5:
            x            = x.copy()
            tmp          = x[:, :63].copy()
            x[:, :63]    = x[:, 63:126]
            x[:, 63:126] = tmp
            x[:, 0::3]   = 1.0 - x[:, 0::3]

        # Time masking
        if np.random.random() < 0.3:
            t     = np.random.randint(0, SEQ_LEN)
            width = np.random.randint(1, max(2, SEQ_LEN // 8))
            x[t:t + width] = 0.0

        # Speed variation
        if np.random.random() < 0.4:
            factor  = np.random.uniform(0.8, 1.2)
            new_len = int(SEQ_LEN * factor)
            new_len = max(min(new_len, SEQ_LEN * 2), SEQ_LEN // 2)
            idxs    = np.linspace(0, len(x) - 1, new_len).astype(np.int32)
            x       = x[idxs]
            if len(x) >= SEQ_LEN:
                x = x[:SEQ_LEN]
            else:
                pad = np.zeros((SEQ_LEN - len(x), x.shape[1]), dtype=np.float32)
                x   = np.concatenate([x, pad])

        # Randomly zero out entire hand to teach robustness
        if np.random.random() < 0.15:
            x = x.copy()
            x[:, :63] = 0.0       # drop left hand
        if np.random.random() < 0.15:
            x = x.copy()
            x[:, 63:126] = 0.0    # drop right hand

        return x.astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.X is None:
            # Lazy load memory-mapped array per worker
            self.X = np.load(f"{DATA_DIR}/X_{self.split}.npy", mmap_mode='r')

        x    = self.X[idx].copy()
        mask = (np.abs(x[:, :201]).sum(axis=-1) < 0.01)  # True = padded frame
        if self.augment:
            x = self.augment_sequence(x)
        return (torch.from_numpy(x).float(),
                int(self.y[idx]),
                torch.from_numpy(mask))


# ── Positional encoding ───────────────────────────────────────────────
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


# ── Transformer model ─────────────────────────────────────────────────
class SignTransformer(nn.Module):
    def __init__(self, feat_dim, d_model, num_heads, num_layers,
                 ff_dim, dropout, seq_len, num_classes):
        super().__init__()

        # Local motion extractor — captures frame-to-frame velocity patterns
        # groups=3 means left hand / right hand / pose each processed separately
        self.local_motion = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=3),
            nn.GELU(),
            nn.Conv1d(feat_dim, d_model, kernel_size=1),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)

        # 6 stacked encoder layers — each one refines the representation
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True       # pre-norm = more stable training
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
        # x: (B, T, F)
        x = x.transpose(1, 2)                    # → (B, F, T) for Conv1d
        x = self.local_motion(x)                 # → (B, d_model, T)
        x = x.transpose(1, 2)                    # → (B, T, d_model)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        x = x.mean(dim=1)                        # global average pool
        return self.head(x)


# ── Label smoothing loss ──────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        smooth  = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(pred, smooth)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_p   = F.log_softmax(pred, dim=1)
        return -(one_hot * log_p).sum(dim=1).mean()


# ── Train ─────────────────────────────────────────────────────────────
def train():
    global NUM_CLASSES
    print(f"Device  : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    with open(f"{MODEL_DIR}/labels.json") as f:
        LABELS = json.load(f)
    NUM_CLASSES = len(LABELS)
    print(f"Classes : {NUM_CLASSES}")

    print("\nLoading datasets...")
    train_ds = KeypointDataset("train", augment=True)
    val_ds   = KeypointDataset("val",   augment=False)

    # WeightedRandomSampler — rare signs get same training attention as common ones
    labels_arr   = train_ds.y[:]
    class_counts = np.bincount(labels_arr)
    weights      = 1.0 / (class_counts[labels_arr].astype(np.float32) + 1e-6)
    sampler      = WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(train_ds),
        replacement=True
    )

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler,              # no shuffle=True when using sampler
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = SignTransformer(
        feat_dim=FEAT_DIM, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        seq_len=SEQ_LEN, num_classes=NUM_CLASSES
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters : {params:,}")

    criterion = LabelSmoothingCE(NUM_CLASSES, smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )

    total_steps  = EPOCHS * len(train_dl)
    warmup_steps = 5 * len(train_dl)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler    = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    patience = 0
    PATIENCE = 20

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Loss':>8} {'Train%':>8} "
          f"{'Val%':>8} {'Top5%':>8} {'LR':>10}  Time")
    print("─" * 65)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()

        for x, labels, mask in train_dl:
            x      = x.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(x, src_key_padding_mask=mask)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total * 100

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_correct = val_top5 = val_total = 0

        with torch.no_grad():
            for x, labels, mask in val_dl:
                x      = x.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                mask   = mask.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(x, src_key_padding_mask=mask)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_top5    += (logits.topk(5, dim=1).indices
                                == labels.unsqueeze(1)).any(1).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total * 100
        top5_acc = val_top5   / val_total * 100
        lr_now   = optimizer.param_groups[0]["lr"]
        elapsed  = time.time() - t0

        print(f"{epoch:6d} "
              f"{total_loss/len(train_dl):8.3f} "
              f"{train_acc:8.1f} "
              f"{val_acc:8.1f} "
              f"{top5_acc:8.1f} "
              f"{lr_now:10.2e}  "
              f"{elapsed:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "val_top5":   top5_acc,
                "config": {
                    "feat_dim":    FEAT_DIM,
                    "d_model":     D_MODEL,
                    "num_heads":   NUM_HEADS,
                    "num_layers":  NUM_LAYERS,
                    "ff_dim":      FF_DIM,
                    "dropout":     DROPOUT,
                    "seq_len":     SEQ_LEN,
                    "num_classes": NUM_CLASSES
                }
            }, f"{MODEL_DIR}/best_model.pt")
            print(f"         ✓ best saved — val {best_acc:.2f}%")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stop — no gain for {PATIENCE} epochs")
                break

    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")

    with open(f"{MODEL_DIR}/training_summary.json", "w") as f:
        json.dump({
            "best_val_accuracy": best_acc,
            "num_classes":       NUM_CLASSES,
            "d_model":           D_MODEL,
            "num_layers":        NUM_LAYERS,
            "seq_len":           SEQ_LEN,
            "feat_dim":          FEAT_DIM,
        }, f, indent=2)


if __name__ == "__main__":
    train()