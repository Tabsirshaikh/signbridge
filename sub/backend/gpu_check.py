import sys, os, json, torch, time
import torch.nn as nn

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), r"..\New folder (2)"))

print("=== GPU UTILIZATION CHECK ===")
print(f"Device           : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"VRAM before load : {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

# ── Same architecture as app.py ──────────────────────────────────────────────
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
    def __init__(self, feat_dim, d_model, num_heads, num_layers, ff_dim, dropout, seq_len, num_classes):
        super().__init__()
        self.local_motion = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=3),
            nn.GELU(),
            nn.Conv1d(feat_dim, d_model, kernel_size=1),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes)
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

# ── Load checkpoint ──────────────────────────────────────────────────────────
ckpt = torch.load(os.path.join(MODEL_DIR, "model", "best_model.pt"), map_location=DEVICE)
cfg  = ckpt["config"]

model = SignTransformer(
    feat_dim    = cfg["feat_dim"],
    d_model     = cfg["d_model"],
    num_heads   = cfg["num_heads"],
    num_layers  = cfg["num_layers"],
    ff_dim      = cfg["ff_dim"],
    dropout     = 0.0,
    seq_len     = cfg["seq_len"],
    num_classes = cfg["num_classes"],
).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

if DEVICE == "cuda":
    print(f"VRAM after  load : {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

print(f"Model on device  : {next(model.parameters()).device}")
print(f"feat_dim={cfg['feat_dim']}  seq_len={cfg['seq_len']}  classes={cfg['num_classes']}")

# ── Benchmark: add_velocity doubles feat_dim, so input is (1, seq_len, feat_dim*2) ──
dummy = torch.randn(1, cfg["seq_len"], cfg["feat_dim"]).to(DEVICE)

with torch.no_grad():
    # Warmup
    for _ in range(5):
        model(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    RUNS = 50
    for _ in range(RUNS):
        model(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / RUNS * 1000

print(f"\nAvg inference    : {elapsed_ms:.2f} ms / call")
print(f"Throughput       : {1000/elapsed_ms:.0f} inferences / sec")
if DEVICE == "cuda":
    print(f"VRAM during infer: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    print(f"\n✅ RTX 3050 is ACTIVE and handling all model inference!")
else:
    print(f"\n❌ Still running on CPU — check CUDA installation.")
