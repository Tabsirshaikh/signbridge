"""
asl_infer_v1.py  ──  Shared v1 inference for the baseline 71% model.
────────────────────────────────────────────────────────────────────
Mirror of asl_infer_v2.py for the original model, exposing the same
Predictor / extract_frame interface so app.py can swap between them.

Feature vector per frame (must match phase3_training_popsign.py):
    left_hand  21 x (x,y,z) = 63
    right_hand 21 x (x,y,z) = 63
    pose[:25]  25 x (x,y,z) = 75
    ------------------------------------------------ position = 201
    + velocity 201  ->  FEAT_DIM = 402

The v1 conv stem uses groups=3 (left hand / right hand / pose processed
separately); loading the checkpoint into a groups=1 model would fail on
weight shape.
"""
import json
import numpy as np
import torch
import torch.nn as nn

N_POS    = 201
FEAT_DIM = 402


# ── Model (identical to phase3_training_popsign.py) ──────────────────────────
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


# ── Landmark extraction (MediaPipe Holistic result -> 201-dim frame) ─────────
def _get_pts(lms_list, n):
    """Flatten a hand/pose landmark list to n*3 (x,y,z), zero-padded."""
    if not lms_list:
        return np.zeros(n * 3, dtype=np.float32)
    item = lms_list[0] if isinstance(lms_list[0], list) else lms_list
    arr  = np.array([[r.x, r.y, r.z] for r in item], dtype=np.float32).flatten()
    if len(arr) < n * 3:
        arr = np.pad(arr, (0, n * 3 - len(arr)))
    return arr[:n * 3]


def extract_frame(results):
    """MediaPipe HolisticLandmarker result -> one (201,) feature frame."""
    lh   = _get_pts(results.left_hand_landmarks,  21)          # 63
    rh   = _get_pts(results.right_hand_landmarks, 21)          # 63
    pose = _get_pts(results.pose_landmarks, 33)[:75]           # 75 (pose[:25])
    return np.concatenate([lh, rh, pose]).astype(np.float32)


# ── Normalization (identical to test.py / training) ──────────────────────────
def normalize_keypoints(seq):
    out = seq.copy()
    for i in range(len(out)):
        f    = out[i]
        lh   = f[0:63].reshape(21, 3)
        rh   = f[63:126].reshape(21, 3)
        pose = f[126:201].reshape(25, 3)
        for hand in (lh, rh):
            if np.abs(hand).sum() > 0.01:
                hand -= hand[0].copy()
                hand /= np.linalg.norm(hand[9] - hand[0]) + 1e-6
        if np.abs(pose).sum() > 0.01:
            mid   = (pose[11] + pose[12]) / 2
            pose -= mid
            pose /= np.linalg.norm(pose[11] - pose[12]) + 1e-6
    return out


def add_velocity(seq):
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=-1)


# ── Predictor ────────────────────────────────────────────────────────────────
class Predictor:
    """Loads the v1 checkpoint and turns a frame buffer into top-k predictions."""
    def __init__(self, model_path, labels_path, device=None, use_amp=None):
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = (self.device == "cuda") if use_amp is None else use_amp
        with open(labels_path) as fh:
            self.labels = json.load(fh)
        ckpt = torch.load(model_path, map_location=self.device)
        self.cfg = ckpt["config"]
        self.model = SignTransformer(
            feat_dim=self.cfg["feat_dim"], d_model=self.cfg["d_model"],
            num_heads=self.cfg["num_heads"], num_layers=self.cfg["num_layers"],
            ff_dim=self.cfg["ff_dim"], dropout=0.0,
            seq_len=self.cfg["seq_len"], num_classes=self.cfg["num_classes"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, sequence, topk=5):
        """sequence: iterable of (201,) frames. Returns [(label, prob0-1), ...]."""
        seq  = np.asarray(sequence, dtype=np.float32)      # (T, 201)
        seq  = normalize_keypoints(seq)
        seq  = add_velocity(seq)                            # (T, 402)
        mask = (np.abs(seq[:, :N_POS]).sum(axis=-1) < 0.01)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                logits = self.model(x, src_key_padding_mask=m)
        else:
            logits = self.model(x, src_key_padding_mask=m)
        probs = torch.softmax(logits, dim=1)[0]
        top   = torch.topk(probs, min(topk, probs.size(0)))
        return [(self.labels[i.item()], v.item())
                for v, i in zip(top.values, top.indices)]
