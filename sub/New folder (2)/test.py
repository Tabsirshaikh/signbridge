# test_inference.py
# Run from backend/ folder: python test_inference.py

import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from collections import deque

os.environ["GLOG_minloglevel"]     = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH  = "model/best_model.pt"
LABELS_PATH = "model/labels.json"
SEQ_LEN     = 64
FEAT_DIM    = 402    # Wait, the config dict has feat_dim=402, we don't need this constant actually but we keep it
CONFIDENCE  = 0.60    # only show prediction if above this
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────

# ── Same model architecture as training ──────────────────────────────
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

# ── Load model ────────────────────────────────────────────────────────
print("Loading model...")
with open(LABELS_PATH) as f:
    LABELS = json.load(f)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
cfg        = checkpoint["config"]

model = SignTransformer(
    feat_dim   = cfg["feat_dim"],
    d_model    = cfg["d_model"],
    num_heads  = cfg["num_heads"],
    num_layers = cfg["num_layers"],
    ff_dim     = cfg["ff_dim"],
    dropout    = 0.0,          # disable dropout at inference
    seq_len    = cfg["seq_len"],
    num_classes= cfg["num_classes"]
).to(DEVICE)

model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Model loaded — {cfg['num_classes']} classes — device: {DEVICE}")

# ── MediaPipe setup ───────────────────────────────────────────────────
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='holistic_landmarker.task')
options = vision.HolisticLandmarkerOptions(
    base_options=base_options
)

def extract_keypoints(results):
    def get_lms(lms_list, num_expected):
        if not lms_list:
            return np.zeros(num_expected * 3)
        item = lms_list[0] if isinstance(lms_list[0], list) else lms_list
        return np.array([[r.x, r.y, r.z] for r in item]).flatten()
        
    lh   = get_lms(results.left_hand_landmarks, 21)
    rh   = get_lms(results.right_hand_landmarks, 21)
    pose = get_lms(results.pose_landmarks, 33)[:75]
    return np.concatenate([lh, rh, pose])

def draw_landmarks_cv2(frame, results):
    """Fallback drawing since solutions.drawing_utils is removed in Python 3.13"""
    h, w = frame.shape[:2]
    def safe_draw(lms_list, color):
        if not lms_list: return
        item = lms_list[0] if isinstance(lms_list[0], list) else lms_list
        for lm in item:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 2, color, -1)
                
    safe_draw(results.left_hand_landmarks, (245, 117, 66))
    safe_draw(results.right_hand_landmarks, (121, 22, 76))
    safe_draw(results.pose_landmarks, (0, 255, 0))

def add_velocity(seq):
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=-1)

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

# ── Inference ─────────────────────────────────────────────────────────
@torch.no_grad()
def predict(sequence):
    """
    sequence: list of SEQ_LEN numpy arrays each (201,)
    returns: list of (word, confidence%) sorted best first
    """
    seq = np.array(sequence, dtype=np.float32)
    seq = normalize_keypoints(seq)
    seq = add_velocity(seq)                             # (64, 402)
    x   = torch.from_numpy(seq).unsqueeze(0).to(DEVICE) # (1, 64, 402)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)[0]
    top5   = torch.topk(probs, 5)
    return [
        (LABELS[idx.item()], round(prob.item() * 100, 1))
        for prob, idx in zip(top5.values, top5.indices)
    ]

# ── Webcam loop ───────────────────────────────────────────────────────
def run():
    cap     = cv2.VideoCapture(0)
    buffer  = deque(maxlen=SEQ_LEN)   # sliding window of keypoints
    last_pred    = None
    last_pred_time = 0
    PRED_INTERVAL  = 0.5              # predict every 0.5 seconds

    print("\nWebcam started. Sign something!")
    print("Press Q to quit.\n")

    with vision.HolisticLandmarker.create_from_options(options) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip so it acts like a mirror
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Run MediaPipe
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = holistic.detect(mp_image)

            # Draw hand landmarks on frame
            draw_landmarks_cv2(frame, results)

            # Add keypoints to sliding window
            kp = extract_keypoints(results)
            buffer.append(kp)

            # Buffer fill progress bar
            fill    = len(buffer) / SEQ_LEN
            bar_w   = 200
            bar_h   = 12
            filled  = int(bar_w * fill)
            cv2.rectangle(frame, (10, h-30), (10+bar_w, h-30+bar_h),
                          (50,50,50), -1)
            cv2.rectangle(frame, (10, h-30), (10+filled, h-30+bar_h),
                          (0,255,0) if fill == 1.0 else (0,165,255), -1)
            cv2.putText(frame,
                        "Ready" if fill == 1.0 else f"Collecting {int(fill*100)}%",
                        (220, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1)

            # Predict every PRED_INTERVAL seconds once buffer is full
            now = time.time()
            if (len(buffer) == SEQ_LEN and
                    now - last_pred_time > PRED_INTERVAL):

                predictions  = predict(list(buffer))
                last_pred    = predictions
                last_pred_time = now

                # Print to terminal
                top_word, top_conf = predictions[0]
                if top_conf / 100 >= CONFIDENCE:
                    print(f"DETECTED: {top_word:30s} {top_conf:.1f}%")
                    print(f"  Top-5: " +
                          "  |  ".join(f"{w} {c}%" for w,c in predictions))
                    print()

            # Draw predictions on frame
            if last_pred:
                top_word, top_conf = last_pred[0]
                color = (0,255,0) if top_conf/100 >= CONFIDENCE else (0,165,255)

                # Big top prediction
                cv2.rectangle(frame, (0,0), (w, 60), (0,0,0), -1)
                cv2.putText(frame,
                            f"{top_word}  {top_conf:.0f}%",
                            (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, color, 2)

                # Top-5 list on right side
                for i, (word, conf) in enumerate(last_pred):
                    bar_color = (0,255,0) if i == 0 else (200,200,200)
                    y = 80 + i * 30
                    cv2.putText(frame,
                                f"{i+1}. {word:20s} {conf:.1f}%",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, bar_color, 1)

            cv2.imshow("Sign Language — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()