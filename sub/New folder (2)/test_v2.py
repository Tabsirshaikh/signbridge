"""
test_v2.py  -  Sign Language to Proper English Sentence
──────────────────────────────────────────────────────────────────
Run:  python test_v2.py

Controls:
  ENTER      -> manually complete the current sentence
  Backspace  -> remove the last added word
  C          -> clear everything and start fresh
  Q          -> quit
"""

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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sentence_builder import SentenceBuilder
from grammar import fix_grammar

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH    = "model_v2/best_model.pt"
LABELS_PATH   = "model_v2/labels.json"
SEQ_LEN       = 64
CONFIDENCE    = 0.35
PRED_INTERVAL = 0.5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Colours (BGR) ─────────────────────────────────────────────────────
C_DARK   = (15,  15,  25)
C_PANEL  = (22,  22,  38)
C_GREEN  = (0,  210,  90)
C_ORANGE = (0,  145, 255)
C_WHITE  = (240, 240, 240)
C_GREY   = (140, 140, 160)
C_YELLOW = (0,  230, 255)

# ── Load v2 model (562-dim lip-aware features, see asl_infer_v2.py) ─────
import asl_infer_v2 as infer

print("Loading model...")
PREDICTOR = infer.Predictor(MODEL_PATH, LABELS_PATH, device=DEVICE)
LABELS    = PREDICTOR.labels
print(f"Model loaded  {len(LABELS)} classes  device: {DEVICE}  "
      f"feat_dim: {PREDICTOR.cfg['feat_dim']}")

# ── MediaPipe ────────────────────────────────────────────────────────
base_opts  = python.BaseOptions(model_asset_path="holistic_landmarker.task")
mp_options = vision.HolisticLandmarkerOptions(base_options=base_opts)


def extract_keypoints(results):
    """One (281,) frame: left_hand + right_hand + pose[:25] + 40 lips."""
    return infer.extract_frame(results)


def draw_landmarks(frame, results):
    h, w = frame.shape[:2]
    def safe_draw(lms_list, color):
        if not lms_list: return
        item = lms_list[0] if isinstance(lms_list[0], list) else lms_list
        for lm in item:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 2, color, -1)
    safe_draw(results.left_hand_landmarks,  (245, 117,  66))
    safe_draw(results.right_hand_landmarks, (121,  22,  76))
    safe_draw(results.pose_landmarks,       (  0, 255,   0))


@torch.no_grad()
def predict(sequence):
    """sequence: list of (281,) frames -> [(word, pct), ...] top-5."""
    return [(w, round(c * 100, 1))
            for w, c in PREDICTOR.predict(sequence, topk=5)]


# ── UI helpers ────────────────────────────────────────────────────────
def put_text(img, text, x, y, scale=0.55, color=C_WHITE, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_hbar(img, x, y, w, h, fill, fg, bg=(40, 40, 55)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + int(w * fill), y + h), fg, -1)


def draw_text_wrapped(img, text, x, y, max_w, scale, color, thickness=1, line_h=26):
    words  = text.split()
    lines, line = [], ""
    for w in words:
        test = line + (" " if line else "") + w
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > max_w and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    for i, ln in enumerate(lines):
        put_text(img, ln, x, y + i * line_h, scale, color, thickness)
    return y + len(lines) * line_h


# ── Main ──────────────────────────────────────────────────────────────
def run():
    cap  = cv2.VideoCapture(0)
    buf  = deque(maxlen=SEQ_LEN)

    last_pred      = None
    last_pred_time = 0.0

    sb = SentenceBuilder()

    # Display state
    english_sentence    = ""
    raw_words_display   = ""
    completed_sentences = []
    flash_msg           = ""
    flash_until         = 0.0

    print("\nWebcam started! Sign something.")
    print("Controls: ENTER=complete  Backspace=undo  C=clear  Q=quit\n")

    with vision.HolisticLandmarker.create_from_options(mp_options) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # ── MediaPipe ───────────────────────────────────────────
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = holistic.detect(mp_img)
            draw_landmarks(frame, results)
            kp = extract_keypoints(results)
            buf.append(kp)

            # ── Prediction ──────────────────────────────────────────
            now = time.time()
            if len(buf) == SEQ_LEN and now - last_pred_time > PRED_INTERVAL:
                last_pred      = predict(list(buf))
                last_pred_time = now
                top_word, top_conf = last_pred[0]
                
                new_word = sb.update(top_word, top_conf / 100)
                if new_word:
                    print(f"  + word: {new_word}  |  raw: {' '.join(sb.words)}")
                    flash_msg = f"+ {new_word}"
                    flash_until = now + 1.0

            # ─────────────── UI ────────────────────────────────────
            overlay = frame.copy()

            # ── TOP BAR: current detected word ─────────────────────
            cv2.rectangle(overlay, (0, 0), (w, 68), C_DARK, -1)
            if last_pred:
                tw, tc = last_pred[0]
                color  = C_GREEN if tc / 100 >= CONFIDENCE else C_ORANGE
                put_text(overlay, tw, 14, 44, 1.15, color, 2)
                conf_x = 14 + cv2.getTextSize(
                    tw, cv2.FONT_HERSHEY_SIMPLEX, 1.15, 2)[0][0] + 12
                put_text(overlay, f"{tc:.0f}%", conf_x, 44, 0.75, C_GREY)

                # Candidate lock progress
                cand = sb.candidate
                if cand:
                    lock_prog = sb.lock_progress
                    draw_hbar(overlay, 14, 56, 180, 6, lock_prog, C_ORANGE)
                    put_text(overlay, f"locking: {cand}",
                             200, 62, 0.35, C_GREY)

                # Top-5 on the right
                for i, (wrd, conf) in enumerate(last_pred[1:], 2):
                    put_text(overlay, f"{i}. {wrd}  {conf:.0f}%",
                             w - 260, 14 + i * 14, 0.38, C_GREY)

            # ── BOTTOM PANEL ───────────────────────────────────────
            bot_y = h - 140
            cv2.rectangle(overlay, (0, bot_y), (w, h), C_PANEL, -1)

            # Row 1: Raw words being collected
            put_text(overlay, "WORDS:", 12, bot_y + 28, 0.42, C_GREY)
            current_raw = " ".join(sb.words) if sb.words else "..."
            put_text(overlay, current_raw, 85, bot_y + 28, 0.50, C_WHITE)

            # Row 2: English sentence (grammar-corrected)
            put_text(overlay, "SENTENCE:", 12, bot_y + 65, 0.42, C_GREY)
            en_display = english_sentence if english_sentence else "..."
            draw_text_wrapped(overlay, en_display,
                              105, bot_y + 65, w - 120,
                              0.62, C_YELLOW, 1, 24)

            # Controls reminder
            put_text(overlay, "ENTER=complete   Backspace=undo   C=clear   Q=quit",
                     12, h - 8, 0.34, C_GREY)

            # ── Buffer fill bar ─────────────────────────────────────
            fill = len(buf) / SEQ_LEN
            draw_hbar(overlay, w - 210, h - 12, 195, 6, fill,
                      C_GREEN if fill == 1.0 else C_ORANGE)

            # ── Flash message ───────────────────────────────────────
            if flash_msg and now < flash_until:
                alpha  = min((flash_until - now) / 0.5, 1.0)
                fcolor = tuple(int(c * alpha) for c in C_GREEN)
                put_text(overlay, flash_msg,
                         w // 2 - 140, h // 2, 1.0, fcolor, 2)

            # ── Blend ───────────────────────────────────────────────
            frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)
            cv2.imshow("Sign Language to Sentence | Q to quit", frame)

            # ── Keyboard ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') or key == ord('C'):
                sb.clear()
                english_sentence  = ""
                raw_words_display = ""
                flash_msg   = "Cleared!"
                flash_until = now + 1.0
                print("Cleared.")
            elif key == 8:  # Backspace
                removed = sb.undo()
                if removed:
                    flash_msg   = f"Removed: {removed}"
                    flash_until = now + 1.0
                    print(f"Removed word: {removed}")
            elif key == 13:  # ENTER
                words = sb.complete()
                if not words:
                    flash_msg   = "No words yet"
                    flash_until = now + 1.0
                else:
                    raw_words_display = " ".join(words)
                    english_sentence = fix_grammar(words)
                    
                    completed_sentences.append({
                        "raw_words": raw_words_display,
                        "english":   english_sentence,
                    })
            
                    flash_msg   = "Sentence complete!"
                    flash_until = time.time() + 2.0
            
                    print()
                    print(f"  ── Sentence Complete ──")
                    print(f"  Raw words : {raw_words_display}")
                    print(f"  Grammar   : {english_sentence}")
                    print()

    cap.release()
    cv2.destroyAllWindows()

    # ── Session summary ─────────────────────────────────────────────
    if completed_sentences:
        print("\n== Session Summary ==")
        for i, s in enumerate(completed_sentences, 1):
            print(f"  {i}. Words   : {s['raw_words']}")
            print(f"     Sentence: {s['english']}")
            print()


if __name__ == "__main__":
    run()
