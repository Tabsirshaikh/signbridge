"""
app.py  ──  SignBridge Flask-SocketIO Backend (gevent async)
──────────────────────────────────────────────────────────────
WebSocket events (Socket.IO):
  CLIENT  → SERVER  : 'frame'           { frame_b64 }
  SERVER  → CLIENT  : 'prediction'      { predictions, landmarks, buf_fill, buf_max }

REST routes (unchanged):
  POST /complete-sentence  { words[] }       → { sentence, translations{}, audio_b64 }
  POST /text-to-sign       { text, lang }    → { english_text, keywords[] }
  POST /speech-to-text     { audio_b64, lang } → { text }

Run:
  pip install -r requirements.txt
  python app.py                          # v1 model (default)
  $env:SIGNBRIDGE_MODEL="v2"; python app.py   # fine-tuned v2 model
"""

# ── gevent monkey-patch MUST be the very first import ───────────────────────
from gevent import monkey
monkey.patch_all()

import os, sys, json, re, base64, io, time, tempfile, threading, subprocess

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND_ROOT, ".env"))
except ImportError:
    pass

import cv2
from collections import deque

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ── Add model directory to path ───────────────────────────────────────────────
MODEL_DIR = os.path.join(_BACKEND_ROOT, r"..\New folder (2)")
sys.path.insert(0, os.path.abspath(MODEL_DIR))

import numpy as np
import torch
import torch.nn as nn

# ── Suppress MediaPipe / TF logs ─────────────────────────────────────────────
os.environ["GLOG_minloglevel"]     = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Gemini ────────────────────────────────────────────────────────────────────
import signbridge_nlp as nlp
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")
SEQ_LEN       = 64
CONFIDENCE    = 0.35
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# Faster throttle on GPU; conservative on CPU to avoid lag
PRED_INTERVAL = 0.15 if DEVICE == "cuda" else 0.4
MP_FRAME_W    = 640          # resize frames to this width before MediaPipe

LANG_CODES = {
    "hindi":   "hi",
    "marathi": "mr",
    "spanish": "es",
    "french":  "fr",
    "arabic":  "ar",
}

# ── Load model: v1 (default) or v2, chosen with SIGNBRIDGE_MODEL ──────────────
#   v1 : 402-dim hands + pose features, groups=3   (asl_infer_v1, model/)
#   v2 : 562-dim lip-aware fine-tuned features     (asl_infer_v2, model_v2/)
# Model class + feature extraction live in asl_infer_v* (importable because
# MODEL_DIR was added to sys.path above) so live inference matches training.
MODEL_VERSION = os.getenv("SIGNBRIDGE_MODEL", "v1").strip().lower()
MODEL_INFO = {
    "v1": {"module": "asl_infer_v1", "dir": "model",    "label": "v1 · baseline"},
    "v2": {"module": "asl_infer_v2", "dir": "model_v2", "label": "v2 · fine-tuned"},
}
if MODEL_VERSION not in MODEL_INFO:
    raise SystemExit(f"SIGNBRIDGE_MODEL must be one of {sorted(MODEL_INFO)}, got '{MODEL_VERSION}'")

import importlib
infer = importlib.import_module(MODEL_INFO[MODEL_VERSION]["module"])
MODEL_LABEL = MODEL_INFO[MODEL_VERSION]["label"]

print(f"Loading model {MODEL_VERSION} …")
_model_subdir = MODEL_INFO[MODEL_VERSION]["dir"]
_labels_path  = os.path.join(MODEL_DIR, _model_subdir, "labels.json")
_model_path   = os.path.join(MODEL_DIR, _model_subdir, "best_model.pt")

PREDICTOR = infer.Predictor(_model_path, _labels_path, device=DEVICE)
LABELS    = PREDICTOR.labels
_cfg      = PREDICTOR.cfg
_USE_AMP  = PREDICTOR.use_amp
print(f"Model {MODEL_VERSION} loaded - {len(LABELS)} classes | device: {DEVICE} | "
      f"amp: {_USE_AMP} | feat_dim: {_cfg['feat_dim']}")

# ── MediaPipe holistic ────────────────────────────────────────────────────────
_task_path = os.path.join(MODEL_DIR, "holistic_landmarker.task")

def _build_holistic():
    """Try GPU delegate first, fall back to CPU silently."""
    for delegate in (mp_python.BaseOptions.Delegate.GPU,
                     mp_python.BaseOptions.Delegate.CPU):
        try:
            base_opts = mp_python.BaseOptions(
                model_asset_path=_task_path,
                delegate=delegate,
            )
            opts = vision.HolisticLandmarkerOptions(base_options=base_opts)
            h = vision.HolisticLandmarker.create_from_options(opts)
            label = "GPU" if delegate == mp_python.BaseOptions.Delegate.GPU else "CPU"
            print(f"MediaPipe holistic delegate: {label}")
            return h
        except Exception as e:
            print(f"MediaPipe delegate {delegate} failed ({e}), trying next …")
    raise RuntimeError("Could not initialise MediaPipe HolisticLandmarker")

HOLISTIC       = _build_holistic()
_holistic_lock = threading.Lock()

# Per-session frame buffer & prediction throttle
FRAME_BUF       = deque(maxlen=SEQ_LEN)
_last_pred_time = 0.0
_last_preds     = []


# ── Keypoint helpers (v1: 201-dim hands+pose, v2: 281-dim incl. 40 lips) ──────
def extract_keypoints(results):
    """One feature frame for the active model (see asl_infer_v1/v2.py)."""
    return infer.extract_frame(results)


@torch.no_grad()
def predict(sequence):
    """sequence: list of per-frame features -> [{'word','conf'}, ...] top-5."""
    return [{"word": w, "conf": round(c, 3)}
            for w, c in PREDICTOR.predict(sequence, topk=5)]


# ── Gemini (sentence + keywords) ─────────────────────────────────────────────
GEMINI_MODEL = nlp.create_generative_model(GEMINI_KEY)
if not GEMINI_KEY:
    print("WARNING: GEMINI_API_KEY not set – sentence/keyword generation uses fallbacks")
nlp.configure(LABELS, GEMINI_MODEL)


def translate_text(text: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text) or text
    except Exception as e:
        print(f"Translation error ({target_lang}): {e}")
        return text


def text_to_audio_b64(text: str, lang: str = "en") -> str:
    buf = io.BytesIO()
    try:
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"TTS error: {e}")
        return ""


# ── Flask + SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# async_mode='eventlet' — uses the monkey-patched eventlet green threads
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="gevent",
    max_http_buffer_size=2 * 1024 * 1024,  # 2 MB per frame message
    ping_timeout=60,
    ping_interval=25,
)


# ── WebSocket frame handler ───────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print(f"[WS] Client connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")

@socketio.on("frame")
def on_frame(data):
    """
    Receives { frame_b64: string } from the frontend.
    Runs MediaPipe → throttled Transformer inference.
    Emits 'prediction' back to the same client.
    """
    global _last_pred_time, _last_preds

    frame_b64 = data.get("frame_b64", "") if isinstance(data, dict) else ""
    if not frame_b64:
        emit("prediction", {"error": "no frame"})
        return

    # ── Decode ───────────────────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(frame_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("decode failed")
    except Exception as e:
        emit("prediction", {"error": str(e)})
        return

    try:
        # ── Resize to fixed width for faster MP ──────────────────────────────
        h_orig, w_orig = frame.shape[:2]
        if w_orig != MP_FRAME_W:
            scale  = MP_FRAME_W / w_orig
            frame  = cv2.resize(frame, (MP_FRAME_W, int(h_orig * scale)),
                                interpolation=cv2.INTER_AREA)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with _holistic_lock:
            results = HOLISTIC.detect(mp_img)

        kp = extract_keypoints(results)
        FRAME_BUF.append(kp)

        # ── Landmark data for frontend overlay ───────────────────────────────
        raw_landmarks = {"pose": [], "lh": [], "rh": []}

        def get_first(lms):
            if not lms: return []
            return lms[0] if isinstance(lms[0], list) else lms

        if results.pose_landmarks:
            raw_landmarks["pose"] = [{"x": lm.x, "y": lm.y}
                                      for lm in get_first(results.pose_landmarks)]
        if results.left_hand_landmarks:
            raw_landmarks["lh"]   = [{"x": lm.x, "y": lm.y}
                                      for lm in get_first(results.left_hand_landmarks)]
        if results.right_hand_landmarks:
            raw_landmarks["rh"]   = [{"x": lm.x, "y": lm.y}
                                      for lm in get_first(results.right_hand_landmarks)]

        # ── Throttled inference ───────────────────────────────────────────────
        predictions = []
        now = time.time()
        if len(FRAME_BUF) == SEQ_LEN:
            if now - _last_pred_time >= PRED_INTERVAL:
                _last_preds     = predict(list(FRAME_BUF))
                _last_pred_time = now
            if _last_preds and _last_preds[0]["conf"] >= CONFIDENCE:
                predictions = _last_preds

        emit("prediction", {
            "predictions": predictions,
            "landmarks":   raw_landmarks,
            "buf_fill":    len(FRAME_BUF),
            "buf_max":     SEQ_LEN,
            "model":       MODEL_VERSION,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        emit("prediction", {"error": str(e)})


# ── REST routes (unchanged) ───────────────────────────────────────────────────
@app.route("/")
def health():
    return jsonify({
        "status":       "ok",
        "model":        MODEL_VERSION,
        "model_label":  MODEL_LABEL,
        "model_classes": len(LABELS),
        "device":       DEVICE,
        "amp":          _USE_AMP,
    })


@app.route("/complete-sentence", methods=["POST"])
def complete_sentence():
    data  = request.get_json(force=True)
    words = data.get("words", [])
    if not words:
        return jsonify({"error": "no words"}), 400

    sentence     = nlp.form_sentence_from_keywords_polished(words)
    audio_b64    = text_to_audio_b64(sentence, "en")
    translations = {
        lang_name: translate_text(sentence, lang_code)
        for lang_name, lang_code in LANG_CODES.items()
    }
    return jsonify({
        "sentence":     sentence,
        "translations": translations,
        "audio_b64":    audio_b64,
    })


@app.route("/text-to-sign", methods=["POST"])
def text_to_sign():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")
    if not text:
        return jsonify({"error": "no text"}), 400

    english_text = translate_text(text, "en") if lang != "en" else text
    keywords     = nlp.animation_keywords(english_text)
    return jsonify({"english_text": english_text, "keywords": keywords})


def _safe_unlink(path):
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _audio_bytes_to_wav_path(audio_bytes, audio_format):
    fmt = (audio_format or "webm").lower().strip().lstrip(".")
    tmp_in = None
    tmp_wav = None
    try:
        if fmt == "wav":
            tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_in.write(audio_bytes)
            tmp_in.close()
            return tmp_in.name, None

        tmp_in = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        tmp_in.write(audio_bytes)
        tmp_in.close()
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", tmp_in.name, "-y",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             tmp_wav],
            check=True, capture_output=True, timeout=120,
        )
        return tmp_wav, tmp_in.name
    except FileNotFoundError:
        _safe_unlink(tmp_wav)
        _safe_unlink(tmp_in.name if tmp_in else None)
        return None, None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _safe_unlink(tmp_wav)
        _safe_unlink(tmp_in.name if tmp_in else None)
        print(f"ffmpeg speech conversion error: {e}")
        return None, None


@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    data         = request.get_json(force=True)
    audio_b64    = data.get("audio_b64", "")
    lang         = data.get("lang", "en-US")
    audio_format = data.get("audio_format", "webm")

    if not audio_b64:
        return jsonify({"error": "no audio"}), 400

    wav_path = None
    extra_in = None
    try:
        audio_bytes = base64.b64decode(audio_b64)
        if not audio_bytes:
            return jsonify({"error": "empty audio"}), 400

        wav_path, extra_in = _audio_bytes_to_wav_path(audio_bytes, audio_format)
        if not wav_path:
            return jsonify({
                "error": "Could not prepare audio. Install ffmpeg and ensure it is on PATH.",
            }), 500

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language=lang)
        return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"text": "", "error": "Could not understand audio"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        _safe_unlink(wav_path)
        _safe_unlink(extra_in)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nSignBridge backend starting on http://localhost:5000")
    print(f"  Model: {MODEL_VERSION} | Device: {DEVICE} | AMP: {_USE_AMP} | pred_interval: {PRED_INTERVAL}s\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
