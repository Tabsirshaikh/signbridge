"""
SignBridge NLP — keyword glosses → fluent English, and English → vocab keywords.
Improved prompts, sanitization, and fallbacks for grammatical output.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

import google.generativeai as genai

# Prefer newer flash models; override with GEMINI_MODEL in environment if needed.
_DEFAULT_MODELS = (
    os.environ.get("GEMINI_MODEL", "").strip(),
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
)

_LABELS: List[str] = []
_LABELS_LOWER: set = set()
_MODEL: Optional[object] = None
_MODEL_NAME: str = ""


def configure(labels: List[str], model: Optional[object]) -> None:
    global _LABELS, _LABELS_LOWER, _MODEL
    _LABELS = list(labels)
    _LABELS_LOWER = {str(x).lower() for x in labels}
    _MODEL = model


def _pick_model_name() -> str:
    for name in _DEFAULT_MODELS:
        if name:
            return name
    return "gemini-1.5-flash"


def create_generative_model(api_key: str):
    """Try a small list of model IDs so installs keep working across API versions."""
    global _MODEL_NAME
    if not api_key or not str(api_key).strip():
        return None
    genai.configure(api_key=api_key)
    names = [n for n in _DEFAULT_MODELS if n]
    if not names:
        names = ["gemini-1.5-flash"]
    last_err = None
    for name in names:
        try:
            m = genai.GenerativeModel(name)
            _MODEL_NAME = name
            print(f"Gemini text model: {name}")
            return m
        except Exception as e:
            last_err = e
    if last_err:
        print(f"Gemini model init failed: {last_err}")
    _MODEL_NAME = ""
    return None


# ── Response helpers ──────────────────────────────────────────────────────────

def _response_text(resp) -> str:
    if not resp:
        return ""
    try:
        t = (resp.text or "").strip()
        if t:
            return t
    except (ValueError, AttributeError):
        pass
    try:
        for c in getattr(resp, "candidates", None) or []:
            parts = getattr(getattr(c, "content", None), "parts", None) or []
            for p in parts:
                if hasattr(p, "text") and p.text:
                    return p.text.strip()
    except Exception:
        pass
    return ""


def sanitize_sentence(text: str) -> str:
    """Strip markdown, labels, and multi-line junk from model output."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```(?:json|text|markdown)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip("`\"'“”‘’*_ ")
    t = re.sub(
        r"^(?:sentence|output|result|english|translation)\s*[:.\-]\s*",
        "",
        t,
        flags=re.I,
    )
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    t = lines[0]
    for noise in (r"\s*\([^)]*\)\s*$", r"\s*\[note:.*$"):
        t = re.split(noise, t, maxsplit=1, flags=re.I)[0].strip()
    t = t.strip("`\"' ")
    return t.strip()


def _sentence_looks_valid(s: str) -> bool:
    if not s or len(s) < 2:
        return False
    if len(s) > 600:
        return False
    lower = s.lower()
    if lower.startswith(("here is", "the sentence", "sure,", "okay,", "note:", "keywords")):
        return False
    if s.count("\n") > 0:
        return False
    return bool(re.search(r"[a-zA-Z]", s))


# ── Fallback when API is off or output is bad ────────────────────────────────

def fallback_sentence_from_keywords(words: List[str]) -> str:
    """Readable list-style English when the model is unavailable."""
    w = []
    for x in words:
        x = (x or "").strip().lower()
        if not x:
            continue
        if w and w[-1] == x:
            continue
        w.append(x)
    if not w:
        return ""
    if len(w) == 1:
        body = w[0].capitalize()
    elif len(w) == 2:
        body = f"{w[0].capitalize()} and {w[1]}"
    else:
        body = ", ".join(w[:-1]) + ", and " + w[-1]
        body = body[0].upper() + body[1:] if body else ""
    if body and body[-1] not in ".?!":
        body += "."
    return body


# ── Gemini: keywords → English sentence ───────────────────────────────────────

_SENTENCE_SYSTEM = """You are an expert editor for American Sign Language glosses.

The user will give an ordered list of English words. Each word is a separate sign gloss from a live recognition system (not a normal English sentence). ASL often omits articles, helping verbs, and inflections that English requires.

Your job: write exactly ONE natural, grammatical English sentence that a hearing person would say in conversation, preserving the signer’s meaning.

Rules:
1. Use standard English word order (subject–verb–object) when you can infer it from the glosses.
2. Add articles (a, an, the), prepositions, copulas (am, is, are, was), auxiliaries (can, will, have), and pronouns (I, you, we, they) wherever English grammar needs them.
3. Choose one tense (usually present) and keep it consistent unless past/future is clearly implied.
4. You may reorder ideas slightly if it fixes grammar, but do not invent new facts or topics not supported by the glosses.
5. Do not repeat the same idea twice unless the gloss list itself repeats.
6. Output only the sentence — no title, no quotes, no bullet, no explanation.
7. End with . or ? or !"""


def form_sentence_from_keywords(words: List[str]) -> str:
    clean_words = [str(w).strip() for w in words if str(w).strip()]
    if not clean_words:
        return ""
    if not _MODEL:
        return fallback_sentence_from_keywords(clean_words)

    gloss_line = ", ".join(clean_words)
    user_part = f"Glosses in order ({len(clean_words)} signs):\n{gloss_line}"

    try:
        generation_config = genai.GenerationConfig(
            temperature=0.35,
            max_output_tokens=220,
            top_p=0.9,
        )
        full_prompt = _SENTENCE_SYSTEM + "\n\n" + user_part
        resp = _MODEL.generate_content(full_prompt, generation_config=generation_config)
        raw = _response_text(resp)
        out = sanitize_sentence(raw)
        if not _sentence_looks_valid(out):
            return fallback_sentence_from_keywords(clean_words)
        if out and out[-1] not in ".?!":
            out = out.rstrip(",;:") + "."
        return out
    except Exception as e:
        print(f"Sentence formation error: {e}")
        return fallback_sentence_from_keywords(clean_words)


# ── Optional polish pass (same model, very low temperature) ───────────────────

_POLISH_INSTRUCTION = """You are a copy editor. Improve ONLY grammar, spelling, and punctuation of the following English sentence. Do not add new facts or change meaning. Output only the corrected single sentence."""


def _maybe_polish(sentence: str) -> str:
    if not _MODEL or not sentence or len(sentence) < 8:
        return sentence
    try:
        generation_config = genai.GenerationConfig(
            temperature=0.15,
            max_output_tokens=180,
        )
        resp = _MODEL.generate_content(
            _POLISH_INSTRUCTION + "\n\n" + sentence,
            generation_config=generation_config,
        )
        polished = sanitize_sentence(_response_text(resp))
        if _sentence_looks_valid(polished) and len(polished) <= max(400, len(sentence) * 3):
            if polished and polished[-1] not in ".?!":
                polished = polished.rstrip(",;:") + "."
            return polished
    except Exception as e:
        print(f"Polish pass skipped: {e}")
    return sentence


def form_sentence_from_keywords_polished(words: List[str]) -> str:
    """Primary entry: sentence + light second pass for cleaner grammar."""
    first = form_sentence_from_keywords(words)
    if not first or not _MODEL:
        return first
    return _maybe_polish(first)


# ── English → animation keywords (vocab-constrained) ──────────────────────────

_KEYWORD_PROMPT = """You map plain English to a sequence of concept words for sign-language animation.

Vocabulary — use ONLY words from this list (comma-separated in your output, each token lowercase):
{voc}

Rules:
1. Output ONLY a comma-separated list of words from the vocabulary above. No other words.
2. Order tokens roughly by how they would be signed (topic before comment when natural).
3. If the sentence has no exact match, choose the closest concepts from the list.
4. Use as few tokens as needed for clarity; omit filler words that have no sign.
5. No explanations, quotes, or line breaks — only the list.

English sentence:
{text}"""


def extract_keywords_raw(english_text: str) -> List[str]:
    text = (english_text or "").strip()
    if not text:
        return []
    if not _MODEL or not _LABELS:
        return re.findall(r"[a-zA-Z]+", text.lower())

    voc = ", ".join(_LABELS)
    prompt = _KEYWORD_PROMPT.format(voc=voc, text=text)
    try:
        generation_config = genai.GenerationConfig(temperature=0.25, max_output_tokens=256)
        resp = _MODEL.generate_content(prompt, generation_config=generation_config)
        raw = _response_text(resp)
        raw = raw.replace("\n", " ").strip()
        return [w.strip().lower() for w in raw.split(",") if w.strip()]
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return re.findall(r"[a-zA-Z]+", text.lower())


def filter_to_vocab(words: List[str]) -> List[str]:
    seen = set()
    out = []
    for w in words:
        lw = w.strip().lower()
        if lw in _LABELS_LOWER and lw not in seen:
            seen.add(lw)
            out.append(lw)
    return out


def animation_keywords(english_text: str) -> List[str]:
    raw = extract_keywords_raw(english_text)
    kw = filter_to_vocab(raw)
    if kw:
        return kw
    return filter_to_vocab(re.findall(r"[a-zA-Z]+", english_text.lower()))
