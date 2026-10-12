"""
grammar.py
──────────
Takes raw sign-language keywords and produces a grammatically
correct English sentence.

Strategy:
  1. Round-trip translation (en → hi → en) through Google Translate.
     Google naturally adds articles, prepositions, and verb forms
     when translating, so the round-trip output is often a proper
     sentence even when the input is just keywords.
  2. Basic cleanup (capitalise, add full stop).
  3. Fallback to simple capitalisation if translation fails.
"""

from deep_translator import GoogleTranslator


_cache: dict[str, str] = {}


def fix_grammar(words: list[str]) -> str:
    """
    Convert a list of raw sign-detected keywords into a proper
    English sentence.

    Example:
        ["fireman", "pretty", "bed", "bedroom"]
        → "The beautiful fireman is in the bedroom."
    """
    if not words:
        return ""

    raw = " ".join(words).strip()
    if not raw:
        return ""

    # Single word — just capitalise
    if len(words) == 1:
        return raw.capitalize() + "."

    # Check cache
    if raw in _cache:
        return _cache[raw]

    try:
        # Step 1: English keywords → Hindi (Google adds grammar)
        hindi = GoogleTranslator(source="en", target="hi").translate(raw)

        # Step 2: Hindi → English (comes back as a proper sentence)
        corrected = GoogleTranslator(source="hi", target="en").translate(hindi)

        if corrected and len(corrected) > 1:
            # Clean up
            corrected = corrected.strip()
            corrected = corrected[0].upper() + corrected[1:]
            if not corrected.endswith(('.', '!', '?')):
                corrected += '.'

            _cache[raw] = corrected
            return corrected

    except Exception as e:
        print(f"[grammar] round-trip failed: {e}")

    # Fallback: basic capitalisation
    fallback = raw.capitalize() + "."
    _cache[raw] = fallback
    return fallback


def get_hindi(words: list[str]) -> str:
    """
    Translate raw keywords directly to Hindi as a phrase.
    Google Translate handles keyword-to-sentence reasonably well
    when given the full phrase at once.
    """
    if not words:
        return ""

    raw = " ".join(words).strip()
    if not raw:
        return ""

    try:
        result = GoogleTranslator(source="en", target="hi").translate(raw)
        return result if result else raw
    except Exception as e:
        print(f"[grammar] Hindi translation failed: {e}")
        return raw
