"""
translator.py
─────────────
Simplified translator — English to Hindi only.
Uses deep_translator's GoogleTranslator.
"""

from deep_translator import GoogleTranslator

_cache: dict[str, str] = {}


def translate_to_hindi(text: str) -> str:
    """
    Translate an English sentence to Hindi.
    Returns the original text if translation fails.
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Check cache
    if text in _cache:
        return _cache[text]

    try:
        result = GoogleTranslator(source="en", target="hi").translate(text)
        if result:
            _cache[text] = result
            return result
    except Exception as e:
        print(f"[translator] Hindi translation failed: {e}")

    return text
