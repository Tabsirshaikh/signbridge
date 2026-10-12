"""
sentence_builder.py
───────────────────
Collects sign-detected words into a sentence.
No auto-timer — user presses ENTER to complete.
"""

import time


class SentenceBuilder:
    WORD_STABLE_SECS  = 1.0    # seconds a word must be stable to lock in
    CONFIDENCE_THRESH = 0.50   # minimum confidence to consider
    MAX_WORDS         = 25

    def __init__(self):
        self._candidate       = None
        self._candidate_since = None
        self._words           = []
        self._last_word       = None

    def update(self, top_word: str, confidence: float):
        """Feed each prediction frame. Returns word if newly locked."""
        now = time.time()

        if confidence >= self.CONFIDENCE_THRESH and len(self._words) < self.MAX_WORDS:
            if top_word != self._candidate:
                self._candidate       = top_word
                self._candidate_since = now

            elapsed = now - self._candidate_since

            if elapsed >= self.WORD_STABLE_SECS:
                if top_word != self._last_word:
                    self._words.append(top_word)
                    self._last_word = top_word
                    self._candidate       = None
                    self._candidate_since = None
                    return top_word
                else:
                    self._candidate       = None
                    self._candidate_since = None
        else:
            self._candidate       = None
            self._candidate_since = None

        return None

    def complete(self) -> list[str]:
        """Return the current words and reset for next sentence."""
        words = list(self._words)
        self._words     = []
        self._last_word = None
        self._candidate = None
        self._candidate_since = None
        return words

    def undo(self) -> str | None:
        if self._words:
            removed = self._words.pop()
            self._last_word = self._words[-1] if self._words else None
            return removed
        return None

    def clear(self):
        self._words           = []
        self._last_word       = None
        self._candidate       = None
        self._candidate_since = None

    @property
    def words(self):
        return list(self._words)

    @property
    def lock_progress(self):
        if self._candidate and self._candidate_since:
            return min((time.time() - self._candidate_since) / self.WORD_STABLE_SECS, 1.0)
        return 0.0

    @property
    def candidate(self):
        return self._candidate
