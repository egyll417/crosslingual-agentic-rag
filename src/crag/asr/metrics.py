"""Word- and character-error rates for ASR output (ORP_SCOPE.md).

Dep-free on purpose: WER/CER is a Levenshtein distance over tokens / characters, and keeping
the normalisation here (rather than in jiwer) lets us control it for Swahili and report exactly
what was compared. Normalisation: Unicode NFKC, lowercase, punctuation -> space, collapse
whitespace. WER tokenises on whitespace; CER counts characters of the normalised string.

Aggregates are micro-averaged (corpus rate = total edits / total reference length), the usual
WER convention, so a longer reference weighs more than a short one.
"""

import re
import unicodedata
from dataclasses import dataclass

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class ErrorCount:
    """Edit distance and reference length; add these up to micro-average a corpus."""

    edits: int
    length: int

    def __add__(self, other: "ErrorCount") -> "ErrorCount":
        return ErrorCount(self.edits + other.edits, self.length + other.length)

    @property
    def rate(self) -> float:
        return self.edits / self.length if self.length else 0.0


def _levenshtein(ref: list | str, hyp: list | str) -> int:
    """Edit distance between two sequences (token lists or strings)."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_errors(reference: str, hypothesis: str) -> ErrorCount:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    return ErrorCount(_levenshtein(ref, hyp), len(ref))


def char_errors(reference: str, hypothesis: str) -> ErrorCount:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    return ErrorCount(_levenshtein(ref, hyp), len(ref))


def corpus_rate(counts: list[ErrorCount]) -> float:
    """Micro-averaged error rate over many items."""
    total = sum(counts, ErrorCount(0, 0))
    return total.rate
