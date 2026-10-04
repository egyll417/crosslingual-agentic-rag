"""Retrieval metrics: answer-string matching and recall@k.

recall@k here is answer recall, which DPR calls top-k retrieval accuracy: the fraction of
questions with at least one gold answer string in the top k passages. It is not passage-level
recall, because the evaluation sets have no gold passage labels. Callers match against the
English answers (`Question.answers_en`), since the knowledge base is English.

See docs/design.md.
"""

import unicodedata
from collections.abc import Iterable, Sequence

# Unicode general categories kept by normalize(): letters, numbers and combining marks.
# Everything else (punctuation, symbols, whitespace, control characters) becomes a space.
_KEPT_CATEGORIES = frozenset("LNM")


def normalize(text: str) -> str:
    """Lowercase, turn punctuation into spaces, and collapse whitespace.

    Punctuation becomes a space rather than being deleted, so the words on either side stay
    separate: "Smith's" -> "smith s", "2016–2017" -> "2016 2017". This follows the DPR
    tokenizer used by Pyserini's DPR retrieval evaluation. NFKC runs first, so composed and
    decomposed accents compare equal and fullwidth digits become ASCII. Diacritics are kept.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    kept = "".join(ch if unicodedata.category(ch)[0] in _KEPT_CATEGORIES else " " for ch in text)
    return " ".join(kept.split())


def has_answer(passage: str, answers: Iterable[str]) -> bool:
    """True if any answer appears in the passage as whole words.

    Both the passage and each answer go through normalize(). An answer matches if its
    normalized words appear as a contiguous run of whole words in the normalized passage:
    "2" does not match "2016", and "New York" does not match "New Yorker". An answer that
    normalizes to the empty string never matches.
    """
    # TODO(human): implement the whole-word match described in the docstring.
    raise NotImplementedError("has_answer not written yet")


def first_hit_rank(passages: Sequence[str], answers: Sequence[str]) -> int | None:
    """1-based rank of the first passage that contains an answer, or None if none does."""
    if isinstance(answers, str):
        raise TypeError("answers must be a collection of strings, not a single str")
    answers = tuple(answers)
    for rank, passage in enumerate(passages, start=1):
        if has_answer(passage, answers):
            return rank
    return None


def recall_at_k(first_hit_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of questions whose first answer-bearing passage is ranked k or better.

    Takes one first_hit_rank() result per question, so every k comes from a single matching
    pass and the per-question ranks can be written to the result file.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not first_hit_ranks:
        raise ValueError("no questions to score")
    hits = sum(1 for rank in first_hit_ranks if rank is not None and rank <= k)
    return hits / len(first_hit_ranks)
