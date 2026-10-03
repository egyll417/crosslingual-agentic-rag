"""MKQA loader: a seeded sample of parallel en/de/fr questions. See docs/design.md."""

import random

from crag.data.io import write_snapshot
from crag.data.schema import Question

MKQA_REPO = "apple/mkqa"
# The default branch only holds a loading script, which datasets no longer runs.
# This is a commit on the Hub's auto-converted refs/convert/parquet branch.
MKQA_REVISION = "d7a2b9681ece319c53f8c2fe850eb4b487cec912"

LANGS = ("en", "de", "fr")
SAMPLE_SIZE = 500
SEED = 0

# Order of the dataset's ClassLabel; answers store the type as an index into this.
ANSWER_TYPES = (
    "entity",
    "long_answer",
    "unanswerable",
    "date",
    "number",
    "number_with_unit",
    "short_phrase",
    "binary",
)
ELIGIBLE_ANSWER_TYPES = frozenset({"entity", "short_phrase", "date", "number", "number_with_unit"})


def answer_type(answer: dict) -> str:
    value = answer["type"]
    return value if isinstance(value, str) else ANSWER_TYPES[value]


def answer_strings(answers: list[dict]) -> tuple[str, ...]:
    """Every answer text and alias, deduplicated, in dataset order."""
    seen = {}
    for answer in answers:
        for candidate in [answer.get("text"), *(answer.get("aliases") or [])]:
            candidate = (candidate or "").strip()
            if candidate:
                seen.setdefault(candidate, None)
    return tuple(seen)


def is_eligible(row: dict) -> bool:
    """True if every English answer has a string-matchable type and there is one to match."""
    english = row["answers"]["en"]
    if not all(answer_type(a) in ELIGIBLE_ANSWER_TYPES for a in english):
        return False
    return bool(answer_strings(english))


def row_to_questions(row: dict) -> list[Question]:
    """One record per language, sharing a parallel id and the English gold answers."""
    parallel_id = f"mkqa-{row['example_id']}"
    english = row["answers"]["en"]
    return [
        Question(
            id=f"{parallel_id}-{lang}",
            parallel_id=parallel_id,
            lang=lang,
            text=row["queries"][lang],
            answers=answer_strings(row["answers"][lang]),
            answers_en=answer_strings(english),
            answer_type=answer_type(english[0]),
            source="mkqa",
        )
        for lang in LANGS
    ]


def select_parallel_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Choose the n parallel questions that make up the MKQA evaluation set.

    rows are the eligible MKQA rows (see is_eligible). The result must be the same
    for a given seed regardless of the order rows arrive in.
    """
    if n > len(rows):
        raise ValueError(f"asked for {n} questions, only {len(rows)} eligible")
    ordered = sorted(rows, key=lambda row: row["example_id"])
    rng = random.Random(seed)
    return rng.sample(ordered, n)


def load_mkqa_rows(revision: str = MKQA_REVISION):
    from datasets import load_dataset

    return load_dataset(MKQA_REPO, revision=revision, split="train")


def build_mkqa_questions(rows, n: int = SAMPLE_SIZE, seed: int = SEED) -> list[Question]:
    eligible = [row for row in rows if is_eligible(row)]
    return [q for row in select_parallel_sample(eligible, n, seed) for q in row_to_questions(row)]


def main() -> None:
    questions = build_mkqa_questions(load_mkqa_rows())
    meta = {
        "source": MKQA_REPO,
        "revision": MKQA_REVISION,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "eligible_answer_types": sorted(ELIGIBLE_ANSWER_TYPES),
    }
    path = write_snapshot(questions, "data/processed", "mkqa", meta)
    print(f"wrote {len(questions)} records to {path}")


if __name__ == "__main__":
    main()
