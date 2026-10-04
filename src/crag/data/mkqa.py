"""MKQA loader: the frozen sample of parallel en/de/fr questions. See docs/design.md."""

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from crag.data.io import write_snapshot
from crag.data.schema import Question

MKQA_REPO = "apple/mkqa"
# The default branch only holds a loading script, which datasets no longer runs.
# This is a commit on the Hub's auto-converted refs/convert/parquet branch.
MKQA_REVISION = "d7a2b9681ece319c53f8c2fe850eb4b487cec912"

LANGS = ("en", "de", "fr")
# The ids in this file define the evaluation set. SAMPLE_SIZE and SEED only say how it was drawn.
SAMPLE_IDS_PATH = Path("configs/mkqa_sample_ids.txt")
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

    Only used to (re)generate the ids file; a normal build reads that file instead.

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


def read_sample_ids(path: str | Path = SAMPLE_IDS_PATH) -> list[str]:
    """The example ids that define the evaluation set, in file order."""
    ids = Path(path).read_text(encoding="utf-8").split()
    if not ids:
        raise ValueError(f"{path} lists no ids")
    duplicates = sorted(i for i, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{path} lists {len(duplicates)} ids more than once, e.g. {duplicates[:3]}"
        )
    return ids


def select_rows_by_id(rows, ids: list[str]) -> list[dict]:
    """The rows named by ids, in that order. Every id must exist and still be eligible."""
    wanted = set(ids)
    found = {row["example_id"]: row for row in rows if row["example_id"] in wanted}
    missing = [i for i in ids if i not in found]
    if missing:
        raise ValueError(f"{len(missing)} sample ids are not in the dataset, e.g. {missing[:3]}")
    ineligible = [i for i in ids if not is_eligible(found[i])]
    if ineligible:
        raise ValueError(
            f"{len(ineligible)} sample ids are no longer eligible, e.g. {ineligible[:3]}"
        )
    return [found[i] for i in ids]


def build_mkqa_questions(rows, ids: list[str]) -> list[Question]:
    return [q for row in select_rows_by_id(rows, ids) for q in row_to_questions(row)]


def write_sample_ids(
    rows, path: str | Path = SAMPLE_IDS_PATH, n: int = SAMPLE_SIZE, seed: int = SEED
) -> list[str]:
    """Redraw the evaluation set and overwrite the ids file. For deliberate resampling only."""
    eligible = [row for row in rows if is_eligible(row)]
    ids = sorted(row["example_id"] for row in select_parallel_sample(eligible, n, seed))
    Path(path).write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resample",
        action="store_true",
        help="redraw the sample and overwrite the ids file instead of building the snapshot",
    )
    args = parser.parse_args()
    rows = load_mkqa_rows()
    if args.resample:
        ids = write_sample_ids(rows)
        print(f"wrote {len(ids)} ids to {SAMPLE_IDS_PATH} (seed {SEED}); check the git diff")
        return

    ids = read_sample_ids()
    questions = build_mkqa_questions(rows, ids)
    meta = {
        "source": MKQA_REPO,
        "revision": MKQA_REVISION,
        "sample_ids_file": str(SAMPLE_IDS_PATH),
        "sample_ids_sha256": hashlib.sha256(SAMPLE_IDS_PATH.read_bytes()).hexdigest(),
        "sample_size": len(ids),
        "seed": SEED,
        "eligible_answer_types": sorted(ELIGIBLE_ANSWER_TYPES),
    }
    path = write_snapshot(questions, "data/processed", "mkqa", meta)
    print(f"wrote {len(questions)} records to {path}")


if __name__ == "__main__":
    main()
