"""AfriQA Swahili loader: every question in sw and in its human English translation.

See docs/design.md.
"""

import ast

from crag.data.io import write_snapshot
from crag.data.schema import Question

AFRIQA_REPO = "masakhane/afriqa"
# The default branch only holds a loading script, which datasets no longer runs.
# This is a commit on the Hub's auto-converted refs/convert/parquet branch.
AFRIQA_REVISION = "8c9e7495b0e14b41d908e08cb1509a985453523f"
SPLIT = "test"


def recover_answers(raw: str) -> tuple[str, ...]:
    """Handle an answer string that is not a valid Python list literal.

    AfriQA writes answers as "['Webuye']"; an unescaped apostrophe, as in
    "['I'm Sprung']", makes the string unparseable. Return the recovered answers,
    or raise ValueError if the string should fail the build.
    """
    # TODO(human): decide what happens to answer strings that do not parse.
    raise NotImplementedError("AfriQA policy for unparseable answers not written yet")


def parse_answers(raw: str) -> tuple[str, ...]:
    """Answers from AfriQA's stringified list, stripped and deduplicated."""
    try:
        values = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return recover_answers(raw)
    if not isinstance(values, list):
        return recover_answers(raw)
    return tuple(dict.fromkeys(v.strip() for v in values if v.strip()))


def row_to_questions(row: dict, parallel_id: str) -> list[Question]:
    """The Swahili question and its human English translation, sharing a parallel id."""
    answers_sw = parse_answers(row["answers"])
    answers_en = parse_answers(row["translated_answer"])
    if not answers_en:
        raise ValueError(f"{parallel_id} has no English gold answer: {row['translated_answer']!r}")
    return [
        Question(
            id=f"{parallel_id}-sw",
            parallel_id=parallel_id,
            lang="sw",
            text=row["question"],
            answers=answers_sw,
            answers_en=answers_en,
            answer_type=None,
            source="afriqa",
        ),
        Question(
            id=f"{parallel_id}-en",
            parallel_id=parallel_id,
            lang="en",
            text=row["translated_question"],
            answers=answers_en,
            answers_en=answers_en,
            answer_type=None,
            source="afriqa",
        ),
    ]


def load_afriqa_rows(revision: str = AFRIQA_REVISION, split: str = SPLIT):
    from datasets import load_dataset

    files = {split: f"swa/{split}/*.parquet"}
    return load_dataset(AFRIQA_REPO, revision=revision, data_files=files, split=split)


def build_afriqa_questions(rows, split: str = SPLIT) -> list[Question]:
    # The parquet rows carry no id, so the id is the row's position in the pinned revision.
    return [
        question
        for index, row in enumerate(rows)
        for question in row_to_questions(row, f"afriqa-swa-{split}-{index:04d}")
    ]


def main() -> None:
    questions = build_afriqa_questions(load_afriqa_rows())
    meta = {"source": AFRIQA_REPO, "revision": AFRIQA_REVISION, "language": "swa", "split": SPLIT}
    path = write_snapshot(questions, "data/processed", "afriqa_sw", meta)
    print(f"wrote {len(questions)} records to {path}")


if __name__ == "__main__":
    main()
