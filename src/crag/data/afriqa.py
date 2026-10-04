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
# A longer English gold answer is a sentence, not something to string-match in a passage.
MAX_ANSWER_WORDS = 10


def recover_answers(raw: str) -> tuple[str, ...]:
    """Handle an answer string that is not a valid Python list literal.

    AfriQA writes answers as "['Webuye']"; an unescaped apostrophe, as in
    "['I'm Sprung']", makes the string unparseable. Return the recovered answers,
    or raise ValueError if the string should fail the build.
    """
    text = raw.strip()
    if not (text.startswith("['") and text.endswith("']")):
        raise ValueError(f"unrecognised answer format: {raw!r}")
    inner = text[2:-2]
    if "', '" in inner:
        raise ValueError(f"looks like several answers: {raw!r}")
    return (inner,)


def parse_answers(raw: str) -> tuple[str, ...]:
    """Answers from AfriQA's stringified list, stripped and deduplicated."""
    try:
        values = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        values = None
    if not isinstance(values, list):
        values = recover_answers(raw)
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


def answer_words(row: dict) -> int:
    """Length in words of the shortest English gold answer, 0 if there is none."""
    answers = parse_answers(row["translated_answer"])
    return min((len(answer.split()) for answer in answers), default=0)


def exclusion_reason(row: dict) -> str | None:
    """Why this question is left out of the evaluation set, or None to keep it."""
    words = answer_words(row)
    if words > MAX_ANSWER_WORDS:
        return f"English gold answer has {words} words (limit {MAX_ANSWER_WORDS})"
    return None


def load_afriqa_rows(revision: str = AFRIQA_REVISION, split: str = SPLIT):
    from datasets import load_dataset

    files = {split: f"swa/{split}/*.parquet"}
    return load_dataset(AFRIQA_REPO, revision=revision, data_files=files, split=split)


def build_afriqa_questions(rows, split: str = SPLIT) -> tuple[list[Question], list[dict]]:
    """Records for the kept questions, and one entry per excluded question."""
    questions, excluded = [], []
    for index, row in enumerate(rows):
        # The parquet rows carry no id, so the id is the row's position in the pinned revision.
        # Excluded questions keep their position, so the ids of the others do not shift.
        parallel_id = f"afriqa-swa-{split}-{index:04d}"
        reason = exclusion_reason(row)
        if reason is None:
            questions.extend(row_to_questions(row, parallel_id))
        else:
            excluded.append(
                {
                    "id": parallel_id,
                    "reason": reason,
                    "question": row["question"],
                    "question_en": row["translated_question"],
                    "answers_en": list(parse_answers(row["translated_answer"])),
                    "answer_words": answer_words(row),
                }
            )
    return questions, excluded


def main() -> None:
    rows = load_afriqa_rows()
    questions, excluded = build_afriqa_questions(rows)
    meta = {
        "source": AFRIQA_REPO,
        "revision": AFRIQA_REVISION,
        "language": "swa",
        "split": SPLIT,
        "max_answer_words": MAX_ANSWER_WORDS,
        "questions_total": len(rows),
        "questions_kept": len(rows) - len(excluded),
        "questions_excluded": len(excluded),
        "excluded": excluded,
    }
    path = write_snapshot(questions, "data/processed", "afriqa_sw", meta)
    print(f"wrote {len(questions)} records to {path}")
    print(f"excluded {len(excluded)} of {len(rows)} questions:")
    for entry in excluded:
        print(f"  {entry['id']}  ({entry['answer_words']} words)")
        print(f"    sw: {entry['question']}")
        print(f"    en: {entry['question_en']}")
        print(f"    answer: {' | '.join(entry['answers_en'])}")


if __name__ == "__main__":
    main()
