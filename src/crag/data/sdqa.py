"""SD-QA Swahili loader (ORP_SCOPE.md).

SD-QA (ffaisal93/SD-QA, Apache-2.0) augments TyDi-QA with read speech. For Swahili it
ships Kenyan and Tanzanian recordings plus Google ASR transcripts, keyed to the TyDi
questions whose gold answers/passages live in Swahili Wikipedia.

Split naming is inverted vs TyDi: SD-QA `test` is the TyDi *dev* set (there is no spoken
training data). We use `test`.

The per-dialect `*-ASR*.jsonl.gz` files are in TyDi *primary-task* format: the full article
(`document_plaintext`, UTF-8 byte-indexed), `passage_answer_candidates` (byte spans), and
`annotations` (each a `passage_answer.candidate_index` + `minimal_answer` byte span). The
gold *written* question is not in the jsonl (there `question_text` is the ASR transcript);
it is in `swahili.test.csv` (`swahili-gold`).

Eval set = questions that are answerable (have a gold passage) AND have a recording in both
varieties. ASR with faster-whisper is a later step; this loader records the shipped Google
transcripts only. Audio (`wav_swa/`) is not fetched here.

Run: `uv run python -m crag.data.sdqa`  (network; files cached under data/raw/sdqa/<sha>/).
"""

import csv
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from crag.data.io import write_snapshot

SDQA_REPO = "ffaisal93/SD-QA"
# Pinned commit on the default branch, so the eval set is reproducible.
SDQA_SHA = "c49e9ebb5adb6a0afc560d31e2248fbcc7984a9d"
RAW_BASE = f"https://raw.githubusercontent.com/{SDQA_REPO}/{SDQA_SHA}"
SPLIT = "test"
VARIETIES = ("kenya", "tanzania")
# Google ASR was run with both the Kenyan (sw-KE) and Tanzanian (sw-TZ) locales per recording.
LOCALES = ("sw-KE", "sw-TZ")
CSV_RELPATH = f"{SPLIT}/swa/swahili.{SPLIT}.csv"


def condition(variety: str, locale: str) -> str:
    """The id for one (speaker variety, Google ASR locale) pair, e.g. 'kenya-sw-KE'."""
    return f"{variety}-{locale}"


def jsonl_relpath(variety: str, locale: str) -> str:
    return f"{SPLIT}/swa/{variety}/swahili.{SPLIT}.{condition(variety, locale)}.jsonl.gz"


@dataclass(frozen=True)
class Gold:
    """The gold passage/answer recovered from one TyDi record's annotations."""

    title: str
    passage_index: int
    passage: str
    answers: tuple[str, ...]
    yes_no_answer: str


@dataclass(frozen=True)
class SDQAItem:
    """One SD-QA Swahili eval question: gold text + the Google transcript per condition."""

    example_id: int
    gold_question: str  # the written TyDi question (swahili-gold)
    document_title: str
    gold_passage: str  # Swahili Wikipedia passage containing the answer
    gold_passage_index: int  # index into the article's passage_answer_candidates
    gold_answers: tuple[str, ...]  # minimal answers from annotators, deduped (may be empty)
    yes_no_answer: str  # NONE / YES / NO
    google_transcripts: dict[str, str]  # {"kenya-sw-KE": transcript, ...}
    varieties: tuple[str, ...]  # speaker varieties with a recording
    lang: str = "sw"
    source: str = "sdqa"

    def to_dict(self) -> dict:
        return asdict(self)


def _plurality(values: list[int]) -> int:
    """Most frequent value; ties resolved to the smallest."""
    return max(sorted(set(values)), key=values.count)


def _mode(values: list[str]) -> str | None:
    return max(values, key=values.count) if values else None


def decode_span(doc_bytes: bytes, start: int, end: int) -> str:
    """Decode a UTF-8 byte span of the article, as TyDi offsets index it."""
    return doc_bytes[start:end].decode("utf-8", "replace").strip()


def parse_gold(record: dict) -> Gold | None:
    """Recover the gold passage/answer, or None if the record is unanswerable.

    A question is answerable if any annotator selected a passage
    (passage_answer.candidate_index >= 0). The gold passage is the plurality choice; the
    gold answers are the minimal-answer spans from annotators who chose that passage.
    Rejected: requiring all annotators to agree, which would shrink an already small set.
    """
    annotations = record["annotations"]
    chosen = [
        a["passage_answer"]["candidate_index"]
        for a in annotations
        if a["passage_answer"]["candidate_index"] >= 0
    ]
    if not chosen:
        return None
    gold_index = _plurality(chosen)
    doc_bytes = record["document_plaintext"].encode("utf-8")
    candidate = record["passage_answer_candidates"][gold_index]
    passage = decode_span(
        doc_bytes, candidate["plaintext_start_byte"], candidate["plaintext_end_byte"]
    )
    on_gold = [a for a in annotations if a["passage_answer"]["candidate_index"] == gold_index]
    answers = []
    for a in on_gold:
        span = a["minimal_answer"]
        if span["plaintext_start_byte"] >= 0:
            answers.append(
                decode_span(doc_bytes, span["plaintext_start_byte"], span["plaintext_end_byte"])
            )
    answers = tuple(dict.fromkeys(a for a in answers if a))
    yes_no = _mode([a["yes_no_answer"] for a in on_gold]) or "NONE"
    return Gold(record["document_title"], gold_index, passage, answers, yes_no)


def build_eval_set(
    gold_by_id: dict[int, Gold | None],
    transcripts: dict[str, dict[int, str]],
    gold_questions: dict[int, str],
) -> tuple[list[SDQAItem], dict]:
    """Assemble the eval set (answerable AND recorded in both varieties) and the stage counts.

    `transcripts` maps a condition id to {example_id: transcript}; a question has a recording
    in a variety if it appears in any of that variety's conditions.
    """
    all_ids = set().union(*(t.keys() for t in transcripts.values())) if transcripts else set()

    def transcripts_for(example_id: int) -> dict[str, str]:
        out = {}
        for variety in VARIETIES:
            for locale in LOCALES:
                text = transcripts.get(condition(variety, locale), {}).get(example_id)
                if text is not None:
                    out[condition(variety, locale)] = text
        return out

    def varieties_of(by_condition: dict[str, str]) -> set[str]:
        return {cond.split("-", 1)[0] for cond in by_condition}

    items: list[SDQAItem] = []
    n_both = n_answerable = n_missing_question = 0
    for example_id in sorted(all_ids):
        item_transcripts = transcripts_for(example_id)
        varieties = varieties_of(item_transcripts)
        both = set(VARIETIES) <= varieties
        gold = gold_by_id.get(example_id)
        n_both += int(both)
        n_answerable += int(gold is not None)
        if not (both and gold is not None):
            continue
        question = gold_questions.get(example_id, "")
        if not question:
            n_missing_question += 1
        items.append(
            SDQAItem(
                example_id=example_id,
                gold_question=question,
                document_title=gold.title,
                gold_passage=gold.passage,
                gold_passage_index=gold.passage_index,
                gold_answers=gold.answers,
                yes_no_answer=gold.yes_no_answer,
                google_transcripts=item_transcripts,
                varieties=tuple(sorted(varieties)),
            )
        )
    counts = {
        "questions_total": len(all_ids),
        "recorded_both_varieties": n_both,
        "answerable_has_gold_passage": n_answerable,
        "eval_set": len(items),
        "eval_set_with_minimal_answer": sum(1 for i in items if i.gold_answers),
        "eval_set_missing_gold_question": n_missing_question,
    }
    return items, counts


# ---- network + file reading (not exercised in CI; covered by the live `main`) ----------------


def _get(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    """GET with exponential backoff, per the API/network retry rule."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch(relpath: str, cache_dir: Path) -> Path:
    """Fetch one repo file to the cache (skipping the download if already present)."""
    dest = cache_dir / relpath
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(f"{RAW_BASE}/{relpath}"))
    return dest


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_gold_questions(csv_path: Path) -> dict[int, str]:
    """{example_id: gold written question} from the language CSV's `swahili-gold` column."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["ID"]): row["swahili-gold"].strip() for row in reader if row.get("ID")}


def load() -> tuple[dict[int, Gold | None], dict[str, dict[int, str]], dict[int, str], dict]:
    """Fetch and parse everything needed to build the eval set."""
    cache_dir = Path(os.environ.get("SDQA_CACHE", "data/raw/sdqa")) / SDQA_SHA
    gold_questions = read_gold_questions(fetch(CSV_RELPATH, cache_dir))
    gold_by_id: dict[int, Gold | None] = {}
    transcripts: dict[str, dict[int, str]] = {}
    recordings: dict[str, int] = {}
    for variety in VARIETIES:
        for locale in LOCALES:
            cond = condition(variety, locale)
            path = fetch(jsonl_relpath(variety, locale), cache_dir)
            per_condition: dict[int, str] = {}
            for record in read_jsonl_gz(path):
                example_id = record["example_id"]
                per_condition[example_id] = record["question_text"]
                # Gold fields are identical across conditions; parse once per id.
                if example_id not in gold_by_id:
                    gold_by_id[example_id] = parse_gold(record)
            transcripts[cond] = per_condition
            recordings[cond] = len(per_condition)
    return gold_by_id, transcripts, gold_questions, recordings


def main() -> None:
    gold_by_id, transcripts, gold_questions, recordings = load()
    items, counts = build_eval_set(gold_by_id, transcripts, gold_questions)
    meta = {
        "source": SDQA_REPO,
        "revision": SDQA_SHA,
        "language": "swahili",
        "split": SPLIT,
        "recordings_per_condition": recordings,
        "conditions": [condition(v, loc) for v in VARIETIES for loc in LOCALES],
        **counts,
    }
    path = write_snapshot(items, "data/processed", "sdqa_sw_test", meta)
    print(f"wrote {len(items)} eval questions to {path}\n")
    for key, value in counts.items():
        print(f"  {key:34s} {value}")
    print("\n  recordings per condition:")
    for cond, n in recordings.items():
        print(f"    {cond:16s} {n}")
    by_variety = Counter(v for item in items for v in item.varieties)
    print(f"\n  eval questions by variety present: {dict(by_variety)}")


if __name__ == "__main__":
    main()
