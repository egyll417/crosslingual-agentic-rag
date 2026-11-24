"""WER/CER of ASR transcripts vs the read reference (ORP_SCOPE.md).

The read reference is the gold written question (`gold_question`) — what the speaker read aloud.
This scores the Google transcripts shipped with SD-QA now; the faster-whisper transcripts get the
same treatment once they exist (run on IMS). Rates are computed over the eval-set snapshot and
micro-averaged by condition. Numbers are written to a result file, per the engineering rules.

Run: `uv run python -m crag.asr.evaluate`  (after building the snapshot with crag.data.sdqa).
"""

import json
from collections.abc import Callable
from pathlib import Path

from crag.asr.metrics import ErrorCount, char_errors, word_errors

SNAPSHOT = Path("data/processed/sdqa_sw_test.jsonl")
RESULT = Path("data/processed/asr_google_wer.json")


def read_items(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def wer_by_condition(
    items: list[dict],
    transcripts_of: Callable[[dict], dict[str, str]],
    reference_of: Callable[[dict], str] = lambda item: item["gold_question"],
) -> dict[str, dict]:
    """Micro-averaged WER/CER per condition, over items with a non-empty reference."""
    words: dict[str, ErrorCount] = {}
    chars: dict[str, ErrorCount] = {}
    counts: dict[str, int] = {}
    for item in items:
        reference = reference_of(item)
        if not reference.strip():
            continue
        for condition, hypothesis in transcripts_of(item).items():
            words[condition] = words.get(condition, ErrorCount(0, 0)) + word_errors(reference, hypothesis)
            chars[condition] = chars.get(condition, ErrorCount(0, 0)) + char_errors(reference, hypothesis)
            counts[condition] = counts.get(condition, 0) + 1
    return {
        condition: {
            "n": counts[condition],
            "wer": round(words[condition].rate, 4),
            "cer": round(chars[condition].rate, 4),
            "word_edits": words[condition].edits,
            "ref_words": words[condition].length,
            "char_edits": chars[condition].edits,
            "ref_chars": chars[condition].length,
        }
        for condition in sorted(words)
    }


def main() -> None:
    items = read_items(SNAPSHOT)
    results = wer_by_condition(items, transcripts_of=lambda item: item["google_transcripts"])
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Google ASR WER/CER over {len(items)} eval questions (reference = gold question):\n")
    print(f"  {'condition':16s} {'n':>5}  {'WER':>7}  {'CER':>7}")
    for condition, r in results.items():
        print(f"  {condition:16s} {r['n']:>5}  {r['wer']:>7.3f}  {r['cer']:>7.3f}")
    print(f"\nwrote {RESULT}")


if __name__ == "__main__":
    main()
