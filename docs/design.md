# Design decisions

Decisions that are settled, with the reason for each. Session state lives in
[PROGRESS.md](../PROGRESS.md); the stack and working rules live in [CLAUDE.md](../CLAUDE.md).

## Evaluation questions

### Record format
Every loader emits `Question` records (`src/crag/data/schema.py`): one record per question per
language.

| Field | Meaning |
|---|---|
| `id` | unique per record, e.g. `mkqa-<example_id>-de` |
| `parallel_id` | shared by all language versions of the same question |
| `lang` | `en`, `de`, `fr` or `sw` |
| `text` | the question in that language |
| `answers` | gold answers and aliases in the question's language |
| `answers_en` | gold answers and aliases in English |
| `answer_type` | kept on every record so results can be sliced by type |
| `source` | `mkqa` or `afriqa` |

Gold matching is against **all English answer aliases** (`answers_en`), because the knowledge base
is English.

### MKQA (en/de/fr)
- A seeded sample of about 500 questions, each emitted in en, de and fr with one `parallel_id`.
  The three languages are therefore compared on identical questions.
- Eligible answer types: `entity`, `short_phrase`, `date`, `number`, `number_with_unit`.
- Excluded answer types:
  - `long_answer`: there is no short gold answer to match.
  - `binary`: yes/no cannot be string-matched in passages.
  - `unanswerable`: the annotators did not find an answer, which is not the same as the answer
    being absent from our passage pool.
- A question is eligible only if every one of its English answers has an eligible type and it has
  at least one English answer string. On the pinned revision this leaves 6,619 of 10,000 questions.
- The 500 are a uniform random draw without replacement from the eligible questions, sorted by
  `example_id` and seeded (`select_parallel_sample` in `src/crag/data/mkqa.py`). The sample
  therefore keeps MKQA's natural mix of answer types rather than balancing them.
- The 500 selected `example_id`s are frozen in `configs/mkqa_sample_ids.txt` (one per line, sorted
  as strings), so the evaluation set is fixed in git. They were drawn with seed 0 from revision
  `d7a2b9681ece319c53f8c2fe850eb4b487cec912` of the parquet branch.
- That file, not the seed, defines the evaluation set. The loader reads the ids, selects exactly
  those rows in file order, and fails if the file has duplicates, an id is missing from the
  dataset, or a listed question is no longer eligible. The manifest records the file's checksum.
  `select_parallel_sample` is kept only to regenerate the file, behind
  `python -m crag.data.mkqa --resample`.
  Rejected: sampling at build time and asserting the result equals the file, which keeps two
  sources of truth and depends on `random.sample` behaving the same on every Python version.

### AfriQA (sw)
- Swahili test split: 302 questions, of which 294 remain after the exclusion rule below.
- Each question is emitted twice with one `parallel_id`: the original Swahili (`lang: sw`) and the
  dataset's human English translation (`lang: en`). This is the Swahili control: the same
  questions asked in sw and in en.
- Machine-translated English is not produced by the loader. It comes later, at retrieval time.
- The parquet rows carry no id, so a question's `parallel_id` is its position in the pinned
  revision, e.g. `afriqa-swa-test-0007`. Rejected: a hash of the Swahili question text, which
  survives re-ordering but is opaque, and the revision pin already fixes the order.
- AfriQA has no answer types, so `answer_type` is empty on these records.
- Answers are stored as a stringified list (`"['Webuye']"`), and an unescaped apostrophe makes the
  string unparseable. `recover_answers` recovers only a single-answer `['...']` wrapper, and
  fails the build if the string is not wrapped or the inside looks like several answers.
- A question is dropped entirely (both its sw and en records) if its English gold answer is
  longer than 10 words (11 or more, counted by splitting on whitespace). This is the same
  reasoning as MKQA's `long_answer` exclusion: a sentence-length gold answer cannot be
  string-matched. If a question has several English answers, it is kept as long as one of them
  is short enough. The build reports how many questions this removes, and the manifest lists
  each excluded question with its id and reason.
- Excluded questions keep their row position, so the ids of the remaining questions do not shift.
- The AfriQA questions are a different set from the MKQA sample, so sw vs. de/fr differences mix
  language with question difficulty. The sw vs. en control above is the clean comparison.

### Where the data comes from
- `apple/mkqa` and `masakhane/afriqa` are script-based Hub repos, which `datasets` no longer loads.
  Both are loaded from the Hub's auto-converted `refs/convert/parquet` branch, pinned to a commit.
- Each loader writes a normalised JSONL snapshot and a manifest (source, revision, seed, counts,
  checksum) under `data/processed/`. `data/` is not committed.

## Retrieval metric
- recall@k is answer recall (DPR's top-k retrieval accuracy): the fraction of questions with at
  least one English gold answer (`answers_en`) in the top k passages. The evaluation sets have no
  gold passage labels, so passage-level recall is not available.
- Matching is on normalized text, as whole words: `normalize` applies NFKC, lowercases, turns
  every character that is not a letter, number or combining mark into a space, and collapses
  whitespace. An answer matches if its words appear as a contiguous run of whole words in the
  passage, so `2` does not match `2016`. An answer that normalizes to nothing never matches.
- Punctuation becomes a space, as in the DPR tokenizer behind Pyserini's DPR retrieval evaluation,
  so `Smith's` still contains `Smith` and `2016–2017` still contains `2017`.
  Rejected: deleting punctuation (SQuAD-style), which is built to compare two answer strings, not
  to search passages; it glues `2016–2017` into `20162017` and only knows ASCII punctuation.
  Known cost, shared with DPR: `1,000` does not match `1000`, and `U.S.` does not match `US`.
  To revisit once the number answers in the MKQA snapshot have been checked.
- Diacritics and articles are kept. Pyserini's implementation is not used because it is in the
  Linux-only `gpu` extra; it can serve as a cross-check on the GPU servers.
- Each question's first-hit rank is computed once and stored per question; recall@k for every k
  comes from those ranks, and they feed the failure attribution.

## Abstention
- Abstention is evaluated on its own controlled set, built at v0.3: answerable questions whose
  gold-containing passages are deleted from the pool. That is the only way to know for certain that
  the answer is absent.
- MKQA's `unanswerable` questions are not used for this (see above).
- Nothing is built for this before v0.3.

## Judge and result files
- No ragas. The LLM judge is our own implementation, validated against human labels.
- The Gemini model id is settled in v0.2, when the API is wired up. From then on, every result file
  records the exact model id that produced it.

## Retrieval models
- bge-m3 dense and sparse output both come from FlagEmbedding (`BGEM3FlagModel`);
  sentence-transformers only exposes the dense output. To be verified on the first GPU install.
