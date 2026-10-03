# Progress

## Current state
- Milestone: v0.1 (target 2026-10-25)
- Status: tooling in place; MKQA loader done with the 500-question sample frozen in git; AfriQA loader
  scaffolded but not buildable yet (2 of 302 rows wait on `recover_answers`)
- Next step: the three numbered items under "Next" in the 2026-10-03 log
- Settled design decisions (data, abstention, judge): docs/design.md

## Milestones
- [ ] v0.1 (Oct 25) — data loaders, passage pool, BM25 + dense recall@k for en/de/fr/sw
- [ ] v0.2 (Nov 22) — hybrid retrieval + reranker, generation with citations, failure-attribution chart
- [ ] v0.3 (Dec 20) — LangGraph agent, agent-vs-pipeline comparison (accuracy, abstention, latency, cost)
- [ ] v1.0 (Jan 24) — LLM-judge validation, FastAPI + docker-compose, CI, architecture.md, eval report

## Log
### 2026-10-03
- Done: added CLAUDE.md and PROGRESS.md; set scope (en/de/fr/sw queries, English knowledge base); vector store set to Qdrant
- Done: configs/config.yaml aligned with the stack (vector_store: qdrant, languages: en/de/fr/sw)
- Done: migrated requirements.txt to uv (pyproject.toml, uv.lock, Python 3.12 pinned in .python-version)
  - core (local, Intel Mac): datasets, google-genai, langgraph, python-dotenv, qdrant-client
  - `gpu` extra, Linux only: torch, sentence-transformers, pyserini (`uv sync --extra gpu`)
  - `dev` group: ruff, pytest; import smoke tests for the crag subpackages
  - dropped lancedb and ragas (ragas 0.4.3 fails to import against langchain-community 0.4.2)
- Decision: no ragas; the LLM judge will be our own implementation
- Decision: requires Python >=3.12, so the lock holds a single pyserini version (2.4.0)
- Results: none yet (no eval scripts exist)
- Blockers: waiting on IMS server specs (GPU/VRAM, scheduler, storage, Java, internet on nodes, Docker/Apptainer);
  servers now also need Python 3.12+, and the `gpu` extra is locked but has never been installed
- Done: added FlagEmbedding to the `gpu` extra. The bge-m3 model card and FlagEmbedding docs give sparse
  (lexical weights) output only via BGEM3FlagModel; sentence-transformers covers dense only.
  To verify on the first GPU install.
- Done: .gitignore covers /data/, /cache/ and /indexes/
- Decision: the Gemini model id is settled in v0.2 when the API is wired up; from then on every result file
  records the exact model id
- Finding: apple/mkqa and masakhane/afriqa are script-based Hub repos, which datasets 5.x refuses to load;
  both load from the `refs/convert/parquet` revision (MKQA: 10,000 rows; AfriQA swa: 415 train / 417 dev / 302 test)
- Done: `Question` record, JSONL snapshot writer with manifest, and MKQA loader (`src/crag/data/`)
  - eligible answer types: entity, short_phrase, date, number, number_with_unit; 6,619 of 10,000 questions eligible
  - uniform seeded sample of 500 (seed 0), three records per question (en/de/fr) with a shared parallel id
  - sample by answer type: entity 326, date 72, number 39, number_with_unit 34, short_phrase 29
- Done: `configs/mkqa_sample_ids.txt` holds the 500 sampled example ids. The loader does not read it yet.
- Done: docs/design.md records the data, abstention and judge decisions
- In progress: AfriQA loader (`src/crag/data/afriqa.py`), Swahili test split, two records per question
  (sw original and en human translation) with position-based ids
  - 300 of 302 rows convert; rows 75 and 290 have answer strings that `ast.literal_eval` rejects
    (unescaped apostrophes), and `recover_answers` is still a `TODO(human)` stub, so the build stops there
- Results: no evaluation results yet. Test suite: 28 passed; `ruff check` clean.
- Blockers: unchanged (IMS server specs; Python 3.12+ on the servers; `gpu` extra never installed)
- Next, in this order:
  1. Implement the approved ids-file proposal for MKQA: `build_mkqa_questions` reads
     `configs/mkqa_sample_ids.txt`, selects exactly those rows from the pinned revision in file order, and
     fails if the file has duplicates, an id is missing from the dataset, or a listed question is no longer
     eligible. `select_parallel_sample` stays only to regenerate the file behind an explicit command. The
     manifest records the file's checksum alongside the seed. Tests: the tracked file has 500 unique ids;
     building from a fixture id list selects the right rows; a missing id raises.
  2. Elliott writes `recover_answers` (the `TODO(human)` in `src/crag/data/afriqa.py`): recover only a
     single-answer `['...']` wrapper, and raise `ValueError` if the inside looks like several answers.
     Claude then adds tests for it.
  3. Add a drop-row hook to the AfriQA loader with this exclusion rule, consistent with MKQA's long_answer
     exclusion: drop the whole question (both sw and en records) if its English gold answer is longer than
     10 words (11 or more, counted by splitting on whitespace). Report how many questions that removes and
     list the excluded ids with reasons in the manifest.
     Then show Elliott the excluded questions (id, question, English gold answer, word count) to check.
     A probe on 2026-10-03 expects 8: rows 20, 47, 123, 190, 199, 258, 285 and 290 of the test split.
  4. Then: build both snapshots, and start on the passage pool.

## Open questions for chat review
- (add anything you want to discuss in the Claude chat here)
- Passage pool source and storage: the AfriQA English Wikipedia pool on the Hub
  (masakhane/afriqa_wiki_en_fr_100, `corpus.jsonl`) is 18.9 GB.
