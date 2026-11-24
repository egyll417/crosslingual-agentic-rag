# Progress

## Current state
- **Scope reset 2026-11-22 (see ORP_SCOPE.md):** collapsed to one monolingual Swahili module. The old
  cross-lingual design, MKQA, AfriQA, English Wikipedia, the LangGraph agent, the reranker and
  BM25/Pyserini are dropped. The MKQA/AfriQA loaders, their tests, the `agent` package and
  `docs/design.md` have been **removed** (recoverable from git history), and `pyproject.toml`/`uv.lock`
  pruned (dropped langgraph + pyserini, sentence-transformers now only transitive via flagembedding;
  added faster-whisper for CPU ASR).
- Status: no SD-QA code yet. Phase 1 is the SD-QA Swahili loader, then ASR (WER/CER); GPU work
  (bge-m3 encode, retrieval) waits on IMS.
- Next step: SD-QA Swahili data loader — both varieties, map each clip to its TyDi question + gold
  answer/passage, define the eval set (gold passage AND recordings in both varieties), print real counts.

## Milestones (reset 2026-11-22)
- [ ] P1 (now → mid-Dec) — SD-QA Swahili loader + ASR (faster-whisper vs Google); WER/CER by variety × system. **CPU only.**
- [ ] P2 (mid-Dec) — Swahili Wikipedia passage pool; bge-m3 dense encode; retrieval recall@k for the 5 conditions. **Needs IMS GPU.**
- [ ] P3 (Jan) — Gemini generation; LLM judge; per-question failure attribution.
- [ ] P4 (Feb–Apr) — ~200 hand labels; judge agreement (Cohen's kappa); final analysis; report + talk.

## IMS server checklist
To confirm with IMS before the first GPU install (still waiting on server specs):
- [ ] GPU model and VRAM
- [ ] NVIDIA driver ≥ 580: the `gpu` extra locks torch 2.14.1+cu130 (a CUDA 13.0 build); on an older
      driver torch still imports but `torch.cuda.is_available()` is False
- [ ] Python 3.12+
      <!-- Java 21 / pyserini line dropped on the 2026-11-22 scope reset: BM25/Pyserini is cut. -->
- [ ] Job scheduler
- [ ] Storage: home-directory quota vs the ~6.3 GB `.venv` from `uv sync --extra gpu` (about 3 GB of
      CUDA wheels), plus model weights and indexes
- [ ] Internet on compute nodes (PyPI, Hugging Face Hub)
- [ ] Docker or Apptainer
- [ ] Install with `UV_HTTP_TIMEOUT=300 uv sync --extra gpu`: uv's default 30 s timeout failed on a large
      wheel download over a slow link

## Log
### 2026-11-22 — scope reset to monolingual Swahili (SD-QA)
- Why: the old design asked AfriQA Swahili questions over English Wikipedia with SD-QA audio, but SD-QA
  records **TyDi-QA** questions, not AfriQA's — the question sets are disjoint, so AfriQA questions had no
  recordings and the speech path was not runnable. Rebuilding entirely on SD-QA aligns text and speech on
  the same questions; TyDi Swahili answers are grounded in Swahili Wikipedia, so the KB is Swahili.
- Done: wrote ORP_SCOPE.md (source of truth); rewrote CLAUDE.md, README.md and `configs/config.yaml`,
  and reset the Current state / Milestones above, to the single-module scope.
- Out of scope now (recoverable from git history): MKQA, AfriQA, English Wikipedia, cross-lingual/
  translation routing, LangGraph agent, reranker, BM25/Pyserini, Qdrant server, FastAPI + docker.
- Done in the reset: deleted `src/crag/data/{mkqa,afriqa}.py` + their tests, `configs/mkqa_sample_ids.txt`,
  the empty `src/crag/agent/` package (dropped from the smoke test), and `docs/design.md`; pruned
  `pyproject.toml`/`uv.lock` (dropped langgraph + pyserini + the direct sentence-transformers dep; added
  faster-whisper). `src/crag/data/{io,schema}.py` kept — reused by the SD-QA loader.
- Verified: `ruff check` clean, `pytest` 5 passed (subpackage import smoke test).
- Next: SD-QA Swahili loader (Phase 1, step 1), then CPU ASR (step 2). Note: huggingface.co was blocked
  in this cloud container on 2026-10-04 — confirm SD-QA download access before building the loader.

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

### 2026-10-04
- Done: first install of the `gpu` extra, in a CPU-only Claude Code cloud container (no GPU). It installs
  from the existing lock (`uv.lock` unchanged) and all four packages import: torch 2.14.1+cu130,
  sentence-transformers 6.1.0, FlagEmbedding 1.4.2 (`BGEM3FlagModel`, `FlagReranker`), pyserini 2.4.0
  (`LuceneSearcher`, `LuceneIndexer`; the JVM starts)
  - the first attempt failed on uv's default 30 s HTTP timeout; `UV_HTTP_TIMEOUT=300` fixed it
    (~25 min at ~3.6 MB/s); `.venv` is 6.3 GB
- Blocked: the bge-m3 sparse check with real weights. The cloud environment's network policy denies
  huggingface.co.
- Done: checked the sparse code path offline with a tiny random XLM-R checkpoint laid out like bge-m3
  (`colbert_linear.pt`, `sparse_linear.pt`). Under transformers 5.18, `encode(..., return_sparse=True)`
  returns `lexical_weights` (one token id → weight dict per sentence), and `convert_id_to_token` and
  `compute_lexical_matching_score` work. The weights are random, so this says nothing about bge-m3's real
  lexical weights or about loading its checkpoint.
- Finding: FlagEmbedding's own `snapshot_download` does not exclude the repo's `onnx/` export. Pre-download
  bge-m3 with `ignore_patterns=["onnx/*"]` and pass the local path.
- Done: merged PR #1 (the IMS server checklist and the entries above)
- Done: items 1 to 3 of the 2026-10-03 "Next" list, and the snapshot half of item 4
  - `recover_answers` (written by Elliott) recovers only a single-answer `['...']` wrapper and raises
    otherwise; it recovers all three unparseable strings in the test split (row 75 sw and en, row 290 en)
  - the MKQA loader builds from `configs/mkqa_sample_ids.txt`; sampling only runs behind
    `python -m crag.data.mkqa --resample`, which reproduces the tracked file byte for byte with seed 0
  - the AfriQA loader drops a question (both records) when its English gold answer is longer than 10 words:
    8 of 302 removed, rows 20, 47, 123, 190, 199, 258, 285 and 290, as the 2026-10-03 probe expected
- Done: built both snapshots under `data/processed/` (not committed; rebuild with
  `uv run python -m crag.data.mkqa` and `uv run python -m crag.data.afriqa`)
  - `mkqa.jsonl`: 1,500 records, 500 each for en/de/fr; by answer type entity 326, date 72, number 39,
    number_with_unit 34, short_phrase 29; sha256 `a54b37d8e7332bc0905f1674840f45a8f9b714dc15c0f32757143e1be0d7bc5b`
  - `afriqa_sw.jsonl`: 588 records, 294 each for sw and en; sha256
    `0b9b7fa964043fc1e68b299802f3d83157dd603295c63cd7bd5033d1476da20c`
- Results: no evaluation results yet. Test suite: 42 passed; `ruff check` clean.
- Blockers: IMS server specs (see IMS server checklist); no Hugging Face access from the cloud container
- Next:
  1. Passage pool: decide the source and where it is stored, then build it
  2. BM25 and dense recall@k for en/de/fr/sw on the two snapshots
  3. The real bge-m3 sparse check (two sentences, `return_sparse=True`) once Hugging Face is reachable from
     a machine with the `gpu` extra, in the cloud container or on the first IMS node

## Open questions for chat review
- (add anything you want to discuss in the Claude chat here)
- SD-QA access + format: exact source (HF repo / GitHub release), the field names linking a clip to its
  TyDi question + gold answer/passage, and whether the cloud container can download it (huggingface.co was
  blocked here on 2026-10-04). Confirm the real eval-set count (questions with a gold passage AND
  recordings in both varieties) — do not reuse the old AfriQA numbers.
- Swahili Wikipedia passage pool: dump/source and how distractors are sampled (Phase 2).

<!-- Resolved by the 2026-11-22 reset: the old AfriQA/English-Wikipedia passage-pool question (the 18.9 GB
masakhane/afriqa_wiki_en_fr_100 corpus) is moot — the KB is now Swahili Wikipedia. -->

