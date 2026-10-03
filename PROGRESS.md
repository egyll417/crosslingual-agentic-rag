# Progress

## Current state
- Milestone: v0.1 (target 2026-10-25)
- Status: scaffold plus tooling; uv project with ruff and pytest in place, no pipeline code yet
- Next step: data loaders for MKQA (en/de/fr) and AfriQA (sw)

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
- Next: data loaders for MKQA (en/de/fr) and AfriQA (sw)

## Open questions for chat review
- (add anything you want to discuss in the Claude chat here)
