# Progress

## Current state
- Milestone: v0.1 (target 2026-10-25)
- Status: scaffold only; tooling being set up
- Next step: migrate requirements.txt to uv (pyproject.toml); data loaders for MKQA (en/de/fr) and AfriQA (sw)

## Milestones
- [ ] v0.1 (Oct 25) — data loaders, passage pool, BM25 + dense recall@k for en/de/fr/sw
- [ ] v0.2 (Nov 22) — hybrid retrieval + reranker, generation with citations, failure-attribution chart
- [ ] v0.3 (Dec 20) — LangGraph agent, agent-vs-pipeline comparison (accuracy, abstention, latency, cost)
- [ ] v1.0 (Jan 24) — LLM-judge validation, FastAPI + docker-compose, CI, architecture.md, eval report

## Log
### 2026-10-03
- Done: added CLAUDE.md and PROGRESS.md; set scope (en/de/fr/sw queries, English knowledge base); vector store set to Qdrant
- Results: none yet
- Blockers: waiting on IMS server specs (GPU/VRAM, scheduler, storage, Java, internet on nodes, Docker/Apptainer)
- Next: uv migration, then data loaders

## Open questions for chat review
- (add anything you want to discuss in the Claude chat here)
