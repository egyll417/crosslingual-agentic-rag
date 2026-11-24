# crosslingual-agentic-rag — working instructions for Claude Code

> **Scope reset (22 Nov 2026).** This is now a single monolingual Swahili module.
> **ORP_SCOPE.md** at the repo root is the source of truth and supersedes any cross-lingual framing
> left in older commits. The repo keeps its original name for git-history continuity.

## Project
Monolingual Swahili retrieval-augmented QA, evaluated end to end. Swahili questions — as text, or as
speech through ASR — answered over a **Swahili Wikipedia** passage pool. The deliverable is the
**evaluation**, not a product: where the pipeline loses answers (ASR → retrieval → generation), how
much the Kenya/Tanzania dialect gap costs, and whether an LLM judge can be trusted in a low-resource
language.

One pipeline, no agent:
spoken question → ASR → dense retrieval (bge-m3, top 5) → generation (Gemini) → evaluation.
Text questions enter at retrieval.
**Conditions (5):** Swahili text; Swahili speech for {Kenya, Tanzania} × {Whisper, Google}.

The centerpiece is the evaluation harness:
- WER / CER vs the read reference, by variety and ASR system
- retrieval answer recall@k at nested pool sizes
- generation scored by an LLM judge (a different Gemini model), validated against 100+ human labels
  (Cohen's kappa)
- per-question failure attribution: lost at ASR, retrieval or generation, or correct
- paired by question, 95% bootstrap CIs

Optional Research Module (ORP) project at IMS, University of Stuttgart.

## Data — one dataset: SD-QA (Swahili)
SD-QA packages TyDi-QA questions + gold answers + gold passages (Swahili Wikipedia) with Kenyan and
Tanzanian read speech (16 kHz) and Google ASR transcripts — one download.
- **Eval set:** SD-QA Swahili test questions that have a TyDi gold passage AND recordings in both
  varieties. Confirm the real counts from the loader; do not reuse the old AfriQA-matched numbers.
- **Retrieval corpus:** Swahili Wikipedia passage pool (100-word passages) — the TyDi gold articles
  plus sampled distractors.

## Stack (keep identical across code, configs and README)
- Python, uv (pyproject.toml), ruff, pytest, GitHub Actions
- ASR: faster-whisper large-v3 (lang=sw), CPU via CTranslate2 (no PyTorch needed locally); plus the
  Google ASR transcripts shipped with SD-QA. Two-system WER/CER, Whisper vs Google.
- Embeddings: BAAI/bge-m3, **dense only**. Retrieval: top 5, **no reranker**.
- Vector store: Qdrant, **local mode** for experiments (no server, no docker-compose).
- Generation: Gemini via google-genai. LLM judge: a *different* Gemini model. Total API budget cap: EUR 50.
- Tracking: MLflow.
- Data: SD-QA Swahili (TyDi questions + gold answers/passages, Kenyan + Tanzanian read speech).

**Cut from the old cross-lingual design** (removed from the repo in the 2026-11-22 reset; recoverable from git history): MKQA, AfriQA,
English Wikipedia, cross-lingual / translation routing, the LangGraph agent, the reranker
(bge-reranker-v2-m3), BM25 / Pyserini, the Qdrant server, and the FastAPI + docker-compose deployment.
Optional only if time allows (nice for the talk): a BM25 baseline and/or the reranker for a
dense-vs-sparse-vs-rerank slide.

## Engineering rules
- Before writing code, state the plan in 2-4 sentences; wait for confirmation on anything non-trivial.
- One component per step. For each design choice, say why and name the alternative rejected.
- After every change, run `ruff check` and `pytest` and show the result. Never claim something works without running it.
- Metric numbers come only from result files written by the eval scripts. Never invent, round up or hand-edit them.
- Cache every LLM/API call to disk; never re-call an API for a result already in the cache.
- Wrap all API calls in retry with exponential backoff and a timeout.
- Never commit secrets (.env), raw data dumps, audio, indexes or model weights.

## Session protocol
- Start of session: read PROGRESS.md, summarise the current state in 3 lines, propose the next step.
- When I say "wrap up": update PROGRESS.md (done / results / next / blockers / open questions),
  propose a commit message, then commit and push after I confirm.
- "Wrap up and push" means commit and push without asking me to confirm.
