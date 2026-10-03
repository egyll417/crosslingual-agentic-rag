# crosslingual-agentic-rag — working instructions for Claude Code

## Project
Cross-lingual retrieval-augmented QA. Users ask in English, German, French or Swahili; the knowledge
base is an English Wikipedia passage pool. The centerpiece is the evaluation harness:
- per-question failure attribution (lost at retrieval, reranking, or generation, or correct)
- agent vs. plain pipeline: accuracy, abstention quality, p50/p95 latency, tokens/cost per query
- validation of LLM-judge metrics against human labels in all four languages

Also an Optional Research Module project at IMS, University of Stuttgart.

## Stack (keep identical across code, configs and README)
- Python, uv (pyproject.toml), ruff, pytest, GitHub Actions
- Embeddings: BAAI/bge-m3 (dense + sparse). Reranker: BAAI/bge-reranker-v2-m3
- Vector store: Qdrant (local mode for experiments, server via docker-compose for the API)
- BM25 baseline: Pyserini
- Agent: LangGraph, built only after the plain pipeline and the harness work
- Generation: one open-weight model on IMS GPUs + Gemini via google-genai (total API budget cap: EUR 50)
- Tracking/tracing: MLflow. Serving: FastAPI + docker-compose
- Data: MKQA (en/de/fr, ~500 parallel questions) and AfriQA (Swahili test split)

## Engineering rules
- Before writing code, state the plan in 2-4 sentences; wait for confirmation on anything non-trivial.
- One component per step. For each design choice, say why and name the alternative rejected.
- After every change, run `ruff check` and `pytest` and show the result. Never claim something works without running it.
- Metric numbers come only from result files written by the eval scripts. Never invent, round up or hand-edit them.
- Cache every LLM/API call to disk; never re-call an API for a result already in the cache.
- Wrap all API calls in retry with exponential backoff and a timeout.
- Never commit secrets (.env), raw data dumps, indexes or model weights.

## Session protocol
- Start of session: read PROGRESS.md, summarise the current state in 3 lines, propose the next step.
- When I say "wrap up": update PROGRESS.md (done / results / next / blockers / open questions),
  propose a commit message, then commit and push after I confirm.
