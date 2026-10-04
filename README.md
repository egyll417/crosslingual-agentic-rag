# crosslingual-agentic-rag

[![CI](https://github.com/egyll417/crosslingual-agentic-rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/egyll417/crosslingual-agentic-rag/actions/workflows/ci.yml)

Users ask in English, German, French or Swahili; the knowledge base is in English.
This repo builds a retrieval-augmented QA system with an agent loop, plus an evaluation harness
that measures **where answers get lost** (retrieval, reranking or generation) and **what the agent
buys** in accuracy at what cost in latency and tokens.

**Status:** work in progress. v0.1 (data + retrieval baselines) is targeted for late October 2026.
See [PROGRESS.md](PROGRESS.md) for the current state.

## Planned
- Hybrid retrieval (BGE-M3 dense + sparse) with cross-encoder reranking
- LangGraph agent: query-translation routing, re-query, abstention
- Evaluation: recall@k, answer correctness, citation accuracy, per-question failure attribution,
  and validation of LLM-judge scores against human labels in all four languages
- Latency and cost per query for every configuration
- FastAPI service + docker-compose

## Data
- MKQA: parallel questions in en/de/fr
- AfriQA: Swahili questions whose answers exist only in English sources
