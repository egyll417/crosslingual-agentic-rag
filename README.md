# crosslingual-agentic-rag

[![CI](https://github.com/egyll417/crosslingual-agentic-rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/egyll417/crosslingual-agentic-rag/actions/workflows/ci.yml)

> **Scope reset (22 Nov 2026):** now a single **monolingual Swahili** module. See
> [ORP_SCOPE.md](ORP_SCOPE.md) for the full scope. The repo keeps its original (cross-lingual) name
> for git-history continuity.

Swahili questions — as text, or as speech through ASR — answered over a **Swahili Wikipedia** passage
pool. The deliverable is an **evaluation**: where the pipeline loses answers (ASR → retrieval →
generation), how much the Kenya/Tanzania dialect gap costs, and whether an LLM judge can be trusted in
a low-resource language. Optional Research Module (ORP) at IMS, University of Stuttgart.

**Status:** work in progress; scope reset 22 Nov 2026. See [PROGRESS.md](PROGRESS.md) for the current state.

## Pipeline
Spoken question → ASR (faster-whisper large-v3 + the Google transcripts shipped with SD-QA) → dense
retrieval (BGE-M3, top 5) → generation (Gemini) → evaluation. Text questions enter at retrieval.
**Conditions (5):** Swahili text; Swahili speech for {Kenya, Tanzania} × {Whisper, Google}.

## Evaluation
- WER / CER vs the read reference, by variety and ASR system
- retrieval answer recall@k at nested pool sizes
- generation scored by an LLM judge, validated against 100+ human labels (Cohen's kappa)
- per-question failure attribution: lost at ASR, retrieval or generation, or correct
- paired by question, 95% bootstrap confidence intervals

## Data
- **SD-QA (Swahili):** TyDi-QA questions + gold answers + gold passages (Swahili Wikipedia), with
  Kenyan and Tanzanian read speech (16 kHz) and Google ASR transcripts.
