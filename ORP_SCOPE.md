# ORP scope — Spoken Swahili RAG evaluation

rev. 22 Nov 2026 · single source of truth · supersedes crosslingual_rag_plan.md and the
cross-lingual framing in CLAUDE.md, README.md and PROGRESS.md.

## What this project is now
One monolingual module. Swahili questions — as text, or as speech through ASR — answered over a
**Swahili Wikipedia** passage pool. The deliverable is the **evaluation**, not a product: where does
the pipeline lose answers (ASR → retrieval → generation), how much does the Kenya/Tanzania dialect gap
cost, and can an LLM judge be trusted in a low-resource language.

## Why it collapsed (the bug that forced it)
The old design asked Swahili (AfriQA) questions over English Wikipedia, using SD-QA audio for the
speech side. But **SD-QA records TyDi-QA questions, not AfriQA's** — the two question sets are
disjoint, so AfriQA questions have no recordings and the design was not runnable as written. Building
entirely on SD-QA aligns text and speech on the same questions. TyDi Swahili answers are grounded in
Swahili Wikipedia, which is why the knowledge base is Swahili, not English. The cross-lingual angle is
dropped.

## One dataset: SD-QA (Swahili)
Packages TyDi-QA questions + gold answers + gold passages (Swahili Wikipedia) with Kenyan + Tanzanian
read speech (16 kHz) and Google ASR transcripts — one download.
- **Eval set:** SD-QA Swahili test questions that have a TyDi gold passage and recordings in both
  varieties. Confirm the real counts in step 1; do not reuse the old AfriQA-matched numbers.
- **Retrieval corpus:** Swahili Wikipedia passage pool (100-word passages): the TyDi gold articles plus
  sampled distractors.

## Pipeline (one, no agent)
Spoken question → ASR (faster-whisper large-v3, lang=sw; Google, as shipped) → dense retrieval
(bge-m3, top 5) → generation (Gemini) → evaluation. Text questions enter at retrieval.
**Conditions (5):** Swahili text; Swahili speech for {Kenya, Tanzania} × {Whisper, Google}.

## Evaluation
- WER / CER vs the read reference, by variety and ASR system.
- Retrieval: answer recall@k at nested pool sizes.
- Generation: LLM judge (a different Gemini model), validated against 100+ hand labels, Cohen's kappa.
- Per-question failure attribution: lost at ASR / retrieval / generation, or correct.
- Paired by question, 95% bootstrap CIs.

## Keep / cut
- **Keep:** SD-QA Swahili · Swahili Wikipedia pool · bge-m3 dense retrieval · Gemini generation ·
  two-ASR WER (Whisper vs Google) · Kenya/Tanzania split · LLM-judge-vs-human · failure attribution ·
  MLflow · uv / ruff / pytest / CI.
- **Cut:** MKQA · AfriQA · English Wikipedia · cross-lingual / translation routing · LangGraph agent ·
  reranker · BM25 / Pyserini · Qdrant server · FastAPI + docker deployment.
- **Optional (only if time; nice for the talk):** a BM25 baseline and/or the reranker, for a
  dense-vs-sparse-vs-rerank retrieval slide. Not required.

## Build plan
**Phase 1 — CPU only, no GPU (now → mid-Dec):**
1. SD-QA Swahili loader: both varieties; map each clip to its TyDi question + gold answer/passage;
   define and count the eval set.
2. ASR: use the shipped Google transcripts; run faster-whisper large-v3 (lang=sw) on CPU (no PyTorch on
   the Intel Mac). WER/CER by variety x system. Cache everything.

**Phase 2 — needs IMS GPU (mid-Dec):**
3. Build the Swahili Wikipedia pool; encode with bge-m3; run retrieval for the 5 conditions; recall@k.

**Phase 3 (Jan):**
4. Gemini generation; LLM judge; failure attribution.

**Phase 4 (Feb–Apr, around exams):**
5. Hand-label ~200 answers; judge agreement (kappa); final analysis; report + talk.

## Division of labour
This chat: design, dataset questions, reading results. Claude Code: all in-repo work. Intel Mac = no
local PyTorch; bge-m3 and any GPU work run on IMS servers or cloud. API budget cap EUR 50; cache every
call.
