# Handoff — Spoken Swahili RAG evaluation

Point-in-time brief for a fresh Claude session, **28 Nov 2026**. Living sources of truth, read them
first: **`ORP_SCOPE.md`** (scope), **`PROGRESS.md`** (state + log), **`CLAUDE.md`** (working rules).
This file just orients you; if it conflicts with those, they win.

Branch: **`claude/happy-pascal-5h9o0t`** (all work here; `main` still holds the old cross-lingual design).

## What the project is
Monolingual Swahili RAG, evaluated end to end. Swahili questions (text or speech-via-ASR) answered over
a **Swahili Wikipedia** passage pool. Deliverable is the **evaluation**, not a product: where answers are
lost (ASR → retrieval → generation), the Kenya/Tanzania dialect-gap cost, and whether an LLM judge can be
trusted in a low-resource language. Pipeline (no agent): ASR → dense retrieval (bge-m3, top 5) → Gemini →
eval. **5 conditions:** Swahili text; speech for {Kenya, Tanzania} × {Whisper, Google}. ORP project, IMS
Stuttgart. API budget cap **EUR 50**, cache every call.

## Status (phases)
- **P1 — SD-QA loader + ASR (CPU).** Loader **done**. ASR metrics + **Google WER done**. faster-whisper
  wrapper written + unit-tested but **not run** (see constraints) → **Whisper WER pending IMS**.
- **P2 — Swahili Wikipedia pool + bge-m3 dense retrieval + recall@k.** Not started. Needs IMS GPU.
- **P3 — Gemini generation + LLM judge + failure attribution.** Not started.
- **P4 — ~200 hand labels, judge agreement (Cohen's κ), report + talk.** Not started.

## Repo map
```
ORP_SCOPE.md CLAUDE.md README.md PROGRESS.md   # scope / rules / state
configs/config.yaml                            # asr/embed/judge models, varieties, top_k=5
pyproject.toml uv.lock .python-version         # uv; Python 3.12
src/crag/data/  schema.py io.py sdqa.py        # Question record, snapshot writer, SD-QA loader
src/crag/asr/   metrics.py transcribe.py evaluate.py
tests/          test_smoke.py test_sdqa.py test_asr.py
.github/workflows/ci.yml                        # uv sync --locked; ruff; pytest (no gpu extra, no network)
```
`data/` is **gitignored** (raw SD-QA cache, processed snapshots, result files) — regenerate, never commit.

## Data: SD-QA (ffaisal93/SD-QA, Apache-2.0), pinned commit `c49e9ebb5adb6a0afc560d31e2248fbcc7984a9d`
Data is committed in the GitHub repo (so **no Hugging Face needed** for it). Loader fetches text files
from the raw CDN at the pinned SHA into `data/raw/sdqa/<sha>/`.
- **Split:** `test` = TyDi **dev** (SD-QA has no spoken training data).
- **Files** per `test/swa/{kenya,tanzania}/`: `swahili.test.{variety}-sw-{KE,TZ}.jsonl.gz` (TyDi
  primary-task format), `metadata.csv`, `wav_swa/{example_id}.wav`. Plus `test/swa/swahili.test.csv`
  with the gold written question (`swahili-gold`) + the Google transcripts.
- **Gold reconstruction** (in `sdqa.py`): a question is answerable if any annotator picked a passage
  (`annotations[].passage_answer.candidate_index >= 0`); gold passage = plurality candidate, sliced from
  `document_plaintext` by UTF-8 **byte** offsets; gold answers = minimal-answer byte spans from
  annotators on that passage (deduped; may be empty for yes/no or passage-only). Gold written question
  comes from the CSV; the jsonl `question_text` is the Google ASR transcript.
- **Eval set = answerable ∩ recorded in both varieties.** Real counts at the pinned SHA:
  total 2383 · both varieties 2322 · answerable 861 · **eval set 845** (776 with a minimal answer, 69
  passage-only incl. yes/no; gold-question join clean, 0 missing).
- **Finding:** the two Google locales (sw-KE, sw-TZ) are **identical for Swahili** (845/845, 0
  disagreements) → loader stores **one Google transcript per variety**. Keeps the 5 scope conditions.
- Audio (`wav_swa/`) is committed upstream but **not fetched by the loader** (only needed for Whisper).

## Results so far
Google ASR vs the gold written question (the read reference), 845 eval questions, micro-averaged
(`data/processed/asr_google_wer.json`, regenerate):

| variety | WER | CER |
|---|---|---|
| kenya | 0.330 | 0.182 |
| tanzania | 0.408 | 0.238 |

Tanzanian ≈ 8 WER points worse than Kenyan — the dialect gap. Re-check once Whisper lands.

## How to run
```bash
uv sync                                  # core+dev; Mac skips faster-whisper (Linux-only)
uv run ruff check . && uv run pytest     # must stay green; 20 tests
uv run python -m crag.data.sdqa          # build the eval snapshot (needs network: GitHub raw)
uv run python -m crag.asr.evaluate       # Google WER/CER -> data/processed/asr_google_wer.json
```
In **this cloud container only**, prefix network commands with `SSL_CERT_FILE=/root/.ccr/ca-bundle.crt`
(proxy CA, for Python urllib). On a normal machine, drop it.

## Environment constraints (important)
- **HF is blocked in the cloud container.** SD-QA data is fine (GitHub). But faster-whisper's large-v3
  **weights come from HF**, so Whisper transcription **cannot run in the cloud container** — run it on IMS
  (or any Linux host with HF access).
- **Intel Mac** (Elliott's laptop): no PyTorch, and faster-whisper won't install (its onnxruntime dep has
  no Intel-macOS wheel). Mac runs the loader, Google WER, tests, Gemini. **ASR + all GPU work → IMS/cloud.**
- **IMS GPU** is needed for bge-m3 (P2) and is the home for Whisper (P1 finish). Still waiting on server
  specs — see the IMS checklist in `PROGRESS.md`.

## Conventions (from CLAUDE.md — follow these)
- Before non-trivial code: state a 2–4 sentence plan, wait for OK. One component per step. Name the
  rejected alternative for each design choice.
- After every change: run `ruff check` + `pytest`, show results. Never claim it works without running it.
- Metric numbers come only from result files the eval scripts write — never invent/round/hand-edit.
- Cache every API/LLM call; wrap network calls in retry+timeout. Never commit secrets, audio, weights,
  indexes, or `data/`.
- Commit on the branch above; push only when asked ("wrap up" / "wrap up and push"). Commit footer:
  `Co-Authored-By: Claude ...` + the session line (see the session's attribution reminder).

## Next steps
1. **Whisper WER on IMS** (finishes P1). Glue to write (not yet built): for each of the 845 eval ids ×
   {kenya, tanzania}, fetch `test/swa/{variety}/wav_swa/{example_id}.wav` from the raw CDN at the pinned
   SHA, transcribe with `crag.asr.transcribe` (faster-whisper large-v3, lang=sw, CPU int8), cache, then
   score with `crag.asr.evaluate.wer_by_condition` using the Whisper transcripts. On IMS:
   `UV_HTTP_TIMEOUT=300 uv sync` then run the glue. Report Whisper WER by variety next to Google.
2. **P2 pool + retrieval** on IMS GPU: build the Swahili Wikipedia 100-word passage pool (gold TyDi
   articles — already in the jsonl `document_plaintext` — plus sampled distractors), encode with bge-m3
   (`uv sync --extra gpu`), dense top-5, recall@k at nested pool sizes for the 5 conditions.
3. **P3** generation (Gemini) + LLM judge + per-question failure attribution. **P4** hand labels + κ.

## Open items / small cleanups
- `src/crag/data/schema.py` (`Question`) and `io.read_snapshot` are cross-lingual leftovers the SD-QA path
  doesn't use (`sdqa.py` only uses `write_snapshot`; `evaluate.py` reads raw JSON). Remove or adapt.
- Decide the Whisper compute: CPU int8 (slow but simple) vs GPU on IMS.
- Confirm IMS specs (GPU/VRAM, scheduler, HF access on nodes) — blocks P1 finish and all of P2.

## Commits this session (on the branch)
`8b9edbe` scope reset → `52cce20` gate faster-whisper to Linux → `47700e3` SD-QA loader (845) →
`cea7a8e` ASR metrics + Google WER + package tidy.
