# Job Finder — Architecture & Build Plan

Living document. Update as decisions change — don't let this drift from reality.

## What this is

An agentic system that searches job postings on my behalf, cross-references
them against my resume/profile (via an existing RAG endpoint), and returns a
ranked list of jobs with per-job gap analysis (matched skills, missing
skills, fit verdict).

Two goals, in priority order:
1. Portfolio-quality architecture (demonstrates real skills: agents, RAG
   integration, self-hosted model serving, clean adapter design).
2. Actually usable for my real job search.

## Locked decisions

| Area | Decision | Why |
|---|---|---|
| RAG service | Already exists, external. Chunk-retrieval style (`ask a question -> get resume text chunks`), not a structured-profile endpoint. | We're a *consumer*, not builder, of RAG. |
| Job source | Aggregator APIs (Adzuna first), not direct scraping of LinkedIn/Naukri/Indeed. | Scraping LinkedIn/Indeed directly is fragile, ToS-violating, and high-maintenance. Aggregators give legal, structured, stable data. |
| Interface | REST API only (FastAPI). No UI/CLI for now. | Keep scope tight; backend-focused. |
| Models | **Fully open source (Hugging Face), no Claude/Anthropic API.** | Explicit goal: avoid vendor lock-in, learn the open-source ML ecosystem. Accepted trade-off: small/mid open models are less reliable than Claude at structured-JSON reasoning for gap analysis. |
| Hardware | **No dedicated GPU available** (superseded earlier local-GPU-box plan). | Ruled out both a local box and free/eval-tier cloud GPU options (Colab/Kaggle: no stable endpoint; NVIDIA NIM free tier: eval-only rate limits; HF ZeroGPU Spaces: cold starts + quota) as unreliable for daily use. |
| Model serving | **Hugging Face Inference Providers** (paid, pay-as-you-go), single OpenAI-compatible endpoint (`https://router.huggingface.co/v1`), routes to third-party providers (Together AI, Fireworks, Groq, etc.) actually hosting the model. | Production-grade reliability, no infra to manage, no markup over provider rates, monthly free credits applied automatically. Same OpenAI-compatible shape already designed into the app, so it's a `base_url` + API key change, nothing else. |
| Models chosen | Extraction + gap analysis (LLM): **`Qwen/Qwen2.5-72B-Instruct`** (fallback: `Llama-3.3-70B-Instruct` if provider availability/pricing differs) via HF Inference Providers. Embeddings (Tier-1 filter): **`BAAI/bge-large-en-v1.5`**, run **locally on the dev machine via CPU** (`sentence-transformers`) — unaffected by the hosting change. | No longer VRAM-constrained since the provider manages the GPU — upgraded from the 14B ceiling to 70B-class for meaningfully better gap-analysis reasoning, closer to Claude quality. Embeddings stay local: free, fast enough on CPU, avoids a network hop for every Tier-1 comparison. |
| Vendor lock-in trade-off (explicit) | Accepted: this reintroduces a hosted-API dependency structurally similar to the original Claude-API concern, just serving open weights instead of closed ones. Self-hosted serving (vLLM/GPU management) is **dropped from the near-term plan**, kept only as a stretch goal (see Phase 6). | Practicality won given repeated hardware/free-tier dead ends. What's preserved from the original motivation: open-weight models, HF ecosystem familiarity. What's given up for now: hands-on inference-serving experience. |
| Auth | HF API token as bearer auth (standard), stored as an env var/secret — not committed. | Standard practice for any hosted API key. |

## Architecture

```
                    ┌─────────────────────────────────────┐
   HTTP (REST)      │            FastAPI app               │
  ───────────────▶  │                                      │
  POST /search      │   ┌──────────────────────────────┐   │
                    │   │   LangGraph agent (the brain) │   │
                    │   └──────────────────────────────┘   │
                    │      │        │            │          │
                    │      ▼        ▼            ▼          │
                    │  JobSource   RAG        HF Inference  │
                    │  adapters    client     Providers     │
                    │                        (OpenAI-compat)│
                    │                                       │
                    │  bge-large (local, CPU, Tier-1         │
                    │  embeddings — in-process, no network) │
                    └──────┼─────────┼────────────┼─────────┘
                           ▼         ▼            ▼
                     Adzuna/JSearch  Existing   Qwen2.5-72B-
                     (job data)      RAG API    Instruct (or
                                                 Llama-3.3-70B),
                                                 hosted by a
                                                 routed provider
```

### Four separate concerns (why they're separate)

1. **`JobSource` adapters** — interface with one concrete impl (Adzuna) to
   start. Adding a second source (JSearch, etc.) later must not touch
   matching logic.
2. **`RAGClient`** — thin HTTP client to the existing chunk-retrieval RAG
   endpoint. Isolated so a RAG API change touches one file.
3. **Matching engine — two-tier funnel** (cost/quality control):
   - **Tier 1 (cheap, wide):** embedding similarity (local `bge-large`)
     between job text and cached profile digest. Filters out obvious
     non-fits fast and for free.
   - **Tier 2 (expensive, narrow):** only the top ~10-15 survivors go to
     Qwen2.5-72B-Instruct (via HF Inference Providers) for real scoring +
     gap analysis.
4. **Gap analysis** — structured JSON output per surviving job:
   `match_score`, `matched_skills`, `missing_skills`, `verdict`. Not free
   text — the API ranks on this.

### RAG usage strategy (two-phase, since RAG is chunk-retrieval)

- **Profile digest (once per search, cached):** fire a small fixed set of
  broad questions at the RAG endpoint up front (languages/frameworks known,
  years of experience/seniority, industries worked in). Feed the returned
  chunks to the LLM (via HF Inference Providers) once to distill a structured
  `ProfileDigest {skills, years, seniority, domains}`. Cache it — Tier-1/2
  matching runs against this in-memory object, zero RAG calls during
  ranking.
- **Targeted evidence (only for the shortlist):** for jobs reaching Tier 2,
  optionally ask the RAG one focused question per job ("what experience is
  relevant to `<job's top requirements>`?") to pull real resume evidence
  into the gap analysis output.

### Response shape (`POST /search`)

Ranked list where each item is:
```
{ job, match_score, matched_skills, missing_skills, verdict, evidence }
```

## Phased roadmap

| # | Phase | Delivers |
|---|---|---|
| 0 | Skeleton | FastAPI app, config (env vars for keys/URLs), Docker + compose, `/health` |
| 1 | Job discovery | Adzuna adapter behind `JobSource` interface; `POST /search` returns normalized raw jobs |
| 2 | RAG integration | `RAGClient` wired to the existing endpoint; verify profile-fact retrieval |
| 2.5 | Model access setup | See sub-phases below — gets HF Inference Providers + local embeddings wired up |
| 3 | Matching | Tier-1 similarity filter -> Tier-2 gap analysis -> ranked JSON response |
| 4 | Agentic upgrade | Wrap the pipeline in LangGraph so the agent plans/expands searches and loops, rather than running a fixed linear script |
| 5 | Polish | Caching, rate-limit handling, tests, README with architecture diagram |
| 6 (stretch) | Self-hosted serving | If/when a GPU (local or cheap dedicated cloud) becomes available: stand up vLLM serving an open model, swap `LLM_BASE_URL` from HF Inference Providers to the self-hosted endpoint. Revisits the original self-hosting/portfolio motivation once the practical blocker (no GPU) is resolved. |

### Phase 2.5 sub-phases (model access)

| Sub-phase | What |
|---|---|
| 2.5a | Create HF account, generate an API token, confirm billing/payment method for pay-as-you-go Inference Providers |
| 2.5b | Confirm `Qwen2.5-72B-Instruct` (or `Llama-3.3-70B-Instruct` fallback) is available through a routed provider at acceptable pricing; pick the provider |
| 2.5c | `pip install sentence-transformers`; load `BAAI/bge-large-en-v1.5` locally for Tier-1 embeddings |
| 2.5d | Point app's `LLM_BASE_URL` at `https://router.huggingface.co/v1` with the HF token; verify with a real chat-completion request |

## External dependencies / accounts needed

- Adzuna `app_id` + `app_key` (free tier)
- Existing RAG endpoint URL
- Hugging Face account + API token (Inference Providers, pay-as-you-go)

## Open questions / things to revisit

- Gap-analysis quality from Qwen2.5-72B vs. Claude hasn't been empirically
  tested yet — watch for confidently-wrong "strong match" verdicts once
  Phase 3 is live; may need prompt tuning or a different open model.
- Whether a second aggregator (JSearch, SerpAPI Google Jobs) gets added
  depends on how much Adzuna's coverage turns out to be missing.
- Actual per-token cost of Qwen2.5-72B/Llama-3.3-70B via HF Inference
  Providers hasn't been checked against real usage volume yet — verify
  after a few days of real searches that the "sporadic use = cheap" assumption
  holds.
- Phase 6 (self-hosted serving) is a stretch goal, not committed — revisit
  if/when a GPU becomes available.
