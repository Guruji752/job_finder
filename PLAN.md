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
| Job source | **JSearch (via RapidAPI)**, not Adzuna, not direct scraping. Aggregates from Google for Jobs (LinkedIn, Indeed, Glassdoor, ZipRecruiter, Monster, etc.) with per-listing source-platform attribution. | Adzuna doesn't expose which original platform a job came from (`redirect_url` points to Adzuna's own landing page, not the source site) — JSearch does, which directly matches the original "which platform" requirement. Avoided unofficial LinkedIn/Naukri "scraper APIs" on RapidAPI — same ToS-violation risk as direct scraping, just outsourced. **Known constraint: free tier is a hard 200 requests/month** — this is now the entire search budget, not just shortlist enrichment; watch usage once Phase 4's agent starts expanding queries. |
| Interface | REST API only (FastAPI). No UI/CLI for now. | Keep scope tight; backend-focused. |
| Models | **Fully open source (Hugging Face), no Claude/Anthropic API.** | Explicit goal: avoid vendor lock-in, learn the open-source ML ecosystem. Accepted trade-off: small/mid open models are less reliable than Claude at structured-JSON reasoning for gap analysis. |
| Hardware | **No dedicated GPU available** (superseded earlier local-GPU-box plan). | Ruled out both a local box and free/eval-tier cloud GPU options (Colab/Kaggle: no stable endpoint; NVIDIA NIM free tier: eval-only rate limits; HF ZeroGPU Spaces: cold starts + quota) as unreliable for daily use. |
| Model serving | **Hugging Face Inference Providers** (paid, pay-as-you-go), single OpenAI-compatible endpoint (`https://router.huggingface.co/v1`), routes to third-party providers (Together AI, Fireworks, Groq, etc.) actually hosting the model. | Production-grade reliability, no infra to manage, no markup over provider rates, monthly free credits applied automatically. Same OpenAI-compatible shape already designed into the app, so it's a `base_url` + API key change, nothing else. |
| Models chosen | Extraction + gap analysis (LLM): **`Qwen/Qwen2.5-72B-Instruct`** via HF Inference Providers OpenAI-compatible chat route (fallback: `Llama-3.3-70B-Instruct`). Embeddings (Tier-1 filter): **`BAAI/bge-large-en-v1.5`** via HF's **`feature_extraction` API** (`huggingface_hub.InferenceClient`), hosted — no local model. Both names in `app/config.py` as `CHAT_MODEL` / `EMBEDDING_MODEL`. | Everything runs on the HF API (no local models). **Key lesson:** HF has TWO embedding surfaces and they are not interchangeable. (1) OpenAI-compatible `router.huggingface.co/v1/embeddings` (`client.embeddings.create()`) routes to third-party providers — none host BGE, so it 404s; this dead end cost several detours (tried `:hf-inference`, `:together`, `intfloat/multilingual-e5-large-instruct`, and even a local `fastembed` fallback). (2) `huggingface_hub.InferenceClient(...).feature_extraction(model=...)` uses HF's own `hf-inference` backend and DOES serve BGE — this is the working path, and the one the existing RAG service already uses. BGE via `feature_extraction` returns token-level embeddings `(seq_len, dim)`, so Tier-1 mean-pools across tokens to get a sentence vector (mirrors the RAG service). Text truncated to 1500 chars per call. Query-instruction prefix applied on the profile/query side (BGE convention). |
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
                       JSearch      Existing   Qwen2.5-72B-
                     (RapidAPI,      RAG API    Instruct (or
                      job data +                Llama-3.3-70B),
                      source platform)           hosted by a
                                                 routed provider
```

### Four separate concerns (why they're separate)

1. **`JobSource` adapters** — interface with one concrete impl (JSearch) to
   start. Adding a second source later must not touch matching logic.
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
| 1 | Job discovery | JSearch adapter behind `JobSource` interface; `POST /search` returns normalized raw jobs (including source platform) |
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

- OpenWeb Ninja account + JSearch API key (direct, no RapidAPI middleman), free tier 200 requests/month, auth via `x-api-key` header
- Existing RAG endpoint URL
- Hugging Face account + API token (Inference Providers, pay-as-you-go)

## Open questions / things to revisit

- Gap-analysis quality from Qwen2.5-72B vs. Claude hasn't been empirically
  tested yet — watch for confidently-wrong "strong match" verdicts once
  Phase 3 is live; may need prompt tuning or a different open model.
- **Confirmed empirically:** `ProfileDigest.years_experience` is unstable
  across separate runs of `build_profile_digest()` — observed 5.47, 6.5,
  and 8.0 across three runs on the same resume (the LLM computing arithmetic
  over unstructured date ranges, not a deterministic parse). `seniority` was
  fixed by adding a job-titles question and instructing the prompt to weigh
  title over computed years (confirmed working: now correctly returns
  `"senior"`). Treat `years_experience` as a soft signal in Phase 3 matching,
  not a hard filter, given this variance — revisit only if it causes visibly
  bad matches in practice.
- **Confirmed empirically:** the provider currently serving
  `Qwen2.5-72B-Instruct` through HF Inference Providers rejects
  `response_format={"type": "json_object"}` (400 error, "not supported").
  Structured output relies on prompt instructions + defensive parsing
  (`app/profile/digest.py::_parse_json_response`, strips markdown fences /
  extracts `{...}` substring) instead of a strict API-enforced mode. Same
  risk applies to Phase 3's gap-analysis JSON output — reuse this parsing
  approach there rather than assuming `response_format` will work.
- JSearch's 200/month free cap is the entire search budget now (no Adzuna
  fallback) — monitor real usage once Phase 4's agent starts expanding a
  single search into multiple queries; may need to add Adzuna back as a
  bulk-search source if the cap turns out too tight.
- **Confirmed empirically:** `num_pages > 1` on `/search-v2` is billed as
  multiple separate requests against the 200/month quota, not one — e.g.
  `num_pages=10` consumes ~10 requests, not 1. Keep `num_pages` low (2 is
  the current default in `JSearchSource.search()`); rely on `date_posted`
  filtering for relevance rather than requesting many pages.
- Actual per-token cost of Qwen2.5-72B/Llama-3.3-70B via HF Inference
  Providers hasn't been checked against real usage volume yet — verify
  after a few days of real searches that the "sporadic use = cheap" assumption
  holds.
- Phase 6 (self-hosted serving) is a stretch goal, not committed — revisit
  if/when a GPU becomes available.
