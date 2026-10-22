# Status and Roadmap: Multi-Turn Chat with File Q&A

**As of:** 2026-10-22 · **Scope:** what is built and verified, what it costs, where it breaks first, and what to do next for scale, cost and quality.

Related: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (architecture and phase checklist), [README](../README.md) (setup).

---

## 1. Summary

- **Working today:** multi-turn chat with streaming replies, saved history, prompt caching, per-message token and cost tracking, and questions answered from uploaded files (PDF, DOCX, TXT, MD, CSV) with versioning. There is a Streamlit UI, and everything runs with one `docker compose up`.
- **Verified:** 75 backend tests and 7 UI tests pass, lint and strict type-checking are clean, and live end-to-end runs against Claude Haiku 5.5 through OpenRouter succeeded.
- **Cost:** a typical turn costs **$0.00005–$0.0002**. Prompt caching cut a follow-up question about a PDF by **about 75%** in a live test. Once caching is working, **output tokens dominate the bill**, so controlling answer length is the next lever.
- **Main gaps before production:** authentication, compaction of long threads, rate limits and budgets, shared file storage across servers, observability, and continuous integration. None of these need architectural changes; the design already allows for each one (section 6).

---

## 2. What is built and verified

| Capability | Status | How it was verified |
|---|---|---|
| Multi-turn chat in one thread | ✅ Done | Live: 4-turn conversation recalled earlier details; integration tests |
| Streaming replies (server-sent events) | ✅ Done | Live in the API and UI; tests for `delta` → `done` / `error` |
| History saved all-or-nothing per turn | ✅ Done | Tests: failure before or during a reply saves nothing; the next turn works |
| One reply at a time per thread | ✅ Done | Test: a second message while one is streaming gets `409 thread_busy`; the lock is released after failures |
| Prompt caching (Anthropic explicit markers, others automatic) | ✅ Done | Live: turn 4 read 528 tokens from cache; a PDF follow-up read 1,456 |
| Same provider for every turn of a thread (`session_id`) | ✅ Done | Test asserts `session_id = thread_id` on every request |
| Per-message usage: tokens, cached tokens, cost | ✅ Done | Stored in `usage`; shown per reply and per thread in the UI |
| File upload: PDF, DOCX, TXT, MD, CSV | ✅ Done | Unit tests per format; live 5-page PDF Q&A with page citations |
| Duplicate detection (sha256) | ✅ Done | Live and tests: `409 file_duplicate`, even under another name |
| File versioning: replace / keep both | ✅ Done | Live: the v2 answer used the new content and the model described the change; the database guarantees one current file per name |
| Model allow-list per thread | ✅ Done | Haiku 5.5 (default) and Sonnet 5.5; anything else gets `422` |
| Health checks, request IDs, structured logs | ✅ Done | `/health`, `/health/ready`; every log line carries `request_id` |
| Migrations with drift detection | ✅ Done | A test fails if the models and migrations diverge |
| Streamlit UI | ✅ Done | Driven headlessly against the live API: chat, upload, conflict prompt, Replace |

**Deliberately deferred** (see [the plan](IMPLEMENTATION_PLAN.md)): RAG and embeddings, since files are about 5 pages and sent in full; background-worker extraction; multiple threads in the UI; authentication.

---

## 3. Architecture as built

```
Streamlit UI ──HTTP/SSE──► FastAPI (stateless, async)
                              │
                              ├── ChatService ── Redis lock per thread
                              │      ├── context_builder: frozen system prompt → history → new turn
                              │      ├── OpenRouterProvider (OpenAI SDK, retries) ──► OpenRouter ──► Claude
                              │      └── one transaction: user msg + assistant msg + usage + file links
                              ├── FileService ── extract (thread pool) → dedupe → version → store
                              │
                              ├── Postgres 17 (threads, messages, files, usage)   [pgvector image, unused so far]
                              ├── Redis 7 (locks; ARQ queue, idle)
                              └── Local disk volume (uploaded bytes)
```

**Design choices that matter for scale and cost:**

1. **Append-only history, replayed exactly as stored.** This is what makes prompt caching work. Editing or reordering history, or putting dynamic data in the system prompt, would silently break caching.
2. **Files go into the user turn after they're uploaded, never into the system prompt**, so uploading a file never invalidates the cached history.
3. **No database connection is held while the model streams.** A 60-second reply doesn't occupy a pooled connection.
4. **Every layer sits behind an interface.** `LLMProvider`, `Storage` and the current-user dependency can each be swapped (another LLM gateway, S3, real authentication) without touching business logic.

---

## 4. Measured cost and performance

### 4.1 Live measurements (Claude Haiku 5.5 through OpenRouter)

| Scenario | Prompt tokens | Read from cache | Written to cache | Output | Cost |
|---|---|---|---|---|---|
| Short chat, turn 1 | 325 | 0 | 0 | 145 | $0.000105 |
| Short chat, turn 3 (first over the cache minimum) | 532 | 0 | 528 | 71 | $0.000102 |
| Short chat, turn 4 | 622 | **528** | 90 | 25 | $0.000029 |
| PDF question, first | 1,460 | 0 | 1,456 | 56 | $0.000210 |
| PDF follow-up | 1,538 | **1,456** | 78 | 48 | **$0.000049** |

What this shows:
- Caching only starts once the prompt passes **512 tokens** (the minimum for Claude 5.x). Very short chats never benefit, and that's fine because they're cheap anyway.
- Once a file or a long history is cached, each later turn pays only for the **new** tokens plus a small cache-read charge.

### 4.2 Cost model for a typical conversation (estimate)

Assumptions: Haiku 5.5 prices ($0.10 input, $0.50 output, $0.01 cache read, $0.125 cache write per million tokens); a 4,000-token file; 20 turns; about 30 tokens per question and 200 per answer.

| | Without caching | With caching (as built) |
|---|---|---|
| Input cost, 20 turns | ~$0.013 | ~$0.002 |
| Output cost, 20 turns | ~$0.002 | ~$0.002 |
| **Total per conversation** | **~$0.015** | **~$0.004 (about 70% less)** |

**Takeaways:**
- **At 100,000 conversations a month, that's about $400 with caching against about $1,500 without** (model cost only).
- **With caching on, output is about half the remaining cost.** Answer length is now the biggest lever (section 6.1).
- **Sonnet 5.5 costs 20× as much per token.** Keep it opt-in per thread for questions that need deeper reasoning.

### 4.3 Throughput characteristics (from the design, not load-tested yet)

- Each streaming reply is one asyncio task, not a thread. A single API process can hold hundreds of open streams. In practice the limits are OpenRouter rate limits and the load balancer's idle timeout, not CPU.
- Each API instance uses at most 20 Postgres connections (pool 10 + overflow 10), and only briefly: two short transactions per turn.
- **Not yet measured.** Phase 5 includes a k6 or Locust load test; the numbers above are design expectations.

---

## 5. Known limitations and risks

| # | Limitation | Impact | Severity | Fix (section) |
|---|---|---|---|---|
| 1 | No authentication; every request acts as `dev@localhost` | Anyone who can reach the API sees and spends on the same account | **High** before any shared deployment | 6.4 |
| 2 | No rate limits or spending budgets | One user or script can run up OpenRouter cost | **High** | 6.1, 6.4 |
| 3 | No compaction of long threads | Context and cost per turn grow with thread length (cheaply, thanks to caching, but without limit); Anthropic prices Haiku prompts over 100K tokens higher | Medium | 6.1 |
| 4 | Uploaded files on a local Docker volume | A second API server on another host can't read them | Medium (blocks horizontal scaling) | 6.2 |
| 5 | If the browser disconnects mid-reply, the turn is discarded | Tokens already generated are paid for but not saved | Low–medium | 6.3 |
| 6 | No metrics, tracing or alerting | Cost spikes or a drop in cache hit rate go unnoticed | Medium | 6.3 |
| 7 | One uvicorn process per container | Uses one CPU core; scale by adding containers | Low (by design) | 6.2 |
| 8 | Token counts before a request are estimated (characters ÷ 4) | File limits are approximate (they lean toward overestimating) | Low | 6.1 |
| 9 | Streamlit UI | Fine for internal use and demos; not a production front end (no auth, one script run per user session) | Low now | 6.2 |
| 10 | No CI pipeline, and the project isn't in git yet | Regressions aren't caught automatically | Medium | 7 |
| 11 | Changing `SYSTEM_PROMPT` invalidates every thread's cache once | One-off cost spike when a new version is deployed | Low (known, expected) | 6.1 |
| 12 | Scanned PDFs are rejected (no OCR) | Some user files can't be used | Low (a product decision) | 6.5 |

---

## 6. Roadmap

Each item lists its expected impact and rough effort (S = under 1 day, M = 1–3 days, L = about 1 week).

### 6.1 Cost optimization

Ranked by expected savings. Items 1–4 cost nothing in answer quality.

| # | Action | Why | Impact | Effort |
|---|---|---|---|---|
| 1 | **Track cache hit rate and cost per conversation** (from the `usage` table) and alert on drops | Caching fails silently; any change that alters the prompt prefix shows up first as a drop in hit rate | Protects the ~70% saving | S |
| 2 | **Per-user and per-thread spending budgets** (Redis counters, checked before each turn) | Caps the worst case: abuse, runaway scripts | Removes unbounded risk | S–M |
| 3 | **Conversation compaction** (Phase 3): above about 60% of a chosen budget (e.g. 60K tokens), have Haiku summarize the oldest turns into a checkpoint; keep file content | Keeps cost per turn flat on long threads, and stays below Haiku's higher price tier over 100K tokens | Large on long threads | M |
| 4 | **Tune answer length per route** (`max_tokens`, plus a "be concise" instruction or a user-selectable short / detailed mode) | Output is about half the cost once caching works | 20–40% of output cost | S |
| 5 | **Set reasoning effort explicitly** through OpenRouter's `reasoning` parameter; low for chat, higher for document analysis | Reasoning tokens are billed as output; simple chat rarely needs much | Varies; measure first | S |
| 6 | **Choose the cache TTL from real gaps between messages**: keep 5 minutes if most follow-ups arrive within 5 minutes; switch `PROMPT_CACHE_TTL=1h` if users often pause 5–60 minutes | A 1-hour cache write costs 2× but survives long pauses | 10–30% on bursty use | S (config only) |
| 7 | **Route by difficulty**: Haiku by default; Sonnet only when the user picks it, or for flagged hard tasks | Sonnet is 20× the price | Keeps the expensive model rare | S (already per thread) |
| 8 | **Exact token counts** for file limits (count with the model's tokenizer, or calibrate against the reported `prompt_tokens`) | Avoids rejecting files that would fit | Better user experience, small savings | S |
| 9 | **Batch API for offline work** (summaries, evaluations, re-processing) | Usually about half price for jobs that aren't latency-sensitive | 50% on that work | M |
| 10 | **Avoid rewriting history** (compaction and any future features must append rather than edit) | Any change to earlier turns throws away the cache | Protects the saving | Ongoing review |

**Metric to watch:** `cache hit rate = sum(cached_tokens) / sum(prompt_tokens)`. Healthy multi-turn traffic should sit **above 70%**. A sudden drop almost always means something started changing the prompt prefix.

### 6.2 Scaling

The API keeps no state in memory and the heavy lifting is done by OpenRouter, so scaling is mostly about the components around it.

**Stage 1: internal or pilot (up to about 50 concurrent users).** What runs today, plus:
- Authentication, budgets and rate limits (6.4, 6.1)
- Postgres backups (a managed Postgres such as RDS, Cloud SQL, Neon or Supabase gives this by default)

**Stage 2: production (hundreds of concurrent users)**

| Area | Change | Effort |
|---|---|---|
| File storage | Implement `S3Storage` behind the existing `Storage` interface; switch with configuration | S |
| API | Run 2+ API containers behind a load balancer; **set the idle timeout to at least 300 s** so streams aren't cut off; disable proxy buffering (the API already sends `X-Accel-Buffering: no`) | S |
| Postgres | Managed instance; add **PgBouncer** (transaction mode) once API instances × 20 connections approaches Postgres's connection limit | M |
| Redis | Managed Redis with failover (it holds the thread locks) | S |
| Extraction | Move file extraction to the ARQ worker if uploads get bigger or more frequent; the service boundary already exists | M |
| Front end | Replace Streamlit with a React or Next.js client on the same API (the endpoints and SSE format stay the same) | L |
| LLM resilience | Add OpenRouter's `models` fallback list (e.g. Haiku 5.5 → another fast model) for provider outages | S |

**Stage 3: large scale (thousands of concurrent users)**
- Autoscale API containers on open connections or CPU; autoscale workers on queue depth.
- Partition hot tables by time or tenant (`messages`, `usage`) once they reach tens of millions of rows; add read replicas for history reads.
- Raise OpenRouter rate limits for the account, or spread traffic across providers or keys.
- If files get large again: add retrieval (pgvector is already in the database image) for files over the token limit, and keep sending small files in full.

### 6.3 Reliability and observability

| Action | Why | Effort |
|---|---|---|
| **OpenTelemetry tracing** (FastAPI, SQLAlchemy and OpenRouter calls) with `request_id` correlation | See where the time goes in each turn | M |
| **Metrics:** time to first token, turn latency, error rate by code, cache hit rate, cost per hour, `409 thread_busy` rate | Service-level targets and alerting | M |
| **Keep generating after the browser disconnects** (finish the reply in a background task, save it, let the client fetch it) | No paid-for tokens are lost; the reply appears on refresh | M |
| **Idempotency key on `POST /messages`** | A client retry after a network blip doesn't create a duplicate turn | S |
| **Alerting** on: error rate, `/health/ready` failures, cost per hour over a threshold, cache hit rate under 50% | Catch cost and outage problems early | S |
| **Database backups plus a tested restore** | Conversations are user data | S (managed database) |

### 6.4 Security

| Action | Priority |
|---|---|
| **Authentication** (OIDC or JWT): replace `get_current_user_id` in [deps.py](../backend/app/api/deps.py); ownership checks are already in every query | **Before any shared deployment** |
| **Rate limits per user** (Redis token bucket) on messages and uploads | High |
| **Secrets in a secret manager**, not `.env`; **rotate the OpenRouter key** that was shared earlier in chat; set a spending limit on the key | High |
| **CORS and HTTPS** at the load balancer; restrict API origins to the front end | High |
| **Upload hardening**: already validates file type against content, sanitizes names and caps size; add antivirus scanning (ClamAV) if files come from untrusted users | Medium |
| **Prompt-injection awareness**: file content is untrusted. It's currently wrapped in `<file>` tags; never give the model tools or actions driven by file content without an approval step | Medium (grows as features are added) |
| **Data retention**: a policy and a delete endpoint for threads and files (also covers GDPR erasure) | Medium |

### 6.5 Quality

| Action | Why | Effort |
|---|---|---|
| **Evaluation set**: 30–50 real questions with expected answers across sample files (factual lookup, cross-page questions, version changes, "not in the file") | The only reliable way to compare models, prompts and cost settings | M |
| **Run the evaluation on Haiku against Sonnet** before changing the default model or prompt | Confirms that cost savings don't hurt answer quality | S once the set exists |
| **Check citations automatically** (does the cited page contain the claimed fact?) | Grounded answers matter most for document Q&A | M |
| **OCR for scanned PDFs** (OpenRouter `mistral-ocr`, $2 per 1,000 pages) behind a setting | Only if users need it | S–M |

---

## 7. Recommended next steps

In order. The first five make the app safe to share; the rest make it cheaper and more robust.

| # | Item | Section | Effort |
|---|---|---|---|
| 1 | Put the project in git and add CI (ruff, mypy, pytest with Postgres and Redis services) | – | S |
| 2 | Rotate the OpenRouter key and set a spending limit on it | 6.4 | S |
| 3 | Authentication (replace the dev user) | 6.4 | M |
| 4 | Per-user rate limits and spending budgets | 6.1, 6.4 | S–M |
| 5 | Cost and cache-hit-rate dashboard, plus alerts | 6.1, 6.3 | S–M |
| 6 | Conversation compaction (Phase 3) | 6.1 | M |
| 7 | Tune answer length and reasoning effort; measure the effect on the evaluation set | 6.1, 6.5 | S–M |
| 8 | `S3Storage` plus 2 API containers behind a load balancer | 6.2 | S–M |
| 9 | Multiple threads in the UI (Phase 4) | Plan | S–M |
| 10 | Load test to replace the design estimates in 4.3 with measurements | 6.2 | M |

---

## Appendix A: Configuration levers

| Setting | Default | Effect |
|---|---|---|
| `DEFAULT_CHAT_MODEL` | `anthropic/claude-haiku-5.5` | Main cost and quality lever |
| `ALLOWED_CHAT_MODELS` | Haiku 5.5, Sonnet 5.5 | Which models a thread may use |
| `CHAT_MAX_TOKENS` | 4096 | Upper bound on output cost per turn |
| `PROMPT_CACHE_TTL` | `5m` | `1h` when users pause 5–60 minutes between messages |
| `MAX_FILE_TOKENS` | 20000 | Upper bound on file input cost; about 25 pages |
| `MAX_UPLOAD_BYTES` | 10 MB | Upload size cap |
| `MAX_MESSAGE_CHARS` | 32000 | Cap on a single message |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | 120, 2 | Resilience against slow or failing providers; also sets how long the thread lock lasts |

## Appendix B: Useful queries

```sql
-- Cost and cache hit rate per day
SELECT date_trunc('day', m.created_at) AS day,
       count(*)                                         AS turns,
       sum(u.cost)                                      AS cost_usd,
       round(sum(u.cached_tokens)::numeric / nullif(sum(u.prompt_tokens), 0), 3) AS cache_hit_rate,
       round(avg(u.completion_tokens))                  AS avg_output_tokens
FROM usage u JOIN messages m ON m.id = u.message_id
GROUP BY 1 ORDER BY 1 DESC;

-- Most expensive threads
SELECT t.id, t.model, count(*) AS turns, sum(u.cost) AS cost_usd, max(t.est_context_tokens) AS context_tokens
FROM usage u JOIN messages m ON m.id = u.message_id JOIN threads t ON t.id = m.thread_id
GROUP BY t.id, t.model ORDER BY cost_usd DESC LIMIT 20;
```
