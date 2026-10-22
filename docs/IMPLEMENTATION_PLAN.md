# Implementation Plan: Multi-Turn Chat with File Q&A

A multi-turn chat application where users can ask follow-up questions in the same thread, upload files, and get answers grounded in those files. Built to keep token cost low without lowering answer quality, and to scale horizontally.

## Contents

1. [Stack and key decisions](#1-stack-and-key-decisions)
2. [Token optimization](#2-token-optimization)
3. [Scalability design](#3-scalability-design)
4. [Data model](#4-data-model)
5. [Project structure](#5-project-structure)
6. [API surface](#6-api-surface)
7. [Phases](#7-phases)
8. [Open decisions](#8-open-decisions)

---

## 1. Stack and key decisions

| Area | Choice | Why |
|---|---|---|
| Backend | Python, FastAPI (async) | Async streaming, Pydantic validation, servers keep no state so they scale horizontally |
| LLM | OpenRouter via the `openai` Python SDK (`base_url` set to OpenRouter) | Many models behind one key; wrapped in an `LLMProvider` interface so the provider can be swapped |
| Database | Postgres + pgvector | Concurrent writes, `JSONB` for message content, vectors in the same database as threads and files |
| Background jobs | Redis + ARQ | Async-native; runs file processing and summarization outside the request |
| File storage | Local disk behind a `Storage` interface | Meets the "store locally for now" requirement; S3 can replace it later without other code changes |
| Migrations | Alembic | Schema changes are versioned from day one |
| Config | pydantic-settings | Typed configuration from environment variables |

---

## 2. Token optimization

Ordered by expected savings. None of these remove information the model needs to answer well.

| # | Lever | How |
|---|---|---|
| 1 | Prompt caching | Keep the request prefix byte-stable (rules below). Cache reads cost about 0.1× normal input on Claude models. |
| 2 | Sticky routing | Send `session_id = thread_id` on every request so OpenRouter keeps the thread on the same provider and its cache stays warm. **Do not set `provider.order`**: it disables sticky routing. |
| 3 | Caching that depends on the model | Anthropic and Qwen models need `cache_control` markers. OpenAI, DeepSeek, Gemini 2.5+ and Grok cache automatically. A `CachePolicy` per model family adds markers only where needed. |
| 4 | Extract file text locally, once | Parse PDF/DOCX to text at upload time and store it. Avoids OpenRouter's per-request PDF parsing fees (Mistral OCR: $2 per 1,000 pages), keeps the text byte-identical so it caches, and gives every model the same input. Native PDF or OCR is used only as a fallback for scanned files. |
| 5 | Retrieval for large files | Files under a threshold (about 20K tokens) go into the conversation whole. Larger files are chunked and embedded, and only the relevant chunks are sent. |
| 6 | Summarize old history in rare, large steps | When a thread passes about 60% of the model's context window, a cheap model summarizes the oldest turns into a checkpoint. Done rarely and in large steps, so the cache is rebuilt only once per summary. |
| 7 | Cheap model for side tasks | Thread titles, summaries and retrieval query rewrites go to a small model. The main model only answers the user. |
| 8 | Output limits | `max_tokens` set per route, reasoning effort set explicitly, reasoning text never resent as visible content. |

### Avoid (these lower quality)

- **OpenRouter's `middle-out` transform.** It silently drops the middle of the conversation. Keep it disabled.
- **Truncating file text.**
- **Re-summarizing on every turn.** It breaks the cache every time and repeatedly loses detail.

### Prompt layout

All prompt assembly happens in `context_builder.py`, and tests check it.

```
[system: frozen text, no timestamps or user data]  <- cache_control (Anthropic/Qwen)
[summary checkpoint, if any]
[user: file text as it was uploaded + question]    <- files appended as turns, never put in the system prompt
[assistant] [user] ...
[user: new question + retrieved chunks, if any]    <- automatic top-level cache_control
```

**Rules:**

- History is append-only. Earlier messages are never edited, reordered or deleted, except by a compaction step.
- The system prompt contains nothing that changes per request: no `datetime.now()`, request IDs or user data.
- Uploaded files are added as a user turn at the point of upload, never to the system prompt.
- Retrieved chunks go only into the current user turn, so the cached prefix is untouched.
- A re-uploaded file (same name, new content) is added as a new turn tagged with its version, e.g. `<file name="report.pdf" version="2" replaces="1">`. The earlier version's turn stays in history so the cache is kept; the frozen system prompt says only the latest version of a file is current, and the next compaction drops the superseded text.
- Store the full assistant message, including `reasoning_details`, and send it back unchanged.

### Measurement

OpenRouter returns these fields automatically, in the last SSE event when streaming. Save them for every message:

- `usage.prompt_tokens`
- `usage.prompt_tokens_details.cached_tokens`
- `usage.prompt_tokens_details.cache_write_tokens`
- `usage.completion_tokens`
- `usage.completion_tokens_details.reasoning_tokens`
- `usage.cost`

Cache hit rate (`cached_tokens / prompt_tokens`) is a tracked metric. A drop means something in the prefix is changing.

### Token counting

OpenRouter has no general count-tokens endpoint. Estimate locally with `tiktoken`, then adjust each thread's estimate using the real `prompt_tokens` from the previous response.

---

## 3. Scalability design

```
            ┌──────────── Load balancer ────────────┐
            ▼                                       ▼
   FastAPI (stateless) × N                 FastAPI (stateless) × N
      │      │       │
      │      │       └── enqueue jobs ──▶ Redis ──▶ ARQ workers × M
      │      │                                      (extract, chunk, embed,
      │      │                                       summarize, title)
      │      └── Storage interface ──▶ Local disk (later S3)
      └── Postgres + pgvector (threads, messages, files, chunks, usage)
                     │
                     ▼
               OpenRouter API
```

- **Stateless servers.** The API keeps no state in memory; every chat request rebuilds its context from Postgres. Scale by running more copies.
- **Slow work runs in the background.** Uploads return `202 Accepted`. A worker extracts, counts, chunks and embeds the file, and moves its status through `pending → processing → ready | failed`.
- **Multi-user from the start.** `user_id` is on every table now, even before login exists.
- **Rate limits and budgets.** A Redis token bucket limits requests per user; each user or thread has a token or cost budget.
- **Resilience.** Timeouts and retries with backoff on 429 and 5xx errors, plus OpenRouter's `models: [...]` fallback list for when a model is unavailable.

---

## 4. Data model

| Table | Key fields |
|---|---|
| `users` | id, email, created_at |
| `threads` | id, user_id, title, model, summary_checkpoint_msg_id, est_context_tokens, created_at |
| `messages` | id, thread_id, seq, role, content_json (full message incl. `reasoning_details`), model, created_at |
| `files` | id, thread_id, user_id, filename, storage_key, mime, size, sha256, token_count, mode (`inline` / `rag`), status, version, superseded_by (file id, null while current), created_at |
| `file_chunks` | id, file_id, ordinal, text, token_count, embedding `vector(n)` |
| `usage` | message_id, prompt_tokens, cached_tokens, cache_write_tokens, completion_tokens, reasoning_tokens, cost, provider |

**Indexes and constraints:**

- `messages (thread_id, seq)`, unique
- `files (thread_id, sha256)`, unique, to stop the same file being stored twice in a thread
- `files (thread_id, filename, version)`, unique, so each version number is taken once even under concurrent uploads
- HNSW index on `file_chunks.embedding`

---

## 5. Project structure

```
backend/
  app/
    main.py
    config.py                 # pydantic-settings
    api/                      # routes only, thin
      threads.py
      messages.py
      files.py
      usage.py
    services/
      chat_service.py         # orchestrates a turn
      context_builder.py      # the one place prompt layout and cache markers live
      compaction_service.py   # summary checkpoints
      file_service.py         # upload, dedupe, status
      retrieval_service.py    # pgvector search
    llm/
      base.py                 # LLMProvider protocol
      openrouter.py           # AsyncOpenAI(base_url=OpenRouter), streaming, retries
      cache_policy.py         # cache_control rules per model family
      tokens.py               # estimation and calibration
    storage/
      base.py
      local.py                # s3.py later
    ingestion/
      extractors.py           # pdf/docx/txt/md/csv -> text
      chunker.py
      embedder.py
    workers/
      tasks.py                # ARQ jobs
    db/
      models.py
      session.py
      repositories/
  alembic/
  tests/
docker-compose.yml            # postgres (pgvector), redis, api, worker
```

---

## 6. API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/threads` | Create a thread |
| `GET` | `/threads` | List threads (Phase 4) |
| `GET` | `/threads/{id}/messages` | Thread history |
| `POST` | `/threads/{id}/messages` | Send a message; reply streamed over SSE |
| `POST` | `/threads/{id}/files` | Upload a file; returns `202` |
| `GET` | `/threads/{id}/files` | List files and their processing status |
| `GET` | `/threads/{id}/usage` | Tokens, cost and cache hit rate |
| `GET` | `/health` | Health check |

---

## 7. Phases

### Phase 0: Foundations (about 1–2 days)

- [x] Docker Compose: Postgres (pgvector), Redis, API, worker
- [x] pydantic-settings config, structured logging, `/health`
- [x] SQLAlchemy async models and initial Alembic migration
- [x] pytest setup with a test database
- [x] `LLMProvider` interface and OpenRouter client with streaming and retries

**Done when:**

- [x] `docker compose up` brings the stack up.
- [x] A smoke test streams a reply from OpenRouter (`pytest -m live`).

### Phase 1: Multi-turn chat in a single thread (about 3–4 days)

- [x] `POST /threads`, `GET /threads/{id}`, `GET /threads/{id}/messages`, `GET /models`
- [x] `POST /threads/{id}/messages`: SSE streaming; saves the full message and usage on completion
- [x] `context_builder` with frozen system prompt, `CachePolicy` and `session_id`
- [x] Usage tracking per message (tokens, cached tokens, cost)
- [x] Minimal UI (Streamlit, `ui/`)
- [x] One turn at a time per thread (Redis lock, `409` otherwise); a turn is saved all-or-nothing

**Done when:**

- [x] A multi-turn conversation remembers earlier context (verified live, 4 turns).
- [x] Cache hits appear once the prompt passes the model's minimum cacheable size (512 tokens on Claude 5.x). Verified live: turn 3 wrote 528 tokens to the cache, and turn 4 read those 528 and wrote only the new 90. *(The original "from turn 2" target assumed a longer system prompt; short first turns are below the minimum.)*
- [x] A test confirms that turns 1 through N−1 are identical across consecutive requests.

### Phase 2: File upload and file Q&A (about 4–5 days)

> **Scope change (decided 2026-10-22):** uploads are small (about 5 pages), so every file is sent to the model in full and there is no RAG, embeddings or `file_chunks` table yet. Files over `MAX_FILE_TOKENS` (20,000 tokens, roughly 25 pages) are rejected with a clear message. Extraction takes milliseconds for files this size, so it runs during the upload request (in a thread pool) rather than in the background worker, and uploads return `201` with the file ready. Both can be revisited if larger files become a requirement.

- [x] `POST /threads/{id}/files`: validate type (by extension and file content) and size, store the file, reject identical content (sha256), return `201`
- [x] Text extraction for PDF (with page markers), DOCX (paragraphs and tables), TXT, MD and CSV; scanned or encrypted PDFs are rejected with a clear message
- [x] Files attached to the user's next message as `<file name="..." version="...">` blocks: never in the system prompt, and never re-sent
- [ ] ~~RAG files: top-k chunk retrieval per question~~ *(deferred, see the scope change above)*
- [x] File versioning: a same-name upload with different content becomes the next version, and the previous one is marked `superseded_by` the new file. A database index guarantees one current file per name.
- [x] Superseded versions are kept, so "what changed between v1 and v2?" can be answered. A new version is marked `replaces="version N"` so the model answers from it, and earlier turns stay unchanged.
- [x] UI: on a same-name upload, ask "Replace, keep both, or cancel?". "Keep both" stores the upload as `name (2).ext`, not as a version.
- [x] File status and version shown in the UI (e.g. "menu.txt updated (v2)"), plus files waiting to be sent and attachments on each message
- [x] Prompt rule: answers cite the file name and page or section

**Done when:**

- [x] The model answers correctly from a 5-page PDF (verified live, with page citations). *The 300-page target is out of scope; see above.*
- [x] A second question about the same file reads it from cache (verified live: 1,456 tokens read from cache, cost down from $0.00021 to $0.00005).
- [x] Uploading the same file twice to a thread is rejected (`409 file_duplicate`, even under a different name).
- [x] After an edited file is re-uploaded under the same name, answers come from the new version only, and earlier turns stay identical (verified by tests and live: the v2 answer said "Porto", and the model noted that v1 had said "Lisbon").

### Phase 3: Long context and cost control (about 3 days)

- [ ] Estimated context size per thread, adjusted from real usage
- [ ] Compaction above 60% of the model's context window: a cheap model summarizes the oldest turns into a checkpoint; file content is kept or re-referenced, not summarized away, except superseded file versions, which are dropped
- [ ] Per-user rate limits and per-thread budgets in Redis
- [ ] `GET /threads/{id}/usage`

**Done when:**

- A thread past the threshold still answers questions about its first turn correctly.
- Cost per turn levels off instead of growing with the length of the thread.

### Phase 4: Multiple threads (about 2 days)

- [ ] `GET /threads` with pagination
- [ ] Rename and delete threads
- [ ] Auto-generated titles (cheap model)
- [ ] Thread sidebar in the UI

**Done when:** switching threads keeps each thread's context and files separate.

### Phase 5: Production hardening (ongoing)

- [ ] Authentication (JWT or OAuth), with `user_id` taken from the token
- [ ] S3 implementation of `Storage`
- [ ] OpenTelemetry tracing; dashboard for cost and cache hit rate
- [ ] Load testing (k6 or Locust); horizontal scaling of API and workers
- [ ] Evaluation set of question/file pairs, so model and prompt changes are checked for quality as well as cost

---

## 8. Open decisions

| # | Decision | Status | Affects |
|---|---|---|---|
| 1 | Default chat model on OpenRouter | **Decided:** `anthropic/claude-haiku-5.5`; `anthropic/claude-sonnet-5.5` also allowed per thread | Default `CachePolicy`, cost per turn |
| 2 | Embedding model for RAG | Open: an embedding model through OpenRouter, or a local one (e.g. `bge-small`) | Per-call cost, `vector(n)` dimension |
| 3 | Frontend | **Decided:** Streamlit | Phase 1 UI effort |
| 4 | File types for version 1 | Open: PDF, DOCX, TXT, MD, CSV; OCR for scanned PDFs? | Extractors, worker dependencies |
