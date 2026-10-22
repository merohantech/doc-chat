# Chat API

A multi-turn chat application where users can ask follow-up questions in the same thread and, in later phases, upload files and ask questions about them. The backend is FastAPI, the UI is Streamlit, and the LLM is reached through [OpenRouter](https://openrouter.ai).

- Architecture and roadmap: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- **Current status:** Phases 1 and 2 are complete: multi-turn chat in a single thread with streaming replies, saved history, prompt caching and per-message token and cost tracking, plus file upload (PDF, DOCX, TXT, MD, CSV) with questions answered from the files and file versioning.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Get the code](#2-get-the-code)
3. [Configure environment variables](#3-configure-environment-variables)
4. [Run the application](#4-run-the-application)
5. [Verify it works](#5-verify-it-works)
6. [Run the tests](#6-run-the-tests)
7. [Everyday commands](#7-everyday-commands)
8. [Troubleshooting](#8-troubleshooting)
9. [Project structure](#9-project-structure)

---

## 1. What you need

### Software

| Tool | Version | Needed for | Install |
|---|---|---|---|
| Docker Desktop (includes Docker Compose) | Recent | Running Postgres, Redis, the API and the worker | https://www.docker.com/products/docker-desktop |
| Git | Any recent | Getting the code | https://git-scm.com/downloads |
| Python | **3.12** | Only for running the API or tests outside Docker | https://www.python.org/downloads |

Check them:

```bash
docker --version
docker compose version
python --version      # on Windows: py -3.12 --version
```

Make sure Docker Desktop is **running** before you continue.

### Accounts and API keys

You need exactly **one** external key:

| Service | What it's for | How to get it | Cost |
|---|---|---|---|
| **OpenRouter** | Access to the LLM (default model: `anthropic/claude-haiku-5.5`) | Sign up at https://openrouter.ai, add credits, then create a key at https://openrouter.ai/keys | Pay per use; a test call costs a fraction of a cent |

Setting a spending limit on the key is recommended.

### Database and other services

You **don't** need to install or sign up for anything else. Docker Compose runs these locally:

| Service | Image | Purpose |
|---|---|---|
| Postgres 17 + pgvector | `pgvector/pgvector:pg17` | Threads, messages, files and usage data; vector search later |
| Redis 7 | `redis:7-alpine` | Background job queue; rate limiting later |

Docker creates both databases (`chat` for the app, `chat_test` for the test suite) automatically on first start.

---

## 2. Get the code

```bash
git clone <repository-url> chat-app
cd chat-app
```

---

## 3. Configure environment variables

All configuration lives in `backend/.env`. Start from the template:

```bash
cp backend/.env.example backend/.env
```

Open `backend/.env` and set **`OPENROUTER_API_KEY`**. Every other value has a working default.

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...your-key...
```

> `backend/.env` holds secrets. Git ignores it and it is excluded from Docker images. Never commit it.

### All variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(none, required)* | Your OpenRouter key. The API refuses to start without it. |
| `DEFAULT_CHAT_MODEL` | `anthropic/claude-haiku-5.5` | Model that answers users. Any [OpenRouter model ID](https://openrouter.ai/models). |
| `ALLOWED_CHAT_MODELS` | `["anthropic/claude-haiku-5.5","anthropic/claude-sonnet-5.5"]` | Models a thread may use (JSON list). Must include `DEFAULT_CHAT_MODEL`. |
| `UTILITY_MODEL` | `anthropic/claude-haiku-5.5` | Cheap model for side tasks (titles, summaries). |
| `CHAT_MAX_TOKENS` | `4096` | Maximum length of one reply, in tokens. |
| `MAX_MESSAGE_CHARS` | `32000` | Maximum length of one user message, in characters. |
| `PROMPT_CACHE_TTL` | `5m` | `5m` or `1h`. Use `1h` if users often pause 5–60 minutes between messages. |
| `DEV_USER_EMAIL` | `dev@localhost` | Until authentication exists, every request acts as this user. |
| `APP_ENV` | `dev` | `dev` (coloured logs), `test`, or `prod` (JSON logs). |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `DATABASE_URL` | `postgresql+asyncpg://chat:chat@localhost:5433/chat` | App database. Docker Compose overrides it inside containers. |
| `TEST_DATABASE_URL` | `postgresql+asyncpg://chat:chat@localhost:5433/chat_test` | Database used by the test suite. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection. Docker Compose overrides it inside containers. |
| `OPENROUTER_APP_NAME` | `chat-backend` | Shown on your OpenRouter activity page. |
| `OPENROUTER_APP_URL` | *(empty)* | Optional; sent to OpenRouter for attribution. |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeout per LLM request. |
| `LLM_MAX_RETRIES` | `2` | Automatic retries on rate limits and server errors. |
| `STORAGE_DIR` | `storage/uploads` | Where uploaded files are stored. |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum upload size in bytes (10 MB). |
| `MAX_FILE_TOKENS` | `20000` | Maximum extracted text per file, in estimated tokens (roughly 25 pages). Files are sent to the model in full, so this keeps cost bounded. |

> **Why port 5433?** The project's Postgres is published on host port **5433**, not the usual 5432, so it doesn't clash with a Postgres you may already have installed. To use another port, set `POSTGRES_HOST_PORT` in your shell and update the two database URLs above to match.

---

## 4. Run the application

There are two ways to run it. Use **Option A** unless you're actively changing backend code.

### Option A: everything in Docker (recommended)

From the repository root:

```bash
docker compose up --build
```

This starts six services in order:

| Service | What it does |
|---|---|
| `postgres` | Database, on host port 5433 |
| `redis` | Job queue and per-thread locks, on host port 6379 |
| `migrate` | Creates or updates the database tables (`alembic upgrade head`), then exits |
| `api` | The FastAPI server at **http://localhost:8000** |
| `worker` | Background job worker |
| `ui` | The chat UI at **http://localhost:8501** |

**Open http://localhost:8501 and start chatting.**

To run in the background instead, add `-d`: `docker compose up --build -d`.

Stop everything with `Ctrl+C`, or `docker compose down` if you used `-d`.

### Option B: API on your machine, databases in Docker (for development)

This gives you auto-reload on code changes and lets you use a debugger.

**Step 1: start only Postgres and Redis**

```bash
docker compose up -d postgres redis
```

**Step 2: create a virtual environment and install dependencies**

Windows (PowerShell):

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

macOS / Linux:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 3: create the database tables**

```bash
alembic upgrade head
```

**Step 4: start the API** (stays in the foreground)

```bash
uvicorn app.main:create_app --factory --reload
```

**Step 5: start the UI** in a second terminal. The UI has its own virtual environment:

Windows (PowerShell):

```powershell
cd ui
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
streamlit run app.py
```

macOS / Linux:

```bash
cd ui
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

The UI opens at http://localhost:8501 and talks to the API at `http://localhost:8000`. Set the `API_URL` environment variable to point it elsewhere.

**Step 6 (optional): start the worker** in a third terminal, with the backend virtual environment activated. Nothing uses it yet; it is reserved for background jobs such as conversation compaction (Phase 3).

```bash
cd backend
arq app.workers.settings.WorkerSettings
```

> If you use VS Code, select `backend/.venv` as the Python interpreter (*Python: Select Interpreter*). Otherwise the editor reports packages as missing.

---

## 5. Verify it works

With the API running (either option):

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/health/ready
# {"status":"ok","checks":{"database":"ok","redis":"ok"}}
```

| URL | What you should see |
|---|---|
| http://localhost:8501 | The chat UI. Send a message, then a follow-up that refers to it. Attach a file from the sidebar and ask about it. |
| http://localhost:8000/health | Liveness: the process is up |
| http://localhost:8000/health/ready | Readiness: database and Redis reachable (`503` if not) |
| http://localhost:8000/docs | Interactive API documentation (Swagger UI) |

Under each reply, the UI shows tokens in, cached tokens, tokens out and cost. The sidebar shows the thread's total cost and cache hit rate. Caching only starts once a conversation passes the model's minimum cacheable size (512 tokens on Haiku 5.5), so the first couple of short turns show 0 cached tokens.

### Try the API directly

```bash
# Create a thread
curl -s -X POST http://localhost:8000/threads -H "Content-Type: application/json" -d '{}'
# -> {"id":"<thread-id>", "model":"anthropic/claude-haiku-5.5", ...}

# Send a message; the reply streams back as server-sent events
curl -N -X POST http://localhost:8000/threads/<thread-id>/messages \
  -H "Content-Type: application/json" -d '{"content":"Hello!"}'

# Read the history, with usage per assistant message
curl -s http://localhost:8000/threads/<thread-id>/messages
```

| Endpoint | Purpose |
|---|---|
| `GET /models` | Default and allowed models |
| `POST /threads` | Create a thread; body `{"title"?, "model"?}` |
| `GET /threads/{id}` | Thread details, including estimated context size |
| `GET /threads/{id}/messages` | Full history, with token and cost usage on assistant messages |
| `POST /threads/{id}/messages` | Send a message; the reply streams as `delta` events, then one `done` (or `error`) event. Files uploaded since the last message are attached to it. |
| `POST /threads/{id}/files` | Upload a file (multipart field `file`; optional form field `on_conflict` = `replace` or `keep_both`) |
| `GET /threads/{id}/files` | All files in the thread, including superseded versions |

### How files work

1. Upload a file. It's stored, its text is extracted, and it waits for your next message.
2. Your next message carries the file's full text to the model, together with your question. After that the file is part of the conversation history and is read from the prompt cache on later turns, at about a tenth of the normal input price.
3. Uploading identical content again is rejected. Uploading a **different** file with the **same name** asks whether to replace it (it becomes version 2, and the model is told it replaces version 1) or keep both (the upload is stored as `name (2).ext`).

Scanned PDFs (images with no text layer) are rejected; OCR is not supported.

Errors return `{"detail": {"code": ..., "message": ...}}`:

| Status | Code | Meaning |
|---|---|---|
| 404 | `thread_not_found` | Thread doesn't exist (or belongs to another user) |
| 409 | `thread_busy` | A reply is already streaming in this thread |
| 409 | `file_duplicate` | This exact file is already in the thread |
| 409 | `file_name_conflict` | A different file with this name exists; retry with `on_conflict` |
| 413 | `file_too_large` | Over `MAX_UPLOAD_BYTES` or `MAX_FILE_TOKENS` |
| 422 | `model_not_allowed`, `message_too_long` | Invalid request |
| 422 | `file_type_unsupported`, `file_unreadable` | Not a supported type, or no extractable text (e.g. a scanned PDF) |
| 429 | `llm_rate_limited` | OpenRouter rate limit; retry shortly |
| 502 / 503 | `llm_request_rejected` / `llm_unavailable` | Provider rejected the request, or is unavailable |

Check that your OpenRouter key works (from `backend/`, with the virtual environment activated):

```bash
pytest -m live
# 1 passed
```

---

## 6. Run the tests

From `backend/`, with the virtual environment activated, and with Postgres and Redis running (`docker compose up -d postgres redis`):

```bash
pytest                  # unit + integration tests (no LLM cost)
pytest -m live          # real call to OpenRouter (costs a fraction of a cent)
ruff check .            # lint
ruff format --check .   # formatting
mypy app tests          # strict type checking
```

The tests use the separate `chat_test` database and Redis database 15, so they never touch your development data. A test also fails if a database model changes without a matching migration.

UI tests, from `ui/` with its virtual environment activated:

```bash
pytest
```

---

## 7. Everyday commands

| Task | Command |
|---|---|
| Start everything | `docker compose up --build -d` |
| Stop everything | `docker compose down` |
| See logs | `docker compose logs -f api worker` |
| Restart the API after changing `.env` | `docker compose up -d --force-recreate api worker` |
| Apply new migrations | `docker compose run --rm migrate`, or `alembic upgrade head` from `backend/` |
| Create a migration after changing models | `alembic revision --autogenerate -m "describe change"`, **then review the generated file** |
| Open a database shell | `docker compose exec postgres psql -U chat -d chat` |
| **Reset all data** (deletes the database) | `docker compose down -v` |

---

## 8. Troubleshooting

**`password authentication failed for user "chat"`**
You're connecting to a different Postgres, usually one installed locally on port 5432. Make sure both database URLs in `backend/.env` use port **5433**.

**The `api` container exits immediately and logs `ValueError: OPENROUTER_API_KEY is not set`**
Set the key in `backend/.env`, then run `docker compose up -d --force-recreate api worker`.

**Tests fail with `database "chat_test" does not exist`**
The test database is created only when the Postgres volume is first initialised. If the volume existed before, create the database manually:
```bash
docker compose exec postgres createdb -U chat chat_test
```
Alternatively, reset everything with `docker compose down -v` (this deletes all data).

**`port is already allocated` when starting Docker**
Another program is using port 5433, 6379 or 8000. Stop it, or change the host port in `docker-compose.yml` (and the matching URL in `.env`).

**The UI says "Can't reach the chat API"**
Check that the API is running (`curl http://localhost:8000/health`). If you run the UI outside Docker against a different address, set `API_URL`.

**"A reply is already being generated in this thread" (409)**
Only one reply streams per thread at a time. Wait for it to finish. If a crashed request left the lock behind, it expires on its own after a few minutes.

**`/health/ready` returns 503**
The response shows which check failed. Make sure the containers are running: `docker compose ps`.

**`Activate.ps1 cannot be loaded because running scripts is disabled` (Windows)**
Allow local scripts for your user once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**The editor reports packages as "not installed"**
Point your editor's Python interpreter at `backend/.venv`.

---

## 9. Project structure

```
.
├── docker-compose.yml          # postgres, redis, migrate, api, worker
├── docker/postgres/init/       # SQL run on first database start (creates chat_test)
├── docs/
│   └── IMPLEMENTATION_PLAN.md  # architecture, phases, progress
├── backend/
│   ├── .env.example            # configuration template
│   ├── Dockerfile
│   ├── pyproject.toml          # dependencies and tool config (pytest, ruff, mypy)
│   ├── alembic/                # database migrations
│   ├── app/
│   │   ├── main.py             # app factory, startup and shutdown
│   │   ├── config.py           # settings from environment variables
│   │   ├── api/                # HTTP routes, schemas, error mapping
│   │   ├── services/           # chat + file services; context_builder = prompt layout
│   │   ├── ingestion/          # text extraction from PDF, DOCX, TXT, MD, CSV
│   │   ├── storage/            # where uploaded bytes live (local disk; S3 later)
│   │   ├── db/                 # SQLAlchemy models, sessions, repositories
│   │   ├── llm/                # LLM provider interface, OpenRouter client, cache policy
│   │   └── workers/            # background job worker
│   └── tests/
│       ├── unit/               # no external services
│       └── integration/        # need Postgres and Redis; `live` tests call OpenRouter
└── ui/
    ├── app.py                  # Streamlit chat UI
    ├── api_client.py           # HTTP + SSE client for the backend
    ├── Dockerfile
    └── tests/
```
