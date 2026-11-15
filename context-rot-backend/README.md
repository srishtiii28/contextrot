# Context Rot platform backend — semester 7

## Local setup

The [root README](../README.md) is the step-by-step guide: install, offline vs real-provider mode, start, and which URLs to open. In short, from this directory:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # Windows: py -3 -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000  # Windows: .\.venv\Scripts\python -m uvicorn ...
```

`requirements-tested.txt` records the exact versions of the core packages used for validation. Install it on top of `requirements.txt` if you need those pins. On its own it does not include `chromadb`, `pymupdf` or `matplotlib`.

No `.env` is needed for offline mode (`mock` is an alias for offline). To customise, copy `.env.example` to `.env` and edit it, but do not overwrite an existing `.env`. The SQLite path is relative to this directory, so start the server here.

- Student app: http://127.0.0.1:8000/workspace/
- Developer lab: http://127.0.0.1:8000/demo/lab.html
- API documentation: http://127.0.0.1:8000/docs
- Capability disclosure: http://127.0.0.1:8000/system/capabilities

## Reliability layer

Each in-scope answer is scored in the background worker against the session history, the student's instructions and the retrieved course text. The worker stores a health score H (0–100) and flags with evidence. Set `RELIABILITY_DETECTOR`:

- `rules` (default): the offline demo baseline. It catches only the scripted faults.
- `judge`: v1, one LLM-judge call per answer. Set `JUDGE_PROVIDER`/`JUDGE_MODEL` to `groq` or `gemini`.
- `nli`: v2, local NLI and MiniLM checks. Run `pip install -r requirements-nli.txt` first.

Evaluate and fit thresholds with `python scripts/eval_reliability.py`. See [docs/RELIABILITY.md](../docs/RELIABILITY.md).

## Key endpoints

- `POST /auth/signup`, `POST /auth/login`, `GET /users/me`
- `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/history`
- `POST /sessions/{id}/messages`, `POST /sessions/{id}/messages/stream`
- `GET /sessions/{id}/health`, `GET /sessions/{id}/analysis`
- `POST /integrations/rag/retrieve`, `POST /integrations/rot/analyze` (team integration contracts)
- `POST /integrations/sessions/{id}/documents`, `GET /integrations/sessions/{id}/documents`
- `POST /knowledge-bases`, `GET /knowledge-bases`
- `POST /knowledge-bases/{id}/documents`, `GET /knowledge-bases/{id}/documents`

## Run checks

```bash
.venv/bin/python -m pytest -q        # Windows: .\.venv\Scripts\python -m pytest -q
# Start the server (offline mode) in another terminal before the following:
.venv/bin/python scripts/demo_check.py
.venv/bin/python scripts/load_test.py --concurrency 20 --rounds 3
```

Generated reports go into `reports/`. The demo-check script creates an isolated demo account, exercises the examiner path, saves non-secret evidence and revokes its temporary API key. Load testing creates test sessions and records first-content and completion timing separately.

## Separate worker

Default: one database-polled worker runs within the API process, so the demo needs only one command. For a separate process, set `EMBEDDED_WORKER=false` in both terminals, then run:

```bash
.venv/bin/python -m app.services.worker      # Windows: .\.venv\Scripts\python -m app.services.worker
```

## Gemini provider

Set the following in `.env` and restart both processes:

```dotenv
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-private-key
GEMINI_MODEL=an-exact-model-id-enabled-in-your-account
ENABLE_DEMO_FAULTS=false
```

## Teammate modules

Configure full endpoint URLs in `RAG_SERVICE_URL` and `ROT_SERVICE_URL`; optional shared service credentials use `INTEGRATION_API_KEY`. See [contracts](../docs/INTEGRATION_CONTRACTS.md).
