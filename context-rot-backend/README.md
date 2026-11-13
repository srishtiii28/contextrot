# Context Rot platform backend — semester 7

## Local setup

Run `Start-Demo.ps1` from the project root. Or, from this directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For exact versions from the tested environment, install `requirements-tested.txt` instead of the broad requirements file.

No `.env` is needed for offline mode (`mock` is retained as an alias for offline). To customize, copy `.env.example` to `.env` and edit it. Do not overwrite an existing `.env`. The SQLite path is relative to this backend directory, so start the server here.

- Demo: http://127.0.0.1:8000/demo/lab.html
- API documentation: http://127.0.0.1:8000/docs
- Capability disclosure: http://127.0.0.1:8000/system/capabilities
- Preserved tutor demo client: http://127.0.0.1:8000/demo/

## RAG & Knowledge Retrieval Slice

The backend supports the complete prototype flow:

1. Create an account and a reference library (`POST /knowledge-bases`).
2. Upload course PDFs to `POST /knowledge-bases/{id}/documents`.
3. Create a study session linked to that library.
4. Send a question to `POST /sessions/{session_id}/messages`.
5. The response includes the grounded answer and the retrieved source pages in `sources`.

Documents are extracted with `pypdf`, split into overlapping chunks, embedded locally, and stored in SQLite / ChromaDB. Retrieval is session-scoped, so one study session cannot use another session's documents.

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

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# Start the server in another terminal before the following:
.\.venv\Scripts\python.exe scripts/demo_check.py
.\.venv\Scripts\python.exe scripts/load_test.py --concurrency 20 --rounds 3
```

Generated reports go into `reports/`. The demo-check script creates an isolated demo account, exercises the examiner path, saves non-secret evidence and revokes its temporary API key. Load testing creates test sessions and records first-content and completion timing separately.

## Separate worker

Default: one database-polled worker runs within the API process, so the demo needs only one command. For a separate process, set `EMBEDDED_WORKER=false` in both terminals, then run:

```powershell
.\.venv\Scripts\python.exe -m app.services.worker
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
