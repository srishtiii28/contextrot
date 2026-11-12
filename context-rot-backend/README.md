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
- Earlier preserved client: http://127.0.0.1:8000/demo/

## Run checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# Start the server in another terminal before the following:
.\.venv\Scripts\python.exe scripts/demo_check.py
.\.venv\Scripts\python.exe scripts/load_test.py --concurrency 20 --rounds 3
```

Generated reports go into `reports/`. The demo-check script creates an isolated demo account, exercises the examiner path, saves non-secret evidence and revokes its temporary API key. Load testing creates test sessions and records first-content and completion timing separately. No test claims detector F1 or vector retrieval quality.

## Separate worker

Default: one database-polled worker runs within the API process, so the demo needs only one command. For a separate process, set `EMBEDDED_WORKER=false` in both terminals, then run:

```powershell
.\.venv\Scripts\python.exe -m app.services.worker
```

Use the same working directory, database and provider settings for API and worker. Jobs are persistent; expired leases can be reclaimed. Do not use Uvicorn `--reload` for load-test measurements. SQLite is the supported local demonstration database; PostgreSQL migration and distributed hardening remain future work.

## Gemini provider

Set the following in `.env` and restart both processes:

```dotenv
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-private-key
GEMINI_MODEL=an-exact-model-id-enabled-in-your-account
ENABLE_DEMO_FAULTS=false
```

Model IDs are intentionally not hardcoded because account availability changes. Non-streaming requests use `generateContent`; streaming requests use `streamGenerateContent?alt=sse`. Implementation follows [Google's generation API](https://ai.google.dev/api/generate-content). Provider integration is contract-tested with mocked HTTP; a real paid call requires your credentials and has not been performed. Offline mode makes no model-network requests.

## Teammate modules

Configure full endpoint URLs in `RAG_SERVICE_URL` and `ROT_SERVICE_URL`; optional shared service credentials use `INTEGRATION_API_KEY`. Do not point these URLs at this API's own integration routes. See [contracts](../docs/INTEGRATION_CONTRACTS.md).

The default local corpus is authored DBMS material with lexical matching, and the checker is a narrow rule baseline. They make the backend independently demonstrable; substitute the team's approved course index and validated detector before claiming final research results.

## Data and security

Existing domain tables remain intact; new runtime tables are additive. Back up the SQLite database before manual data changes. User passwords use PBKDF2; legacy bcrypt hashes can still be verified. JWTs expire; API keys can be revoked. Set a private random JWT secret for shared deployments and bind publicly only after agreeing the deployment scope. The launcher binds loopback by default.

Root documentation contains the [demo guide](../docs/DEMO_GUIDE.md), [personal implementation checklist](../docs/ARIJIT_SEMESTER_7.md), [measurements](../docs/VALIDATION.md) and [eighth-semester plan](../docs/SEMESTER_8_PLAN.md).
