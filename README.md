# Context Rot Detection — T01 Swifties

An AI exam-prep tutor that answers from a course's own notes, plus a **reliability layer** that checks every answer in the background and corrects answers that fail. The checks cover:
- contradictions with earlier answers
- ignored student instructions
- conflicts with the course material
- answers that drift off the question

Students see each answer's check result, any correction, and a session health score in the chat.

DSC4195 Project II · Amar Sinha, Arijit Samanta, Debayush Dey, Srishti Dutta.

## What runs where

| Part | Folder | What it does |
|---|---|---|
| Backend API | `context-rot-backend/` | FastAPI: accounts, sessions, course libraries (PDF upload and retrieval), tutor, background reliability worker, REST + WebSocket |
| Reliability layer | `context-rot-backend/app/reliability/` | Scores each answer: health score 0–100 plus flags. Low scores get one automatic correction. See [docs/RELIABILITY.md](docs/RELIABILITY.md) |
| Student app | `context-rot-demo/` → `/workspace/` | Chat with your course notes, the reliability checks and corrections, standing instructions, dashboard |
| Developer lab | `context-rot-frontend/lab.html` → `/demo/lab.html` | Bare client for demos with scripted faults (offline mode) |

Both web clients are served by the backend, so there is only one process to start.

## Run it

Needs **Python 3.11+**. All commands run from `context-rot-backend/`.

### 1. Install

macOS / Linux:

```bash
cd context-rot-backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd context-rot-backend
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Optional, only for the local NLI detector (`RELIABILITY_DETECTOR=nli`): `pip install -r requirements-nli.txt`. It pulls in PyTorch and runs on the CPU.

### 2. Choose a mode

**Offline demo (no API keys).** With no `.env` the backend runs offline:
- the tutor answers extractively from built-in DBMS notes
- the reliability layer uses the narrow `rules` baseline
- the scripted fault buttons work

Use this to check the system end to end. It does not show real answer quality.

**Real tutor and judge (Groq).** Copy the example settings and fill in two keys:

```bash
cp .env.example .env
```

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=...            # tutor
GROQ_MODEL=openai/gpt-oss-120b
RELIABILITY_DETECTOR=judge
JUDGE_PROVIDER=groq
JUDGE_API_KEY=...           # judge: use a key from a DIFFERENT Groq account
JUDGE_MODEL=qwen/qwen3.8-27b
JWT_SECRET_KEY=<a long random string>
```

Groq limits apply per account and per model: on the free tier, 8k tokens a minute and 200k a day. Two keys from the same account share one quota. A judge check costs about 1,100 tokens, so one free judge account covers roughly 180 checked answers a day. `.env` is gitignored, so never commit keys.

Gemini works too: set `LLM_PROVIDER=gemini`, `GOOGLE_API_KEY` and `GEMINI_MODEL`, and the same for `JUDGE_*`. Every setting is described in `.env.example`.

### 3. Start

macOS / Linux:

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows: `.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`. Alternatively, from the repository root, `powershell -ExecutionPolicy Bypass -File .\Start-Demo.ps1` creates the environment if needed and starts the server.

Start the server from `context-rot-backend/`: the SQLite database and `.env` are read relative to that folder. The reliability worker runs inside the same process.

### 4. Open

| URL | For |
|---|---|
| http://127.0.0.1:8000/workspace/ | **The student app.** Create an account, then in *Content library* create a library and upload the course PDF. Create a study session on that library (optionally with instructions such as "Answer in at most 3 sentences") and ask questions. |
| http://127.0.0.1:8000/demo/lab.html | Developer lab. *Start local demo account*, then use the fault buttons in offline mode. |
| http://127.0.0.1:8000/docs | API documentation (Swagger) |
| http://127.0.0.1:8000/system/capabilities | Which tutor, model and detector are active |

`http://127.0.0.1:8000/` on its own only returns a status message.

**What a student sees after each answer:**
- "Checking…" while the answer is verified. The next question waits until the check finishes.
- "✓ Verified" with a score, when the answer passes.
- "Replaced" with the reason, followed by a labelled "Corrected answer", when it fails.
- "Still unreliable", when the correction also fails.
- "Unverified" with a Retry button, when the check could not run.
- "Not checked", when the question was treated as outside the notes.

Replies to "thanks" or "ok" are not checked.

## Test and evaluate

From `context-rot-backend/`:

```bash
.venv/bin/python -m pytest -q                      # tests never read .env or call a real API
.venv/bin/python scripts/demo_check.py             # examiner path; needs the server running in offline mode
.venv/bin/python scripts/eval_reliability.py --help
```

`eval_reliability.py` scores labelled sessions with a detector (`rules`, `judge` or `nli`). It reports:
- rot F1 with a bootstrap interval
- false alarms per clean session
- latency and cost
- the v1 vs v2 decision, plus a premise study and recovery rate

Thresholds are fitted on the dev split only, and the held-out set can be scored once. See [docs/RELIABILITY.md](docs/RELIABILITY.md).

Known test state: 11 tests in `test_biology_*` and `test_math_enhancements.py` need a pre-indexed textbook database that is not in the repository, and fail without it.

## Repository layout

```text
context-rot-backend/
  app/core/           settings and authentication
  app/database/       tables; new columns are added to existing SQLite databases at start-up
  app/routes/         REST, streaming (SSE) and WebSocket endpoints
  app/services/       tutor prompts, retrieval, pipeline, background worker
  app/reliability/    rules, judge (v1), nli (v2), health score, evaluation
  app/integrations/   replaceable retrieval and detector boundaries
  scripts/            demo check, load test, reliability evaluation
  tests/
context-rot-demo/     student app (/workspace/)
context-rot-frontend/ developer lab (/demo/lab.html) and an older client
context-rot-extension/ earlier browser extension, not needed to run the system
docs/                 reliability layer, integration contracts, validation results, plans
Start-Demo.ps1        Windows launcher
```

## Further reading

- [Reliability layer: definition of rot, detectors, health score, evaluation](docs/RELIABILITY.md)
- [Validation results, including the real-provider audit](docs/VALIDATION.md)
- [API and teammate integration contracts](docs/INTEGRATION_CONTRACTS.md)
- [Seven-minute demo walkthrough](docs/DEMO_GUIDE.md)
- [Backend details: endpoints, separate worker, teammate services](context-rot-backend/README.md)

## Data and keys

`.gitignore` keeps `.env`, databases (`*.db`), vector stores (`chroma_db/`, `chroma_store/`) and PDFs out of git. Uploaded course PDFs are stored in the database, not as files. Course material is used with permission and is not redistributed.
