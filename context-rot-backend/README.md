# Context Rot Detection API

Member 1 backend for the AI tutoring platform. It owns authentication, tutoring sessions, chat history, and stable integration points for the RAG and context-rot modules.

## Run locally

```powershell
cd context-rot-backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation. The default `LLM_PROVIDER=mock` needs no API key and makes the chat pipeline testable immediately.

## RAG slice

The backend supports the complete prototype flow:

1. Create an account and a reference library.
2. Upload platform-owned PDFs to `POST /knowledge-bases/{id}/documents`.
3. Create a study session linked to that library.
4. Send a question to `POST /sessions/{session_id}/messages`.
5. The response includes the grounded answer and the retrieved source pages in `sources`.

Documents are extracted with `pypdf`, split into overlapping chunks, embedded locally, and stored in SQLite. Retrieval is session-scoped, so one study session cannot use another session's documents.

The demo uses the authenticated local user as both content manager and learner. In a client integration, these content-management routes should be protected by the client's admin/author role, while learner sessions should be created with the course's `knowledge_base_id`. Legacy session document endpoints remain available for backwards compatibility but are not used by the learner UI.


To use Groq instead of the deterministic local mock:

```powershell
Copy-Item .env.example .env
# Set LLM_PROVIDER=groq and GROQ_API_KEY=<your key> in .env
uvicorn app.main:app --reload
```

Get a key from [console.groq.com/keys](https://console.groq.com/keys). Keep it only in `.env`; never commit that file.

## Key endpoints

- `POST /auth/signup`, `POST /auth/login`, `GET /users/me`
- `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/history`
- `POST /sessions/{id}/messages`
- `GET /sessions/{id}/health`, `GET /sessions/{id}/analysis`
- `POST /integrations/rag/retrieve`, `POST /integrations/rot/analyze` (team integration contracts)
- `POST /integrations/sessions/{id}/documents`, `GET /integrations/sessions/{id}/documents`
- `POST /knowledge-bases`, `GET /knowledge-bases`
- `POST /knowledge-bases/{id}/documents`, `GET /knowledge-bases/{id}/documents`

## Integration contracts

Member 3 can replace `app/integrations/rag_client.py` with a real retriever that returns context chunks. Member 2 can replace `app/integrations/rot_client.py` with their analyzer; its results are persisted as `analysis_results` and returned by the health APIs. Keep the existing request/response shapes stable for the frontend.
