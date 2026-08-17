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

## Key endpoints

- `POST /auth/signup`, `POST /auth/login`, `GET /users/me`
- `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/history`
- `POST /sessions/{id}/messages`
- `GET /sessions/{id}/health`, `GET /sessions/{id}/analysis`
- `POST /integrations/rag/retrieve`, `POST /integrations/rot/analyze` (team integration contracts)

## Integration contracts

Member 3 can replace `app/integrations/rag_client.py` with a real retriever that returns context chunks. Member 2 can replace `app/integrations/rot_client.py` with their analyzer; its results are persisted as `analysis_results` and returned by the health APIs. Keep the existing request/response shapes stable for the frontend.

