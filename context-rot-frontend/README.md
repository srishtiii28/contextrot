# Context Compass frontend (Member 4)

A dependency-free frontend for the Context Rot Detection API. It supports account creation and login, study-session management, chat, conversation-health monitoring, alerts, and latest analysis signals.

## Run locally

1. Start the backend from `../context-rot-backend`:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload
   ```

2. In this `context-rot-frontend` directory, serve the static app:

   ```powershell
   py -m http.server 5500
   ```

3. Open `http://127.0.0.1:5500`.

The default API endpoint is `http://127.0.0.1:8000`. It can be changed from the **API connection** control on the sign-in page.

## API contract used

- `POST /auth/signup`, `POST /auth/login`, `GET /users/me`
- `POST /sessions`, `GET /sessions`, `GET /sessions/{id}/history`
- `POST /sessions/{id}/messages`
- `GET /sessions/{id}/health`, `GET /sessions/{id}/analysis`
