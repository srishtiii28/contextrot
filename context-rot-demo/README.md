# Context Rot Demo Client

This is a small end-user-style frontend for the Context Rot API. It is intentionally separate from the browser extension.

## Run it

Start the backend first:

```powershell
cd ..\context-rot-backend
uvicorn app.main:app --reload
```

Then serve this folder on port 3000:

```powershell
cd ..\context-rot-demo
py -m http.server 3000
```

Open <http://127.0.0.1:3000>.

The API URL defaults to `http://127.0.0.1:8000`. It can be changed in the sign-in screen if the backend is hosted elsewhere.


## End-user flow

1. Create an account or sign in.
2. Create a study session.
3. Upload a text-based PDF.
4. Ask questions in the tutor chat.
5. Inspect the source filename, page, and retrieval score shown under each answer.

The browser stores only the JWT and the selected session id in local storage. The Groq key stays on the backend and is never exposed to this client.
