# Context Rot Detection — T01 Swifties

Arijit Samanta's semester 7 backend, with a reference browser client and reproducible demo scripts.

## Start the demo

From this folder in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\Start-Demo.ps1
```

Open **http://127.0.0.1:8000/demo/lab.html** and click **Start local demo account**. API documentation: **http://127.0.0.1:8000/docs**. If port 8000 is occupied, add `-Port 8007` and use that port in the links.

The default is fully offline: authored DBMS notes, extractive answers, lexical retrieval, and a narrow rule-based checker. It demonstrates backend integration and failure handling. It is not a validated detector or the team's final vector RAG implementation. A Gemini adapter and HTTP contracts for teammates' retrieval and detector services are included.

## Read these next

- [Seven-minute demo and Swagger walkthrough](docs/DEMO_GUIDE.md)
- [Your responsibilities, architecture and completion checklist](docs/ARIJIT_SEMESTER_7.md)
- [Your proposed eighth-semester plan](docs/SEMESTER_8_PLAN.md)
- [Teammate integration contracts](docs/INTEGRATION_CONTRACTS.md)
- [Measurements and remaining evaluation work](docs/VALIDATION.md)
- [Backend setup](context-rot-backend/README.md)

## Structure

```text
context-rot-backend/
  app/core/                 configuration and authentication
  app/database/             original domain tables + additive runtime tables
  app/routes/               REST, streaming SSE and WebSocket endpoints
  app/services/             generation, durable worker, recovery, demo corpus
  app/integrations/         replaceable retrieval and detector boundaries
  scripts/                  examiner-path check and concurrent load test
  tests/                    isolated backend regression tests
  reports/                  locally generated measurement JSON
context-rot-frontend/
  lab.html, lab.js, lab.css  reference client served by the API
  index.html, app.js, ...    preserved earlier frontend
context-rot-extension/      preserved earlier extension; not required for this demo
docs/                      handover, viva and semester plans
Start-Demo.ps1              local launcher with embedded database worker
```

Existing database content is retained. New tables are additive; there is no destructive schema reset. Tests use a disposable database. Keep `.env`, API keys, database files and private logs out of Git.
