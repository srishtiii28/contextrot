# Integration contracts

## Authentication

REST accepts either `Authorization: Bearer <JWT>` or `X-API-Key: <platform key>`. Create a local user with `/auth/signup` then log in at `/auth/login`. `POST /api-keys` returns a secret once; the database stores its SHA-256 digest. `GET /api-keys` lists metadata; `DELETE /api-keys/{id}` revokes it. Do not place provider secrets in browser code.

## Session and message API

```json
POST /sessions
{"title":"DBMS revision","subject":"DBMS","constraints":["At most 2 sentences."]}
```

```json
POST /sessions/1/messages
{"content":"Explain primary keys.","demo_fault":null}
```

Normal responses include `user_message`, `assistant_message` and `job_id`. Each assistant message exposes `status`, `sources`, `correction_of` and `generation_ms`. `generation_ms` is full generation/persistence-path timing, NOT first-token latency.

`demo_fault` accepts `contradiction`, `instruction`, `drift`, or `persistent` only when demo faults are enabled AND the model provider is offline/mock. It is metadata for scripted failures, not a feature for production clients. The checker sees the corrupted answer content; local scoring does not read the fault label.

`GET /sessions/{id}/history`, `/analysis`, `/health`, `/jobs`, `/constraints` and `/events?after=0` provide canonical persisted state. Add a canonical session instruction with `POST /sessions/{id}/constraints`, body `{"content":"Use bullet points."}`. Constraints added after a job was queued apply to future answers; jobs evaluate the captured snapshot.

## SSE answer stream

`POST /sessions/{id}/messages/stream` accepts the same message request. Parse `text/event-stream` frames:

```text
event: token
data: {"text":"A primary key..."}

event: saved
data: {"user_message":{...},"assistant_message":{...},"job_id":1}
```

An `error` frame means generation/persistence did not complete; partial displayed text is provisional. HTTP 401/403/404/409/422/503 can occur before streaming. First-content latency starts before sending the request and stops on the first nonempty `token` frame. It excludes background analysis completion.

## WebSocket replay

Connect to `/sessions/{id}/events?after=<last-event-id>`. Within 10 seconds send one JSON authentication frame:

```json
{"token":"JWT"}
```

or `{"api_key":"cr_..."}`. The server sends `connected`, then persisted events in ID order. Supported events are `answer.saved`, `analysis.completed`, `answer.corrected`, `analysis.failed`. Events include IDs and payloads. IDs are global database IDs and can have gaps within a session. Store the largest ID seen, deduplicate by ID and resume from it. REST history remains authoritative. Send `ping` to receive `pong`. Event replay pages contain at most 100 entries; REST callers paginate with the last ID.

## Debayush's retrieval service

Set `RAG_SERVICE_URL` to the FULL POST endpoint of a separate service. Do not point it back at this application's `/integrations/rag/retrieve`, which would recurse. Requests:

```json
{"query":"Explain primary keys.","session_id":1,"top_k":5}
```

Responses:

```json
{"in_scope":true,"chunks":[{"text":"A primary key...","source":"Approved course notes.pdf","page":4,"score":0.88,"topic":"Keys"}]}
```

Return `in_scope:false` with an empty list for out-of-syllabus queries. The backend refuses before generation if no chunks remain. The local fallback is lexical retrieval over authored notes, not embeddings/FAISS/Chroma. Replace it for the vector-retrieval evaluation. `/course/topics` currently lists the local demo corpus only; connect the real course catalogue when integrating.

## Srishti's detector service

Set `ROT_SERVICE_URL` to the FULL POST endpoint of a separate service. Do not point it at this application's own integration route. Input:

```json
{"session_id":1,"message_id":2,"current_question":"Explain primary keys.","current_answer":"...","conversation_history":[{"role":"user","content":"..."}],"constraints":[{"type":"instruction","content":"At most 2 sentences."}],"retrieved_context":[{"text":"...","source":"notes.pdf","page":4,"score":0.9}]}
```

Required result:

```json
{"topic_score":1,"contradiction_score":0,"memory_score":1,"scope_score":1,"health_score":100,"topic_drift":false,"contradiction_detected":false,"memory_violation":false,"alert":null}
```

Scores are validated: component scores must be 0–1, health 0–100. Agree on higher-is-better semantics except `contradiction_score`, where higher means greater contradiction. `RECOVERY_THRESHOLD` defaults to 65 only for demonstration; tune it on development data. The service receives history preceding the current question/answer. Recovery is checked using the same constraints and updated retrieved context.

Both external services optionally receive `X-API-Key` from `INTEGRATION_API_KEY`. Calls time out and malformed/non-2xx responses fail visibly. The detector should return an explanation, not claim its score is calibrated confidence.
