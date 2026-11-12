# Arijit's seventh-semester backend

## Implemented responsibilities

| Mid-semester promise | Implementation | How to demonstrate |
|---|---|---|
| Public API and OpenAPI docs | FastAPI routes; JWT and revocable hashed API keys | `/docs`, create an API key, create a session using X-API-Key |
| Session database | Questions, original answers, sources, constraints, analyses, corrections, jobs and events | Reload history after a correction; restart the API and reopen the same session |
| LLM adapter | Offline extractive generation and Gemini HTTP adapter | Offline demo now; configure a model and credentials for real-provider testing |
| Worker and queue | Persistent database-polled jobs with atomic lease claims and fenced completion | Jobs API and persisted events; run worker separately if desired |
| Recovery loop | Re-retrieve, bound the history, reapply constraints, regenerate and recheck once | Inject contradiction; inspect the original and linked correction |
| Push to clients | Authenticated WebSocket with persisted event cursor; REST event replay | Live update and reconnect; `after` resumes from an event ID |
| Load test | SSE first-content timings at configurable concurrency | `scripts/load_test.py --concurrency 20 --rounds 3` |
| Labelling support | Documented independent annotation workflow | Use team dataset and guideline; actual research labels still require human work |

The detector and retrieval modules have replaceable boundaries. Local demo implementations make Arijit's portion independently runnable. They do not complete Srishti's detector research or Debayush's vector retrieval work. The lab is a plain JavaScript reference client, not Amar's promised final React client.

## Request lifecycle

1. Authenticate using JWT or X-API-Key; check session ownership.
2. Read canonical constraints and recent history; retrieve course passages.
3. Refuse questions with no supported course passages before calling the generation model.
4. Generate normally or stream content. On completion, atomically save the question, answer, source metadata, a queued job and an `answer.saved` event.
5. A worker atomically claims the job. It calls the detector using the answer, prior history, course passages and constraints.
6. If the score is below the configured threshold, retrieve again, truncate history to a bounded recent window while retaining canonical constraints, regenerate once and recheck.
7. Atomically save analyses, a linked correction if needed, message statuses and events. A low-scoring correction remains `flagged`. Original answers remain available.
8. The browser receives events, rereads canonical history, and labels scores and corrections.

History compression here means bounded recent-history truncation, not learned or abstractive summarization. Earlier facts can be lost; canonical instructions are separate and always reapplied. A richer structured memory is future work.

## Reliability of the infrastructure

- Restarted workers reclaim jobs whose leases expire. Normal embedded-worker shutdown returns its active job to the queue.
- Atomic claim tokens fence stale workers: an old worker cannot commit a second correction after another worker takes its job.
- One correction attempt is distinct from job processing attempts. Manual retry is available for infrastructure failures, capped at three processing attempts; successful jobs cannot be manually rerun.
- External calls have timeouts; errors mark answers unverified rather than healthy. There is no silent cloud-to-offline fallback.
- WebSocket authentication is sent in the first frame, not in URL query parameters. JWT expiry and key revocation are rechecked on the connection.
- SQLite uses WAL, foreign keys, a busy timeout and non-pooled connections to avoid pool starvation under streaming load.
- The demo is intended for a single local API process. Database leases serialize overlapping generation requests for the same session, including streaming requests. Pending reliability jobs also block the next question. Request idempotency and broader distributed deployment hardening remain future work. Use separate sessions for concurrent users.
- During streaming, partial content is provisional. Only the `saved` frame confirms database persistence. A disconnect may prevent saving; inspect history before resubmitting.

## Remaining dependencies before team submission

1. Connect approved course material and Debayush's scope-aware vector retrieval service.
2. Connect Srishti's validated detector, freeze its result contract and use development data to choose the recovery threshold.
3. Connect Amar's React interface to these same API and event contracts.
4. You and Amar independently annotate the team's natural/manual examples. Debayush adjudicates; Srishti should not label her own detector's test set.
5. Run real-model load testing and all held-out evaluation metrics. Record model version, dataset version and configuration.
6. Agree with the supervisor on the deck's conflict between scope refusal and warnings. This build follows the explicit scope slide's refusal policy.

## Viva preparation

- Why background checking? It avoids adding detector latency to first content, but the initial answer is provisional and may be unreliable until checked.
- Why keep the original? It supports an auditable correction history and shows what failed.
- Why one retry? It bounds cost and prevents infinite recovery loops.
- Why persisted events? Reconnecting clients can recover missed updates without relying on in-memory broadcasts.
- Is 100 a probability of correctness? No. It is a baseline score, not calibrated confidence.
- Did you prove context rot? No. The backend demonstrates monitoring and recovery; the premise study must establish whether natural failures increase over turns.
- Why no Celery? The slides allow a database-polled worker. It runs on Windows without requiring Redis, Docker or WSL.
- What did you personally implement? The API boundary, persistence, model adapter, queued processing, recovery orchestration, event delivery and repeatable load testing. Explain each from its source file and test.
