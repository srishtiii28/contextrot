# Arijit's proposed eighth-semester plan

This is a recommended personal plan derived from the features deferred in the mid-semester deck. It is not a new faculty-approved allocation. Keep ownership of platform/backend engineering; coordinate model training and detection research with the relevant teammates.

## Outcome

Turn the single-course demonstration into an evaluable integration service with tenant/course separation, richer learning features and reproducible deployment. Begin only after the seventh-semester core and measurements are stable.

| Phase | Your backend work | Teammate dependency | Demonstrable output |
|---|---|---|---|
| Weeks 1–2: harden the foundation | Version API contracts, add database migrations, request idempotency, distributed deployment hardening, and structured error/latency logs | Freeze detector and retrieval schemas | Repeatable deployment and restart/concurrency tests |
| Weeks 3–4: course/platform isolation | Add tenant, course and material-version IDs; scope API keys to tenants; enforce isolation in every query and event subscription; migrate toward PostgreSQL | Debayush supplies multiple course indices | Two platforms/courses operate without cross-access |
| Weeks 5–6: uploads and learner features | Authenticated upload jobs, file limits, progress, metadata, deletion and consent flow; expose gap/difficulty results via versioned APIs | Debayush's ingestion/gap analysis; teammate difficulty module | Uploaded material is processed with traceable ownership and deletable history |
| Weeks 7–8: model experiments | Extend the adapter to route selected tenant/course requests to approved LoRA adapters; record model/prompt/index versions, latency and usage | Team provides a trained adapter, data permission and evaluation evidence | Reproducible base-vs-adapted comparison through the same API |
| Weeks 9–10: analytics and pilot | Aggregate session metrics, recovery outcomes, latency and usage; expose consent-aware pilot exports and a documented retention policy | Educator recruitment and study design | Analytics report and pilot dataset without unnecessary personal identifiers |
| Weeks 11–12: integration and handover | Add MCP integration only if justified by a real platform use case; run acceptance/security/load tests, deploy the agreed environment and complete API documentation | Platform integration and supervisor approval of scope | Versioned release, integration guide, rollback procedure and final demonstration |

## Priorities if time is limited

Finish tenant/course isolation, reliable uploads, deployment, and real-provider performance first. Treat billing and MCP integration as lower-priority additions requiring a concrete use case. A learning-outcome pilot needs supervisor-approved study design and participant consent; do not promise improved marks without evidence.

## Your deliverables

- Versioned integration API and database migration history.
- Tenant/course-scoped authentication and cross-access tests.
- Durable upload and processing pipeline, if uploads remain in scope.
- Model-routing/experiment support; training itself is shared research work.
- Session analytics and reproducible performance/recovery measurements.
- Deployment, configuration, monitoring and handover documentation.

## What to say during the seventh-semester presentation

“This semester I implemented the API, persistent session history, background reliability workflow, one-retry recovery, live event delivery and load-test tooling for one course. In eighth semester I plan to extend the backend for multiple courses and platforms, integrate student uploads and the team's personalization modules, support controlled fine-tuned-model experiments, and harden deployment and analytics. Each addition will be evaluated against the current baseline.”
