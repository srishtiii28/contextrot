# Validation and evidence

## Recorded local run — 12 November 2026

The live offline backend was tested on this workspace's Windows environment at `127.0.0.1:8007`. An initial load run exposed a pooled-connection stall; the SQLite engine was changed to non-pooled connections and the test was rerun successfully.

- Automated regression suite passed 17 tests: persistence, one-retry cap, instruction/drift faults, refusal, API-key security, session isolation, WebSocket replay, expired lease recovery, failed service retry, SSE flow, provider contracts, bounded history and overlapping-submission leases.
- Live examiner-path script passed sourced answers, refusal, two correction cases, persistent failure, reconnected history and revoked-key rejection.
- Load test: **20 concurrent sessions × 3 rounds = 60 requests; 60 successful, 0 errors**.
- Measured offline P95 first-content latency: **1919.08 ms**; median **1278.92 ms**.
- Raw evidence: `context-rot-backend/reports/demo-check.json` and `context-rot-backend/reports/load-test.json`. Re-running replaces those files; use their latest timestamp and configuration when reporting results.

This run is a small local integration benchmark, not proof of production capacity. Three rounds give only 60 timing samples; repeat under the final configuration with enough requests for a stable tail estimate. The application runs SQLite and an embedded worker. Offline generation is extractive and the local checker is a narrow rule baseline. Real-provider latency, application-only overhead, vector retrieval quality and research metrics are not established by this run.

## Evaluation status against the mid-semester plan

| Metric / evidence | Current status | Required final work |
|---|---|---|
| Persisted answer / correction / event flow | Implemented and integration-tested | Repeat with teammate modules and on second laptop |
| 20-session P95 first content <3s | Passed in the recorded offline run only | Repeat with exact hosted model and final services; report application overhead separately |
| Score available within 10s | Background workflow implemented; no formal percentile study yet | Measure end-to-end save-to-score time under real load |
| Rot detection F1 ≥0.80 + confidence interval | Not established | Score the held-out labelled conversations using the validated detector |
| ≤0.5 false alarms per clean session | Not established | Measure on the clean held-out sessions, including realistic low failure prevalence |
| Scope precision ≥0.90 / recall ≥0.70 | Not established | Evaluate Debayush's scope filter on the held-out questions |
| Retrieval NDCG@5 ≥0.75 | Not established | Integrate vector retrieval, annotate relevance and compare with BM25 |
| Grounding gain ≥15% | Not established | Blind comparison of the same model with and without retrieval |
| Recovery rate | Scripted correction paths work; not a research estimate | Measure corrected flagged answers on held-out examples, including persistent failures |
| Failure growth in later turns | Not established | Premise study on natural 30-turn sessions across two models |
| Label agreement κ≥0.6 | Human work remains | Arijit and Amar label independently, Debayush adjudicates disagreements |

## Annotation procedure for Arijit

Use the agreed rot definition and dataset split. For each non-injected answer record session ID, turn ID, label (clean / contradiction / ignored instruction / other agreed failure), supporting passage or previous turn, and a short reason. Keep injected labels separate. Do not copy detector predictions into your labels. Compare your labels to Amar's only after independent annotation, calculate agreement, revise unclear guidelines, and have Debayush resolve disagreements. Tune thresholds on development data and score the held-out set once after freezing the method.

The report should replace old placeholder results with actual measurements, and list missed targets honestly. Do not replace a missing experiment with results from scripted fault injection.
