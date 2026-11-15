# Validation and evidence

## Recorded local run — 12 November 2026

The live offline backend was tested on this workspace's Windows environment at `127.0.0.1:8007`. An initial load run exposed a pooled-connection stall; the SQLite engine was changed to non-pooled connections and the test was rerun successfully.

- Automated regression suite passed 17 tests: persistence, one-retry cap, instruction/drift faults, refusal, API-key security, session isolation, WebSocket replay, expired lease recovery, failed service retry, SSE flow, provider contracts, bounded history and overlapping-submission leases.
- Live examiner-path script passed sourced answers, refusal, two correction cases, persistent failure, reconnected history and revoked-key rejection.
- Load test: **20 concurrent sessions × 3 rounds = 60 requests; 60 successful, 0 errors**.
- Measured offline P95 first-content latency: **1919.08 ms**; median **1278.92 ms**.
- Raw evidence: `context-rot-backend/reports/demo-check.json` and `context-rot-backend/reports/load-test.json`. Re-running replaces those files; use their latest timestamp and configuration when reporting results.

This run is a small local integration benchmark, not proof of production capacity. Three rounds give only 60 timing samples; repeat under the final configuration with enough requests for a stable tail estimate. The application runs SQLite and an embedded worker. Offline generation is extractive and the local checker is a narrow rule baseline. Real-provider latency, application-only overhead, vector retrieval quality and research metrics are not established by this run.

## Reliability layer on a real provider — 15 November 2026

Setup: tutor `openai/gpt-oss-120b` and judge `qwen/qwen3.8-27b` on Groq (free tier; two keys, but in one organisation, so they share its quota), the 157-page *Distributed Databases (R22CSE3146)* notes ingested as the course (346 chunks), `RELIABILITY_DETECTOR=judge`, 10 s check budget.

**Live sessions.** A 13-step revision session (with "at most 3 sentences", a later "use bullet points", follow-ups, "thanks" and an off-syllabus question) and a 14-step adversarial session (with "at most 2 sentences", "only give hints", and the student repeatedly insisting on wrong claims). Results:

- The judge caught the real failures the tutor produced: a 21-sentence answer under a 3-sentence limit, and full answers under "only give hints". The tutor held its ground under pushback, and there were no false alarms on clean turns.
- Problems found and fixed during the audit:
  - The Groq tutor prompt did not include the session's instructions, so the one-retry recovery could never fix an ignored instruction.
  - The judge prompt exceeded the free tier's 8k tokens/minute.
  - Rate limits and timeouts marked answers `check_failed`.
  - The judge sometimes ran out of output tokens before closing its JSON.
  - NLI models loaded on the shared Apple GPU ran out of memory next to a local LLM.
- Under the free tier's limit of 8k tokens per minute, checks during a busy session wait in the queue. Typical end-to-end check time was 0.5–14 s, so the 10 s target is met only when not rate-limited. A provider back-off longer than 60 s now fails the check visibly, instead of blocking the student's next question.

**Labelled realistic set** (60 turns, 16 sessions, 13 rot turns: real course excerpts as retrieved context, two sessions of real Groq tutor answers). This is not the held-out evaluation: the turns were written and labelled by the detector's developer, and the set is small.

| Detector | Precision | Recall | F1 (95% CI) | False alarms per clean session | p95 latency | Cost per answer |
|---|---|---|---|---|---|---|
| Flag every answer | 0.22 | 1.00 | 0.36 | 25 | — | — |
| rules | 1.00 | 0.23 | 0.38 | 0 | <1 ms | free |
| nli (v2) | 0.57 | 0.31 | 0.40 | 2.78 | 412 ms | free (CPU) |
| judge (v1) | 1.00 | 1.00 | 1.00 (1.00–1.00) | 0 | 625 ms | ~1,100 tokens |

**A cheaper judge was tried and rejected.** On `gpt-oss-20b`, three changes together cut tokens per check by 30% (1,503 → 1,059) and p95 latency from 3.0 s to 2.0 s:
- a shorter system prompt
- sending only the earlier-answer and excerpt sentences that share terms with the answer
- low reasoning effort

But F1 fell from 1.00 to 0.89 and false alarms rose from 0 to 0.93 per clean session, so the measured configuration was kept. Capacity comes instead from giving the judge a key from a separate Groq account, since quotas are per account and per model.

v2 raises false alarms on real course text and misses most course conflicts. Under the slide-8 rule it does not replace v1. These numbers still need the held-out sessions labelled by Arijit and Amar.

**Not caused by the reliability layer, but visible in the audit:**
- The lexical scope filter labels student pushback ("Are you sure? My professor said…") as out of syllabus. Those answers are then never checked, which is where an answer caving to pressure would hide.
- `/health` reports "unavailable" after any out-of-syllabus answer.

## Evaluation status against the mid-semester plan

| Metric / evidence | Current status | Required final work |
|---|---|---|
| Persisted answer / correction / event flow | Implemented and integration-tested | Repeat with teammate modules and on second laptop |
| 20-session P95 first content <3s | Passed in the recorded offline run only | Repeat with exact hosted model and final services; report application overhead separately |
| Score available within 10s | Background workflow implemented; no formal percentile study yet | Measure end-to-end save-to-score time under real load |
| Rot detection F1 ≥0.80 + confidence interval | Detectors (v1 judge, v2 NLI) and harness `scripts/eval_reliability.py` implemented and unit-tested; no labelled sessions yet | Run the harness on the 60 labelled sessions: fit on dev, score held-out once |
| ≤0.5 false alarms per clean session | Harness reports false alarms per 25-turn clean session and precision at 2% prevalence; not yet measured | Measure on the 13 held-out clean sessions |
| Scope precision ≥0.90 / recall ≥0.70 | Not established | Evaluate Debayush's scope filter on the held-out questions |
| Retrieval NDCG@5 ≥0.75 | Not established | Integrate vector retrieval, annotate relevance and compare with BM25 |
| Grounding gain ≥15% | Not established | Blind comparison of the same model with and without retrieval |
| Recovery rate | Scripted correction paths work; not a research estimate | Measure corrected flagged answers on held-out examples, including persistent failures |
| Failure growth in later turns | Not established | Premise study on natural 30-turn sessions across two models |
| Label agreement κ≥0.6 | Human work remains | Arijit and Amar label independently, Debayush adjudicates disagreements |

## Annotation procedure for Arijit

Use the agreed rot definition and dataset split. For each non-injected answer record session ID, turn ID, label (clean / contradiction / ignored instruction / other agreed failure), supporting passage or previous turn, and a short reason. Keep injected labels separate. Do not copy detector predictions into your labels. Compare your labels to Amar's only after independent annotation, calculate agreement, revise unclear guidelines, and have Debayush resolve disagreements. Tune thresholds on development data and score the held-out set once after freezing the method.

The report should replace old placeholder results with actual measurements, and list missed targets honestly. Do not replace a missing experiment with results from scripted fault injection.
