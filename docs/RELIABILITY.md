# Reliability layer (context rot detection)

Owner: Srishti. Code: `context-rot-backend/app/reliability/`. This is the orange "Reliability layer" block in the midsem block diagram (slide 4). It runs inside the background worker after each answer is saved. It scores the answer against the session history, the student's instructions and the retrieved course material. It outputs a **health score H (0–100) and flags**. If H < τ (`RECOVERY_THRESHOLD`), the worker re-retrieves, compresses the history and regenerates the answer once. If the score is still low, the answer is flagged. The worker, queue and recovery code already existed in `app/services/worker.py`.

## Rot definition

An answer has rotted if it contains at least one of these:

| Flag | Meaning | Not rot |
|---|---|---|
| `contradiction` | A claim is logically incompatible with an earlier assistant answer in the same session | Elaboration, rephrasing, simplification, added detail |
| `ignored_instruction` | Breaks an instruction the student set for the session (`constraints` table) | An instruction given only for one earlier question (a transient format request) |
| `course_conflict` | A claim conflicts with the retrieved course text | A claim the course text simply does not mention |
| `topic_drift` | The answer is mainly about something other than the question | Brief context or a link to a related topic |

Answers to short social messages ("thanks", "ok", "got it") are not checked (slide 5). They are detected before retrieval: the tutor sends a fixed short reply, with no retrieval, no LLM call and no out-of-syllabus banner. The reply is saved with status `skipped` and no job is queued. A social message is checked as normal if the tutor replies with real content, for example "yes" after "Shall I explain joins?". Out-of-syllabus answers are warned by the scope filter and are not checked either.

Answers that the layer replaced (`superseded`), or could not fix (`flagged`), are removed from the history that later turns see. This applies to both the tutor and the checker. A known-wrong answer therefore never makes a correct later answer look like a contradiction.

## Detectors (`RELIABILITY_DETECTOR`)

| Mode | What it does | Cost | Notes |
|---|---|---|---|
| `rules` | The original regex baseline | free | Offline demo only. It catches the scripted primary-key, sentence-limit and photosynthesis faults. Do not report it as a detector. |
| `judge` (v1) | Deterministic instruction rules, then **one** LLM call that judges contradictions with the last `RELIABILITY_HISTORY_TURNS` answers, conflicts with the excerpts, unmeasurable instructions and drift. The output is strict JSON, validated with pydantic. | 1 LLM call per answer | `JUDGE_PROVIDER`/`JUDGE_MODEL` can differ from the tutor's model, which limits self-preference bias. The prompt treats history, excerpts and the answer as data, not instructions. |
| `nli` (v2) | Deterministic instruction rules. An NLI cross-encoder compares claims with claims from earlier answers, and with single excerpt sentences, under two gates. First, both sentences must be about the same thing: MiniLM cosine ≥ `NLI_PAIR_MIN_SIMILARITY`, default 0.70. Second, the model must call the pair a contradiction in **both** directions. A course conflict also needs that no excerpt passage entails the claim. Drift is the answer's MiniLM similarity to the question, the previous answer and the excerpts. | CPU, no LLM call | `pip install -r requirements-nli.txt`. The models load when the worker starts. It cannot check free-form instructions such as "only give hints". |

Every mode runs under `RELIABILITY_TIMEOUT_SECONDS` (10 s target, slide 3). Errors propagate, so the worker marks the answer `check_failed` ("unverified") instead of passing it silently. The client can retry with `POST /sessions/{id}/jobs/{job_id}/retry`.

Measurable instructions are checked by rules in every mode. These are sentence and word limits ("at most 2 sentences", "in two sentences", "3 sentences max", "50 words or less"), bullets and "no bullets", numbered lists, "include an example" and "no code". Limits per item ("2 sentences per bullet") are left to the judge. Sentence counting ignores headings, tables, citations, "e.g."/"i.e." and decimals.

## Health score

`H = 100 − Σ w_k · s_k`, clipped to [0, 100]. Here `s_k` is the strongest severity (0–1) of flag type k. The defaults are `W_CONTRADICTION=50`, `W_INSTRUCTION=45`, `W_COURSE_CONFLICT=45` and `W_DRIFT=45`. They are placeholders chosen so that one confident failure (s ≥ 0.8) of any type pushes H below τ=65. Fit them on development data. Severities are detector signals, **not calibrated probabilities**, and the UI says so.

Contract fields (`docs/INTEGRATION_CONTRACTS.md`):

- `contradiction_score` = max(contradiction, course_conflict)
- `memory_score` = 1 − instruction severity
- `topic_score` = 1 − drift severity
- `scope_score` = share of claims supported by the excerpts
- The boolean fields are set when a severity is ≥ 0.5.
- Flags are always stored. The student-facing `alert` is set only when H < τ, the same rule that triggers recovery and that the evaluation counts as a flag. This makes the measured false-alarm rate the rate of alerts students actually see.
- For a contradiction, `turn` counts answers back from the current one: 1 is the previous answer. Counting back stays correct when the history has been cut to a recent window.

## Evaluation (slide 9)

The labelled sessions are JSONL, one session per line. The format is in the `app/reliability/evaluation.py` docstring, and an example is in `tests/fixtures/reliability_sessions.jsonl`, which is test data only. Run these from `context-rot-backend/`:

```bash
python scripts/eval_reliability.py run --detector judge --data data/rot_sessions.jsonl   # caches scores
python scripts/eval_reliability.py fit --detector judge [--fit-weights]                    # dev split only
python scripts/eval_reliability.py evaluate --detector judge --split test                  # held-out, once
python scripts/eval_reliability.py compare --v1 judge --v2 nli                              # slide 8 rule, dev split
python scripts/eval_reliability.py premise --data data/premise_sessions.jsonl               # slide 9 premise
python scripts/eval_reliability.py recovery --database-url sqlite:///./context_rot.db       # slide 9 recovery rate
```

- `run` replays each session turn by turn. Each answer sees only the history before it, cut exactly as in production by `compress_history`. It writes `reports/reliability-scores-<detector>.jsonl` with per-type severities and the cost per answer (judge tokens, latency). Later fitting never calls the model again. A detector error on one turn is recorded and counted, and the run continues. NLI models load before the first turn, so loading doesn't count as latency.
- `fit` sweeps τ (and the weights, with `--fit-weights`) for the best dev F1, subject to ≤ 0.5 false alarms per 25-turn clean session. It refuses to run without a fully clean dev session, and once the held-out set has been scored. If several τ values tie, it picks the middle of the range. It writes `reports/reliability-thresholds-<detector>.json`. To apply the result, copy τ to `RECOVERY_THRESHOLD` and the weights to the `W_*` settings.
- `evaluate` uses the frozen thresholds and reports:
  - P, R and F1 with a 95% bootstrap CI that resamples whole sessions
  - false alarms per clean session
  - precision at 2% prevalence
  - injected vs natural failures separately
  - recall per failure type
  - P50 and P95 latency
  - the flag-every-answer baseline

  It also reports cost per answer and the number of detector errors. The held-out split is scored once, a second run needs `--force`, and the thresholds must come from the same data file (checked by SHA-256).
- `compare` applies the slide-8 rule: v2 replaces v1 only if its F1 is within 0.05 **and** it costs less (LLM tokens per answer, then latency). It defaults to the dev split, as in the week-8 plan.
- `premise` uses human labels only. For each tutor model (the session `model` field) it reports the failure rate in turns 21–30 vs 1–10, with a 95% session-bootstrap interval for the difference.
- `recovery` reads the live database and reports the share of retried answers that came back `corrected` rather than `flagged`.

Keep the validation protocol: the detector builder (Srishti) does not label, thresholds are set only on dev, and injected and natural failures are reported separately.

## Known limitations

- The `judge` quality depends on the judge model. With no labelled sessions, there is no evidence yet of F1, the false-alarm rate or the 10 s latency on a real provider.
- `nli` works sentence by sentence. It misses contradictions spread over several sentences, and it is weak on maths and notation-heavy answers. The similarity gate trades recall for fewer false alarms: a contradiction worded very differently from the earlier answer can fall below 0.70. In the audit, small NLI models (xsmall, small, base) all called pairs like "mitosis…" vs "meiosis…" contradictory; the gates are what suppress this, not model size.
- The live layer sees only the history kept by `compress_history` (about 6000 characters, at most `RELIABILITY_HISTORY_TURNS` answers). A contradiction with an answer from much earlier in a long session can't be caught. This is a deliberate cost and latency bound; widen it if the premise study shows late-session contradictions matter.
- Instruction rules only cover measurable constraints. Unmeasurable ones are checked by the judge only.
- The tutor's own retrieval is still lexical (`rag_service.embed` is a hashed bag of words), so `course_conflict` is only as good as the excerpts it is given.
