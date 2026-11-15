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

Answers to short social messages ("thanks", "ok", "got it") are not checked. They are saved with status `skipped` and no job is queued (slide 5). Out-of-syllabus answers are refused or warned by the scope filter, and are not checked either.

## Detectors (`RELIABILITY_DETECTOR`)

| Mode | What it does | Cost | Notes |
|---|---|---|---|
| `rules` | The original regex baseline | free | Offline demo only. It catches the scripted primary-key, sentence-limit and photosynthesis faults. Do not report it as a detector. |
| `judge` (v1) | Deterministic instruction rules, then **one** LLM call that judges contradictions with the last `RELIABILITY_HISTORY_TURNS` answers, conflicts with the excerpts, unmeasurable instructions and drift. The output is strict JSON, validated with pydantic. | 1 LLM call per answer | `JUDGE_PROVIDER`/`JUDGE_MODEL` can differ from the tutor's model, which limits self-preference bias. The prompt treats history, excerpts and the answer as data, not instructions. |
| `nli` (v2) | Deterministic instruction rules. NLI cross-encoder on (earlier claim, new claim) pairs, limited to topically related pairs. NLI on (excerpt window, claim) pairs, where a claim is flagged only if something contradicts it and nothing entails it. MiniLM cosine for drift. | CPU, no LLM call | `pip install -r requirements-nli.txt`. It cannot check free-form instructions such as "only give hints". |

Every mode runs under `RELIABILITY_TIMEOUT_SECONDS` (10 s target, slide 3). Errors propagate, so the worker marks the answer `check_failed` ("unverified") instead of passing it silently. The client can retry with `POST /sessions/{id}/jobs/{job_id}/retry`.

Measurable instructions are checked by rules in every mode: sentence and word limits, bullets, numbered lists, "include an example", "no code".

## Health score

`H = 100 − Σ w_k · s_k`, clipped to [0, 100]. Here `s_k` is the strongest severity (0–1) of flag type k. The defaults are `W_CONTRADICTION=50`, `W_INSTRUCTION=45`, `W_COURSE_CONFLICT=45` and `W_DRIFT=45`. They are placeholders chosen so that one confident failure (s ≥ 0.8) of any type pushes H below τ=65. Fit them on development data. Severities are detector signals, **not calibrated probabilities**, and the UI says so.

Contract fields (`docs/INTEGRATION_CONTRACTS.md`):

- `contradiction_score` = max(contradiction, course_conflict)
- `memory_score` = 1 − instruction severity
- `topic_score` = 1 − drift severity
- `scope_score` = share of claims supported by the excerpts
- The boolean fields are set when a severity is ≥ 0.5.

## Evaluation (slide 9)

The labelled sessions are JSONL, one session per line. The format is in the `app/reliability/evaluation.py` docstring, and an example is in `tests/fixtures/reliability_sessions.jsonl`, which is test data only. Run these from `context-rot-backend/`:

```bash
python scripts/eval_reliability.py run --detector judge --data data/rot_sessions.jsonl   # caches scores
python scripts/eval_reliability.py fit --detector judge [--fit-weights]                    # dev split only
python scripts/eval_reliability.py evaluate --detector judge --split test                  # held-out, once
python scripts/eval_reliability.py compare --v1 judge --v2 nli                              # slide 8 rule
```

- `run` replays each session turn by turn, so each answer sees only the history before it. It writes `reports/reliability-scores-<detector>.jsonl` with per-type severities. Later fitting never calls the model again.
- `fit` sweeps τ (and the weights, with `--fit-weights`) for the best dev F1, subject to ≤ 0.5 false alarms per 25-turn clean session. If several τ values tie, it picks the middle of the range. It writes `reports/reliability-thresholds-<detector>.json`. To apply the result, copy τ to `RECOVERY_THRESHOLD` and the weights to the `W_*` settings.
- `evaluate` uses the frozen thresholds and reports:
  - P, R and F1 with a 95% bootstrap CI that resamples whole sessions
  - false alarms per clean session
  - precision at 2% prevalence
  - injected vs natural failures separately
  - recall per failure type
  - P50 and P95 latency
  - the flag-every-answer baseline

  The held-out split is scored once. A second run needs `--force`.
- `compare` applies the slide-8 rule: v2 replaces v1 only if its F1 is within 0.05.

Keep the validation protocol: the detector builder (Srishti) does not label, thresholds are set only on dev, and injected and natural failures are reported separately.

## Known limitations

- The `judge` quality depends on the judge model. With no labelled sessions, there is no evidence yet of F1, the false-alarm rate or the 10 s latency on a real provider.
- `nli` works sentence by sentence. It misses contradictions spread over several sentences, and it is weak on maths and notation-heavy answers.
- Instruction rules only cover measurable constraints. Unmeasurable ones are checked by the judge only.
- The tutor's own retrieval is still lexical (`rag_service.embed` is a hashed bag of words), so `course_conflict` is only as good as the excerpts it is given.
