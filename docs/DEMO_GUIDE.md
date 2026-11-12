# Seven-minute demonstration

## Before the evaluation

1. On two charged laptops, install dependencies and run the offline prototype once while connectivity is available. Keep the repository, Python environment, course data and generated reports locally.
2. Start `Start-Demo.ps1`. Open `/demo/lab.html` and `/docs` on the selected port. Use a fresh DBMS session and verify that WebSocket status says connected.
3. Run `scripts/demo_check.py` and `scripts/load_test.py --concurrency 20 --rounds 3`. Save the JSON reports and inspect mode, timestamp, sample count and errors. Never present an offline measurement as hosted-model performance.
4. Rehearse with the other three teammates. Every member must contribute to the total seven-minute demonstration.
5. Prepare one opening slide showing promised vs demonstrated scope. This is required by the supplied assessment note.

## Suggested team sequence

| Time | Demonstration | Suggested speaker |
|---|---|---|
| 0:00–0:35 | One-slide scope summary; disclose offline or live-provider mode | Team introduction |
| 0:35–1:25 | Course material, syllabus topics, sourced answer and out-of-course refusal | Debayush |
| 1:25–2:40 | Inject a contradiction and an ignored instruction; show score drop and detector explanation | Srishti |
| 2:40–3:35 | Live alert, labelled correction and persistence after page reload | Amar |
| 3:35–5:30 | Repeat the session flow in Swagger, show jobs/events and explain one-retry orchestration | Arijit |
| 5:30–6:20 | Show measured load report and clearly state its mode/limits | Arijit |
| 6:20–7:00 | Actual evaluation results, remaining limitations and eighth-semester plan | Team |

The reference browser lab can demonstrate independently before team integration. Replace its local retrieval/checker with teammate services for the team's final claims.

## Browser lab steps

1. Click **Start local demo account**. This creates a disposable local account and DBMS session with the constraint “At most 2 sentences.” The browser keeps a JWT so reload can restore the session.
2. Click **1. Sourced answer**, then **Send through streaming API**. Inspect source passages and checked status.
3. Click **2. Inject contradiction**, then send. The incorrect primary-key statement should score 30, followed by a linked corrected answer. Read the original's score to show the failure, rather than only the latest corrected score.
4. Click **3. Ignore instruction**, then send. Five sentences violate the stored limit. The recovery reapplies it and produces a shorter answer.
5. Refresh the page. The same session and labelled correction should return from the database; events replay from storage.
6. **4. Fail after one retry** deliberately leaves the correction incorrect and labelled `flagged`. No third answer should appear.
7. **5. Out-of-course question** is refused before generation. No reliability job is created for a refusal.

Local scores are fixed rule-based demonstration scores. These examples do not measure natural model degradation or establish a general detector's accuracy.

## Run a session entirely from Swagger

1. `POST /auth/signup` with a new username, valid email and a password of at least eight characters.
2. `POST /auth/login`; copy `access_token`.
3. Click **Authorize**, paste the token into HTTPBearer, authorize and close. Paste the token only, without adding the word Bearer.
4. `POST /api-keys` with `{"name":"Panel demo"}`. Copy the returned secret once. Log out HTTPBearer in Authorize, and authorize using the **X-API-Key** scheme to demonstrate platform integration.
5. `POST /sessions`:

```json
{"title":"Panel DBMS session","subject":"DBMS","constraints":["At most 2 sentences."]}
```

6. Note the returned session ID. Call `POST /sessions/{id}/messages`:

```json
{"content":"Explain primary keys.","demo_fault":null}
```

7. Read `/jobs` until done, then `/history` and `/analysis`. Repeat with `demo_fault:"contradiction"`, then `"instruction"`. Inspect `correction_of` and statuses. To demonstrate failure after one retry, use `"persistent"`.
8. Read `/events?after=0`. Close and reopen the docs tab, authorize again, and retrieve the same history. For actual process restart persistence, stop and relaunch the API from the same backend directory and reopen the session.
9. Revoke the API key with `DELETE /api-keys/{key_id}`. A subsequent request using that key should return 401. You can use the original JWT to manage the account again.

## Your load-test explanation

“I used 20 separate sessions and three requests per session. The script measures time from request start to the first nonempty streamed content, records full completion separately, waits for background checks, and calculates nearest-rank P95. The saved report identifies the provider and detector mode. Offline results measure this backend on this laptop; we need a separate run for the hosted model and final teammate services.”

## Troubleshooting

- **Port in use:** launch `Start-Demo.ps1 -Port 8007`; use matching URLs and pass `--url http://127.0.0.1:8007` to scripts.
- **401:** the JWT expired or the key was revoked. Reconnect with a new local demo account, or log in again through Swagger.
- **Checking forever:** check `/jobs`; start a separate worker if embedded mode is disabled. A crashed worker's lease may need up to 300 seconds to expire before another process reclaims the job.
- **Check failed:** inspect service configuration, fix it, then use the explicit failed-job retry endpoint. Do not call an unverified answer healthy.
- **503 or SSE error:** check provider credentials/model ID or teammate service availability. The application does not silently switch providers.
- **No matching notes:** use one of the displayed DBMS topics, or integrate the actual course retriever.
- **Connectivity lost:** return to the prepared offline configuration and restart; disclose that mode to the panel.
