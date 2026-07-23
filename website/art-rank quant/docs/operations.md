# Operations

Run ingestion, quality, features, inference, portfolio and publication under one deterministic `run_id`. Each step is persisted and resumable. A critical failure keeps the prior stable publication and marks the run failed.

## Training isolation and recovery

- Long-term and short-term runs are stored separately under `artifacts/training_runs/{long_term,short_term}`.
- A single resource lock prevents both memory-heavy pipelines from running together. A cross-mode submission returns HTTP 409 and never reuses the other mode's run id.
- Heartbeats are written every 15 seconds. A worker with no heartbeat for 90 seconds is treated as unresponsive and its lock is recovered.
- Eastmoney is the default daily and 30-minute provider; AKShare/Sina is the default fallback. BaoStock is disabled unless explicitly enabled.
- Missing 30-minute coverage blocks publication. Existing stable predictions remain untouched.
- Set `EASTMONEY_PERMISSION_CONFIRMED=true` only after confirming that both daily and historical 30-minute page data are permitted for the intended personal-research use. If it is false, preflight blocks the run before bulk download.
- Requests have bounded timeouts and retry with jittered backoff. A provider is suspended when 20 consecutive symbols fail or failures reach 20% in the latest 100 symbols; validated symbol caches remain resumable.
- Browser API calls time out after 10 seconds (task submission after 20 seconds). Status polling reports an interruption after three failures and retries after 2, 5, 10 and 30 seconds.
- Use `make api` for normal operation and `make api-dev` only while editing code.
