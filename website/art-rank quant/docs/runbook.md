# Local runbook

1. Stop publication if a critical dataset fails completeness or PIT checks.
2. Keep the last stable snapshot visible with `is_stale=true` and its original timestamp.
3. Re-run by the same idempotency key; never overwrite an existing prediction version.
4. Promote models manually after out-of-sample review.

