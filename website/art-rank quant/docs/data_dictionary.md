# Data dictionary

Every time-sensitive dataset contains `event_time`, `available_time`, `ingested_at`, `source`, `schema_version`, and `data_version`. Prices retain raw values and adjustment factors. Predictions additionally contain `as_of_date`, `feature_version`, `model_version`, `run_id`, and `is_stale`.

`eastmoney_stock_daily` stores one licensed daily snapshot for every returned A-share code and name, including unquoted/suspended rows. `eastmoney_stock_daily_training` contains only rows with a non-empty name and valid positive OHLC values. Extra fields include `pre_close`, `amount`, `pct_change`, `amplitude`, `turnover_rate`, `pe_dynamic`, `volume_ratio`, `pb`, `total_market_cap` and `float_market_cap`.

Eastmoney prices are raw, unadjusted values (`adjust_factor=1`). They may enter fitting only after multiple immutable trading-day snapshots have accumulated and corporate-action adjustment has been materialized; a single daily snapshot is not a valid training history.
