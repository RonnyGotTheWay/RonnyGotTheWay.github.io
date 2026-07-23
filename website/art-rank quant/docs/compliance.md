# Compliance boundary

- Research, backtest and simulated portfolios only.
- No customer, account, order or execution-detail data.
- No unlicensed commercial market-data scraping.
- Any future real-data adapter requires written scope, retention and redistribution review.
- Model contribution explanations are not causal claims or investment advice.

## Eastmoney personal-research permission record

- Operator confirmation date: 2026-07-14.
- Operator states that permission has been obtained to collect Eastmoney quote-center data for personal research.
- Source: `https://quote.eastmoney.com/center/gridlist.html#hs_a_board` and its page-backed quote feed.
- Scope implemented: current Shanghai, Shenzhen and Beijing A-share daily quote snapshots only.
- Storage: local raw and processed Parquet with immutable manifests and checksums.
- Prohibited by project policy: redistribution, resale, public hosting, account/order data and live order execution.
- The crawler refuses to run unless `EASTMONEY_PERMISSION_CONFIRMED=true`; the operator remains responsible for keeping the permission valid.

## Two-month report source policy

- Shanghai and Shenzhen 60-session history uses AKShare/Sina as the configured primary source and BaoStock as the validator/fallback.
- Conflicting closes are quarantined; vendor prices are never averaged.
- Eastmoney remains the Beijing Stock Exchange and security-metadata fallback under the personal-research permission above.
- Every published report records source health and does not present unavailable fund-flow or news evidence as verified.
