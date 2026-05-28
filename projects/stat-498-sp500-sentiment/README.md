# Does Financial News Sentiment Move the S&P 500?

**Course:** DATA 498 / Capstone Project<br>
**Project type:** Financial NLP, market direction modeling, backtesting, written report<br>
**Source repository:** [Guojiawei-01/sp500](https://github.com/Guojiawei-01/sp500)<br>
**My portfolio focus:** Written report and presentation narrative

[Project page](index.html) | [Written report](report/final-report.pdf) | [Proposal](proposal.pdf) | [Presentation folder](https://github.com/Guojiawei-01/sp500/tree/main/presentation) | [Source repo](https://github.com/Guojiawei-01/sp500)

---

## Project Overview

This capstone project asks whether daily financial news headlines contain useful information for predicting next-day S&P 500 movement. The repository includes data cleaning, sentiment extraction, macro context, predictive modeling, backtest evaluation, figures, a final written report, and presentation materials.

The raw Kaggle dataset contains **19,127 headline rows** from January 2, 2008 through March 4, 2024. After duplicate removal and daily aggregation, the project uses **3,506 prepared daily rows** with next-day return and direction targets.

---

## My Contribution

I participated in the checklist item **Written report**. The current source contribution file also lists Duli Lei as writing and presentation lead. My work focused on:

- shaping the report narrative;
- organizing figures and final takeaways;
- explaining why weak predictive signal still makes the project valuable;
- keeping the conclusion cautious instead of overstating fragile backtest results;
- supporting the presentation flow and dashboard materials.

---

## Methods Identified from the Repository

- Daily headline aggregation and next-day S&P 500 target construction.
- FRED macro context including VIX, Treasury rates, federal funds rate, CPI, unemployment, and recession indicator.
- Sentiment methods: dictionary baseline, VADER, FinBERT, and LDA topic modeling.
- Direction models: Logistic Regression and XGBoost.
- Time-aware split: training before 2020, validation in 2020-2021, and testing from 2022 to March 2024.
- Validation-selected probability thresholds.
- Long/flat backtesting with transaction-cost and regime robustness checks.
- Static presentation dashboard and recorded group presentation materials.

---

## Main Results

| Result | Value |
| --- | ---: |
| Raw headline rows | 19,127 |
| Clean headline rows | 18,153 |
| Prepared daily rows | 3,506 |
| Next-day up-day rate | 54.3% |
| Best test balanced accuracy | 51.4% |
| Logit + FinBERT test return | 13.1% |
| Buy-and-hold test return | 7.0% |

The final conclusion is cautious: daily headline sentiment showed weak overall predictive power. Some finance-specific signals looked promising in selected backtests, but those results were sensitive to regime, threshold, exposure, and trading assumptions.

---

## Files

| File | Description |
| --- | --- |
| [report/final-report.pdf](report/final-report.pdf) | Final written report. |
| [proposal.pdf](proposal.pdf) | Project proposal PDF from the repository. |
| [source-readme.md](source-readme.md) | Local copy of the source repository README. |
| [report/backtest-regime-summary.md](report/backtest-regime-summary.md) | Backtest and regime evaluation summary. |
| [figures/](figures/) | Portfolio figures from the project. |
| [Source presentation folder](https://github.com/Guojiawei-01/sp500/tree/main/presentation) | Static dashboard and recorded-presentation assets in the source repository. |
