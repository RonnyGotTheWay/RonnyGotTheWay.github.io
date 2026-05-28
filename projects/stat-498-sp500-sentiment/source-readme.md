# Does Financial News Sentiment Move the S&P 500?

DATA 498D capstone project exploring whether daily financial news headlines contain useful information about next-day S&P 500 movements.

## Project Question

Financial markets are often assumed to incorporate public information quickly. This project tests whether news headline sentiment is associated with short-run S&P 500 direction and whether that relationship changes across different market conditions.

Main questions:

1. Is headline sentiment statistically related to next-day S&P 500 returns?
2. Do finance-specific sentiment methods perform better than general sentiment tools?
3. Does predictive performance differ across calm and high-volatility market regimes?

## Data

The primary dataset is the Kaggle dataset **S&P 500 with Financial News Headlines (2008-2024)** by dyutidasmahaptra, licensed under CC BY-SA 4.0.

Included raw file:

```text
data/raw/sp500_headlines_2008_2024.csv
```

The working CSV contains 19,127 rows, 3 columns, and 3,507 unique trading dates from January 2, 2008 through March 4, 2024.

Macro context files are managed by `scripts/prepare_data.py`. The script uses cached FRED files in `data/raw/macro/` by default and only downloads fresh copies when run with `--refresh-macro`.

- `VIXCLS`: CBOE Volatility Index
- `DGS10`: 10-year Treasury constant maturity rate
- `DGS2`: 2-year Treasury constant maturity rate
- `FEDFUNDS`: Federal funds effective rate
- `CPIAUCSL`: Consumer Price Index for All Urban Consumers
- `UNRATE`: Unemployment rate
- `USREC`: NBER recession indicator

Important data limitations:

- The original news source is not fully documented.
- Some headlines may be off-topic or only loosely related to the S&P 500.
- The dataset has daily closing prices only, so intraday reactions cannot be measured.
- Headline volume changes strongly over time, especially after 2019.
- The current file includes duplicate rows that are removed during cleaning.
- Monthly macro variables are joined by most recent available observation date for EDA context.

## Repository Structure

```text
.
├── README.md
├── CONTRIBUTIONS.md
├── data_dictionary.md
├── requirements.txt
├── new.pdf
├── data/
│   ├── raw/
│   │   ├── sp500_headlines_2008_2024.csv
│   │   └── macro/
│   └── processed/
│       ├── daily_with_macro.csv
│       ├── daily_with_sentiment.csv
│       ├── daily_with_sentiment_v2.csv
│       ├── model_predictions.csv
│       ├── model_metrics_summary.csv
│       ├── backtest_performance_summary.csv
│       └── backtest_regime_summary.csv
├── figures/
│   ├── fig1_split_timeline.png
│   ├── fig2_sentiment_method_comparison.png
│   ├── fig3_logit_coefficients.png
│   ├── fig4_xgb_feature_importance.png
│   ├── fig5_roc_test.png
│   ├── fig6_backtest_equity_curves.png
│   ├── fig7_regime_backtest_returns.png
│   ├── fig8_backtest_robustness.png
│   ├── fig9_return_model_predictions.png
│   └── sentiment_analysis_summary.png
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 03b_advanced_sentiment.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_backtest_regime_analysis.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── modeling_pipeline.py
│   └── evaluate_backtest.py
├── presentation/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── video/
│   │   └── meeting_04.mp4
│   └── assets/figures/
└── report/
    ├── final_report.docx
    ├── final_report.pdf
    ├── data_preparation_summary.md
    └── backtest_regime_summary.md
```

## How to Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

The processed outputs are already included in the repository. To reproduce them from the raw files, run the project in this order.

1. Prepare the daily dataset and macro context:

```bash
python scripts/prepare_data.py
```

Use this only if the FRED macro files need to be refreshed:

```bash
python scripts/prepare_data.py --refresh-macro
```

2. Run the notebooks in order:

```text
notebooks/01_data_cleaning.ipynb
notebooks/02_eda.ipynb
notebooks/03_sentiment_analysis.ipynb
notebooks/03b_advanced_sentiment.ipynb
notebooks/04_modeling.ipynb
notebooks/05_backtest_regime_analysis.ipynb
```

3. The modeling and backtest steps can also be rerun directly:

```bash
python scripts/modeling_pipeline.py
python scripts/evaluate_backtest.py
```

4. View the static presentation dashboard:

```text
presentation/index.html
```

The dashboard is self-contained and uses the exported project figures in `presentation/assets/figures/`. The recorded group presentation is included at `presentation/video/meeting_04.mp4`.

Key intermediate tables:

```text
data/processed/daily_with_macro.csv          # 01 output
data/processed/daily_with_sentiment.csv      # 03 output (dictionary baseline)
data/processed/daily_with_sentiment_v2.csv   # 03b output (+ VADER + FinBERT + topics)
data/processed/model_predictions.csv         # 04 output, consumed by 05
data/processed/model_metrics_summary.csv     # 04 output, includes balanced accuracy and Brier score
data/processed/model_thresholds.csv          # 04 output, validation-selected classification thresholds
data/processed/model_tuning_summary.csv      # 04 output, XGBoost validation tuning grid
data/processed/backtest_performance_summary.csv
data/processed/backtest_regime_summary.csv
data/processed/return_model_metrics_summary.csv
report/final_report.docx
report/final_report.pdf
report/backtest_regime_summary.md
```

## Analysis Plan

Completed steps:

- `01_data_cleaning.ipynb`: validate raw data, remove duplicates, create daily targets, join macro context fields
- `02_eda.ipynb`: explore headline volume, price movement, returns, macro context, and regimes
- `03_sentiment_analysis.ipynb`: dictionary-based sentiment baseline, daily aggregation, descriptive links to next-day returns
- `03b_advanced_sentiment.ipynb`: VADER (general), FinBERT (finance-specific BERT), and LDA topic modeling on the headline corpus; merges into `daily_with_sentiment_v2.csv`
- `04_modeling.ipynb`: feature engineering (sentiment lags, rolling means, macro interactions, topic distributions), time-aware train/val/test split, two baselines, **five-way sentiment-method comparison** (dict / general / VADER / FinBERT / all) under both Logistic Regression and XGBoost, balanced Logistic Regression, XGBoost `scale_pos_weight`, calibrated probabilities, validation-selected thresholds, validation-tuned XGBoost hyperparameters, balanced accuracy, and expanding-window OOS predictions
- `05_backtest_regime_analysis.ipynb`: regime-segmented evaluation, validation-threshold test backtest, transaction-cost robustness, model-failure cases, and continuous next-day return modeling supplement

## Deliverables Checklist

- [x] Project proposal
- [x] Dataset added to repository
- [x] Data dictionary
- [x] Reproducible notebooks for cleaning, EDA, sentiment, modeling, and backtesting
- [x] Reproducible scripts for data preparation, modeling, and backtest evaluation
- [x] README with run instructions
- [x] Processed output tables generated
- [x] Figures generated for report and presentation use
- [x] Advanced sentiment notebook (03b) — VADER, FinBERT, topic modeling
- [x] Modeling notebook (04) with five-way method comparison and OOS predictions
- [x] Backtest and regime evaluation (05)
- [x] Static presentation dashboard
- [x] CONTRIBUTIONS.md finalized with real team member roles
- [x] Written report
- [x] Recorded presentation
- [x] Peer review submission

## Note on Written Work

The final report and individual reflection paragraphs should be written by team members in their own words, following the course academic integrity policy.
