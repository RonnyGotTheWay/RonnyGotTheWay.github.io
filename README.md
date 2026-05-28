# Lei Duli

## Data Analytics | Regression | Statistical Modeling | Machine Learning

Welcome to my personal data science portfolio. I build projects that connect statistical reasoning, reproducible analysis, model evaluation, and business-readable interpretation.

[Portfolio Website](https://ronnygottheway.github.io/) | [GitHub](https://github.com/RonnyGotTheWay) | [Email](mailto:leiduli@arizona.edu)

---

## About

I am a statistics student at the University of Arizona with strong interests in data analysis, regression modeling, machine learning, and applied statistical reasoning. My work focuses on building clean analysis pipelines, comparing models fairly, and explaining what the results mean beyond a single score.

---

## Featured Projects

### 2026 MCM Problem C: Audience Vote Estimation and Fairness Optimization

**Course/Competition:** 2026 MCM/ICM Problem C<br>
**Role:** Coding Member / Code Lead<br>
**Tools:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, statsmodels, matplotlib<br>
**Project page:** [2026 MCM Problem C](projects/mcm-2026-dwts-voting/index.html)

This project builds a data-driven framework for Dancing with the Stars to estimate hidden audience support, compare ranking-based and percentage-based voting mechanisms, analyze contestant and partner effects, and propose the STARS scoring system.

| Result | Value |
| --- | --- |
| Award | **Successful Participant, S Award** |
| XGBoost AUC | **0.8106** |
| Elimination Accuracy | **45.70%** |
| Bottom-2 Hit Rate | **69.23%** |
| Ranking vs Percentage Accuracy | **52.24% vs 48.06%** |
| STARS Backtesting Accuracy | **91.86%** |

Links: [Project page](projects/mcm-2026-dwts-voting/index.html) | [Final paper](projects/mcm-2026-dwts-voting/files/mcm-2026-problem-c-report.pdf) | [Notebooks](projects/mcm-2026-dwts-voting/index.html#files)

### Data Fest: Sequence-Based Segmentation of Chronic Disease Care Pathways

**Competition:** Data Fest<br>
**Award:** Best Insights Award<br>
**Role:** Data Analysis / Pipeline Implementation<br>
**Tools:** Python, pandas, NumPy, scikit-learn, K-Means, PCA, matplotlib, seaborn<br>
**Project page:** [Data Fest Chronic Care Pathways](projects/datafest-2026-chronic-care-pathways/)<br>
**Project README:** [README](projects/datafest-2026-chronic-care-pathways/README.md)

This project identifies distinct longitudinal care pathways among chronic disease patients with diabetes and hypertension. We transformed encounter histories into behavioral sequences, inserted silence markers for care gaps longer than 90 days, engineered patient-level features, and used K-Means clustering to identify clinically meaningful patient segments.

| Result                                    |               Value |
| ----------------------------------------- | ------------------: |
| Award                                     | Best Insights Award |
| Patients analyzed                         |              46,714 |
| Unique encounters                         |           3,073,410 |
| Number of clusters                        |                   4 |
| Acute-unstable cluster size               |                4.7% |
| ED/inpatient ratio in Cluster 2           |               12.5% |
| Mortality rate in Cluster 2               |                7.3% |
| MyChart activation in Cluster 2           |               71.7% |
| Silence-to-acute probability in Cluster 2 |                 59% |

**Best insight:** The highest-risk patients were not simply those with the most visits. Risk concentrated among patients whose care trajectories became unstable after long silent periods. In the acute-unstable cluster, nearly six out of ten post-silence transitions ended in ED or inpatient care, making silence in the clinical record a predictive intervention window.

### Does Financial News Sentiment Move the S&P 500?

**Course:** DATA 498 / Capstone Project<br>
**Focus:** Written report and presentation narrative<br>
**Tools:** Python, VADER, FinBERT, LDA, Logistic Regression, XGBoost, backtesting<br>
**Project page:** [S&P 500 News Sentiment](projects/stat-498-sp500-sentiment/index.html)

This project tests whether daily financial news sentiment contains useful information for next-day S&P 500 direction. The repository includes cleaning, daily aggregation, sentiment methods, market context, predictive modeling, backtesting, regime evaluation, figures, a final report, and presentation materials. My contribution foregrounds the checklist item **Written report**, with narrative, figure organization, final takeaways, and presentation flow.

| Result | Value |
| --- | --- |
| Raw headline rows | **19,127** |
| Prepared daily rows | **3,506** |
| Best balanced accuracy | **51.4%** |
| Logit + FinBERT test return | **13.1%** versus **7.0%** buy-and-hold |
| Main interpretation | Daily headline sentiment produced weak overall predictive signal; selected backtests were interesting but fragile. |

Links: [Written report](projects/stat-498-sp500-sentiment/report/final-report.pdf) | [Proposal](projects/stat-498-sp500-sentiment/proposal.pdf) | [Presentation folder](https://github.com/Guojiawei-01/sp500/tree/main/presentation) | [Source repo](https://github.com/Guojiawei-01/sp500)

### Predicting Airbnb Listing Prices in New York City

**Course:** DATA 467 / Linear Regression<br>
**Tools:** Python, pandas, statsmodels, scipy, matplotlib<br>
**Project page:** [Airbnb NYC Price Modeling](projects/stat-467-airbnb-nyc/index.html)

This project uses the 2019 NYC Airbnb open dataset to study which listing characteristics most strongly influence nightly price. The analysis models log(price) with OLS regression, compares three specifications, checks diagnostics, and uses a logistic high-price indicator as a second confirmation of the main pricing structure.

| Result | Value |
| --- | --- |
| Raw observations | **48,895** listings |
| Cleaned analytic sample | **38,833** listings |
| Best model fit | Model 3, **adjusted R2 = 0.507** |
| Entire-home premium | About **118%** higher than private rooms in Model 2 |
| Manhattan premium | About **35.6%** higher than Brooklyn in Model 2 |
| Main interpretation | Room type and borough location dominate Airbnb pricing patterns. |

Links: [Notebook](projects/stat-467-airbnb-nyc/code/data_analysis.ipynb) | [Final paper](projects/stat-467-airbnb-nyc/final-paper.pdf) | [Proposal](projects/stat-467-airbnb-nyc/project-proposal.pdf) | [Dataset](projects/stat-467-airbnb-nyc/data/AB_NYC_2019.csv)

### From Browsing to Buying: Predicting Online Purchase Intention

**Course:** STAT 474<br>
**Focus:** Modeling Strategy and Evaluation<br>
**Tools:** R, tidyverse, caret, pROC, glmnet, randomForest, gbm, kernlab<br>
**Project page:** [Online Purchase Intention](projects/stat-474-online-shoppers/index.html)

This project uses the UCI Online Shoppers Purchasing Intention Dataset to predict whether an e-commerce browsing session will end in purchase. Because only **15.5%** of sessions generated revenue, the evaluation emphasizes F1, recall, balanced accuracy, and ROC AUC rather than accuracy alone.

| Result | Value |
| --- | --- |
| Best F1 | Random Forest, **0.667** |
| Best ROC AUC | Gradient Boosting, **0.922** |
| Best recall | Random Forest, **0.769** |
| High-intent group conversion rate | **41.9%** |
| Main interpretation | PageValues is the dominant predictive signal, but it should be treated as predictive rather than causal. |

Links: [R code](projects/stat-474-online-shoppers/online_shoppers_analysis.R) | [Final report](projects/stat-474-online-shoppers/final-report.pdf) | [Proposal](projects/stat-474-online-shoppers/proposal.pdf)

---

## Additional Projects

### Large-Scale Data Preprocessing and Feature Discretization

A data preprocessing and feature engineering project for a large used-car transaction dataset. The workflow focuses on missing value handling, outlier detection, discretization, and encoding.

**Skills:** data cleaning, outlier detection, equal-width and equal-frequency discretization, ordinal encoding, one-hot encoding<br>
**Tools:** Python, pandas, NumPy, scikit-learn

### Heuristic Optimization of LSTM-Based Models for Time-Series Prediction

A course research paper reviewing heuristic optimization strategies for LSTM-based time-series forecasting. The work compares PSO, GA, SA, and ACO across applied forecasting domains.

**Skills:** research synthesis, model comparison, time-series analysis<br>
**Methods:** LSTM, BiLSTM, GRU, PSO, GA, SA, ACO

---

## Technical Toolkit

| Category | Tools and Methods |
| --- | --- |
| Languages | R, Python, SQL |
| Data work | data cleaning, EDA, feature engineering, visualization, reporting |
| Modeling | OLS, GLM, logistic regression, LASSO, PCA, random forest, gradient boosting, SVM |
| Evaluation | cross-validation, regression diagnostics, ROC AUC, F1, precision, recall, balanced accuracy |
| Reporting | reproducible notebooks, model interpretation, sensitivity analysis, executive summaries |

---

## Education

**B.S. / Undergraduate / Statistics**  
University of Arizona

---

## Contact

- GitHub: [RonnyGotTheWay](https://github.com/RonnyGotTheWay)
- Email: [leiduli@arizona.edu](mailto:leiduli@arizona.edu)
