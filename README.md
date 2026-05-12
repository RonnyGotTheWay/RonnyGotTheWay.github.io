# Lei Duli

## Data Analytics | Statistical Modeling | Machine Learning

Welcome to my personal data science portfolio. I build projects that connect statistical reasoning, machine learning workflows, and business-readable interpretation.

[Portfolio Website](https://ronnygottheway.github.io/) | [GitHub](https://github.com/RonnyGotTheWay) | [Email](mailto:leiduli@arizona.edu)

---

## About

I am a statistics student at the University of Arizona with strong interests in data analysis, machine learning, and applied statistical modeling. My work focuses on reproducible analysis, fair model comparison, and translating model results into useful decisions.

---

## Featured Project

### From Browsing to Buying: Predicting Online Purchase Intention

**Course:** STAT 474<br>
**Focus:** Modeling Strategy and Evaluation<br>
**Tools:** R, tidyverse, caret, pROC, glmnet, randomForest, gbm, kernlab

This project uses the UCI Online Shoppers Purchasing Intention Dataset to predict whether an e-commerce browsing session will end in purchase. The target is highly imbalanced: only **15.5%** of sessions generated revenue, so the evaluation emphasizes F1, recall, balanced accuracy, and ROC AUC rather than accuracy alone.

My work focused on designing the modeling workflow, comparing six model families, selecting thresholds from training-set cross-validated predictions, and evaluating final performance on a held-out test set.

| Result | Value |
| --- | --- |
| Best F1 | Random Forest, **0.667** |
| Best ROC AUC | Gradient Boosting, **0.922** |
| Best recall | Random Forest, **0.769** |
| High-intent group conversion rate | **41.9%** |
| Main interpretation | PageValues is the dominant predictive signal, but it should be treated as predictive rather than causal. |

Links: [Project page](projects/stat-474-online-shoppers/) | [R code](projects/stat-474-online-shoppers/online_shoppers_analysis.R) | [Final report](projects/stat-474-online-shoppers/final-report.pdf) | [Proposal](projects/stat-474-online-shoppers/proposal.pdf)

---

## Additional Projects

### Large-Scale Data Preprocessing and Feature Discretization

A data preprocessing and feature engineering project for a large used-car transaction dataset. The workflow focuses on cleaning raw data, handling missing values, identifying outliers, discretizing skewed numerical features, and preparing structured inputs for downstream modeling.

**Skills:** data cleaning, outlier detection, equal-width and equal-frequency discretization, ordinal encoding, one-hot encoding<br>
**Tools:** Python, pandas, NumPy, scikit-learn

### Heuristic Optimization of LSTM-Based Models for Time-Series Prediction

A course research paper reviewing heuristic optimization strategies for LSTM-based time-series forecasting. The work compares how PSO, GA, SA, and ACO can support hyperparameter tuning and network structure optimization across energy, transportation, finance, and weather applications.

**Skills:** research synthesis, model comparison, time-series analysis<br>
**Methods:** LSTM, BiLSTM, GRU, PSO, GA, SA, ACO

---

## Technical Toolkit

| Category | Tools and Methods |
| --- | --- |
| Languages | R, Python, SQL |
| Data work | data cleaning, EDA, feature engineering, visualization |
| Modeling | logistic regression, LASSO, PCA, random forest, gradient boosting, SVM |
| Evaluation | cross-validation, ROC AUC, F1, precision, recall, balanced accuracy |
| Reporting | reproducible scripts, model interpretation, sensitivity analysis, executive summaries |

---

## Education

**B.S. / Undergraduate / Statistics**  
University of Arizona

---

## Contact

- GitHub: [RonnyGotTheWay](https://github.com/RonnyGotTheWay)
- Email: [leiduli@arizona.edu](mailto:leiduli@arizona.edu)
