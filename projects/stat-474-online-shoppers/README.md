# From Browsing to Buying: Predicting Online Purchase Intention

**Course:** STAT 474<br>
**Project type:** Machine learning classification, model evaluation, business analytics<br>
**My portfolio focus:** Modeling Strategy and Evaluation<br>
**Dataset:** UCI Online Shoppers Purchasing Intention Dataset<br>

[Project page](./) | [R analysis code](online_shoppers_analysis.R) | [Final report](final-report.pdf) | [Proposal](proposal.pdf)

---

## Project Overview

Most e-commerce visitors leave without buying, but their browsing sessions contain signals about intent: pages viewed, time spent, exit behavior, visitor type, seasonality, and analytics-derived page value. This project predicts whether a session ends in purchase using the UCI Online Shoppers Purchasing Intention Dataset.

The target variable is imbalanced: only **1,908 of 12,330 sessions**, or **15.5%**, generated revenue. Because of that imbalance, the evaluation emphasizes precision, recall, F1, balanced accuracy, and ROC AUC rather than accuracy alone.

---

## My Contribution

I focused on designing and evaluating the modeling workflow:

- Built a stratified **80/20 train-test split** to preserve the minority purchase rate.
- Compared Logistic Regression, LASSO Logistic Regression, PCA + Logistic Regression, Random Forest, Gradient Boosting, and Radial SVM.
- Used **5-fold cross-validation** on the training set for hyperparameter tuning.
- Selected classification thresholds from training-set cross-validated predictions to improve purchase-class F1.
- Evaluated final models only on the held-out test set.
- Added a PageValues sensitivity analysis to separate predictive performance from causal interpretation.

---

## Final Test-Set Results

| Model | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0.25 | 0.877 | 0.828 | 0.577 | 0.759 | 0.655 | **0.922** |
| Random Forest | 0.30 | 0.882 | **0.836** | 0.590 | **0.769** | **0.667** | 0.921 |
| Radial SVM | 0.14 | **0.886** | 0.810 | **0.614** | 0.701 | 0.654 | 0.894 |
| LASSO Logistic Regression | 0.15 | 0.867 | 0.808 | 0.553 | 0.722 | 0.626 | 0.892 |
| PCA + Logistic Regression | 0.27 | 0.880 | 0.766 | 0.612 | 0.601 | 0.607 | 0.883 |
| Logistic Regression | 0.24 | 0.871 | 0.771 | 0.575 | 0.627 | 0.600 | 0.881 |

**Main result:** Random Forest had the strongest F1 and balanced accuracy, while Gradient Boosting had the highest ROC AUC.

---

## Key Insights

- Purchase sessions had much higher PageValues, more product-page engagement, and lower ExitRates and BounceRates.
- PageValues was the strongest predictor in both Random Forest and Gradient Boosting.
- PCA did not improve performance, likely because useful class information was spread across many predictors rather than concentrated in a few components.
- Removing PageValues sharply reduced performance: Random Forest ROC AUC dropped from **0.921 to 0.762**, and F1 dropped from **0.667 to 0.414**.
- The strongest conclusion is predictive, not causal: PageValues is a powerful intent signal, but it should not be interpreted as direct proof that a page caused the purchase.

---

## Files

| File | Description |
| --- | --- |
| [online_shoppers_analysis.R](online_shoppers_analysis.R) | Full reproducible R analysis script. |
| [final-report.pdf](final-report.pdf) | Final project report with results, visuals, interpretation, and limitations. |
| [proposal.pdf](proposal.pdf) | Project proposal with research questions, data overview, and planned modeling approach. |
