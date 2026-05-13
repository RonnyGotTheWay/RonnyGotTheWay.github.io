# Predicting Airbnb Listing Prices in New York City

**Course:** DATA 467 / Linear Regression<br>
**Project type:** Regression modeling, diagnostics, applied pricing analysis<br>
**Dataset:** New York City Airbnb Open Data, 2019<br>

[Project page](index.html) | [Final paper](final-paper.pdf) | [Proposal](project-proposal.pdf) | [Notebook](code/data_analysis.ipynb) | [Dataset](data/AB_NYC_2019.csv)

---

## Project Overview

This project asks what factors most strongly influence nightly Airbnb listing prices in New York City. The analysis uses the 2019 NYC Airbnb dataset with **48,895 raw listings** across Brooklyn, Manhattan, Queens, the Bronx, and Staten Island.

Because raw price is strongly right-skewed, the main response variable is **log(price)**. After removing zero-price listings and rows with missing modeling variables, the analytic sample contains **38,833 listings**.

---

## Modeling Strategy

- Used ordinary least squares regression for log(price).
- Compared three specifications: categorical-only, categorical plus quantitative controls, and an enhanced model with interactions and log-transformed predictors.
- Treated **Private room** as the room-type reference group and **Brooklyn** as the borough reference group.
- Evaluated model fit with adjusted R2, AIC, BIC, and RMSE.
- Checked residuals, Q-Q behavior, scale-location patterns, leverage, Cook's distance, and VIF.
- Used a logistic high-price indicator as a second check on the main pricing structure.

---

## Main Results

| Model | Adjusted R2 | AIC | BIC | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Model 1 | 0.480 | 53000.55 | 53060.52 | 0.4787 |
| Model 2 | 0.498 | 51591.63 | 51694.43 | 0.4700 |
| Model 3 | **0.507** | **50910.83** | **51082.17** | **0.4658** |

Key findings:

- Room type and borough location were the dominant pricing factors.
- In Model 2, entire-home listings were about **118% higher** than private rooms on the expected price scale.
- Manhattan listings were about **35.6% higher** than Brooklyn listings after controls.
- Model 3 improved fit by adding room type by borough interactions and log-transforming skewed numeric predictors.
- The final model is useful for broad pricing patterns, but individual listing prediction remains uncertain because amenities, bedrooms, transit distance, seasonality, and booking prices are not observed.

---

## Files

| File | Description |
| --- | --- |
| [final-paper.pdf](final-paper.pdf) | Final written report. |
| [project-proposal.pdf](project-proposal.pdf) | Project proposal. |
| [code/data_analysis.ipynb](code/data_analysis.ipynb) | Reproducible Python notebook. |
| [data/AB_NYC_2019.csv](data/AB_NYC_2019.csv) | Original Airbnb NYC dataset. |
| [figures/](figures/) | Figures used in the project report and portfolio page. |
