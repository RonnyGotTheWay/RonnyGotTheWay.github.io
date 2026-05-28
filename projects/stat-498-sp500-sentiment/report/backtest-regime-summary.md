# Backtest, Regime Evaluation, and Return Modeling Summary

This report uses calibrated direction-model probabilities and validation-selected thresholds for the held-out test set.

## Test Classification, All Features

| model_name   | regime                    |   n |   threshold |   accuracy |   balanced_accuracy |    auc |     f1 |   up_day_rate |
|:-------------|:--------------------------|----:|------------:|-----------:|--------------------:|-------:|-------:|--------------:|
| Logit        | 2022-2023 rate-hike cycle | 495 |        0.54 |     0.4848 |              0.4978 | 0.4991 | 0.6521 |        0.4869 |
| Logit        | Other                     |  42 |        0.54 |     0.5952 |              0.5    | 0.6071 | 0.7463 |        0.5952 |
| XGB          | 2022-2023 rate-hike cycle | 495 |        0.5  |     0.4869 |              0.5    | 0.5132 | 0.6549 |        0.4869 |
| XGB          | Other                     |  42 |        0.5  |     0.5952 |              0.5    | 0.4894 | 0.7463 |        0.5952 |

## Test Long/Flat Backtest, All Features

| model_name   |   threshold |   strategy_cumulative_return |   buy_hold_cumulative_return |   strategy_sharpe |   strategy_max_drawdown |   exposure |   trades |
|:-------------|------------:|-----------------------------:|-----------------------------:|------------------:|------------------------:|-----------:|---------:|
| Logit        |        0.54 |                       0.0715 |                       0.0697 |            0.2657 |                 -0.2524 |     0.9944 |        7 |
| XGB          |        0.5  |                       0.0696 |                       0.0697 |            0.2611 |                 -0.2538 |     1      |        1 |

## Robustness Snapshot

Robustness checks vary threshold and transaction costs. Thresholds are selected on validation for the main test report.

| split         | model_name   |   threshold |   transaction_cost |   strategy_cumulative_return |   strategy_sharpe |   strategy_max_drawdown |   exposure |   trades |
|:--------------|:-------------|------------:|-------------------:|-----------------------------:|------------------:|------------------------:|-----------:|---------:|
| oos_expanding | XGB          |        0.5  |             0      |                       5.2185 |            0.8292 |                 -0.3392 |     0.9518 |      175 |
| oos_expanding | XGB          |        0.5  |             0.0001 |                       5.1106 |            0.8222 |                 -0.3392 |     0.9518 |      175 |
| oos_expanding | XGB          |        0.5  |             0.0005 |                       4.6972 |            0.7938 |                 -0.3392 |     0.9518 |      175 |
| oos_expanding | Logit        |        0.4  |             0      |                       4.6592 |            0.7535 |                 -0.3392 |     1      |        1 |
| oos_expanding | Logit        |        0.4  |             0.0001 |                       4.6587 |            0.7535 |                 -0.3392 |     1      |        1 |
| oos_expanding | Logit        |        0.4  |             0.0005 |                       4.6564 |            0.7533 |                 -0.3392 |     1      |        1 |
| oos_expanding | Logit        |        0.45 |             0      |                       4.4744 |            0.7427 |                 -0.3392 |     0.9941 |       33 |
| oos_expanding | Logit        |        0.45 |             0.0001 |                       4.4564 |            0.7415 |                 -0.3392 |     0.9941 |       33 |

## Continuous Return Modeling

| model        | feature_set   |    rmse |     mae |       r2 |   information_coefficient |   direction_accuracy |
|:-------------|:--------------|--------:|--------:|---------:|--------------------------:|---------------------:|
| XGBRegressor | general       | 0.01193 | 0.00893 |  0.00518 |                   0.08974 |              0.51024 |
| Ridge        | all           | 0.01204 | 0.00911 | -0.01227 |                   0.08944 |              0.54749 |
| Ridge        | general       | 0.01198 | 0.00902 | -0.00354 |                   0.07837 |              0.49534 |
| Ridge        | vader         | 0.012   | 0.00903 | -0.00556 |                   0.04301 |              0.49348 |
| Ridge        | dict          | 0.01202 | 0.00906 | -0.01003 |                   0.03845 |              0.49162 |
| XGBRegressor | dict          | 0.01218 | 0.00916 | -0.03641 |                   0.02202 |              0.48976 |
| XGBRegressor | all           | 0.01221 | 0.00907 | -0.04224 |                   0.0109  |              0.54935 |
| Ridge        | finbert       | 0.01201 | 0.00903 | -0.00766 |                  -0.01158 |              0.50093 |

## Interpretation

- Balanced class weighting and calibrated probabilities make the models less dependent on raw accuracy.
- Thresholds are chosen on validation data, then applied to the test set.
- Predictive value remains weak; improvements should be framed as methodological cleanup rather than proof of a reliable trading edge.
