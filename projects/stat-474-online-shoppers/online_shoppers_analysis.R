# STAT 474 final project analysis
# Dataset: UCI Online Shoppers Purchasing Intention
#
# This script contains only the analyses used in the final report:
#   1. data overview, class balance, EDA, outlier and correlation checks
#   2. PCA diagnostic for the PCA + Logistic Regression baseline
#   3. six-model comparison on a stratified held-out test set
#   4. threshold selection from training-set cross-validated predictions
#   5. predicted-intent grouping, variable importance, PageValues profiling,
#      and PageValues sensitivity analysis
#
# By default it writes report tables to prelim_outputs/ and report figures to
# figures/. Set STAT474_CONSOLE_ONLY=1 or run with --console-only to print the
# same research outputs without writing files.

suppressPackageStartupMessages({
  required_pkgs <- c(
    "tidyverse", "caret", "pROC", "glmnet", "randomForest",
    "gbm", "kernlab"
  )
  missing_pkgs <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
  if (length(missing_pkgs) > 0) {
    stop(
      "Missing required R packages: ",
      paste(missing_pkgs, collapse = ", "),
      ". Install them before running this script."
    )
  }
  invisible(lapply(required_pkgs, library, character.only = TRUE))
})

set.seed(474)

args_all <- commandArgs(trailingOnly = FALSE)
args_trailing <- commandArgs(trailingOnly = TRUE)
console_only <- Sys.getenv("STAT474_CONSOLE_ONLY") == "1" ||
  "--console-only" %in% args_trailing

file_arg <- "--file="
script_path <- sub(file_arg, "", args_all[grepl(file_arg, args_all)])
script_dir <- if (length(script_path) > 0 && file.exists(script_path[1])) {
  dirname(normalizePath(script_path[1]))
} else {
  getwd()
}
if (!file.exists(file.path(script_dir, "online_shoppers_intention.csv")) &&
    file.exists(file.path(getwd(), "market", "online_shoppers_intention.csv"))) {
  script_dir <- file.path(getwd(), "market")
}

data_path <- file.path(script_dir, "online_shoppers_intention.csv")
out_dir <- file.path(script_dir, "prelim_outputs")
fig_dir <- file.path(script_dir, "figures")

if (!console_only) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
}

save_table <- function(x, filename) {
  if (!console_only) {
    readr::write_csv(x, file.path(out_dir, filename))
  }
  invisible(x)
}

save_figure <- function(filename, plot, width, height, dpi = 180) {
  if (!console_only) {
    ggplot2::ggsave(
      filename = file.path(fig_dir, filename),
      plot = plot,
      width = width,
      height = height,
      dpi = dpi
    )
  }
  invisible(plot)
}

df_raw <- readr::read_csv(data_path, show_col_types = FALSE)

numeric_behavior_vars <- c(
  "Administrative", "Administrative_Duration",
  "Informational", "Informational_Duration",
  "ProductRelated", "ProductRelated_Duration",
  "BounceRates", "ExitRates", "PageValues", "SpecialDay"
)

categorical_vars <- c(
  "Month", "OperatingSystems", "Browser", "Region",
  "TrafficType", "VisitorType", "Weekend"
)

df <- df_raw %>%
  mutate(
    RevenueClass = factor(
      if_else(Revenue, "Purchase", "NoPurchase"),
      levels = c("Purchase", "NoPurchase")
    ),
    RevenueNum = as.integer(Revenue)
  ) %>%
  mutate(across(all_of(categorical_vars), as.factor))

model_df <- df %>%
  dplyr::select(all_of(numeric_behavior_vars), all_of(categorical_vars), RevenueClass)

# -----------------------------------------------------------------------------
# 1. Data overview and EDA used in the report
# -----------------------------------------------------------------------------
dataset_overview <- tibble(
  observations = nrow(df_raw),
  predictors = ncol(df_raw) - 1,
  behavior_numeric_predictors = length(numeric_behavior_vars),
  categorical_predictors = length(categorical_vars),
  missing_values = sum(is.na(df_raw))
)

target_distribution <- df %>%
  count(RevenueClass, name = "n") %>%
  mutate(percent = n / sum(n))

variable_overview <- tibble(
  variable = names(df_raw),
  source_class = purrr::map_chr(df_raw, ~ class(.x)[1]),
  role = case_when(
    variable %in% numeric_behavior_vars ~ "behavior numeric predictor",
    variable %in% categorical_vars ~ "categorical predictor",
    variable == "Revenue" ~ "binary response",
    TRUE ~ "not used"
  ),
  n_missing = purrr::map_int(df_raw, ~ sum(is.na(.x))),
  n_unique = purrr::map_int(df_raw, dplyr::n_distinct)
)

numeric_summary <- df_raw %>%
  summarise(
    across(
      all_of(numeric_behavior_vars),
      list(
        min = ~ min(.x, na.rm = TRUE),
        q1 = ~ quantile(.x, 0.25, na.rm = TRUE),
        median = ~ median(.x, na.rm = TRUE),
        mean = ~ mean(.x, na.rm = TRUE),
        q3 = ~ quantile(.x, 0.75, na.rm = TRUE),
        max = ~ max(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE)
      )
    )
  ) %>%
  pivot_longer(
    everything(),
    names_to = c("feature", "metric"),
    names_pattern = "(.+)_(min|q1|median|mean|q3|max|sd)$",
    values_to = "value"
  ) %>%
  pivot_wider(names_from = metric, values_from = value)

numeric_by_revenue <- purrr::map_dfr(numeric_behavior_vars, function(v) {
  x_no <- df_raw[[v]][!df_raw$Revenue]
  x_yes <- df_raw[[v]][df_raw$Revenue]
  pooled_sd <- sd(df_raw[[v]], na.rm = TRUE)
  mean_difference <- mean(x_yes, na.rm = TRUE) - mean(x_no, na.rm = TRUE)
  tibble(
    feature = v,
    mean_no_purchase = mean(x_no, na.rm = TRUE),
    mean_purchase = mean(x_yes, na.rm = TRUE),
    mean_difference = mean_difference,
    standardized_difference = if_else(pooled_sd == 0, 0, mean_difference / pooled_sd),
    median_no_purchase = median(x_no, na.rm = TRUE),
    median_purchase = median(x_yes, na.rm = TRUE)
  )
}) %>%
  arrange(desc(abs(standardized_difference)))

numeric_target_correlation <- tibble(
  feature = numeric_behavior_vars,
  correlation_with_purchase = purrr::map_dbl(
    numeric_behavior_vars,
    ~ suppressWarnings(cor(df_raw[[.x]], as.integer(df_raw$Revenue)))
  )
) %>%
  mutate(abs_correlation = abs(correlation_with_purchase)) %>%
  arrange(desc(abs_correlation))

cor_mat <- cor(df_raw[, numeric_behavior_vars], use = "pairwise.complete.obs")
cor_mat[lower.tri(cor_mat, diag = TRUE)] <- NA
high_correlations <- which(abs(cor_mat) >= 0.70, arr.ind = TRUE) %>%
  as_tibble() %>%
  transmute(
    var1 = rownames(cor_mat)[row],
    var2 = colnames(cor_mat)[col],
    correlation = cor_mat[cbind(row, col)]
  ) %>%
  arrange(desc(abs(correlation)))

outlier_summary <- purrr::map_dfr(numeric_behavior_vars, function(v) {
  q1 <- quantile(df_raw[[v]], 0.25, na.rm = TRUE)
  q3 <- quantile(df_raw[[v]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  n_outliers <- sum(df_raw[[v]] < lower | df_raw[[v]] > upper, na.rm = TRUE)
  tibble(
    feature = v,
    q1 = q1,
    q3 = q3,
    iqr = iqr,
    lower = lower,
    upper = upper,
    n_iqr_outliers = n_outliers,
    pct_iqr_outliers = n_outliers / nrow(df_raw)
  )
}) %>%
  arrange(desc(pct_iqr_outliers))

categorical_purchase_rates <- purrr::map_dfr(categorical_vars, function(v) {
  df %>%
    group_by(.data[[v]]) %>%
    summarise(
      n = n(),
      purchases = sum(RevenueClass == "Purchase"),
      purchase_rate = purchases / n,
      .groups = "drop"
    ) %>%
    mutate(feature = v) %>%
    rename(level = 1) %>%
    dplyr::select(feature, level, n, purchases, purchase_rate) %>%
    arrange(feature, desc(purchase_rate))
})

focus_categorical_purchase_rates <- categorical_purchase_rates %>%
  filter(feature %in% c("Month", "VisitorType", "Weekend"))

save_table(dataset_overview, "dataset_overview.csv")
save_table(target_distribution, "target_distribution.csv")
save_table(variable_overview, "variable_overview.csv")
save_table(numeric_summary, "numeric_summary.csv")
save_table(numeric_by_revenue, "numeric_by_revenue.csv")
save_table(numeric_target_correlation, "numeric_target_correlation.csv")
save_table(high_correlations, "high_correlations.csv")
save_table(outlier_summary, "outlier_summary.csv")
save_table(focus_categorical_purchase_rates, "focus_categorical_purchase_rates.csv")

theme_set(theme_minimal(base_size = 11))

target_plot <- target_distribution %>%
  mutate(label = paste0(n, "\n", sprintf("%.1f%%", 100 * percent))) %>%
  ggplot(aes(x = RevenueClass, y = percent, fill = RevenueClass)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = label), vjust = -0.25, size = 3.5) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 0.95)) +
  scale_fill_manual(values = c("Purchase" = "#2A9D8F", "NoPurchase" = "#6C757D")) +
  labs(title = "Target Distribution", x = NULL, y = "Share of sessions")
save_figure("target_distribution.png", target_plot, width = 5.8, height = 3.5)

behavior_gap_plot <- numeric_by_revenue %>%
  slice_max(abs(standardized_difference), n = 8) %>%
  mutate(
    feature = fct_reorder(feature, standardized_difference),
    direction = if_else(
      standardized_difference >= 0,
      "Higher in purchase sessions",
      "Lower in purchase sessions"
    )
  ) %>%
  ggplot(aes(x = feature, y = standardized_difference, fill = direction)) +
  geom_col(width = 0.72) +
  geom_hline(yintercept = 0, color = "gray40", linewidth = 0.4) +
  coord_flip() +
  scale_fill_manual(values = c(
    "Higher in purchase sessions" = "#2A9D8F",
    "Lower in purchase sessions" = "#D95F59"
  )) +
  labs(
    title = "Behavioral Signal Gap Between Purchase and NoPurchase Sessions",
    subtitle = "Standardized mean differences; positive values are higher among purchase sessions.",
    x = NULL,
    y = "Standardized difference"
  ) +
  theme(legend.position = "bottom", legend.title = element_blank())
save_figure("behavioral_signal_gap.png", behavior_gap_plot, width = 7.4, height = 4.8)

# -----------------------------------------------------------------------------
# 2. PCA diagnostic used to justify the PCA baseline interpretation
# -----------------------------------------------------------------------------
predictor_df <- model_df %>% dplyr::select(-RevenueClass)
dummy_encoder <- caret::dummyVars(~ ., data = predictor_df, fullRank = TRUE)
encoded_predictors <- as.data.frame(predict(dummy_encoder, newdata = predictor_df))
zero_var_cols <- caret::nearZeroVar(encoded_predictors)
if (length(zero_var_cols) > 0) {
  encoded_predictors <- encoded_predictors[, -zero_var_cols, drop = FALSE]
}
encoded_scaled <- scale(encoded_predictors)
pca_fit <- prcomp(encoded_scaled, center = FALSE, scale. = FALSE)
pca_variance <- pca_fit$sdev^2 / sum(pca_fit$sdev^2)
pca_cumulative_variance <- cumsum(pca_variance)

pca_diagnostic <- tibble(
  diagnostic = c(
    "Encoded predictor dimensions",
    "PC1 explained variance",
    "PCs for 80% variance",
    "PCs for 90% variance",
    "PCs for 95% variance"
  ),
  value = c(
    ncol(encoded_predictors),
    pca_variance[1],
    which(pca_cumulative_variance >= 0.80)[1],
    which(pca_cumulative_variance >= 0.90)[1],
    which(pca_cumulative_variance >= 0.95)[1]
  )
)

pca_variance_table <- tibble(
  component = paste0("PC", seq_along(pca_variance)),
  explained_variance = pca_variance,
  cumulative_variance = pca_cumulative_variance
)

save_table(pca_diagnostic, "pca_diagnostic.csv")
save_table(pca_variance_table, "pca_variance_table.csv")

# -----------------------------------------------------------------------------
# 3. Modeling and final evaluation
# -----------------------------------------------------------------------------
train_idx <- caret::createDataPartition(model_df$RevenueClass, p = 0.80, list = FALSE)
train_df <- model_df[train_idx, ]
test_df <- model_df[-train_idx, ]

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

model_formula <- RevenueClass ~ .

calc_metrics <- function(obs, probs, threshold) {
  pred <- factor(
    if_else(probs >= threshold, "Purchase", "NoPurchase"),
    levels = levels(obs)
  )
  cm <- confusionMatrix(pred, obs, positive = "Purchase")
  roc_auc <- pROC::roc(
    obs,
    probs,
    levels = c("NoPurchase", "Purchase"),
    quiet = TRUE
  )$auc
  precision <- as.numeric(cm$byClass["Pos Pred Value"])
  recall <- as.numeric(cm$byClass["Sensitivity"])
  f1 <- if_else(
    is.na(precision + recall) || precision + recall == 0,
    NA_real_,
    2 * precision * recall / (precision + recall)
  )
  tibble(
    Threshold = threshold,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    BalancedAccuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    Precision = precision,
    Recall = recall,
    Specificity = as.numeric(cm$byClass["Specificity"]),
    F1 = f1,
    ROC_AUC = as.numeric(roc_auc)
  )
}

confusion_counts <- function(obs, probs, threshold, model_name) {
  pred <- factor(
    if_else(probs >= threshold, "Purchase", "NoPurchase"),
    levels = levels(obs)
  )
  tab <- table(Predicted = pred, Actual = obs)
  tibble(
    Model = model_name,
    TN = as.integer(tab["NoPurchase", "NoPurchase"]),
    FP = as.integer(tab["Purchase", "NoPurchase"]),
    FN = as.integer(tab["NoPurchase", "Purchase"]),
    TP = as.integer(tab["Purchase", "Purchase"])
  )
}

filter_best_cv_predictions <- function(fit) {
  pred <- fit$pred
  if (!is.null(fit$bestTune) && nrow(fit$bestTune) > 0) {
    for (nm in names(fit$bestTune)) {
      pred <- pred[pred[[nm]] == fit$bestTune[[nm]], , drop = FALSE]
    }
  }
  pred
}

best_f1_threshold <- function(fit) {
  pred <- filter_best_cv_predictions(fit)
  thresholds <- seq(0.05, 0.95, by = 0.01)
  scores <- purrr::map_dfr(thresholds, function(threshold) {
    calc_metrics(pred$obs, pred$Purchase, threshold) %>%
      dplyr::select(Threshold, Precision, Recall, F1)
  })
  scores %>%
    arrange(desc(F1), desc(Recall), desc(Precision)) %>%
    slice(1) %>%
    pull(Threshold)
}

evaluate_model <- function(fit, model_name, newdata, threshold = NULL) {
  probs <- predict(fit, newdata = newdata, type = "prob")[, "Purchase"]
  if (is.null(threshold)) {
    threshold <- best_f1_threshold(fit)
  }
  calc_metrics(newdata$RevenueClass, probs, threshold) %>%
    mutate(Model = model_name, .before = 1)
}

train_logit <- function(data) {
  train(
    model_formula,
    data = data,
    method = "glm",
    family = binomial(),
    metric = "ROC",
    preProcess = c("zv", "center", "scale"),
    trControl = ctrl
  )
}

train_lasso <- function(data) {
  train(
    model_formula,
    data = data,
    method = "glmnet",
    metric = "ROC",
    tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4, 0.5, length = 20)),
    preProcess = c("zv", "center", "scale"),
    trControl = ctrl
  )
}

train_pca_logit <- function(data) {
  train(
    model_formula,
    data = data,
    method = "glm",
    family = binomial(),
    metric = "ROC",
    preProcess = c("zv", "center", "scale", "pca"),
    trControl = ctrl
  )
}

train_rf <- function(data) {
  train(
    model_formula,
    data = data,
    method = "rf",
    metric = "ROC",
    tuneGrid = expand.grid(mtry = c(3, 5, 7)),
    ntree = 300,
    trControl = ctrl
  )
}

train_gbm <- function(data) {
  train(
    model_formula,
    data = data,
    method = "gbm",
    metric = "ROC",
    tuneGrid = expand.grid(
      interaction.depth = c(1, 3),
      n.trees = c(100, 200, 300),
      shrinkage = 0.05,
      n.minobsinnode = 10
    ),
    verbose = FALSE,
    trControl = ctrl
  )
}

train_svm_radial <- function(data) {
  train(
    model_formula,
    data = data,
    method = "svmRadial",
    metric = "ROC",
    tuneGrid = expand.grid(
      sigma = c(0.005, 0.01, 0.02),
      C = c(0.5, 1, 2)
    ),
    preProcess = c("zv", "center", "scale"),
    trControl = ctrl
  )
}

cat("\nFitting final model set...\n")
set.seed(474)
fit_logit <- train_logit(train_df)
set.seed(474)
fit_lasso <- train_lasso(train_df)
set.seed(474)
fit_pca_logit <- train_pca_logit(train_df)
set.seed(474)
fit_rf <- train_rf(train_df)
set.seed(474)
fit_gbm <- train_gbm(train_df)
set.seed(474)
fit_svm <- train_svm_radial(train_df)

fits <- list(
  "Logistic Regression" = fit_logit,
  "LASSO Logistic Regression" = fit_lasso,
  "PCA + Logistic Regression" = fit_pca_logit,
  "Random Forest" = fit_rf,
  "Gradient Boosting" = fit_gbm,
  "Radial SVM" = fit_svm
)

model_thresholds <- purrr::imap_dfr(fits, function(fit, model_name) {
  tibble(Model = model_name, Threshold = best_f1_threshold(fit))
})

model_benchmark <- purrr::imap_dfr(fits, function(fit, model_name) {
  threshold <- model_thresholds %>%
    filter(Model == model_name) %>%
    pull(Threshold)
  evaluate_model(fit, model_name, test_df, threshold)
}) %>%
  arrange(desc(ROC_AUC))

model_confusion_counts <- purrr::imap_dfr(fits, function(fit, model_name) {
  threshold <- model_thresholds %>%
    filter(Model == model_name) %>%
    pull(Threshold)
  probs <- predict(fit, newdata = test_df, type = "prob")[, "Purchase"]
  confusion_counts(test_df$RevenueClass, probs, threshold, model_name)
}) %>%
  mutate(Model = factor(Model, levels = model_benchmark$Model)) %>%
  arrange(Model) %>%
  mutate(Model = as.character(Model))

best_tuning <- tibble(
  Model = c(
    "Logistic Regression",
    "LASSO Logistic Regression",
    "PCA + Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "Radial SVM"
  ),
  Selected_Tuning_Setting = c(
    "default glm after preprocessing",
    paste0(
      "alpha = ", fit_lasso$bestTune$alpha,
      ", lambda = ", signif(fit_lasso$bestTune$lambda, 4)
    ),
    "default glm after preprocessing",
    paste0("mtry = ", fit_rf$bestTune$mtry),
    paste0(
      "trees = ", fit_gbm$bestTune$n.trees,
      ", depth = ", fit_gbm$bestTune$interaction.depth,
      ", shrinkage = ", fit_gbm$bestTune$shrinkage
    ),
    paste0(
      "sigma = ", fit_svm$bestTune$sigma,
      ", C = ", fit_svm$bestTune$C
    )
  )
)

rf_importance <- varImp(fit_rf)$importance %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall))

gbm_importance <- varImp(fit_gbm)$importance %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall))

save_table(model_thresholds, "model_thresholds.csv")
save_table(model_benchmark, "final_model_benchmark.csv")
save_table(model_confusion_counts, "model_confusion_counts.csv")
save_table(best_tuning, "best_tuning_parameters.csv")
save_table(rf_importance, "rf_variable_importance.csv")
save_table(gbm_importance, "gbm_variable_importance.csv")

roc_data <- purrr::imap_dfr(fits, function(fit, model_name) {
  probs <- predict(fit, newdata = test_df, type = "prob")[, "Purchase"]
  roc_obj <- pROC::roc(
    test_df$RevenueClass,
    probs,
    levels = c("NoPurchase", "Purchase"),
    quiet = TRUE
  )
  pROC::coords(
    roc_obj,
    x = "all",
    ret = c("specificity", "sensitivity"),
    transpose = FALSE
  ) %>%
    as_tibble() %>%
    mutate(fpr = 1 - specificity, Model = model_name)
})

roc_plot <- ggplot(roc_data, aes(x = fpr, y = sensitivity, color = Model)) +
  geom_line(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray55") +
  coord_equal() +
  labs(
    title = "ROC Curves on Held-Out Test Set",
    x = "False positive rate",
    y = "True positive rate"
  ) +
  theme(legend.position = "bottom")
save_figure("roc_curves.png", roc_plot, width = 7.2, height = 5.2)

performance_plot <- model_benchmark %>%
  dplyr::select(Model, ROC_AUC, F1) %>%
  mutate(Model = fct_reorder(Model, ROC_AUC, .fun = max)) %>%
  pivot_longer(-Model, names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = Model, y = value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  scale_fill_manual(values = c("F1" = "#2A9D8F", "ROC_AUC" = "#1F4D78")) +
  labs(
    title = "Final Model Comparison on Held-Out Test Set",
    subtitle = "F1 reflects the Purchase-class precision/recall balance; ROC AUC reflects overall separation.",
    x = NULL,
    y = "Metric value"
  ) +
  theme(legend.position = "bottom")
save_figure("model_performance.png", performance_plot, width = 8, height = 5.4)

importance_plot_df <- bind_rows(
  rf_importance %>% slice_head(n = 10) %>% mutate(Model = "Random Forest"),
  gbm_importance %>% slice_head(n = 10) %>% mutate(Model = "Gradient Boosting")
) %>%
  mutate(feature = fct_reorder(feature, Overall))

importance_plot <- ggplot(importance_plot_df, aes(x = feature, y = Overall, fill = Model)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  coord_flip() +
  facet_wrap(~ Model, scales = "free_y") +
  labs(
    title = "What the Best Models Learned",
    subtitle = "Page value and product engagement dominate the ensemble models.",
    x = NULL,
    y = "Relative importance"
  )
save_figure("variable_importance.png", importance_plot, width = 8, height = 5.4)

# -----------------------------------------------------------------------------
# 4. Prediction-to-intent grouping
# -----------------------------------------------------------------------------
rf_test_probabilities <- predict(fit_rf, newdata = test_df, type = "prob")[, "Purchase"]
intent_group_purchase_rates <- tibble(
  observed = test_df$RevenueClass,
  predicted_purchase_probability = rf_test_probabilities
) %>%
  mutate(
    intent_rank = ntile(predicted_purchase_probability, 3),
    Intent_Group = factor(
      case_when(
        intent_rank == 1 ~ "Low predicted intent",
        intent_rank == 2 ~ "Medium predicted intent",
        TRUE ~ "High predicted intent"
      ),
      levels = c("Low predicted intent", "Medium predicted intent", "High predicted intent")
    )
  ) %>%
  group_by(Intent_Group) %>%
  summarise(
    n = n(),
    purchases = sum(observed == "Purchase"),
    observed_purchase_rate = purchases / n,
    mean_predicted_probability = mean(predicted_purchase_probability),
    .groups = "drop"
  )

save_table(intent_group_purchase_rates, "intent_group_purchase_rates.csv")

overall_purchase_rate <- mean(test_df$RevenueClass == "Purchase")
intent_group_plot <- intent_group_purchase_rates %>%
  ggplot(aes(x = Intent_Group, y = observed_purchase_rate, fill = Intent_Group)) +
  geom_col(width = 0.62, show.legend = FALSE) +
  geom_hline(
    yintercept = overall_purchase_rate,
    linetype = "dashed",
    color = "gray45",
    linewidth = 0.55
  ) +
  geom_text(
    aes(label = paste0(scales::percent(observed_purchase_rate, accuracy = 0.1), "\n", "n=", n)),
    vjust = -0.25,
    size = 3.5
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 0.55)) +
  scale_fill_manual(values = c(
    "Low predicted intent" = "#7A8A99",
    "Medium predicted intent" = "#4C9CA0",
    "High predicted intent" = "#2A9D8F"
  )) +
  labs(
    title = "Observed Purchase Rate by Predicted Intent Group",
    subtitle = "Random Forest probabilities on the held-out test set, split into three equal-sized groups.",
    x = NULL,
    y = "Observed purchase rate"
  )
save_figure("intent_group_purchase_rate.png", intent_group_plot, width = 6.8, height = 4.4)

# -----------------------------------------------------------------------------
# 5. PageValues profile and sensitivity analysis
# -----------------------------------------------------------------------------
top_month_string <- function(months, n_show = 3) {
  month_rate <- prop.table(table(months))
  month_tbl <- tibble(
    month = names(month_rate),
    pct = as.numeric(month_rate)
  ) %>%
    arrange(desc(pct), month)
  cutoff <- month_tbl$pct[min(n_show, nrow(month_tbl))]
  month_tbl %>%
    filter(pct >= cutoff - 1e-12) %>%
    mutate(label = paste0(month, " ", scales::percent(pct, accuracy = 0.1))) %>%
    pull(label) %>%
    paste(collapse = ", ")
}

pagevalue_q90 <- quantile(df$PageValues, 0.90, na.rm = TRUE, names = FALSE)
pagevalue_profile_df <- df %>%
  mutate(
    PageValues_Group = factor(
      case_when(
        PageValues == 0 ~ "PageValues = 0",
        PageValues >= pagevalue_q90 ~ "Top 10% PageValues",
        TRUE ~ "Positive but not top 10%"
      ),
      levels = c("PageValues = 0", "Positive but not top 10%", "Top 10% PageValues")
    )
  )

pagevalues_group_summary <- pagevalue_profile_df %>%
  group_by(PageValues_Group) %>%
  summarise(
    n = n(),
    purchase_rate = mean(RevenueClass == "Purchase"),
    product_pages = mean(ProductRelated),
    exit_rate = mean(ExitRates),
    bounce_rate = mean(BounceRates),
    .groups = "drop"
  ) %>%
  mutate(
    behavior_profile = case_when(
      PageValues_Group == "PageValues = 0" ~ "Mostly ordinary browsing",
      PageValues_Group == "Positive but not top 10%" ~ "Deep product browsing",
      TRUE ~ "24.7% new visitors; concentrated in Nov, May, Dec"
    )
  )

pagevalues_detailed_profile <- pagevalue_profile_df %>%
  group_by(PageValues_Group) %>%
  summarise(
    mean_pagevalues = mean(PageValues),
    product_duration_sec = mean(ProductRelated_Duration),
    admin_pages = mean(Administrative),
    info_pages = mean(Informational),
    new_visitors = mean(VisitorType == "New_Visitor"),
    weekend = mean(Weekend == TRUE),
    top_months = top_month_string(Month),
    .groups = "drop"
  )

save_table(pagevalues_group_summary, "pagevalues_group_summary.csv")
save_table(pagevalues_detailed_profile, "pagevalues_detailed_profile.csv")

cat("\nRunning PageValues sensitivity analysis...\n")
train_no_page <- train_df %>% dplyr::select(-PageValues)
test_no_page <- test_df %>% dplyr::select(-PageValues)

set.seed(474)
fit_logit_no_page <- train_logit(train_no_page)
set.seed(474)
fit_rf_no_page <- train_rf(train_no_page)
set.seed(474)
fit_gbm_no_page <- train_gbm(train_no_page)

sensitivity_fits <- list(
  "Logistic Regression" = fit_logit,
  "Random Forest" = fit_rf,
  "Gradient Boosting" = fit_gbm
)
sensitivity_no_page_fits <- list(
  "Logistic Regression" = fit_logit_no_page,
  "Random Forest" = fit_rf_no_page,
  "Gradient Boosting" = fit_gbm_no_page
)

pagevalues_sensitivity <- bind_rows(
  purrr::imap_dfr(
    sensitivity_fits,
    ~ evaluate_model(.x, .y, newdata = test_df) %>%
      mutate(Feature_Set = "Full model")
  ),
  purrr::imap_dfr(
    sensitivity_no_page_fits,
    ~ evaluate_model(.x, .y, newdata = test_no_page) %>%
      mutate(Feature_Set = "Without PageValues")
  )
) %>%
  relocate(Feature_Set, .after = Model) %>%
  arrange(Model, Feature_Set)

save_table(pagevalues_sensitivity, "pagevalues_sensitivity.csv")

sensitivity_plot <- pagevalues_sensitivity %>%
  dplyr::select(Model, Feature_Set, ROC_AUC, F1) %>%
  pivot_longer(c(ROC_AUC, F1), names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = Model, y = value, fill = Feature_Set)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.62) +
  facet_wrap(~ metric, nrow = 1) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  scale_fill_manual(values = c("Full model" = "#2A9D8F", "Without PageValues" = "#D95F59")) +
  labs(
    title = "Sensitivity Analysis: Removing PageValues",
    x = NULL,
    y = "Metric value"
  ) +
  theme(legend.position = "bottom")
save_figure("pagevalues_sensitivity.png", sensitivity_plot, width = 7.6, height = 4.2)

# -----------------------------------------------------------------------------
# 6. Concise console summary
# -----------------------------------------------------------------------------
cat("\nDataset overview:\n")
print(dataset_overview)

cat("\nTarget distribution:\n")
print(target_distribution)

cat("\nTop EDA differences by Revenue class:\n")
print(numeric_by_revenue %>% slice_head(n = 8))

cat("\nHigh numeric correlations used to motivate regularization/tree models:\n")
print(high_correlations)

cat("\nIQR outlier summary for behavior variables:\n")
print(outlier_summary)

cat("\nPCA diagnostic:\n")
print(pca_diagnostic)

cat("\nBest tuning parameters:\n")
print(best_tuning)

cat("\nFinal model benchmark with CV-selected F1 thresholds:\n")
print(model_benchmark)

cat("\nConfusion counts on held-out test set:\n")
print(model_confusion_counts)

cat("\nObserved purchase rate by predicted intent group:\n")
print(intent_group_purchase_rates)

cat("\nTop Random Forest variable importance:\n")
print(rf_importance %>% slice_head(n = 10))

cat("\nTop Gradient Boosting variable importance:\n")
print(gbm_importance %>% slice_head(n = 10))

cat("\nPageValues group summary:\n")
print(pagevalues_group_summary)

cat("\nDetailed PageValues behavior profile:\n")
print(pagevalues_detailed_profile)

cat("\nPageValues sensitivity:\n")
print(pagevalues_sensitivity)

if (console_only) {
  cat("\nConsole-only mode complete. No files or images were written.\n")
} else {
  cat("\nOutputs written to:\n")
  cat("  Tables:", out_dir, "\n")
  cat("  Figures:", fig_dir, "\n")
}
