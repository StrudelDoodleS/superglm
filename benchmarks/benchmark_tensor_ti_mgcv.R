# Discrete MTPL tensor benchmark — mgcv side.
#
# Uses the exact train/test split exported by benchmark_tensor_ti_freq.py and
# compares:
#
#   baseline
#   +1 tensor
#   +2 tensors
#   +3 tensors
#   +1 spline-by-categorical
#   +2 spline-by-categorical
#   mixed tensor + spline-by-categorical
#
# with bam(..., discrete=TRUE, method="fREML"), so the result can be used as a
# behavior oracle without copying any mgcv source into SuperGLM.

.libPaths(c("~/R/libs", .libPaths()))
suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

RESULTS_DIR <- file.path("benchmarks", "results")
TRAIN_CSV <- file.path(RESULTS_DIR, "tensor_ti_freq_train.csv")
TEST_CSV <- file.path(RESULTS_DIR, "tensor_ti_freq_test.csv")
OUT_JSON <- file.path(RESULTS_DIR, "tensor_ti_mgcv.json")

weighted_gini <- function(y, mu, w) {
  ord <- order(mu)
  y <- y[ord]
  w <- w[ord]
  yw <- y * w
  total_w <- sum(w)
  total_yw <- sum(yw)
  if (total_w <= 0 || total_yw <= 0) {
    return(NA_real_)
  }
  cw <- cumsum(w) / total_w
  cy <- cumsum(yw) / total_yw
  area <- sum(diff(c(0, cw)) * (head(c(0, cy), -1) + cy) / 2)
  1 - 2 * area
}

base_formula <- y_freq ~ s(DrivAge, k = 20, bs = "cr") +
  s(VehAge, k = 15, bs = "cr") +
  s(BonusMalus, k = 15, bs = "cr") +
  Area

case_definitions <- list(
  list(
    name = "mgcv_baseline_discrete",
    interactions = c(),
    formula = base_formula
  ),
  list(
    name = "mgcv_baseline_plus_ti_discrete",
    interactions = c("DrivAge:BonusMalus"),
    formula = y_freq ~ s(DrivAge, k = 20, bs = "cr") +
      s(VehAge, k = 15, bs = "cr") +
      s(BonusMalus, k = 15, bs = "cr") +
      ti(DrivAge, BonusMalus, k = c(20, 15), bs = c("cr", "cr")) +
      Area
  ),
  list(
    name = "mgcv_baseline_plus_2_tensors_discrete",
    interactions = c("DrivAge:BonusMalus", "DrivAge:VehAge"),
    formula = y_freq ~ s(DrivAge, k = 20, bs = "cr") +
      s(VehAge, k = 15, bs = "cr") +
      s(BonusMalus, k = 15, bs = "cr") +
      ti(DrivAge, BonusMalus, k = c(20, 15), bs = c("cr", "cr")) +
      ti(DrivAge, VehAge, k = c(20, 15), bs = c("cr", "cr")) +
      Area
  ),
  list(
    name = "mgcv_baseline_plus_3_tensors_discrete",
    interactions = c("DrivAge:BonusMalus", "DrivAge:VehAge", "VehAge:BonusMalus"),
    formula = y_freq ~ s(DrivAge, k = 20, bs = "cr") +
      s(VehAge, k = 15, bs = "cr") +
      s(BonusMalus, k = 15, bs = "cr") +
      ti(DrivAge, BonusMalus, k = c(20, 15), bs = c("cr", "cr")) +
      ti(DrivAge, VehAge, k = c(20, 15), bs = c("cr", "cr")) +
      ti(VehAge, BonusMalus, k = c(15, 15), bs = c("cr", "cr")) +
      Area
  ),
  list(
    name = "mgcv_baseline_plus_spline_cat_discrete",
    interactions = c("DrivAge:Area"),
    formula = y_freq ~ s(DrivAge, k = 20, bs = "cr") +
      s(VehAge, k = 15, bs = "cr") +
      s(BonusMalus, k = 15, bs = "cr") +
      s(DrivAge, by = Area, k = 20, bs = "cr") +
      Area
  ),
  list(
    name = "mgcv_baseline_plus_2_spline_cat_discrete",
    interactions = c("DrivAge:Area", "BonusMalus:Area"),
    formula = y_freq ~ s(DrivAge, k = 20, bs = "cr") +
      s(VehAge, k = 15, bs = "cr") +
      s(BonusMalus, k = 15, bs = "cr") +
      s(DrivAge, by = Area, k = 20, bs = "cr") +
      s(BonusMalus, by = Area, k = 15, bs = "cr") +
      Area
  ),
  list(
    name = "mgcv_baseline_plus_mixed_tensor_spline_cat_discrete",
    interactions = c("DrivAge:BonusMalus", "VehAge:Area"),
    formula = y_freq ~ s(DrivAge, k = 20, bs = "cr") +
      s(VehAge, k = 15, bs = "cr") +
      s(BonusMalus, k = 15, bs = "cr") +
      ti(DrivAge, BonusMalus, k = c(20, 15), bs = c("cr", "cr")) +
      s(VehAge, by = Area, k = 15, bs = "cr") +
      Area
  )
)

fit_case <- function(case_def, train_df, test_df) {
  t0 <- proc.time()
  fit <- bam(
    case_def$formula,
    family = poisson(link = "log"),
    weights = train_df$Exposure,
    data = train_df,
    method = "fREML",
    discrete = TRUE
  )
  elapsed <- (proc.time() - t0)["elapsed"]

  pred <- as.numeric(predict(fit, newdata = test_df, type = "response"))
  gini_model <- weighted_gini(test_df$y_freq, pred, test_df$Exposure)
  sm <- summary(fit)
  edf_smooth <- if (!is.null(sm$s.table)) sum(sm$s.table[, "edf"]) else 0
  total_edf <- sum(fit$edf)

  list(
    model = case_def$name,
    interactions = case_def$interactions,
    n_interactions = length(case_def$interactions),
    fit_s = unname(elapsed),
    gini_model = unname(gini_model),
    effective_df = total_edf,
    smooth_edf = edf_smooth,
    converged = fit$converged,
    n_outer_iter = if (!is.null(fit$outer.info)) fit$outer.info$iter else NA,
    deviance = deviance(fit)
  )
}

delta_row <- function(row, baseline) {
  list(
    fit_s = row$fit_s - baseline$fit_s,
    gini_model = row$gini_model - baseline$gini_model,
    effective_df = row$effective_df - baseline$effective_df,
    n_outer_iter = row$n_outer_iter - baseline$n_outer_iter
  )
}

if (!file.exists(TRAIN_CSV) || !file.exists(TEST_CSV)) {
  stop("Missing tensor benchmark split CSVs. Run benchmark_tensor_ti_freq.py first.")
}

train_df <- read.csv(TRAIN_CSV)
test_df <- read.csv(TEST_CSV)
train_df$Area <- factor(train_df$Area)
test_df$Area <- factor(test_df$Area, levels = levels(train_df$Area))

cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("mgcv version: %s\n", as.character(packageVersion("mgcv"))))
cat(strrep("=", 72), "\n")
cat("Discrete MTPL tensor benchmark (mgcv)\n")
cat(strrep("=", 72), "\n")

rows <- lapply(case_definitions, fit_case, train_df = train_df, test_df = test_df)
baseline <- rows[[1]]
with_ti <- rows[[2]]
by_case <- list()
for (row in rows[-1]) {
  by_case[[row$model]] <- delta_row(row, baseline)
}

out <- list(
  tool = "mgcv",
  dataset = "freMTPL2freq",
  split = "tensor_ti_freq_train/test.csv",
  discrete = TRUE,
  case_matrix = lapply(case_definitions, function(case_def) {
    list(name = case_def$name, interactions = case_def$interactions)
  }),
  results = rows,
  deltas = list(
    fit_s = with_ti$fit_s - baseline$fit_s,
    gini_model = with_ti$gini_model - baseline$gini_model,
    effective_df = with_ti$effective_df - baseline$effective_df,
    by_case = by_case
  )
)

write(toJSON(out, auto_unbox = TRUE, pretty = TRUE), OUT_JSON)

for (row in rows) {
  cat(sprintf("%-56s fit=%7.2fs  gini=% .6f  edf=%8.2f  outer=%s  converged=%s\n",
              row$model, row$fit_s, row$gini_model,
              row$effective_df, row$n_outer_iter, row$converged))
}
cat("\n")
cat(sprintf("Delta vs baseline: fit=%+.2fs  gini=%+.6f  edf=%+.2f\n",
            out$deltas$fit_s, out$deltas$gini_model, out$deltas$effective_df))
cat(sprintf("Saved JSON: %s\n", OUT_JSON))
