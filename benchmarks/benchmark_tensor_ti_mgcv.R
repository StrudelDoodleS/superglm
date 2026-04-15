# Discrete MTPL tensor benchmark — mgcv side.
#
# Uses the exact train/test split exported by benchmark_tensor_ti_freq.py and
# compares:
#
#   y_freq ~ s(DrivAge) + s(VehAge) + s(BonusMalus) + Area
#
# against:
#
#   y_freq ~ s(DrivAge) + s(VehAge) + s(BonusMalus) + ti(DrivAge, BonusMalus) + Area
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

fit_case <- function(name, formula_obj, train_df, test_df, discrete_bins) {
  t0 <- proc.time()
  fit <- bam(
    formula_obj,
    family = poisson(link = "log"),
    weights = train_df$Exposure,
    data = train_df,
    method = "fREML",
    discrete = discrete_bins
  )
  elapsed <- (proc.time() - t0)["elapsed"]

  pred <- as.numeric(predict(fit, newdata = test_df, type = "response"))
  gini_model <- weighted_gini(test_df$y_freq, pred, test_df$Exposure)
  sm <- summary(fit)
  edf_smooth <- if (!is.null(sm$s.table)) sum(sm$s.table[, "edf"]) else 0
  total_edf <- sum(fit$edf)

  list(
    model = name,
    fit_s = unname(elapsed),
    gini_model = unname(gini_model),
    effective_df = total_edf,
    smooth_edf = edf_smooth,
    converged = fit$converged,
    n_outer_iter = if (!is.null(fit$outer.info)) fit$outer.info$iter else NA,
    deviance = deviance(fit)
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

baseline <- fit_case(
  "mgcv_baseline_discrete",
  y_freq ~ s(DrivAge, k = 20, bs = "cr") +
    s(VehAge, k = 15, bs = "cr") +
    s(BonusMalus, k = 15, bs = "cr") +
    Area,
  train_df,
  test_df,
  discrete_bins = c(256, 256, 256)
)

with_ti <- fit_case(
  "mgcv_plus_ti_discrete",
  y_freq ~ s(DrivAge, k = 20, bs = "cr") +
    s(VehAge, k = 15, bs = "cr") +
    s(BonusMalus, k = 15, bs = "cr") +
    ti(DrivAge, BonusMalus, k = c(20, 15), bs = c("cr", "cr")) +
    Area,
  train_df,
  test_df,
  discrete_bins = c(256, 256, 256, 256)
)

rows <- list(baseline, with_ti)
out <- list(
  tool = "mgcv",
  dataset = "freMTPL2freq",
  split = "tensor_ti_freq_train/test.csv",
  discrete_bins = list(
    baseline = c(256, 256, 256),
    with_ti = c(256, 256, 256, 256)
  ),
  results = rows,
  deltas = list(
    fit_s = with_ti$fit_s - baseline$fit_s,
    gini_model = with_ti$gini_model - baseline$gini_model,
    effective_df = with_ti$effective_df - baseline$effective_df
  )
)

write(toJSON(out, auto_unbox = TRUE, pretty = TRUE), OUT_JSON)

cat(sprintf("%-24s fit=%7.2fs  gini=% .6f  edf=%8.2f  converged=%s\n",
            baseline$model, baseline$fit_s, baseline$gini_model,
            baseline$effective_df, baseline$converged))
cat(sprintf("%-24s fit=%7.2fs  gini=% .6f  edf=%8.2f  converged=%s\n",
            with_ti$model, with_ti$fit_s, with_ti$gini_model,
            with_ti$effective_df, with_ti$converged))
cat("\n")
cat(sprintf("Delta vs baseline: fit=%+.2fs  gini=%+.6f  edf=%+.2f\n",
            out$deltas$fit_s, out$deltas$gini_model, out$deltas$effective_df))
cat(sprintf("Saved JSON: %s\n", OUT_JSON))
