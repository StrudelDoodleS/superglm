# REML benchmark harness — mgcv side.
#
# Runs mgcv::gam (REML) and mgcv::bam (fREML, discrete=TRUE) on the same
# datasets exported by the Python companion script.
#
# Usage:
#   Rscript benchmarks/reml_benchmark_harness.R [--no-mtpl2]
#
# Outputs:
#   benchmarks/results/mgcv_results.json

.libPaths(c("~/R/libs", .libPaths()))
suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
skip_mtpl2 <- "--no-mtpl2" %in% args

RESULTS_DIR <- file.path("benchmarks", "results")

cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("mgcv version: %s\n", as.character(packageVersion("mgcv"))))
cat(strrep("=", 60), "\n")
cat("mgcv REML Benchmark Harness\n")
cat(strrep("=", 60), "\n\n")

results <- list()

# ── Helper: extract structured results from a gam/bam fit ────────

extract_results <- function(fit, elapsed, name, n, family_name, method) {
  sm <- summary(fit)
  edf_smooth <- if (!is.null(sm$s.table)) sum(sm$s.table[, "edf"]) else 0
  # Total edf = smooth edf + parametric terms (intercept + categoricals)
  total_edf <- sum(fit$edf) + 1  # +1 for intercept

  # Extract smoothing parameters
  sp_list <- as.list(fit$sp)
  names(sp_list) <- names(fit$sp)

  list(
    name = name,
    n = n,
    family = family_name,
    method = method,
    wall_time_s = elapsed,
    deviance = deviance(fit),
    effective_df = total_edf,
    smooth_edf = edf_smooth,
    scale = fit$scale,
    n_outer_iter = if (!is.null(fit$outer.info)) fit$outer.info$iter else NA,
    converged = fit$converged,
    smoothing_params = sp_list
  )
}

# ── Synthetic benchmarks ─────────────────────────────────────────

run_synthetic <- function(family_name) {
  csv_path <- file.path(RESULTS_DIR, sprintf("bench_synthetic_%s.csv", family_name))
  if (!file.exists(csv_path)) {
    cat(sprintf("  %s not found — run Python harness first\n", csv_path))
    return(list())
  }

  df <- read.csv(csv_path)
  n <- nrow(df)
  cat(sprintf("  Loaded %s (n=%d)\n", csv_path, n))

  if (family_name == "poisson") {
    fam <- poisson(link = "log")
  } else if (family_name == "gamma") {
    fam <- Gamma(link = "log")
  }

  # mgcv::gam with REML, bs="cr" k=10 (comparable to CRS n_knots=8)
  label <- sprintf("synthetic_%s_gam_reml", family_name)
  cat(sprintf("  Running %s...", label))
  flush.console()
  t0 <- proc.time()
  fit <- gam(
    y ~ s(x1, k = 10, bs = "cr") + s(x2, k = 10, bs = "cr"),
    family = fam,
    data = df,
    method = "REML"
  )
  elapsed <- (proc.time() - t0)["elapsed"]
  r <- extract_results(fit, elapsed, label, n, family_name, "gam_REML")
  cat(sprintf(" %.2fs, dev=%.1f, edf=%.2f\n", elapsed, r$deviance, r$effective_df))

  list(r)
}

cat("── Synthetic benchmarks ──\n")
for (fam in c("poisson", "gamma")) {
  results <- c(results, run_synthetic(fam))
}

# ── MTPL2 benchmarks ─────────────────────────────────────────────

if (!skip_mtpl2) {
  cat("\n── MTPL2 benchmarks ──\n")

  csv_path <- file.path(RESULTS_DIR, "bench_mtpl2.csv")
  if (!file.exists(csv_path)) {
    # Fall back to the prepared CSV
    csv_path <- file.path("scratch", "r_experiments", "mtpl2_prepared.csv")
  }

  if (file.exists(csv_path)) {
    df <- read.csv(csv_path)
    df$Area <- factor(df$Area)
    n <- nrow(df)
    cat(sprintf("  Loaded %s (n=%d)\n", csv_path, n))

    # For Poisson gam: need integer counts + offset
    if ("y_freq" %in% names(df) && "Exposure" %in% names(df)) {
      df$ClaimNb <- round(df$y_freq * df$Exposure)
      df$logExposure <- log(df$Exposure)
    }

    # mgcv::gam with REML on 200k subsample (full dataset too slow for gam)
    SUB_N <- min(200000, n)
    set.seed(42)
    idx <- sample(n, SUB_N)
    df_sub <- df[idx, ]

    label <- "mtpl2_poisson_gam_reml_200k"
    cat(sprintf("  Running %s (n=%d)...", label, SUB_N))
    flush.console()
    t0 <- proc.time()
    fit_gam <- gam(
      ClaimNb ~ s(DrivAge, k = 20, bs = "cr") +
                s(VehAge, k = 15, bs = "cr") +
                s(BonusMalus, k = 15, bs = "cr") +
                Area +
                offset(logExposure),
      family = poisson(link = "log"),
      data = df_sub,
      method = "REML"
    )
    elapsed_gam <- (proc.time() - t0)["elapsed"]
    r_gam <- extract_results(fit_gam, elapsed_gam, label, SUB_N, "poisson", "gam_REML")
    cat(sprintf(" %.2fs, dev=%.1f, edf=%.2f\n", elapsed_gam, r_gam$deviance, r_gam$effective_df))
    results <- c(results, list(r_gam))

    # mgcv::bam with fREML on full dataset
    label <- "mtpl2_poisson_bam_freml"
    cat(sprintf("  Running %s (n=%d)...", label, n))
    flush.console()
    t0 <- proc.time()
    fit_bam <- bam(
      ClaimNb ~ s(DrivAge, k = 20, bs = "cr") +
                s(VehAge, k = 15, bs = "cr") +
                s(BonusMalus, k = 15, bs = "cr") +
                Area +
                offset(logExposure),
      family = poisson(link = "log"),
      data = df,
      method = "fREML",
      discrete = TRUE
    )
    elapsed_bam <- (proc.time() - t0)["elapsed"]
    r_bam <- extract_results(fit_bam, elapsed_bam, label, n, "poisson", "bam_fREML")
    cat(sprintf(" %.2fs, dev=%.1f, edf=%.2f\n", elapsed_bam, r_bam$deviance, r_bam$effective_df))
    results <- c(results, list(r_bam))

    # mgcv::bam with fREML, weights-based (rate response + exposure weights)
    label <- "mtpl2_poisson_bam_freml_weights"
    cat(sprintf("  Running %s (n=%d)...", label, n))
    flush.console()
    t0 <- proc.time()
    fit_bam_w <- bam(
      y_freq ~ s(DrivAge, k = 20, bs = "cr") +
               s(VehAge, k = 15, bs = "cr") +
               s(BonusMalus, k = 15, bs = "cr") +
               Area,
      family = poisson(link = "log"),
      weights = df$Exposure,
      data = df,
      method = "fREML",
      discrete = TRUE
    )
    elapsed_bam_w <- (proc.time() - t0)["elapsed"]
    r_bam_w <- extract_results(fit_bam_w, elapsed_bam_w, label, n, "poisson", "bam_fREML_weights")
    cat(sprintf(" %.2fs, dev=%.1f, edf=%.2f\n", elapsed_bam_w, r_bam_w$deviance, r_bam_w$effective_df))
    results <- c(results, list(r_bam_w))

  } else {
    cat("  MTPL2 data not found, skipping\n")
  }
}

# ── Write results ────────────────────────────────────────────────

out <- list(
  tool = "mgcv",
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  r_version = R.version.string,
  mgcv_version = as.character(packageVersion("mgcv")),
  results = results
)

out_path <- file.path(RESULTS_DIR, "mgcv_results.json")
write(toJSON(out, auto_unbox = TRUE, pretty = TRUE), out_path)
cat(sprintf("\nResults written to %s\n", out_path))
