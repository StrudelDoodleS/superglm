# 30-rep timing study for mgcv::bam fREML discrete on MTPL2.
#
# Measures wall time for bam(method="fREML", discrete=TRUE) across 30
# repetitions with fixed thread count (nthreads=1).
# Reports median, mean, std, min, max.
#
# Usage:
#     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#         Rscript benchmarks/timing_30rep_mgcv.R

.libPaths(c("~/R/libs", .libPaths()))
suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

N_REPS <- 30
RESULTS_DIR <- file.path("benchmarks", "results")

# Report thread settings
for (var in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) {
  val <- Sys.getenv(var, unset = "(not set)")
  cat(sprintf("  %s=%s\n", var, val))
}

cat(sprintf("  R: %s, mgcv: %s\n", R.version.string, as.character(packageVersion("mgcv"))))

# Load data
csv_path <- file.path(RESULTS_DIR, "bench_mtpl2.csv")
if (!file.exists(csv_path)) {
  stop("bench_mtpl2.csv not found — run Python harness first")
}
df <- read.csv(csv_path)
df$Area <- factor(df$Area)
n <- nrow(df)
cat(sprintf("  MTPL2: %s rows\n", format(n, big.mark = ",")))

times <- numeric(N_REPS)
deviances <- numeric(N_REPS)
edfs <- numeric(N_REPS)

for (i in seq_len(N_REPS)) {
  t0 <- proc.time()
  fit <- bam(
    y_freq ~ s(DrivAge, k = 20, bs = "cr") +
             s(VehAge, k = 15, bs = "cr") +
             s(BonusMalus, k = 15, bs = "cr") +
             Area,
    family = poisson(link = "log"),
    weights = df$Exposure,
    data = df,
    method = "fREML",
    discrete = TRUE,
    nthreads = 1
  )
  elapsed <- (proc.time() - t0)["elapsed"]
  times[i] <- elapsed
  deviances[i] <- deviance(fit)
  edfs[i] <- sum(fit$edf)

  tag <- if (i == 1) " (warmup)" else ""
  cat(sprintf("  rep %2d/%d: %.3fs  dev=%.1f  edf=%.2f%s\n",
              i, N_REPS, elapsed, deviances[i], edfs[i], tag))
}

warmup <- times[1]
steady <- times[2:N_REPS]

result <- list(
  tool = "mgcv",
  method = "bam_fREML_discrete_weights",
  r_version = R.version.string,
  mgcv_version = as.character(packageVersion("mgcv")),
  n = n,
  n_reps = N_REPS,
  threads = Sys.getenv("OMP_NUM_THREADS", unset = "default"),
  nthreads_bam = 1,
  warmup_s = warmup,
  all_times_s = times,
  steady_times_s = as.numeric(steady),
  median_s = median(steady),
  mean_s = mean(steady),
  std_s = sd(steady),
  min_s = min(steady),
  max_s = max(steady),
  p10_s = quantile(steady, 0.1, names = FALSE),
  p90_s = quantile(steady, 0.9, names = FALSE),
  deviance = median(deviances),
  effective_df = median(edfs)
)

cat(sprintf("\n  === mgcv bam fREML discrete — %d reps (excluding warmup) ===\n", N_REPS))
cat(sprintf("  warmup:  %.3fs\n", warmup))
cat(sprintf("  median:  %.3fs\n", result$median_s))
cat(sprintf("  mean:    %.3fs\n", result$mean_s))
cat(sprintf("  std:     %.3fs\n", result$std_s))
cat(sprintf("  min:     %.3fs\n", result$min_s))
cat(sprintf("  max:     %.3fs\n", result$max_s))
cat(sprintf("  p10:     %.3fs\n", result$p10_s))
cat(sprintf("  p90:     %.3fs\n", result$p90_s))
cat(sprintf("  dev:     %.1f\n", result$deviance))
cat(sprintf("  edf:     %.2f\n", result$effective_df))

out_path <- file.path(RESULTS_DIR, "mgcv_30rep.json")
write(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out_path)
cat(sprintf("\n  Saved to %s\n", out_path))
