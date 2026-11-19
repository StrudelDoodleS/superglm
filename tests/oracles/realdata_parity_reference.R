# Print the mgcv references of MGCV_GAMMA in tests/test_realdata_parity.py.
#
# The severity book is freMTPL2's claim amounts summed per policy, joined to
# the policies' clipped features: the test module's own _load_gamma_data, which
# reads the local freMTPL2 parquet files (scripts/fetch_fremtpl.py) and so is
# written out first. The model is the test's: cubic regression splines
# (bs = "cr") of DrivAge (k = 15), VehAge and BonusMalus (k = 10), and Area, a
# Gamma log-link fit by REML. scale is mgcv's Pearson estimate. mgcv is GPL:
# this runs it as a black box and prints only what the fitted object reports.
#
#   uv run python -c "from tests.test_realdata_parity import _load_gamma_data as book; \
#     book()[0].to_csv('severity_book.csv', index=False)"
#   Rscript tests/oracles/realdata_parity_reference.R severity_book.csv
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript tests/oracles/realdata_parity_reference.R BOOK.csv")
suppressMessages(library(mgcv))
d <- read.csv(args[1])
d$Area <- factor(d$Area)
g <- gam(ClaimAmount ~ s(DrivAge, bs = "cr", k = 15) + s(VehAge, bs = "cr", k = 10) +
           s(BonusMalus, bs = "cr", k = 10) + Area,
         family = Gamma(link = "log"), data = d, method = "REML")
edf <- sapply(g$smooth, function(s) sum(g$edf[s$first.para:s$last.para]))
cat(sprintf(
  "gamma: rows %d | deviance %.2f | scale %.2f | edf %s total %.2f | intercept %.3f | pred_mean %.1f | pred_cor_y %.5f\n",
  nrow(d), g$deviance, g$scale, paste(sprintf("%.2f", edf), collapse = " "), sum(g$edf),
  coef(g)[["(Intercept)"]], mean(fitted(g)), cor(d$ClaimAmount, fitted(g))
))
cat(R.version.string, "mgcv", as.character(packageVersion("mgcv")), "\n")
