# Print the mgcv references of tests/test_reml_mgcv_parity.py.
#
# The model is the one SuperGLM fits there: cubic P-splines with a second-order
# difference penalty, ten basis functions a smooth (mgcv bs = "ps", k = 10,
# m = c(2, 2); SuperGLM Spline(kind = "ps", n_knots = 6)), on SuperGLM's knots,
# by REML. SuperGLM places six interior knots evenly on the data's range, pads
# its ends by 0.1% of the range and adds three knots beyond each end at the end
# interval's width. mgcv's default spaces all fourteen evenly on the padded
# range, 0.002 from SuperGLM's here, which moves the Poisson x2's lambda by
# 0.35%, so the script passes mgcv SuperGLM's knots.
#
# mgcv scales each penalty by S.scale before it reports sp, so the smoothing
# parameter of the penalty as built is sp / S.scale. The Newton step left at
# mgcv's stop, H^-1 g in log sp from its outer gradient and Hessian, is how far
# its REML search stopped from the optimum. reml.scale is the scale that
# maximises the REML criterion, the estimator SuperGLM's phi is (mgcv's scale
# is the Pearson estimate). mgcv is GPL: this runs it as a black box and prints
# only what the fitted object reports.
#
#   Rscript tests/oracles/reml_parity_reference.R
suppressMessages(library(mgcv))
superglm_knots <- function(x) {
  r <- range(x)
  pad <- diff(r) * 0.001
  inner <- c(r[1] - pad, seq(r[1], r[2], length.out = 8)[2:7], r[2] + pad)
  c(inner[1] - (3:1) * (inner[2] - inner[1]), inner, inner[8] + (1:3) * (inner[8] - inner[7]))
}
for (fam in c("poisson", "gamma")) {
  d <- read.csv(file.path("tests", "fixtures", paste0("reml_parity_data_", fam, ".csv")))
  family <- if (fam == "poisson") poisson() else Gamma(link = "log")
  g <- gam(y ~ s(x1, bs = "ps", k = 10) + s(x2, bs = "ps", k = 10),
           family = family, data = d, method = "REML",
           knots = list(x1 = superglm_knots(d$x1), x2 = superglm_knots(d$x2)))
  lambda <- g$sp / sapply(g$smooth, function(s) s$S.scale)
  step <- solve(g$outer.info$hess, g$outer.info$grad)
  edf <- sapply(g$smooth, function(s) sum(g$edf[s$first.para:s$last.para]))
  cat(sprintf(
    "%s: lambda %.7g %.7g | edf %.6f %.6f total %.6f | deviance %.7f | reml.scale %.9g | step %s\n",
    fam, lambda[1], lambda[2], edf[1], edf[2], sum(g$edf), g$deviance, g$reml.scale,
    paste(signif(step, 3), collapse = ", ")
  ))
}
cat(R.version.string, "mgcv", as.character(packageVersion("mgcv")), "\n")
