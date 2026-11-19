# Print the mgcv references of tests/test_reml_mgcv_parity.py.
#
# The model is the one SuperGLM fits there: cubic P-splines with a second-order
# difference penalty, ten basis functions a smooth (mgcv bs = "ps", k = 10,
# m = c(2, 2); SuperGLM Spline(kind = "ps", n_knots = 6)), by REML. mgcv scales
# each penalty by S.scale before it reports sp, so the smoothing parameter of
# the penalty as built is sp / S.scale. The Newton step left at mgcv's stop,
# H^-1 g in log sp from its outer gradient and Hessian, is how far its REML
# search stopped from the optimum. mgcv is GPL: this runs it as a black box and
# prints only what the fitted object reports.
#
#   Rscript tests/oracles/reml_parity_reference.R
suppressMessages(library(mgcv))
for (fam in c("poisson", "gamma")) {
  d <- read.csv(file.path("tests", "fixtures", paste0("reml_parity_data_", fam, ".csv")))
  family <- if (fam == "poisson") poisson() else Gamma(link = "log")
  g <- gam(y ~ s(x1, bs = "ps", k = 10) + s(x2, bs = "ps", k = 10),
           family = family, data = d, method = "REML")
  lambda <- g$sp / sapply(g$smooth, function(s) s$S.scale)
  step <- solve(g$outer.info$hess, g$outer.info$grad)
  cat(sprintf(
    "%s: lambda %.7g %.7g | edf %.6f %.6f total %.6f | deviance %.7f | scale %.7f | step %s\n",
    fam, lambda[1], lambda[2], sum(g$edf[g$smooth[[1]]$first.para:g$smooth[[1]]$last.para]),
    sum(g$edf[g$smooth[[2]]$first.para:g$smooth[[2]]$last.para]), sum(g$edf), g$deviance,
    g$scale, paste(signif(step, 3), collapse = ", ")
  ))
}
cat(R.version.string, "mgcv", as.character(packageVersion("mgcv")), "\n")
