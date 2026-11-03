# Pin scam's unpenalised shape-constrained Poisson fits for three tests in
# tests/test_shape_rules_reference.py: test_scam_shapes_hold,
# test_matches_scams_stated_problem_solved_exactly and test_scam_shapes_agree.
# The second solves the problem the paper states, on the knots recorded here,
# with a general-purpose solver; it reads the data and knots, not scam's fit.
#
# scam fits shape-constrained P-splines by reparameterising the coefficients
# with exponentiated elements (Pya, N. and Wood, S. N. (2015), "Shape
# constrained additive models", Statistics and Computing 25, 543-559; Pya
# Arnqvist, N. (2024), "On some extensions of shape-constrained generalized
# additive modelling in R", arXiv:2403.09438, section 2).  scam is GPL: this
# script runs it as a black box and records only what the fitted object reports
# (fitted values, linear predictor, EDF, knots, convergence); nothing here reads
# its source.
#
# Regenerate the fixture (R >= 4.5, scam in a private library):
#   mkdir -p ~/.cache/superglm-oracles/R-lib
#   R_LIBS_USER=~/.cache/superglm-oracles/R-lib Rscript -e \
#     'install.packages("scam", lib=Sys.getenv("R_LIBS_USER"), repos="https://cloud.r-project.org")'
#   uv run python tests/oracles/scam_shape_rules_data.py DATA_DIR
#   R_LIBS_USER=~/.cache/superglm-oracles/R-lib Rscript tests/oracles/scam_shape_rules.R \
#     DATA_DIR tests/fixtures/scam_shape_rules.json
#
# The JSON holds the data scam fitted (as R read them) beside its outputs, and
# the R, scam and mgcv versions, so the test needs neither R nor the generator.

args <- commandArgs(trailingOnly=TRUE)
usage <- "Usage: Rscript tests/oracles/scam_shape_rules.R DATA_DIR OUTPUT.json"
if (length(args) != 2) stop(usage)
data_dir <- normalizePath(args[1], mustWork=TRUE)
out <- args[2]
suppressPackageStartupMessages(library(scam))

shapes <- c("micv", "micx", "mdcv", "mdcx")
k <- 10L

num <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse=", "), "]")
str_ <- function(s) paste0('"', gsub('"', '\\\\"', s), '"')

cases <- character()
for (shape in shapes) {
    d <- read.csv(file.path(data_dir, paste0(shape, ".csv")))
    formula <- as.formula(sprintf('y ~ s(x, bs="%s", k=%d)', shape, k))
    fit <- scam(formula, family=poisson(), data=d, sp=0)
    smooth <- fit$smooth[[1]]
    fields <- c(
        sprintf('"formula": %s', str_(sprintf('y ~ s(x, bs="%s", k=%d)', shape, k))),
        '"family": "poisson"',
        '"sp": 0',
        sprintf('"data": {"x": %s, "y": %s}', num(d$x), num(d$y)),
        sprintf('"fitted_values": %s', num(fit$fitted.values)),
        sprintf('"linear_predictor": %s', num(fit$linear.predictors)),
        sprintf('"edf": %.17g', sum(fit$edf)),
        sprintf('"knots": %s', num(smooth$knots)),
        sprintf('"converged": %s', if (isTRUE(fit$conv)) "true" else "false"),
        sprintf('"deviance": %.17g', fit$deviance)
    )
    cases <- c(cases, sprintf('  %s: {\n    %s\n  }', str_(shape),
                              paste(fields, collapse=",\n    ")))
}

header <- c(
    sprintf('  "r_version": %s', str_(R.version.string)),
    sprintf('  "scam_version": %s', str_(as.character(packageVersion("scam")))),
    sprintf('  "mgcv_version": %s', str_(as.character(packageVersion("mgcv")))),
    sprintf('  "reference": %s', str_(paste(
        "Pya, N. and Wood, S. N. (2015), Shape constrained additive models,",
        "Statistics and Computing 25, 543-559")))
)
writeLines(c("{", paste(c(header, cases), collapse=",\n"), "}"), out)
