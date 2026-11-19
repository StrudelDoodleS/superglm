# Pin mgcv REML fits of the fs factor smooth for tests/test_factor_smooth_mgcv_parity.py.
#
# mgcv's bs="fs" penalizes each null-space column of its base smooth separately
# (the smooth.construct.fs.smooth.spec help page), in whatever basis of the
# penalty's zero eigenspace its eigensolver returns. Run as a black box on these
# rows with xt=list(bs="ps"), its two null curves are a rotation of the constant
# and the line that swaps and changes sign when the rows are shuffled. With the
# default "tp" base, a shuffle rotates them freely. SuperGLM's fs puts null_j on
# degree j of the penalty's polynomial null space, orthonormalized over the rows.
# This script gives mgcv that model through paraPen: the per-level ps basis, its
# second-difference penalty and one rank-one penalty per null_j, which mgcv's
# REML then fits. mgcv is GPL: this script runs it as a black box and records
# only what the fitted object reports.
#
# The discrete case is fitted by bam(method="fREML", discrete=TRUE), the
# criterion SuperGLM's discrete fit follows; the others by gam(method="REML").
#
# Regenerate the fixture (R >= 4.5 with mgcv):
#   uv run python tests/oracles/factor_smooth_mgcv_reference.py export DIR
#   Rscript tests/oracles/factor_smooth_mgcv_reference.R DIR
#   uv run python tests/oracles/factor_smooth_mgcv_reference.py merge DIR

suppressMessages(library(mgcv))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1) stop("Usage: Rscript tests/oracles/factor_smooth_mgcv_reference.R DIR")
dir <- normalizePath(args[1], mustWork=TRUE)
num <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse=", "), "]")
read <- function(name, part) read.csv(file.path(dir, sprintf("%s_%s.csv", name, part)))

reference <- function(name) {
    rows <- read(name, "data")
    prediction <- read(name, "prediction_data")
    unseen <- read(name, "unseen_data")
    levels <- sort(unique(rows$f))
    base <- smoothCon(s(x, bs="ps", k=6, m=2), data=rows, absorb.cons=FALSE)[[1]]
    k <- ncol(base$X)
    # null_j: degree j of the polynomial coefficient sequences, orthonormal over the rows.
    G <- crossprod(base$X) / nrow(rows)
    P <- cbind(1, seq(-1, 1, length.out=k))
    N <- P %*% solve(chol(t(P) %*% G %*% P))
    D <- diff(diag(k), differences=2)
    penalties <- list(crossprod(D), G %*% tcrossprod(N[, 1]) %*% G, G %*% tcrossprod(N[, 2]) %*% G)
    design <- function(frame) {
        B <- PredictMat(base, frame)
        do.call(cbind, lapply(levels, function(level) B * (frame$f == level)))
    }
    rows$Xfs <- design(rows)
    global <- grepl("global", name)
    formula <- as.formula(paste0(
        "y ~ ", if (global) 's(x, bs="ps", k=7, m=2) + ' else "", "Xfs",
        if (is.null(rows$exposure)) "" else " + offset(log(exposure))"
    ))
    family <- if (name == "gaussian") gaussian() else poisson()
    paraPen <- list(Xfs=lapply(penalties, function(S) kronecker(diag(length(levels)), S)))
    discrete <- grepl("discrete", name)
    fit <- if (discrete) {
        bam(formula, data=rows, family=family, method="fREML", discrete=TRUE, paraPen=paraPen)
    } else {
        gam(formula, data=rows, family=family, method="REML", paraPen=paraPen,
            control=gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))
    }
    if (!isTRUE(fit$converged) || !(discrete || identical(fit$outer.info$conv, "full convergence"))) {
        stop(sprintf("%s: mgcv did not converge", name))
    }
    beta <- coef(fit)
    fs <- grep("^Xfs", names(beta))
    smooth <- if (global) fit$smooth[[1]] else NULL
    # The fitted values from the coefficients, as predict() on a discrete bam fit
    # takes no matrix covariate: f(x) for level l is B(x) beta_l.
    response <- function(frame, conditional) {
        eta <- rep(beta[[1]], nrow(frame))
        if (conditional) eta <- eta + drop(design(frame) %*% beta[fs])
        if (global) eta <- eta + drop(PredictMat(smooth, frame) %*% beta[smooth$first.para:smooth$last.para])
        if (!is.null(frame$exposure)) eta <- eta + log(frame$exposure)
        family$linkinv(eta)
    }
    lambdas <- fit$sp[c("Xfs1", "Xfs2", "Xfs3")]
    named <- function(values) sprintf(
        '{"wiggle": %.17g, "null_0": %.17g, "null_1": %.17g}', values[1], values[2], values[3]
    )
    fields <- c(
        sprintf('"unscaled_lambdas": %s', named(lambdas)),
        sprintf('"scale": %.17g', fit$sig2),
        sprintf('"variance_components": %s', named(fit$sig2 / lambdas)),
        sprintf('"factor_smooth_edf": %.17g', sum(fit$edf[fs])),
        sprintf('"total_edf": %.17g', sum(fit$edf)),
        sprintf('"deviance": %.17g', fit$deviance),
        sprintf('"intercept": %.17g', beta[[1]]),
        sprintf('"conditional_prediction": %s', num(response(prediction, TRUE))),
        sprintf('"population_prediction": %s', num(response(prediction, FALSE))),
        sprintf('"factor_smooth_link": %s', num(design(prediction) %*% beta[fs])),
        sprintf('"unseen_population_prediction": %s', num(response(unseen, FALSE)))
    )
    if (global) {
        fields <- c(
            fields,
            sprintf('"global_unscaled_lambda": %.17g', fit$sp[[smooth$label]] / smooth$S.scale),
            sprintf('"global_edf": %.17g', sum(fit$edf[smooth$first.para:smooth$last.para]))
        )
    }
    writeLines(c("{", paste0("  ", fields, collapse=",\n"), "}"), file.path(dir, sprintf("%s_reference.json", name)))
}

for (name in c("gaussian", "poisson", "poisson_global", "poisson_global_discrete")) reference(name)
writeLines(
    sprintf('{"r_version": "%s", "mgcv_version": "%s"}', R.version.string, packageVersion("mgcv")),
    file.path(dir, "versions.json")
)
