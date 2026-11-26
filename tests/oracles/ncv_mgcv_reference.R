# Pin mgcv's neighbourhood cross-validation fits for tests/test_fit_ncv_mgcv.py.
#
# Regenerate the fixture (R >= 4.5 with mgcv; DIR any empty directory):
#   uv run python tests/oracles/ncv_mgcv_reference.py export DIR
#   Rscript tests/oracles/ncv_mgcv_reference.R DIR
#   uv run python tests/oracles/ncv_mgcv_reference.py merge DIR
#
# Each case is the library's own design X (intercept first) and penalties S_j, fitted by
#   gam(y ~ X - 1, weights = w, paraPen = list(X = S), method = "NCV", nei = nei)
# with nei = list(a, ma, d, md) the library's neighbourhoods (the ?NCV help page): a the rows
# each neighbourhood removes, d the rows it scores, ma and md the end of each in a and d. No case
# has an offset: mgcv 1.9.3 scores held-out rows without it. The ordinary Bayesian covariance is
# read from a refit at the chosen sp with nei$jackknife = FALSE (?NCV: otherwise the variance
# estimate accounts for the neighbourhood structure); its score is the free fit's score at that
# sp. Its IRLS runs to epsilon = 1e-12 (?gam.control): at the default 1e-7 its Vp and edf are
# formed at weights about 5e-9 from those of its reported coefficients. A paraPen smoothing parameter is passed as paraPen = list(X = c(S, list(sp = ...))).
#
# mgcv is GPL: this script runs it as a black box and records only what the fitted object reports.
# It prints with base R (no jsonlite).

suppressMessages(library(mgcv))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript tests/oracles/ncv_mgcv_reference.R DIR")
dir <- normalizePath(args[1], mustWork = TRUE)
num <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse = ","), "]")
matrix_csv <- function(name) unname(as.matrix(read.csv(file.path(dir, name), header = FALSE)))
ints <- function(name) as.integer(scan(file.path(dir, name), quiet = TRUE))

for (family in c("gaussian", "poisson")) {
  X <- matrix_csv(sprintf("%s_X.csv", family))
  files <- sort(list.files(dir, pattern = sprintf("^%s_S[0-9]+[.]csv$", family)))
  # the penalties are written as the library holds them; mgcv's root needs exact symmetry
  S <- lapply(files, function(f) { M <- matrix_csv(f); (M + t(M)) / 2 })
  rows <- read.csv(file.path(dir, sprintf("%s_rows.csv", family)))
  dat <- data.frame(y = rows$y, w = rows$w)
  dat$X <- X
  fam <- if (family == "gaussian") gaussian() else poisson()
  for (form in c("rows", "groups", "folds", "neighbours", "forward")) {
    part <- function(p) ints(sprintf("%s_%s_%s.txt", family, form, p))
    nei <- list(a = part("a"), ma = part("ma"), d = part("d"), md = part("md"))
    # a Poisson rate is not an integer, which only the family's AIC remarks on
    free <- suppressWarnings(gam(y ~ X - 1, data = dat, family = fam, weights = w,
                                 paraPen = list(X = S), method = "NCV", nei = nei))
    bayes <- suppressWarnings(gam(y ~ X - 1, data = dat, family = fam, weights = w,
                                  paraPen = list(X = c(S, list(sp = free$sp))), method = "NCV",
                                  nei = c(nei, list(jackknife = FALSE)),
                                  control = gam.control(epsilon = 1e-12)))
    fields <- c(
      sprintf('"sp":%s', num(free$sp)),
      sprintf('"score":%.17g', free$gcv.ubre),
      sprintf('"score_at_sp":%.17g', bayes$gcv.ubre),
      sprintf('"edf":%.17g', sum(free$edf)),
      sprintf('"scale":%.17g', free$sig2),
      sprintf('"deviance":%.17g', deviance(free)),
      sprintf('"outer_iterations":%d', as.integer(free$outer.info$iter)),
      sprintf('"outer_convergence":"%s"', free$outer.info$conv),
      sprintf('"outer_gradient":%s', num(free$outer.info$grad)),
      sprintf('"coefficients":%s', num(coef(free))),
      sprintf('"fitted":%s', num(fitted(free))),
      sprintf('"se":%s', num(sqrt(diag(free$Vp)))),
      sprintf('"se_bayesian":%s', num(sqrt(diag(bayes$Vp)))),
      sprintf('"coefficients_bayesian":%s', num(coef(bayes))),
      sprintf('"edf_bayesian":%.17g', sum(bayes$edf)),
      sprintf('"scale_bayesian":%.17g', bayes$sig2)
    )
    writeLines(paste0("{", paste(fields, collapse = ","), "}"),
               file.path(dir, sprintf("%s_%s_reference.json", family, form)))
    cat(family, form, "sp", signif(free$sp, 6), "score", sprintf("%.12g", free$gcv.ubre),
        "iter", free$outer.info$iter, free$outer.info$conv, "\n")
  }
}
writeLines(sprintf('{"r_version":"%s","mgcv_version":"%s"}', R.version.string,
                   as.character(packageVersion("mgcv"))),
           file.path(dir, "versions.json"))
