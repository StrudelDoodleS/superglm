# Generate pinned mgcv bs="fs" reference values for SuperGLM parity tests.
#
# Usage from the repository root:
#   Rscript tests/fixtures/factor_smooth_mgcv_reference.R
#
# The script deliberately uses only base R plus mgcv. In particular, it does
# not require jsonlite, so the fixture is reproducible in a minimal R install.

library(mgcv)

json_escape <- function(value) {
  value <- gsub("\\", "\\\\", value, fixed = TRUE)
  value <- gsub("\"", "\\\"", value, fixed = TRUE)
  value <- gsub("\n", "\\n", value, fixed = TRUE)
  value <- gsub("\r", "\\r", value, fixed = TRUE)
  value <- gsub("\t", "\\t", value, fixed = TRUE)
  paste0("\"", value, "\"")
}

to_json <- function(value) {
  if (is.null(value)) {
    return("null")
  }
  if (is.factor(value)) {
    value <- as.character(value)
  }
  if (is.list(value)) {
    named <- !is.null(names(value)) && all(nzchar(names(value)))
    entries <- vapply(value, to_json, character(1), USE.NAMES = FALSE)
    if (named) {
      fields <- paste0(json_escape(names(value)), ":", entries)
      return(paste0("{", paste(fields, collapse = ","), "}"))
    }
    return(paste0("[", paste(entries, collapse = ","), "]"))
  }
  if (is.character(value)) {
    entries <- vapply(value, json_escape, character(1), USE.NAMES = FALSE)
  } else if (is.logical(value)) {
    entries <- ifelse(is.na(value), "null", ifelse(value, "true", "false"))
  } else if (is.numeric(value)) {
    entries <- ifelse(
      is.finite(value),
      vapply(value, function(x) sprintf("%.17g", x), character(1)),
      "null"
    )
  } else {
    stop(paste("Unsupported JSON value type:", typeof(value)))
  }
  if (length(entries) == 1) {
    return(entries)
  }
  paste0("[", paste(entries, collapse = ","), "]")
}

factor_smooth_reference <- function(
  fit,
  prediction_data,
  unseen_reference_data,
  term_label = "s(x,f)",
  global_label = NULL
) {
  smooth_index <- which(vapply(
    fit$smooth,
    function(item) identical(item$label, term_label),
    logical(1)
  ))
  if (length(smooth_index) != 1) {
    stop("Could not uniquely identify the factor smooth")
  }
  smooth <- fit$smooth[[smooth_index]]
  sp_index <- which(startsWith(names(fit$sp), term_label))
  if (length(sp_index) != 3) {
    stop("Expected wiggle plus two null-space smoothing parameters")
  }
  unscaled_lambda <- unname(fit$sp[sp_index] / smooth$S.scale)
  names(unscaled_lambda) <- c("wiggle", "null_0", "null_1")
  scale <- unname(summary(fit)$scale)

  output <- list(
    raw_mgcv_sp = list(
      wiggle = unname(fit$sp[sp_index[1]]),
      null_0 = unname(fit$sp[sp_index[2]]),
      null_1 = unname(fit$sp[sp_index[3]])
    ),
    penalty_scale = list(
      wiggle = unname(smooth$S.scale[1]),
      null_0 = unname(smooth$S.scale[2]),
      null_1 = unname(smooth$S.scale[3])
    ),
    unscaled_lambdas = list(
      wiggle = unname(unscaled_lambda[["wiggle"]]),
      null_0 = unname(unscaled_lambda[["null_0"]]),
      null_1 = unname(unscaled_lambda[["null_1"]])
    ),
    scale = scale,
    variance_components = list(
      wiggle = scale / unname(unscaled_lambda[["wiggle"]]),
      null_0 = scale / unname(unscaled_lambda[["null_0"]]),
      null_1 = scale / unname(unscaled_lambda[["null_1"]])
    ),
    factor_smooth_edf = unname(sum(
      fit$edf[smooth$first.para:smooth$last.para]
    )),
    total_edf = unname(sum(fit$edf)),
    deviance = unname(deviance(fit)),
    intercept = unname(coef(fit)[["(Intercept)"]]),
    conditional_prediction = unname(predict(
      fit,
      newdata = prediction_data,
      type = "response"
    )),
    population_prediction = unname(predict(
      fit,
      newdata = prediction_data,
      type = "response",
      exclude = term_label
    )),
    factor_smooth_link = unname(predict(
      fit,
      newdata = prediction_data,
      type = "terms",
      terms = term_label
    )),
    unseen_population_prediction = unname(predict(
      fit,
      newdata = unseen_reference_data,
      type = "response",
      exclude = term_label
    ))
  )

  if (!is.null(global_label)) {
    global_index <- which(vapply(
      fit$smooth,
      function(item) identical(item$label, global_label),
      logical(1)
    ))
    if (length(global_index) != 1) {
      stop("Could not uniquely identify the global smooth")
    }
    global_smooth <- fit$smooth[[global_index]]
    global_sp_index <- which(names(fit$sp) == global_label)
    output$global_unscaled_lambda <- unname(
      fit$sp[global_sp_index] / global_smooth$S.scale[1]
    )
    output$global_edf <- unname(sum(
      fit$edf[global_smooth$first.para:global_smooth$last.para]
    ))
  }
  output
}

gaussian_levels <- sprintf("g%02d", seq_len(5))
gaussian_f <- factor(
  rep(gaussian_levels, each = 30),
  levels = gaussian_levels
)
gaussian_x <- rep(seq(-1, 1, length.out = 30), times = 5) +
  rep(seq(-0.03, 0.03, length.out = 5), each = 30)
gaussian_amplitude <- c(0.70, -0.45, 0.30, -0.60, 0.20)
gaussian_y <- 1.10 +
  0.25 * sin(2.3 * gaussian_x) +
  gaussian_amplitude[as.integer(gaussian_f)] *
    (gaussian_x + 0.30 * gaussian_x^2) +
  0.05 * cos(seq_along(gaussian_x) * 1.7)
gaussian_data <- data.frame(
  x = gaussian_x,
  f = gaussian_f,
  y = gaussian_y
)
gaussian_curve_grid <- seq(-0.9, 0.9, length.out = 9)
gaussian_prediction_data <- expand.grid(
  x = gaussian_curve_grid,
  f = gaussian_levels,
  KEEP.OUT.ATTRS = FALSE
)
gaussian_prediction_data$f <- factor(
  gaussian_prediction_data$f,
  levels = gaussian_levels
)
gaussian_unseen_data <- data.frame(
  x = c(-0.82, 0.05, 0.88),
  f = factor(rep(gaussian_levels[1], 3), levels = gaussian_levels)
)
gaussian_fit <- gam(
  y ~ s(x, f, bs = "fs", k = 6, xt = list(bs = "ps"), m = 2),
  family = gaussian(),
  data = gaussian_data,
  method = "REML"
)

set.seed(20260726)
poisson_levels <- sprintf("p%02d", seq_len(6))
poisson_f <- factor(
  rep(poisson_levels, each = 45),
  levels = poisson_levels
)
poisson_x <- rep(seq(-1.3, 1.3, length.out = 45), times = 6) +
  rep(seq(-0.045, 0.045, length.out = 6), each = 45)
poisson_exposure <- 0.45 + ((seq_along(poisson_x) * 19) %% 37) / 25
poisson_amplitude <- c(0.55, -0.38, 0.30, -0.48, 0.18, -0.12)
poisson_eta <- -0.42 +
  0.34 * sin(2.15 * poisson_x) +
  poisson_amplitude[as.integer(poisson_f)] *
    (poisson_x + 0.28 * poisson_x^2)
poisson_y <- rpois(
  length(poisson_x),
  poisson_exposure * exp(poisson_eta)
)
poisson_data <- data.frame(
  x = poisson_x,
  f = poisson_f,
  exposure = poisson_exposure,
  y = poisson_y
)
poisson_curve_grid <- seq(-1.15, 1.15, length.out = 11)
poisson_prediction_data <- expand.grid(
  x = poisson_curve_grid,
  f = poisson_levels,
  KEEP.OUT.ATTRS = FALSE
)
poisson_prediction_data$f <- factor(
  poisson_prediction_data$f,
  levels = poisson_levels
)
poisson_prediction_data$exposure <- rep(
  c(0.65, 1.20, 1.85),
  length.out = nrow(poisson_prediction_data)
)
poisson_unseen_data <- data.frame(
  x = c(-1.0, 0.1, 1.05),
  f = factor(rep(poisson_levels[1], 3), levels = poisson_levels),
  exposure = c(0.70, 1.25, 1.90)
)

poisson_fit <- gam(
  y ~ s(x, f, bs = "fs", k = 6, xt = list(bs = "ps"), m = 2) +
    offset(log(exposure)),
  family = poisson(),
  data = poisson_data,
  method = "REML"
)
poisson_global_fit <- gam(
  y ~ s(x, bs = "ps", k = 7, m = 2) +
    s(x, f, bs = "fs", k = 6, xt = list(bs = "ps"), m = 2) +
    offset(log(exposure)),
  family = poisson(),
  data = poisson_data,
  method = "REML"
)
poisson_global_discrete_fit <- bam(
  y ~ s(x, bs = "ps", k = 7, m = 2) +
    s(x, f, bs = "fs", k = 6, xt = list(bs = "ps"), m = 2) +
    offset(log(exposure)),
  family = poisson(),
  data = poisson_data,
  method = "fREML",
  discrete = TRUE,
  nthreads = 1
)

gaussian_case <- list(
  data = list(
    x = unname(gaussian_x),
    f = as.character(gaussian_f),
    y = unname(gaussian_y)
  ),
  curve_grid = unname(gaussian_curve_grid),
  prediction_data = list(
    x = unname(gaussian_prediction_data$x),
    f = as.character(gaussian_prediction_data$f)
  ),
  unseen_data = list(x = unname(gaussian_unseen_data$x)),
  reference = factor_smooth_reference(
    gaussian_fit,
    gaussian_prediction_data,
    gaussian_unseen_data
  )
)

poisson_case <- function(fit, global = FALSE) {
  list(
    data = list(
      x = unname(poisson_x),
      f = as.character(poisson_f),
      exposure = unname(poisson_exposure),
      y = unname(poisson_y)
    ),
    curve_grid = unname(poisson_curve_grid),
    prediction_data = list(
      x = unname(poisson_prediction_data$x),
      f = as.character(poisson_prediction_data$f),
      exposure = unname(poisson_prediction_data$exposure)
    ),
    unseen_data = list(
      x = unname(poisson_unseen_data$x),
      exposure = unname(poisson_unseen_data$exposure)
    ),
    reference = factor_smooth_reference(
      fit,
      poisson_prediction_data,
      poisson_unseen_data,
      global_label = if (global) "s(x)" else NULL
    )
  )
}

fixture <- list(
  metadata = list(
    generated_at = "2026-07-24",
    r_version = R.version.string,
    mgcv_version = as.character(packageVersion("mgcv")),
    poisson_seed = 20260726,
    factor_smooth = "s(x, f, bs=\"fs\", k=6, xt=list(bs=\"ps\"), m=2)"
  ),
  gaussian = gaussian_case,
  poisson = poisson_case(poisson_fit),
  poisson_global = poisson_case(poisson_global_fit, global = TRUE),
  poisson_global_discrete = poisson_case(
    poisson_global_discrete_fit,
    global = TRUE
  )
)

arguments <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(arguments) > 0) {
  arguments[[1]]
} else {
  "tests/fixtures/factor_smooth_mgcv_reference.json"
}
writeLines(to_json(fixture), output_path)
cat("Wrote", output_path, "\n")
