# Generate pinned mgcv bs="re" reference values for SuperGLM parity tests.
#
# Usage from the repository root:
#   Rscript tests/fixtures/random_effect_mgcv_reference.R
#
# The script deliberately uses only base R plus mgcv.  In particular, it does
# not require jsonlite, so the committed fixture can be regenerated in a
# minimal R installation.

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

random_effect_reference <- function(fit, data, term_label = "s(level)") {
  smooth <- fit$smooth[[which(vapply(
    fit$smooth,
    function(item) identical(item$label, term_label),
    logical(1)
  ))]]
  coefficient_index <- smooth$first.para:smooth$last.para
  variance_components <- gam.vcomp(fit)$vc
  summary_fit <- summary(fit)

  list(
    lambda = unname(fit$sp[[term_label]]),
    scale = unname(summary_fit$scale),
    variance_component = unname(variance_components[term_label, "std.dev"]^2),
    standard_deviation = unname(variance_components[term_label, "std.dev"]),
    smooth_edf = unname(summary_fit$s.table[term_label, "edf"]),
    total_edf = unname(sum(fit$edf)),
    deviance = unname(deviance(fit)),
    intercept = unname(coef(fit)[["(Intercept)"]]),
    slope = unname(coef(fit)[["x"]]),
    random_effects = unname(coef(fit)[coefficient_index]),
    conditional_prediction = unname(predict(fit, newdata = data, type = "response")),
    population_prediction = unname(predict(
      fit,
      newdata = data,
      type = "response",
      exclude = term_label
    ))
  )
}

set.seed(20260724)

gaussian_levels <- sprintf("g%02d", seq_len(9))
gaussian_level <- factor(
  rep(gaussian_levels, each = 14),
  levels = gaussian_levels
)
gaussian_x <- rep(seq(-1.2, 1.2, length.out = 14), times = 9) +
  rep(seq(-0.04, 0.04, length.out = 9), each = 14)
gaussian_effect <- c(-0.72, -0.43, -0.19, -0.04, 0.13, 0.29, 0.51, 0.37, 0.08)
gaussian_y <- 1.25 + 0.58 * gaussian_x +
  gaussian_effect[as.integer(gaussian_level)] +
  rnorm(length(gaussian_x), sd = 0.16)
gaussian_data <- data.frame(
  x = gaussian_x,
  level = gaussian_level,
  y = gaussian_y
)
gaussian_fit <- gam(
  y ~ x + s(level, bs = "re"),
  family = gaussian(),
  data = gaussian_data,
  method = "REML"
)

set.seed(20260725)

poisson_levels <- sprintf("p%02d", seq_len(12))
poisson_level <- factor(
  rep(poisson_levels, each = 30),
  levels = poisson_levels
)
poisson_x <- rep(seq(-1.4, 1.4, length.out = 30), times = 12) +
  rep(seq(-0.06, 0.06, length.out = 12), each = 30)
poisson_exposure <- 0.55 + ((seq_along(poisson_x) * 17) %% 31) / 22
poisson_effect <- c(-0.58, -0.41, -0.26, -0.12, -0.02, 0.09, 0.19, 0.31, 0.46, 0.25, 0.03, -0.21)
poisson_mean <- poisson_exposure * exp(
  -0.34 + 0.42 * poisson_x + poisson_effect[as.integer(poisson_level)]
)
poisson_y <- rpois(length(poisson_x), poisson_mean)
poisson_data <- data.frame(
  x = poisson_x,
  level = poisson_level,
  exposure = poisson_exposure,
  y = poisson_y
)
poisson_fit <- gam(
  y ~ x + s(level, bs = "re") + offset(log(exposure)),
  family = poisson(),
  data = poisson_data,
  method = "REML"
)
poisson_discrete_fit <- bam(
  y ~ x + s(level, bs = "re") + offset(log(exposure)),
  family = poisson(),
  data = poisson_data,
  method = "fREML",
  discrete = TRUE,
  nthreads = 1
)

fixture <- list(
  metadata = list(
    generated_at = "2026-07-24",
    r_version = R.version.string,
    mgcv_version = as.character(packageVersion("mgcv")),
    seed_gaussian = 20260724,
    seed_poisson = 20260725
  ),
  gaussian = list(
    data = list(
      x = unname(gaussian_x),
      level = as.character(gaussian_level),
      y = unname(gaussian_y)
    ),
    reference = random_effect_reference(gaussian_fit, gaussian_data)
  ),
  poisson = list(
    data = list(
      x = unname(poisson_x),
      level = as.character(poisson_level),
      exposure = unname(poisson_exposure),
      y = unname(poisson_y)
    ),
    reference = random_effect_reference(poisson_fit, poisson_data)
  ),
  poisson_discrete = list(
    data = list(
      x = unname(poisson_x),
      level = as.character(poisson_level),
      exposure = unname(poisson_exposure),
      y = unname(poisson_y)
    ),
    reference = random_effect_reference(poisson_discrete_fit, poisson_data)
  )
)

arguments <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(arguments) > 0) {
  arguments[[1]]
} else {
  "tests/fixtures/random_effect_mgcv_reference.json"
}
writeLines(to_json(fixture), output_path)
cat("Wrote", output_path, "\n")
