# Generate pinned mgcv bs="sz" reference values for SuperGLM parity tests.
#
# This clean-room fixture calls documented mgcv APIs and records numerical
# outputs only. It neither embeds nor translates mgcv implementation source.

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

sz_reference <- function(
  fit,
  prediction_data,
  unseen_reference_data,
  term_label = "s(x,f)",
  global_label = "s(x)"
) {
  smooth_index <- which(vapply(
    fit$smooth,
    function(item) identical(item$label, term_label),
    logical(1)
  ))
  if (length(smooth_index) != 1) {
    stop("Could not uniquely identify the SZ smooth")
  }
  smooth <- fit$smooth[[smooth_index]]
  if (smooth$first.sp != smooth$last.sp || length(smooth$S.scale) != 1) {
    stop("Expected one shared SZ smoothing parameter")
  }
  raw_sz_sp <- unname(fit$sp[smooth$first.sp])
  sz_scale <- unname(smooth$S.scale[1])

  global_index <- which(vapply(
    fit$smooth,
    function(item) identical(item$label, global_label),
    logical(1)
  ))
  if (length(global_index) != 1) {
    stop("Could not uniquely identify the global smooth")
  }
  global_smooth <- fit$smooth[[global_index]]
  raw_global_sp <- unname(fit$sp[global_smooth$first.sp])
  global_scale <- unname(global_smooth$S.scale[1])

  list(
    raw_mgcv_sp = list(
      global = raw_global_sp,
      sz_wiggle = raw_sz_sp
    ),
    penalty_scale = list(
      global = global_scale,
      sz_wiggle = sz_scale
    ),
    unscaled_lambdas = list(
      global = raw_global_sp / global_scale,
      sz_wiggle = raw_sz_sp / sz_scale
    ),
    scale = unname(summary(fit)$scale),
    total_edf = unname(sum(fit$edf)),
    global_edf = unname(sum(
      fit$edf[global_smooth$first.para:global_smooth$last.para]
    )),
    sz_edf = unname(sum(
      fit$edf[smooth$first.para:smooth$last.para]
    )),
    deviance = unname(deviance(fit)),
    intercept = unname(coef(fit)[["(Intercept)"]]),
    conditional_prediction = unname(predict(
      fit,
      newdata = prediction_data,
      type = "response"
    )),
    global_only_prediction = unname(predict(
      fit,
      newdata = prediction_data,
      type = "response",
      exclude = term_label
    )),
    sz_deviation_link = unname(predict(
      fit,
      newdata = prediction_data,
      type = "terms",
      terms = term_label
    )),
    global_link = unname(predict(
      fit,
      newdata = prediction_data,
      type = "terms",
      terms = global_label
    )),
    unseen_population_prediction = unname(predict(
      fit,
      newdata = unseen_reference_data,
      type = "response",
      exclude = term_label
    ))
  )
}

construction_levels <- sprintf("c%02d", seq_len(4))
construction_x <- rep(seq(-1, 1, length.out = 12), times = 4) +
  rep(seq(-0.02, 0.02, length.out = 4), each = 12)
construction_f <- factor(
  rep(construction_levels, each = 12),
  levels = construction_levels
)
construction_data <- data.frame(
  x = construction_x,
  f = construction_f
)
construction_spec <- s(
  x,
  f,
  bs = "sz",
  k = 6,
  xt = list(bs = "ps"),
  m = 2,
  id = 1
)
construction <- smoothCon(
  construction_spec,
  construction_data,
  absorb.cons = TRUE
)[[1]]
construction_no_id <- smoothCon(
  s(x, f, bs = "sz", k = 6, xt = list(bs = "ps"), m = 2),
  construction_data,
  absorb.cons = TRUE
)[[1]]
construction_predict_data <- data.frame(
  x = 0.137,
  f = factor(construction_levels[4], levels = construction_levels)
)
construction_prediction <- PredictMat(
  construction,
  construction_predict_data
)

gaussian_levels <- sprintf("g%02d", seq_len(5))
gaussian_f <- factor(
  rep(gaussian_levels, each = 40),
  levels = gaussian_levels
)
gaussian_x <- rep(seq(-1.2, 1.2, length.out = 40), times = 5) +
  rep(seq(-0.035, 0.035, length.out = 5), each = 40)
gaussian_amplitude <- c(0.65, -0.40, 0.28, -0.50, -0.03)
gaussian_y <- 1.05 +
  0.36 * sin(2.1 * gaussian_x) +
  gaussian_amplitude[as.integer(gaussian_f)] *
    (gaussian_x + 0.25 * gaussian_x^2) +
  0.04 * cos(seq_along(gaussian_x) * 1.3)
gaussian_data <- data.frame(
  x = gaussian_x,
  f = gaussian_f,
  y = gaussian_y
)
gaussian_curve_grid <- seq(-1.05, 1.05, length.out = 11)
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
  x = c(-0.95, 0.08, 0.97),
  f = factor(rep(gaussian_levels[1], 3), levels = gaussian_levels)
)
gaussian_fit <- gam(
  y ~ s(x, bs = "ps", k = 7, m = 2) +
    s(x, f, bs = "sz", k = 6, xt = list(bs = "ps"), m = 2, id = 1),
  family = gaussian(),
  data = gaussian_data,
  method = "REML"
)

set.seed(20260725)
poisson_levels <- sprintf("p%02d", seq_len(6))
poisson_f <- factor(
  rep(poisson_levels, each = 50),
  levels = poisson_levels
)
poisson_x <- rep(seq(-1.3, 1.3, length.out = 50), times = 6) +
  rep(seq(-0.045, 0.045, length.out = 6), each = 50)
poisson_exposure <- 0.45 + ((seq_along(poisson_x) * 19) %% 41) / 27
poisson_amplitude <- c(0.52, -0.38, 0.29, -0.46, 0.18, -0.15)
poisson_eta <- -0.38 +
  0.37 * sin(2.0 * poisson_x) +
  poisson_amplitude[as.integer(poisson_f)] *
    (poisson_x + 0.24 * poisson_x^2)
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
poisson_formula <- y ~ s(x, bs = "ps", k = 7, m = 2) +
  s(x, f, bs = "sz", k = 6, xt = list(bs = "ps"), m = 2, id = 1) +
  offset(log(exposure))
poisson_fit <- gam(
  poisson_formula,
  family = poisson(),
  data = poisson_data,
  method = "REML"
)
poisson_discrete_fit <- bam(
  poisson_formula,
  family = poisson(),
  data = poisson_data,
  method = "fREML",
  discrete = TRUE,
  nthreads = 1
)

fit_case <- function(
  data,
  curve_grid,
  prediction_data,
  unseen_data,
  fit
) {
  output_data <- list(
    x = unname(data$x),
    f = as.character(data$f),
    y = unname(data$y)
  )
  prediction_output <- list(
    x = unname(prediction_data$x),
    f = as.character(prediction_data$f)
  )
  unseen_output <- list(x = unname(unseen_data$x))
  if ("exposure" %in% names(data)) {
    output_data$exposure <- unname(data$exposure)
    prediction_output$exposure <- unname(prediction_data$exposure)
    unseen_output$exposure <- unname(unseen_data$exposure)
  }
  list(
    data = output_data,
    curve_grid = unname(curve_grid),
    prediction_data = prediction_output,
    unseen_data = unseen_output,
    reference = sz_reference(
      fit,
      prediction_data,
      unseen_data
    )
  )
}

fixture <- list(
  metadata = list(
    generated_at = "2026-07-25",
    r_version = R.version.string,
    mgcv_version = as.character(packageVersion("mgcv")),
    seed = 20260725,
    formula = paste(deparse(poisson_formula), collapse = " "),
    sz_term = paste(
      "s(x, f, bs=\"sz\", k=6,",
      "xt=list(bs=\"ps\"), m=2, id=1)"
    )
  ),
  construction = list(
    data = list(
      x = unname(construction_x),
      f = as.character(construction_f)
    ),
    levels = construction_levels,
    design_dim = unname(dim(construction$X)),
    design_flat = unname(as.numeric(construction$X)),
    penalty_count = length(construction$S),
    penalty_dim = unname(dim(construction$S[[1]])),
    penalty_flat = unname(as.numeric(construction$S[[1]])),
    penalty_scale = unname(construction$S.scale[1]),
    penalty_rank = unname(construction$rank[1]),
    nullity = unname(construction$null.space.dim),
    no_id_smoothing_parameter_count = length(construction_no_id$S),
    prediction_data = list(
      x = unname(construction_predict_data$x),
      f = as.character(construction_predict_data$f)
    ),
    prediction_dim = unname(dim(construction_prediction)),
    prediction_flat = unname(as.numeric(construction_prediction))
  ),
  gaussian = fit_case(
    gaussian_data,
    gaussian_curve_grid,
    gaussian_prediction_data,
    gaussian_unseen_data,
    gaussian_fit
  ),
  poisson = fit_case(
    poisson_data,
    poisson_curve_grid,
    poisson_prediction_data,
    poisson_unseen_data,
    poisson_fit
  ),
  poisson_discrete = fit_case(
    poisson_data,
    poisson_curve_grid,
    poisson_prediction_data,
    poisson_unseen_data,
    poisson_discrete_fit
  )
)

arguments <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(arguments) > 0) {
  arguments[[1]]
} else {
  "tests/fixtures/factor_smooth_sz_mgcv_reference.json"
}
writeLines(to_json(fixture), output_path)
cat("Wrote", output_path, "\n")
