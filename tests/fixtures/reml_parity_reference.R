# Generate mgcv reference values for REML parity tests.
#
# Produces a JSON-like output that can be pasted into the Python test.
# Uses B-splines (bs="bs") with second-difference penalty to match
# SuperGLM's default Spline class.

library(mgcv)

set.seed(42)
n <- 800

x1 <- runif(n, 0, 1)
x2 <- runif(n, 0, 1)
mu <- exp(0.5 + sin(2 * pi * x1) + 0.3 * x2)

# ------- Poisson -------
y_pois <- rpois(n, mu)
df <- data.frame(x1 = x1, x2 = x2, y = y_pois)

# Write data for Python reproducibility
write.csv(df, "reml_parity_data_poisson.csv", row.names = FALSE)

m_pois <- gam(y ~ s(x1, bs = "bs", k = 10, m = c(3, 2))
                + s(x2, bs = "bs", k = 10, m = c(3, 2)),
              family = poisson, data = df, method = "REML")

cat("=== Poisson ===\n")
cat("sp:", m_pois$sp, "\n")
cat("reml_score:", m_pois$gcv.ubre, "\n")
cat("edf:", m_pois$edf, "\n")
cat("sum_edf:", sum(m_pois$edf), "\n")
cat("deviance:", deviance(m_pois), "\n")
cat("intercept:", coef(m_pois)[1], "\n")

# ------- Gamma -------
shape <- 5.0
y_gamma <- rgamma(n, shape = shape, rate = shape / mu)
y_gamma <- pmax(y_gamma, 1e-4)
df_g <- data.frame(x1 = x1, x2 = x2, y = y_gamma)
write.csv(df_g, "reml_parity_data_gamma.csv", row.names = FALSE)

m_gamma <- gam(y ~ s(x1, bs = "bs", k = 10, m = c(3, 2))
                 + s(x2, bs = "bs", k = 10, m = c(3, 2)),
               family = Gamma(link = "log"), data = df_g, method = "REML")

cat("\n=== Gamma ===\n")
cat("sp:", m_gamma$sp, "\n")
cat("reml_score:", m_gamma$gcv.ubre, "\n")
cat("edf:", m_gamma$edf, "\n")
cat("sum_edf:", sum(m_gamma$edf), "\n")
cat("deviance:", deviance(m_gamma), "\n")
cat("intercept:", coef(m_gamma)[1], "\n")
cat("scale:", m_gamma$scale, "\n")

# ------- Poisson with select=TRUE -------
# Add a pure noise variable
x3 <- runif(n, 0, 1)
df_sel <- data.frame(x1 = x1, x2 = x2, x3 = x3, y = y_pois)
write.csv(df_sel, "reml_parity_data_select.csv", row.names = FALSE)

m_sel <- gam(y ~ s(x1, bs = "bs", k = 10, m = c(3, 2))
               + s(x2, bs = "bs", k = 10, m = c(3, 2))
               + s(x3, bs = "bs", k = 10, m = c(3, 2)),
             family = poisson, data = df_sel, method = "REML", select = TRUE)

cat("\n=== Poisson select=TRUE ===\n")
cat("sp:", m_sel$sp, "\n")
cat("edf:", m_sel$edf, "\n")
cat("sum_edf:", sum(m_sel$edf), "\n")
cat("noise_edf_x3:", sum(m_sel$edf[grepl("x3", names(m_sel$edf))]), "\n")

cat("\nDone.\n")
