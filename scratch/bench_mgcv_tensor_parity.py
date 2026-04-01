"""mgcv parity audit: anisotropic tensor REML (ti() + main effects).

Compares superglm's multi-penalty tensor REML against mgcv::gam with
ti() interaction + s() main effects. Acceptance gate is fitted surface
similarity, not raw lambda matching.

Our TensorInteraction is ti()-style (interaction-only), so we compare
against mgcv::gam(y ~ s(x1) + s(x2) + ti(x1, x2), ...).

Usage:
    uv run python scratch/bench_mgcv_tensor_parity.py
"""

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.spline import CubicRegressionSpline

# ── Generate shared dataset ─────────────────────────────────────

rng = np.random.default_rng(42)
n = 2000
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)

# DGP: smooth main effects + mild interaction surface
eta = 0.5 + np.sin(2 * np.pi * x1) + 0.5 * np.cos(np.pi * x2) + 0.3 * x1 * np.sin(2 * np.pi * x2)
mu = np.exp(eta)
y = rng.poisson(mu).astype(float)

df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
data_path = os.path.join(tempfile.gettempdir(), "tensor_parity_data.csv")
df.to_csv(data_path, index=False)

# ── Evaluation grid ─────────────────────────────────────────────

g1 = np.linspace(0, 1, 30)
g2 = np.linspace(0, 1, 30)
G1, G2 = np.meshgrid(g1, g2)
grid_df = pd.DataFrame({"x1": G1.ravel(), "x2": G2.ravel()})
grid_path = os.path.join(tempfile.gettempdir(), "tensor_parity_grid.csv")
grid_df.to_csv(grid_path, index=False)

# ── Fit with mgcv ───────────────────────────────────────────────

mgcv_pred_path = os.path.join(tempfile.gettempdir(), "tensor_parity_mgcv_pred.csv")
mgcv_summary_path = os.path.join(tempfile.gettempdir(), "tensor_parity_mgcv_summary.txt")

r_script = f"""
library(mgcv)

d <- read.csv("{data_path}")
grid <- read.csv("{grid_path}")

# Fit with ti() + main effects, matching basis dimensions
# SuperGLM CRS n_knots=6 -> 8 basis functions (6+2 for CRS),
# 7 after identifiability. mgcv k=8 gives 8 basis functions,
# 7 after identifiability. So k=8 matches n_knots=6.
m <- gam(y ~ s(x1, bs="cr", k=8) + s(x2, bs="cr", k=8) + ti(x1, x2, bs="cr", k=c(8, 8)),
         family=poisson(), data=d, method="REML")

# Predict on grid
grid$pred <- predict(m, newdata=grid, type="response")
write.csv(grid, "{mgcv_pred_path}", row.names=FALSE)

# Summary
sink("{mgcv_summary_path}")
cat("=== mgcv summary ===\\n")
cat("Deviance:", deviance(m), "\\n")
cat("EDF:", sum(m$edf), "\\n")
cat("Per-smooth EDF:\\n")
print(summary(m)$s.table[, c("edf", "Ref.df")])
cat("\\nSmoothing parameters (sp):\\n")
print(m$sp)
cat("\\nFitted values range:", range(fitted(m)), "\\n")
sink()

cat("mgcv fit complete\\n")
"""

print("Fitting mgcv model...")
result = subprocess.run(
    ["Rscript", "-e", r_script],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(f"R error:\n{result.stderr}")
    raise RuntimeError("mgcv fit failed")
print(result.stdout.strip())
print(result.stderr.strip())

# ── Fit with superglm ──────────────────────────────────────────

print("\nFitting superglm model...")
model = SuperGLM(
    family="poisson",
    selection_penalty=0,
    features={
        "x1": CubicRegressionSpline(n_knots=6),
        "x2": CubicRegressionSpline(n_knots=6),
    },
    interactions=[("x1", "x2")],
)
model.fit_reml(df, y, max_reml_iter=40)

print(f"  Converged: {model._reml_result.converged}")
print(f"  REML iters: {model._reml_result.n_reml_iter}")
print(f"  Deviance: {model.result.deviance:.1f}")
print(f"  EDF: {model.result.effective_df:.2f}")
print("  Lambdas:")
for k, v in model._reml_lambdas.items():
    print(f"    {k}: {v:.4f}")

# Predict on grid
sg_pred = model.predict(grid_df)

# ── Compare surfaces ────────────────────────────────────────────

mgcv_grid = pd.read_csv(mgcv_pred_path)
mgcv_pred = mgcv_grid["pred"].values

# Print mgcv summary
with open(mgcv_summary_path) as f:
    print(f"\n{f.read()}")

print("\n=== Surface comparison ===")

# Overall RMSE
rmse = np.sqrt(np.mean((sg_pred - mgcv_pred) ** 2))
mean_pred = np.mean(mgcv_pred)
print(f"RMSE on grid: {rmse:.4f} (mean pred: {mean_pred:.4f}, relative: {rmse / mean_pred:.4f})")

# Correlation
corr = np.corrcoef(sg_pred, mgcv_pred)[0, 1]
print(f"Correlation: {corr:.6f}")

# Fixed-x2 slices
print("\nFixed-x2 slices (correlation):")
for x2_val in [0.25, 0.5, 0.75]:
    mask = np.abs(G2.ravel() - x2_val) < 0.02
    if mask.sum() > 5:
        corr_slice = np.corrcoef(sg_pred[mask], mgcv_pred[mask])[0, 1]
        print(f"  x2={x2_val:.2f}: r={corr_slice:.4f} (n={mask.sum()})")

# Fixed-x1 slices
print("\nFixed-x1 slices (correlation):")
for x1_val in [0.25, 0.5, 0.75]:
    mask = np.abs(G1.ravel() - x1_val) < 0.02
    if mask.sum() > 5:
        corr_slice = np.corrcoef(sg_pred[mask], mgcv_pred[mask])[0, 1]
        print(f"  x1={x1_val:.2f}: r={corr_slice:.4f} (n={mask.sum()})")

# EDF comparison
print(f"\nEDF: superglm={model.result.effective_df:.2f}")

print("\n=== Verdict ===")
if corr > 0.99 and rmse / mean_pred < 0.05:
    print("PASS: surfaces are similar (corr > 0.99, relative RMSE < 5%)")
elif corr > 0.95:
    print("MARGINAL: surfaces are broadly similar but not tight")
else:
    print("FAIL: surfaces diverge meaningfully")
