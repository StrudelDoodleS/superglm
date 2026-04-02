"""mgcv parity audit: multi-order derivative penalties.

Compares Spline(kind='cr', m=(1,2)) against mgcv s(x, bs='cr', m=c(1,2)).

Usage:
    uv run python scratch/bench_mgcv_multi_m_parity.py
"""

import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import Spline, SuperGLM

rng = np.random.default_rng(42)
n = 2000
x = rng.uniform(0, 1, n)
eta = 0.5 + np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)
mu = np.exp(eta)
y = rng.poisson(mu).astype(float)
df = pd.DataFrame({"x": x, "y": y})

data_path = os.path.join(tempfile.gettempdir(), "multi_m_data.csv")
df.to_csv(data_path, index=False)

mgcv_out = os.path.join(tempfile.gettempdir(), "multi_m_mgcv.csv")
r_script = f"""
library(mgcv)
d <- read.csv("{data_path}")
grid <- data.frame(x=seq(0, 1, length.out=200))

# m=2 (default)
m2 <- gam(y ~ s(x, bs="cr", k=10), family=poisson(), data=d, method="REML")
# m=c(1,2)
m12 <- gam(y ~ s(x, bs="cr", k=10, m=c(1,2)), family=poisson(), data=d, method="REML")
# m=c(2,3)
m23 <- gam(y ~ s(x, bs="cr", k=10, m=c(2,3)), family=poisson(), data=d, method="REML")

get_partial <- function(model) {{
    p <- predict(model, newdata=grid, type="terms")
    return(p[, 1])
}}

out <- data.frame(
    x = grid$x,
    mgcv_m2 = get_partial(m2),
    mgcv_m12 = get_partial(m12),
    mgcv_m23 = get_partial(m23)
)
write.csv(out, "{mgcv_out}", row.names=FALSE)

cat("mgcv m=2:     edf=", summary(m2)$s.table[1, "edf"], "\\n")
cat("mgcv m=c(1,2): edf=", summary(m12)$s.table[1, "edf"], "\\n")
cat("mgcv m=c(2,3): edf=", summary(m23)$s.table[1, "edf"], "\\n")
cat("done\\n")
"""

print("Fitting mgcv...")
result = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
if result.returncode != 0:
    print(f"R error:\n{result.stderr}")
    raise RuntimeError("mgcv failed")
print(result.stdout.strip())

mgcv = pd.read_csv(mgcv_out)

print("\nFitting superglm...")
configs = {
    "m=2": {"m": 2},
    "m=(1,2)": {"m": (1, 2)},
    "m=(2,3)": {"m": (2, 3)},
}

sg_results = {}
for label, kwargs in configs.items():
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={"x": Spline(kind="cr", n_knots=8, **kwargs)},
    )
    model.fit_reml(df[["x"]], y, max_reml_iter=30)
    r = model.reconstruct_feature("x")
    sg_results[label] = {
        "x": r["x"],
        "log_rel": r["log_relativity"],
        "converged": model._reml_result.converged,
        "edf": model.result.effective_df,
        "lambdas": model._reml_lambdas,
    }
    print(
        f"  {label}: converged={model._reml_result.converged}, "
        f"edf={model.result.effective_df:.2f}, lambdas={model._reml_lambdas}"
    )

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Multi-order penalties: superglm vs mgcv (CRS, Poisson REML)", fontsize=13)

pairs = [
    ("m=2", "mgcv_m2"),
    ("m=(1,2)", "mgcv_m12"),
    ("m=(2,3)", "mgcv_m23"),
]

for ax, (sg_label, mg_col) in zip(axes, pairs):
    sg = sg_results[sg_label]
    ax.plot(mgcv["x"], np.exp(mgcv[mg_col]), "b-", lw=2, label=f"mgcv {sg_label}")
    ax.plot(sg["x"], np.exp(sg["log_rel"]), "r--", lw=2, label=f"superglm {sg_label}")
    ax.set_title(sg_label)
    ax.set_ylabel("relativity")
    ax.set_xlabel("x")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scratch/multi_m_parity.png", dpi=150)
print("\nPlots saved to scratch/multi_m_parity.png")
