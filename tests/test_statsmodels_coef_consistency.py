"""
Compare SuperGLM vs statsmodels coefficient consistency.

Fits the same Tweedie GLM on the same data with:
  - All-categorical factors
  - All-numeric factors
  - Mixed (categorical + numeric)

Both libraries should produce nearly identical coefficients when
SuperGLM runs unpenalised (selection_penalty=0).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.categorical import Categorical
from superglm.tweedie_profile import generate_tweedie_cpg

# ── Shared settings ──────────────────────────────────────────
P = 1.5
PHI = 2.0
N = 5_000
SEED = 42
rng = np.random.default_rng(SEED)

# ── Generate data ────────────────────────────────────────────
# Numeric columns
x1 = rng.uniform(-2, 2, N)
x2 = rng.uniform(0, 5, N)

# Categorical columns (will be dummy-coded)
cat_a = rng.choice(["lo", "mid", "hi"], N, p=[0.3, 0.4, 0.3])
cat_b = rng.choice(["north", "south", "east", "west"], N, p=[0.25, 0.25, 0.25, 0.25])

# True linear predictor (log scale)
eta_true = (
    0.5                                          # intercept
    + 0.3 * x1
    - 0.1 * x2
    + 0.4 * (cat_a == "hi")
    - 0.2 * (cat_a == "lo")
    + 0.15 * (cat_b == "north")
    - 0.10 * (cat_b == "south")
)
mu_true = np.exp(eta_true)
y = generate_tweedie_cpg(N, mu=mu_true, phi=PHI, p=P, rng=rng)

df = pd.DataFrame({
    "x1": x1, "x2": x2,
    "cat_a": cat_a, "cat_b": cat_b,
})


def fit_statsmodels(X_df, y, feature_names, cat_cols):
    """Fit statsmodels GLM with Tweedie(p) + log link, return result."""
    # Build design matrix with dummies (drop_first to match base level)
    parts = []
    col_names = []
    for col in feature_names:
        if col in cat_cols:
            dummies = pd.get_dummies(X_df[col], prefix=col, drop_first=True, dtype=float)
            parts.append(dummies.values)
            col_names.extend(dummies.columns.tolist())
        else:
            parts.append(X_df[[col]].values)
            col_names.append(col)

    X_mat = np.column_stack(parts)
    X_with_const = sm.add_constant(X_mat)
    col_names_full = ["const"] + col_names

    family = sm.families.Tweedie(var_power=P, link=sm.families.links.Log())
    model = sm.GLM(y, X_with_const, family=family)
    result = model.fit(maxiter=100)
    return result, col_names_full


def fit_superglm(X_df, y, features_dict):
    """Fit SuperGLM unpenalised, return model."""
    model = SuperGLM(
        family=Tweedie(p=P),
        link="log",
        selection_penalty=0.0,
        features=features_dict,
    )
    model.fit(X_df, y)
    return model


def print_comparison(title, sm_result, sm_names, sg_model):
    """Print side-by-side summary tables."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

    print("\n── statsmodels summary ──")
    print(sm_result.summary())

    print("\n── SuperGLM summary ──")
    print(sg_model.summary())

    # Extract and compare
    sm_params = sm_result.params
    sg_intercept = sg_model._result.intercept
    sg_beta = sg_model._result.beta

    print("\n── Coefficient comparison ──")
    print(f"{'Parameter':<25} {'statsmodels':>12} {'SuperGLM':>12} {'diff':>12}")
    print("-" * 63)
    print(f"{'Intercept':<25} {sm_params[0]:>12.6f} {sg_intercept:>12.6f} {abs(sm_params[0] - sg_intercept):>12.2e}")

    # Map SuperGLM beta to statsmodels param ordering
    # SuperGLM stores beta in feature-spec order
    for i, name in enumerate(sm_names[1:]):  # skip const
        sm_val = sm_params[i + 1]
        # For simple numeric/categorical features with no penalty,
        # beta ordering should match the design matrix column order
        if i < len(sg_beta):
            sg_val = sg_beta[i]
            diff = abs(sm_val - sg_val)
            print(f"{name:<25} {sm_val:>12.6f} {sg_val:>12.6f} {diff:>12.2e}")
        else:
            print(f"{name:<25} {sm_val:>12.6f} {'N/A':>12}")

    print()


# ══════════════════════════════════════════════════════════════
# TEST 1: All numeric factors
# ══════════════════════════════════════════════════════════════
print("\n\n>>> TEST 1: All numeric factors")
num_features = {"x1": Numeric(), "x2": Numeric()}
sm_res1, sm_names1 = fit_statsmodels(df, y, ["x1", "x2"], cat_cols=[])
sg_model1 = fit_superglm(df, y, num_features)
print_comparison("All Numeric", sm_res1, sm_names1, sg_model1)

# ══════════════════════════════════════════════════════════════
# TEST 2: All categorical factors
# ══════════════════════════════════════════════════════════════
print("\n\n>>> TEST 2: All categorical factors")
cat_features = {
    "cat_a": Categorical(base="first"),
    "cat_b": Categorical(base="first"),
}
sm_res2, sm_names2 = fit_statsmodels(df, y, ["cat_a", "cat_b"], cat_cols=["cat_a", "cat_b"])
sg_model2 = fit_superglm(df, y, cat_features)
print_comparison("All Categorical", sm_res2, sm_names2, sg_model2)

# ══════════════════════════════════════════════════════════════
# TEST 3: Mixed (numeric + categorical)
# ══════════════════════════════════════════════════════════════
print("\n\n>>> TEST 3: Mixed (numeric + categorical)")
mixed_features = {
    "x1": Numeric(),
    "x2": Numeric(),
    "cat_a": Categorical(base="first"),
    "cat_b": Categorical(base="first"),
}
sm_res3, sm_names3 = fit_statsmodels(df, y, ["x1", "x2", "cat_a", "cat_b"], cat_cols=["cat_a", "cat_b"])
sg_model3 = fit_superglm(df, y, mixed_features)
print_comparison("Mixed (Numeric + Categorical)", sm_res3, sm_names3, sg_model3)
