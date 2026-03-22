"""
Compare SuperGLM vs statsmodels coefficient and SE consistency.

Fits the same GLM on the same data with numeric, categorical, mixed,
and spline features.  Confirms coefficients and standard errors match
when SuperGLM runs unpenalised (selection_penalty=0).
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric

# ── Shared data fixtures ─────────────────────────────────────────


@pytest.fixture(scope="module")
def tweedie_data():
    """Generate Tweedie response data with known DGP."""
    sm = pytest.importorskip("statsmodels")  # noqa: F841
    from superglm.distributions import Tweedie
    from superglm.tweedie_profile import generate_tweedie_cpg

    P, PHI, N, SEED = 1.5, 2.0, 5_000, 42
    rng = np.random.default_rng(SEED)

    x1 = rng.uniform(-2, 2, N)
    x2 = rng.uniform(0, 5, N)
    cat_a = rng.choice(["lo", "mid", "hi"], N, p=[0.3, 0.4, 0.3])
    cat_b = rng.choice(["north", "south", "east", "west"], N)

    eta_true = (
        0.5
        + 0.3 * x1
        - 0.1 * x2
        + 0.4 * (cat_a == "hi")
        - 0.2 * (cat_a == "lo")
        + 0.15 * (cat_b == "north")
        - 0.10 * (cat_b == "south")
    )
    mu_true = np.exp(eta_true)
    y = generate_tweedie_cpg(N, mu=mu_true, phi=PHI, p=P, rng=rng)

    df = pd.DataFrame({"x1": x1, "x2": x2, "cat_a": cat_a, "cat_b": cat_b})
    return df, y, P, Tweedie


# ── Helpers ──────────────────────────────────────────────────────


def _fit_statsmodels(X_df, y, feature_names, cat_cols, var_power):
    """Fit statsmodels GLM with Tweedie(p) + log link."""
    import statsmodels.api as sm

    parts, col_names = [], []
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

    family = sm.families.Tweedie(var_power=var_power, link=sm.families.links.Log())
    result = sm.GLM(y, X_with_const, family=family).fit(maxiter=100)
    return result


def _fit_superglm(X_df, y, features_dict, Tweedie, p):
    """Fit SuperGLM unpenalised."""
    model = SuperGLM(
        family=Tweedie(p=p),
        link="log",
        selection_penalty=0.0,
        features=features_dict,
    )
    model.fit(X_df, y)
    return model


def _assert_coef_and_se_match(sm_result, sg_model, atol_coef=1e-4, atol_se=1e-3):
    """Assert coefficients AND standard errors match."""
    sm_params = sm_result.params
    sm_bse = sm_result.bse

    np.testing.assert_allclose(
        sg_model._result.intercept,
        sm_params[0],
        atol=atol_coef,
        err_msg="Intercept coef mismatch",
    )
    np.testing.assert_allclose(
        sg_model._result.beta,
        sm_params[1:],
        atol=atol_coef,
        err_msg="Feature coefs mismatch",
    )

    summary = sg_model.summary()
    sg_ses = [row.se for row in summary._coef_rows]
    np.testing.assert_allclose(
        sg_ses,
        list(sm_bse),
        atol=atol_se,
        err_msg="Standard errors mismatch",
    )


# ── Tests ────────────────────────────────────────────────────────


class TestStatsmodelsCoefConsistency:
    """Unpenalised SuperGLM should match statsmodels coefficients and SEs."""

    def test_all_numeric(self, tweedie_data):
        df, y, P, Tweedie = tweedie_data
        features = {"x1": Numeric(), "x2": Numeric()}
        sm_res = _fit_statsmodels(df, y, ["x1", "x2"], cat_cols=[], var_power=P)
        sg_model = _fit_superglm(df, y, features, Tweedie, P)
        _assert_coef_and_se_match(sm_res, sg_model)

    def test_all_categorical(self, tweedie_data):
        df, y, P, Tweedie = tweedie_data
        features = {
            "cat_a": Categorical(base="first"),
            "cat_b": Categorical(base="first"),
        }
        sm_res = _fit_statsmodels(
            df,
            y,
            ["cat_a", "cat_b"],
            cat_cols=["cat_a", "cat_b"],
            var_power=P,
        )
        sg_model = _fit_superglm(df, y, features, Tweedie, P)
        _assert_coef_and_se_match(sm_res, sg_model)

    def test_mixed_numeric_categorical(self, tweedie_data):
        df, y, P, Tweedie = tweedie_data
        features = {
            "x1": Numeric(),
            "x2": Numeric(),
            "cat_a": Categorical(base="first"),
            "cat_b": Categorical(base="first"),
        }
        sm_res = _fit_statsmodels(
            df,
            y,
            ["x1", "x2", "cat_a", "cat_b"],
            cat_cols=["cat_a", "cat_b"],
            var_power=P,
        )
        sg_model = _fit_superglm(df, y, features, Tweedie, P)
        _assert_coef_and_se_match(sm_res, sg_model)


class TestSplineSEConsistency:
    """Spline curve SEs should use the augmented (marginal) covariance."""

    def test_spline_se_uses_augmented_inverse(self):
        """Spline curve SEs from summary must match term_inference SEs.

        Both paths should use the augmented inverse (marginal SEs
        accounting for intercept), not the conditional p×p inverse.
        This also verifies spline basis construction produces
        consistent SEs across the two code paths.
        """
        from superglm.features.spline import Spline

        rng = np.random.default_rng(99)
        n = 2_000
        x = rng.uniform(0, 10, n)
        # Uncentered design — intercept correction matters
        y = rng.poisson(np.exp(2.0 + 0.3 * np.sin(x))).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit_reml(X, y)

        # Curve SEs from summary (uses augmented inverse)
        s = model.summary()
        spline_row = next(r for r in s._coef_rows if r.is_spline)
        assert spline_row.curve_se_min is not None
        assert spline_row.curve_se_max is not None
        assert spline_row.curve_se_min > 0

        # Curve SEs from term_inference (uses augmented inverse via inference.py)
        ti = model.term_inference("x")
        assert ti.se_log_relativity is not None
        se_curve = np.asarray(ti.se_log_relativity)
        assert np.all(se_curve > 0)

        # Both should give the same SE range (they use the same augmented inverse).
        # term_inference evaluates on a 200-point grid so the range should be
        # consistent with (but not identical to) the summary's grid.
        np.testing.assert_allclose(
            spline_row.curve_se_min,
            float(np.min(se_curve)),
            rtol=0.1,
            err_msg="Curve SE min disagrees between summary and term_inference",
        )
        np.testing.assert_allclose(
            spline_row.curve_se_max,
            float(np.max(se_curve)),
            rtol=0.1,
            err_msg="Curve SE max disagrees between summary and term_inference",
        )

    def test_spline_se_larger_than_conditional(self):
        """Marginal SEs (augmented) should be >= conditional SEs (p×p only).

        For an uncentered spline basis, the augmented inverse gives larger
        SEs because it accounts for intercept estimation uncertainty.
        """
        from superglm.features.spline import Spline

        rng = np.random.default_rng(77)
        n = 1_000
        # Deliberately uncentered — large intercept correction expected
        x = rng.uniform(5, 15, n)
        y = rng.poisson(np.exp(1.0 + 0.2 * x)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X, y)

        # Get both inverses from the metrics machinery
        metrics = model.metrics(X, y)
        _, _, XtWX_inv, XtWX_inv_aug, active_groups = metrics._active_info
        phi = metrics.phi

        ag = active_groups[0]
        aug_sl = slice(1 + ag.start, 1 + ag.end)
        se_marginal = np.sqrt(phi * np.diag(XtWX_inv_aug[aug_sl, aug_sl]))
        se_conditional = np.sqrt(phi * np.diag(XtWX_inv[ag.sl, ag.sl]))

        # Marginal SEs should be >= conditional for every coefficient
        assert np.all(se_marginal >= se_conditional - 1e-12), (
            "Marginal SEs should be >= conditional SEs"
        )
        # And strictly larger for at least some (uncentered basis)
        assert np.any(se_marginal > se_conditional + 1e-10), (
            "Expected at least some marginal SEs to be strictly larger"
        )
