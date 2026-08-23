"""Independent GLM oracles for the replication weight contract.

The oracles are statsmodels' ``freq_weights`` and glum, both of which count a
weighted row as that many rows in the dispersion denominator, so every fit here
declares ``weight_semantics="frequency"`` to be compared against the same
likelihood.  The prior contract has its own oracles -- ``scipy.stats``
densities -- in ``test_weight_semantics.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2
from scipy.stats import f as f_dist

from superglm import Numeric, SuperGLM


def _frequency_problem(
    family: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260804)
    n = 180
    x1 = rng.normal(size=n)
    x2 = 0.25 * x1 + rng.normal(scale=0.9, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    weights = rng.integers(1, 5, size=n).astype(np.float64)
    eta = 0.35 + 0.42 * x1 - 0.28 * x2

    if family == "gaussian":
        y = eta + rng.normal(scale=0.75, size=n)
    elif family == "gamma":
        y = rng.gamma(shape=3.5, scale=np.exp(eta) / 3.5)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    else:  # pragma: no cover - helper is only called by the parametrization
        raise AssertionError(f"unsupported test family {family!r}")
    return frame, np.asarray(y, dtype=np.float64), weights


def _statsmodels_family(sm, family: str):
    if family == "gaussian":
        return sm.families.Gaussian(sm.families.links.Identity())
    if family == "gamma":
        return sm.families.Gamma(sm.families.links.Log())
    if family == "poisson":
        return sm.families.Poisson(sm.families.links.Log())
    raise AssertionError(f"unsupported test family {family!r}")


def _fit_superglm(
    family: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> SuperGLM:
    return SuperGLM(
        family=family,
        selection_penalty=0.0,
        features={name: Numeric() for name in frame.columns},
        tol=1e-11,
        max_iter=300,
        weight_semantics="frequency",
    ).fit(frame, y, sample_weight=weights)


def _model_standard_errors(model: SuperGLM, metrics, family: str) -> np.ndarray:
    scale = 1.0 if family == "poisson" else model.result.phi
    raw = np.r_[
        metrics.intercept_se_raw,
        *(metrics.coefficient_se_raw[name] for name in model._feature_order),
    ]
    return np.sqrt(scale) * raw


def _quasi_standard_errors(model: SuperGLM, metrics) -> np.ndarray:
    return np.r_[
        metrics.intercept_se,
        *(metrics.coefficient_se[name] for name in model._feature_order),
    ]


@pytest.mark.parametrize("family", ["gaussian", "gamma", "poisson"])
def test_frequency_weight_fit_and_inference_match_statsmodels(family: str) -> None:
    sm = pytest.importorskip("statsmodels.api")
    frame, y, weights = _frequency_problem(family)
    design = sm.add_constant(frame.to_numpy())

    model = _fit_superglm(family, frame, y, weights)
    metrics = model.metrics(frame, y, sample_weight=weights)
    reference = sm.GLM(
        y,
        design,
        family=_statsmodels_family(sm, family),
        freq_weights=weights,
    ).fit(maxiter=300, tol=1e-12)

    actual_parameters = np.r_[model.result.intercept, model.result.beta]
    np.testing.assert_allclose(
        actual_parameters,
        reference.params,
        rtol=3e-6,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        model.predict(frame),
        reference.fittedvalues,
        rtol=3e-6,
        atol=3e-7,
    )
    assert metrics.deviance == pytest.approx(reference.deviance, rel=3e-9, abs=3e-9)

    pearson_dispersion = float(reference.pearson_chi2 / reference.df_resid)
    assert metrics._coefficient_dispersion == pytest.approx(
        pearson_dispersion,
        rel=3e-7,
        abs=3e-9,
    )
    if family == "poisson":
        assert model.result.phi == 1.0
    else:
        assert model.result.phi == pytest.approx(reference.scale, rel=3e-7, abs=3e-9)

    np.testing.assert_allclose(
        _model_standard_errors(model, metrics, family),
        reference.bse,
        rtol=3e-6,
        atol=3e-8,
    )

    quasi_reference = sm.GLM(
        y,
        design,
        family=_statsmodels_family(sm, family),
        freq_weights=weights,
    ).fit(maxiter=300, tol=1e-12, scale="X2")
    np.testing.assert_allclose(
        _quasi_standard_errors(model, metrics),
        quasi_reference.bse,
        rtol=3e-6,
        atol=3e-8,
    )


@pytest.mark.parametrize("family", ["gaussian", "gamma", "poisson"])
def test_frequency_weight_fit_and_covariance_match_glum(family: str) -> None:
    glum = pytest.importorskip("glum")
    frame, y, weights = _frequency_problem(family)
    design = frame.to_numpy()
    glum_family = "normal" if family == "gaussian" else family

    model = _fit_superglm(family, frame, y, weights)
    metrics = model.metrics(frame, y, sample_weight=weights)
    reference = glum.GeneralizedLinearRegressor(
        family=glum_family,
        alpha=0.0,
        robust=False,
        expected_information=True,
        gradient_tol=1e-10,
        step_size_tol=1e-12,
        max_iter=300,
    ).fit(design, y, sample_weight=weights)

    actual_parameters = np.r_[model.result.intercept, model.result.beta]
    reference_parameters = np.r_[reference.intercept_, reference.coef_]
    np.testing.assert_allclose(
        actual_parameters,
        reference_parameters,
        rtol=3e-6,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        model.predict(frame),
        reference.predict(design),
        rtol=3e-6,
        atol=3e-7,
    )
    reference_deviance = reference._family_instance.deviance(
        y,
        reference.predict(design),
        weights,
    )
    assert metrics.deviance == pytest.approx(reference_deviance, rel=3e-9, abs=3e-9)

    reference_dispersion = reference._family_instance.dispersion(
        y,
        reference.predict(design),
        sample_weight=weights,
        ddof=design.shape[1] + 1,
    )
    assert metrics._coefficient_dispersion == pytest.approx(
        reference_dispersion,
        rel=3e-7,
        abs=3e-9,
    )

    model_dispersion = 1.0 if family == "poisson" else model.result.phi
    reference_covariance = reference.covariance_matrix(
        design,
        y,
        sample_weight=weights,
        dispersion=model_dispersion,
        robust=False,
        expected_information=True,
    )
    # glum applies an additional finite-sample multiplier to its public
    # model-based covariance. Remove that library-specific convention before
    # comparing the common inverse-Fisher covariance.
    n_parameters = design.shape[1] + 1
    glum_correction = weights.sum() / (weights.sum() - n_parameters)
    reference_se = np.sqrt(np.diag(reference_covariance / glum_correction))
    np.testing.assert_allclose(
        _model_standard_errors(model, metrics, family),
        reference_se,
        rtol=3e-6,
        atol=3e-8,
    )


@pytest.mark.parametrize(
    ("family", "test"),
    [
        pytest.param("gaussian", "Chisq"),
        pytest.param("gaussian", "F"),
        pytest.param("gamma", "Chisq"),
        pytest.param("gamma", "F"),
        pytest.param("poisson", "Chisq"),
    ],
)
def test_drop1_matches_statsmodels_nested_deviance_test(
    family: str,
    test: str,
) -> None:
    sm = pytest.importorskip("statsmodels.api")
    frame, y, weights = _frequency_problem(family)
    full_design = sm.add_constant(frame.to_numpy())

    model = _fit_superglm(family, frame, y, weights)
    actual = model.drop1(frame, y, sample_weight=weights, test=test).set_index("feature")
    full = sm.GLM(
        y,
        full_design,
        family=_statsmodels_family(sm, family),
        freq_weights=weights,
    ).fit(maxiter=300, tol=1e-12)
    scale = 1.0 if family == "poisson" else float(full.scale)

    for dropped_index, feature in enumerate(frame.columns):
        kept_index = 1 - dropped_index
        reduced = sm.GLM(
            y,
            sm.add_constant(frame.to_numpy()[:, [kept_index]]),
            family=_statsmodels_family(sm, family),
            freq_weights=weights,
        ).fit(maxiter=300, tol=1e-12)
        delta_deviance = float(reduced.deviance - full.deviance)
        row = actual.loc[feature]

        assert row["delta_deviance"] == pytest.approx(
            delta_deviance,
            rel=3e-7,
            abs=3e-8,
        )
        assert row["delta_df"] == pytest.approx(1.0, rel=3e-9, abs=3e-9)
        expected_statistic = delta_deviance / scale
        if test == "F":
            expected_p_value = float(f_dist.sf(expected_statistic, 1.0, full.df_resid))
        else:
            expected_p_value = float(chi2.sf(expected_statistic, 1.0))
        assert row["statistic"] == pytest.approx(
            expected_statistic,
            rel=3e-6,
            abs=3e-8,
        )
        assert row["p_value"] == pytest.approx(
            expected_p_value,
            rel=3e-6,
            abs=1e-12,
        )
