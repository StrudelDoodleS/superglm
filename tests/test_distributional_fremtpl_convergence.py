"""Regressions for isolated-tail Gamma fits on uncapped freMTPL claims."""

from __future__ import annotations

import numpy as np
import pytest

from superglm import Categorical, GammaLS, Predictor, Spline
from tests.bound_predictor_fixtures import model_from_templates

from . import _datasets

_NUMERIC = ["VehAge", "DrivAge", "BonusMalus", "LogDensity", "VehPower"]
_CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
_MISSING = _datasets.skip_reason("freMTPL2freq.parquet") or _datasets.skip_reason(
    "freMTPL2sev.parquet"
)
pytestmark = pytest.mark.skipif(_MISSING is not None, reason=_MISSING or "")


def _weak_starts(data, predictors) -> dict[str, float]:
    """The weak start of 0.1 these fits were calibrated on, per spline penalty.

    That start weighed the curvature integral over each covariate; the
    penalty is now integrated over the covariate in knot intervals, ``hbar**3``
    times as large (``hbar`` the mean knot interval), so the same start is
    ``0.1 / hbar**3``.
    """
    return {
        f"{name}:{feature}#wiggle": 0.1
        * float((np.ptp(data[feature]) / (features[feature].n_knots + 1)) ** -3)
        for name, features in predictors
        for feature in features
        if feature in _NUMERIC
    }


@pytest.fixture(scope="module")
def uncapped_claims():
    claims = _datasets.load_sev().merge(
        _datasets.load_freq(), on="IDpol", how="inner", validate="many_to_one"
    )
    assert (claims.ClaimAmount > 0).all()
    claims["LogDensity"] = np.log(claims.Density)
    claims["y"] = claims.ClaimAmount / 1000.0
    rng = np.random.default_rng(20260909)
    policies = np.sort(claims.IDpol.unique())
    rng.shuffle(policies)
    split = {
        policy: "train"
        if i < int(0.6 * len(policies))
        else "valid"
        if i < int(0.8 * len(policies))
        else "test"
        for i, policy in enumerate(policies)
    }
    claims["split"] = claims.IDpol.map(split)
    claims["budget_order"] = rng.permutation(len(claims))
    for name in _CATEGORICAL:
        claims[name] = claims[name].astype(str)
    train = claims.loc[claims.split == "train"].sort_values("budget_order").reset_index(drop=True)
    full = claims.loc[claims.split != "test"].reset_index(drop=True)
    return train, full


@pytest.mark.parametrize("case", ["all_train", "all_full", "selected_full"])
def test_uncapped_gamma_mean_and_scale_converge(uncapped_claims, case):
    """Kills weak automatic starts and bitwise-only exact-face revalidation."""
    train, full = uncapped_claims
    data = train if case == "all_train" else full
    levels = {name: sorted(train[name].unique()) for name in _CATEGORICAL}

    def features(names):
        return {
            name: Spline("cr", k=8) if name in _NUMERIC else Categorical(levels=levels[name])
            for name in names
        }

    all_features = _NUMERIC + _CATEGORICAL
    scale_features = ["DrivAge", "VehPower"] if case == "selected_full" else all_features
    named = [("mean", features(all_features)), ("scale", features(scale_features))]
    options = {}
    if case == "selected_full":
        starts = _weak_starts(data, named)
        # VehAge's lambda reached the default cap of 1e10 over the covariate,
        # an exact face; in its knot intervals that cap is 1e10 times the same
        # hbar**-3, and every other lambda stops far below it.
        cap = 1.0e10 * starts["mean:VehAge#wiggle"] / 0.1
        options = {"initial_lambda": 0.1, "lambdas": starts, "max_lambda": cap}
    model = model_from_templates(
        family=GammaLS(),
        predictors=[Predictor(name, spec) for name, spec in named],
    ).fit_reml(data[all_features], data.y.to_numpy(), **options)
    state = model._require_fitted().fit_state
    fit = state.solver_result
    assert len(fit.theta) == len(data)
    assert model.result_.converged
    assert fit.converged
    assert fit.score_relative <= fit.config.tolerance
    retained_width = (
        len(fit.coefficients)
        if fit.coefficient_face is None
        else fit.coefficient_face.reduced_width
    )
    assert fit.terminal_rank.rank == retained_width
    assert fit.terminal_curvature.actual_source == "observed"
    assert all(
        item.terminal_curvature.fallback_count == 0 for item in state.smoothing.coefficient_fits
    )
    if case == "selected_full":
        assert state.exact_face_components == ("mean:VehAge#wiggle",)


def test_uncapped_gamma_explicit_weak_start_refuses_unresolved_stationarity(uncapped_claims):
    """The isolated claim stays in training; an unresolved score cannot certify a fit."""
    from scipy.special import digamma

    train, _ = uncapped_claims
    levels = {name: sorted(train[name].unique()) for name in _CATEGORICAL}
    all_features = _NUMERIC + _CATEGORICAL

    def features():
        return {
            name: Spline("cr", k=8) if name in _NUMERIC else Categorical(levels=levels[name])
            for name in all_features
        }

    named = [("mean", features()), ("scale", features())]
    starts = _weak_starts(train, named)
    model = model_from_templates(
        family=GammaLS(),
        predictors=[Predictor(name, spec) for name, spec in named],
    ).fit_reml(train[all_features], train.y.to_numpy(), initial_lambda=0.1, lambdas=starts)
    state = model._require_fitted().fit_state
    fit = state.solver_result
    np.testing.assert_array_equal(state.retained_rows.response, train.y.to_numpy())
    assert len(fit.theta) == len(train)
    assert all(
        value == starts.get(name, 0.1) for name, value in state.smoothing.initial_lambdas.items()
    )
    mean, scale = fit.theta.T
    shape = 1.0 / scale**2
    ratio = train.y.to_numpy() / mean
    bracket = np.log(shape) + 1 - digamma(shape) + np.log(ratio) - ratio
    row_score = (shape * (ratio - 1), -2 * shape * bracket)
    score, magnitudes = [], []
    for k, predictor in enumerate(state.layout.predictors):
        matrix = np.column_stack((np.ones(len(train)), predictor.design.toarray()))
        score.append(matrix.T @ row_score[k])
        # Include the terms before cancellation in the Gamma shape derivative.
        row_magnitude = (
            shape * (ratio + 1)
            if k == 0
            else 2
            * shape
            * (abs(np.log(shape)) + 1 + abs(digamma(shape)) + abs(np.log(ratio)) + ratio)
        )
        magnitudes.append(abs(matrix).T @ row_magnitude)
    independent = np.concatenate(score) - fit.penalty @ fit.coefficients
    magnitude = np.concatenate(magnitudes) + abs(fit.penalty) @ abs(fit.coefficients)
    operations = 64 * (len(train) + len(fit.coefficients))
    arithmetic = operations * np.finfo(float).eps / (1 - operations * np.finfo(float).eps)
    error = arithmetic * (1 + magnitude)
    assert np.all(np.isfinite(independent))
    assert np.linalg.norm(independent - fit.terminal_score, np.inf) <= max(error)
    denominator = 1 + abs(fit.penalized_optimizing_log_likelihood)
    unresolved = np.max(np.maximum(abs(independent) - error, 0.0)) / denominator
    assert unresolved > fit.config.tolerance
    assert not fit.converged
    assert not model.result_.coefficient_converged
    assert not model.result_.converged
    report = model.diagnose()
    assert any(finding.identifier == "fit.not_converged:coefficient" for finding in report.findings)


@pytest.fixture(scope="module")
def frequency_book_train():
    """The training rows of the books.py frequency book (seed 20261012, 80/20 split)."""
    freq = _datasets.load_freq_corrected()
    freq["LogDensity"] = np.log1p(freq["Density"].to_numpy(dtype=np.float64))
    for column in ("DrivAge", "VehAge", "BonusMalus", "VehPower"):
        freq[column] = freq[column].astype(np.float64)
    for column in ("VehBrand", "Region", "VehGas", "Area"):
        freq[column] = freq[column].astype(str)
    order = np.random.default_rng(20261012).permutation(len(freq))
    train = np.zeros(len(freq), dtype=bool)
    train[order[: int(round(0.8 * len(freq)))]] = True
    return freq[train].reset_index(drop=True)


def _book_location_terms():
    from superglm import cat, s

    return (
        s("DrivAge", kind="cr", k=10),
        s("VehAge", kind="cr", k=10),
        s("BonusMalus", kind="cr", k=10),
        s("LogDensity", kind="cr", k=8),
        s("VehPower", kind="cr", k=6),
        cat("VehBrand"),
        cat("Region"),
        cat("VehGas"),
        cat("Area"),
    )


def _fit_disclosing(model, sample, response, **options):
    """Fit, then check the result is usable and any non-convergence was warned."""
    import warnings

    from superglm.diagnostics.convergence import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(sample, response, **options)
    assert np.all(np.isfinite(model.covariance_))
    offsets = options.get("offsets")
    assert np.all(np.isfinite(np.asarray(model.predict(sample, offsets=offsets))))
    if not model.result_.converged:
        assert any(issubclass(item.category, ConvergenceWarning) for item in caught)
    # Every final-curvature fallback of these families used to refuse (NB2 had
    # no expected information, Tweedie has none), so each one is disclosed.
    telemetry = model._require_fitted().fit_state.solver_result.terminal_curvature
    if telemetry.fallback_count > 0:
        assert model.result_.warnings
        assert any(issubclass(item.category, ConvergenceWarning) for item in caught)
    for message in model.result_.warnings:
        assert any(str(item.message) == message for item in caught)


@pytest.mark.parametrize("discrete", [False, True])
def test_negative_binomial_book_sample_finishes_instead_of_refusing(frequency_book_train, discrete):
    """Kills the refusal on the 20,000-row freMTPL2 sample of the lss_nb benchmark case.

    Master raised RepeatedCurvatureIndefinitenessError on this sample, dense
    and binned: its unpenalized start fitted -log(exposure) with vehicle-age
    spline coefficients near 2e5, the solver stalled there, and the terminal
    observed curvature stayed indefinite with no Fisher fallback. The fit now
    returns and predicts finite means. Both routes end on Fisher curvature and
    stop with a smoothing parameter at its cap (lambda_cap_unresolved), and
    both are disclosed. The test pins the disclosures, not that outcome.
    """
    from superglm import NegativeBinomialLS, SuperLSS, cat, s

    sample = frequency_book_train.iloc[:20000]
    log_exposure = np.log(sample["Exposure"].to_numpy(dtype=np.float64))
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(*_book_location_terms()),
        family.theta(s("DrivAge", kind="cr", k=5), cat("VehGas"), cat("Area")),
        discrete=discrete,
    )
    _fit_disclosing(
        model,
        sample,
        sample["ClaimNb"].to_numpy(dtype=np.float64),
        offsets={"mean": log_exposure, "theta": log_exposure},
    )


def test_binned_tweedie_book_sample_finishes_instead_of_refusing(frequency_book_train):
    """Kills the refusal on a 3,000-row binned TweedieLSS fit of the frequency book.

    Master accepted a chunked line-search trial on its value, then raised when
    the family refused that trial's derivatives ("required float64 work is not
    representable"). The trial is now backtracked from, as a dense one is.
    """
    from superglm import SuperLSS, TweedieLSS, cat, s

    sample = frequency_book_train.iloc[:3000]
    exposure = sample["Exposure"].to_numpy(dtype=np.float64)
    family = TweedieLSS()
    model = SuperLSS(
        family,
        family.mu(*_book_location_terms()),
        family.phi(s("DrivAge", kind="cr", k=5), cat("VehGas"), cat("Area")),
        family.p(),
        discrete=True,
    )
    _fit_disclosing(
        model,
        sample,
        sample["ClaimTotal"].to_numpy(dtype=np.float64) / exposure,
        sample_weight=exposure,
    )
