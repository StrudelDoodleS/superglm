"""Uncertainty under ``centering="mean"``.

Mean centering is a change of identifiability constraint, not a change of
origin: the shift subtracted is the mean of the SAME fitted coefficients, so
the reported vector is the linear contrast ``C b`` with
``C = I - 11'/L`` and its covariance is ``C V C'`` -- see Searle,
*Linear Models* (Wiley, 1971), Ch. 5 on estimable functions of a
reparametrised model.  Translating the interval endpoints and leaving the
standard errors alone is only valid for a shift that is a known constant.

Two observable consequences pinned here:

* the reference level stops being the reference, so it must stop carrying the
  reference's exactly-zero standard error and zero-width interval;
* nothing about the report may depend on which level the fit happened to
  drop, because the whole premise of mean centering is that no level is
  privileged.

Tolerances.  Every comparison below is between two orders of association of
the same exact quadratic form (``diag(C M Cov M' C')`` against
``sqrt(diag(C @ (M Cov M') @ C.T))``).  The classical dot-product bound gives
a relative error on a variance of at most ``k*eps/(1 - k*eps)`` times the
condition number of the sum, with ``k`` the number of active coefficients
(<= 32 in these fixtures) and ``eps = 2.22e-16``; the fixtures are
well-conditioned by construction (unpenalised selection, balanced levels,
n=3000), keeping that condition number below 1e3.  The resulting bound is
``32 * 2.22e-16 * 1e3 ~ 7e-12`` on the variance and half that on its square
root.  ``rtol=1e-10`` is that bound with an order of magnitude spare.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from superglm import (
    Categorical,
    Numeric,
    Piecewise,
    Polynomial,
    PSpline,
    RandomEffect,
    SuperGLM,
)

RTOL = 1e-10
ATOL = 1e-13


def _design(seed=7, n=3000):
    rng = np.random.default_rng(seed)
    region = rng.choice(["L0", "L1", "L2", "L3"], n, p=[0.40, 0.30, 0.20, 0.10])
    age = rng.uniform(18.0, 80.0, n)
    dens = rng.uniform(0.0, 10.0, n)
    term = rng.uniform(1.0, 12.0, n)
    agent = rng.choice([f"A{i:02d}" for i in range(12)], n)
    weight = rng.uniform(0.3, 1.0, n)
    effect = {"L0": -0.4, "L1": 0.0, "L2": 0.25, "L3": 0.5}
    eta = -1.5 + np.array([effect[r] for r in region]) + 0.01 * (age - 50.0) + 0.05 * dens
    y = rng.poisson(np.exp(eta) * weight).astype(float)
    X = pd.DataFrame({"region": region, "age": age, "dens": dens, "term": term, "agent": agent})
    return X, y, weight


@pytest.fixture(scope="module")
def centering_model():
    X, y, weight = _design()
    model = SuperGLM(
        features={
            "region": Categorical(base="first"),
            "age": PSpline(n_knots=8),
            "dens": Polynomial(degree=2),
            "term": Piecewise(breaks=[3.0, 6.0, 9.0]),
        },
        selection_penalty=0.0,
    )
    model.fit(X, y, sample_weight=weight)
    return model


@pytest.fixture(scope="module")
def random_effect_model():
    X, y, weight = _design()
    model = SuperGLM(
        features={"agent": RandomEffect(), "dens": Numeric()},
        selection_penalty=0.0,
    )
    model.fit_reml(X, y, sample_weight=weight)
    return model


def _coefficient_block(model, name):
    """Dense covariance of the term's own active coefficients."""
    covariance, active_groups = model._coef_covariance
    subgroups = [g for g in active_groups if g.feature_name == name]
    indices = np.concatenate([np.arange(g.start, g.end) for g in subgroups])
    return np.asarray(covariance, dtype=np.float64)[np.ix_(indices, indices)]


def _report_map(model, name, n_points=200):
    """The explicit map from the term's active coefficients to what it reports.

    Built from the feature spec's own basis, independently of whatever route
    the inference code takes to the same quantity.
    """
    spec = model._specs[name]
    feature_groups = [g for g in model._groups if g.feature_name == name]
    beta = np.concatenate([model._result.beta[g.sl] for g in feature_groups])
    if isinstance(spec, PSpline):
        grid = spec.reconstruct(beta, n_points=n_points)["x"]
        return np.asarray(spec.transform(grid), dtype=np.float64)
    if isinstance(spec, Polynomial):
        grid = spec.reconstruct(beta)["x"]
        return np.asarray(spec.transform(grid), dtype=np.float64)
    if isinstance(spec, Piecewise):
        return np.asarray(
            spec._raw_basis_matrix(spec._knots)[:, spec._non_base_indices], dtype=np.float64
        )
    if isinstance(spec, Categorical):
        columns = {level: j for j, level in enumerate(spec._non_base)}
        rows = np.zeros((len(spec._levels), len(columns)))
        for i, level in enumerate(spec._levels):
            if level in columns:
                rows[i, columns[level]] = 1.0
        return rows
    if isinstance(spec, RandomEffect):
        return np.eye(len(spec.reconstruct(beta)["levels"]))
    raise TypeError(f"no reference map for {type(spec).__name__}")


def _contrast_se(model, name, n_points=200):
    """``sqrt(diag(C V C'))`` formed densely, C = I - 11'/L."""
    basis = _report_map(model, name, n_points=n_points)
    V = basis @ _coefficient_block(model, name) @ basis.T
    size = V.shape[0]
    C = np.eye(size) - np.ones((size, size)) / size
    return np.sqrt(np.maximum(np.diag(C @ V @ C.T), 0.0))


# ── The contrast covariance, term kind by term kind ────────────────


@pytest.mark.parametrize("name", ["region", "age", "dens", "term"])
def test_mean_centered_se_is_the_contrast_covariance(centering_model, name):
    ti = centering_model.term_inference(name, centering="mean")
    np.testing.assert_allclose(
        ti.se_log_relativity,
        _contrast_se(centering_model, name),
        rtol=RTOL,
        atol=ATOL,
    )


def test_random_effect_mean_centered_se_is_the_contrast_covariance(random_effect_model):
    ti = random_effect_model.term_inference("agent", centering="mean")
    np.testing.assert_allclose(
        ti.se_log_relativity,
        _contrast_se(random_effect_model, "agent"),
        rtol=RTOL,
        atol=ATOL,
    )


def test_compact_covariance_centers_without_materialising_its_block():
    """Above 256 structured coefficients the accessor REFUSES a dense block.

    So the correction cannot be ``C @ V @ C.T`` -- it has to reach ``Vp``
    through one matvec.  The dense ``gram`` backend fits the same model and
    supplies the reference the compact one may not form.
    """
    rng = np.random.default_rng(5)
    n, n_levels = 6000, 300
    codes = rng.integers(0, n_levels, n)
    agent = np.array([f"A{c:03d}" for c in codes])
    x = rng.normal(0.0, 1.0, n)
    weight = rng.uniform(0.3, 1.0, n)
    level_effect = rng.normal(0.0, 0.35, n_levels)
    y = rng.poisson(np.exp(-1.6 + level_effect[codes] + 0.2 * x) * weight).astype(float)
    X = pd.DataFrame({"agent": agent, "x": x})

    features = {"agent": RandomEffect(), "x": Numeric()}
    compact = SuperGLM(features=features, selection_penalty=0.0, direct_solve="structured")
    compact.fit_reml(X, y, sample_weight=weight)
    dense = SuperGLM(features=features, selection_penalty=0.0, direct_solve="gram")
    dense.fit_reml(X, y, sample_weight=weight)

    covariance, _ = compact._coef_covariance
    assert covariance.backend == "structured"
    with pytest.raises(ValueError, match="Refusing to materialize"):
        np.asarray(covariance, dtype=np.float64)

    np.testing.assert_allclose(
        compact.term_inference("agent", centering="mean").se_log_relativity,
        _contrast_se(dense, "agent"),
        rtol=1e-8,
        atol=1e-12,
    )


@pytest.mark.parametrize("name", ["region", "age", "dens", "term"])
def test_mean_centering_moves_the_standard_errors(centering_model, name):
    """The whole defect in one assertion: the two must not be the same array."""
    native = centering_model.term_inference(name, centering="native")
    mean = centering_model.term_inference(name, centering="mean")
    assert not np.allclose(native.se_log_relativity, mean.se_log_relativity, rtol=1e-8)


# ── The reference level stops being the reference ──────────────────


def test_mean_centered_base_level_carries_uncertainty(centering_model):
    native = centering_model.term_inference("region", centering="native")
    mean = centering_model.term_inference("region", centering="mean")
    base = int(np.argmin(native.se_log_relativity))
    assert native.se_log_relativity[base] == 0.0
    assert mean.se_log_relativity[base] > 0.0
    assert mean.ci_lower[base] < mean.relativity[base] < mean.ci_upper[base]


def test_mean_centered_base_knot_carries_uncertainty(centering_model):
    native = centering_model.term_inference("term", centering="native")
    mean = centering_model.term_inference("term", centering="mean")
    base = int(np.argmin(native.se_log_relativity))
    assert native.se_log_relativity[base] == 0.0
    assert mean.se_log_relativity[base] > 0.0
    assert mean.ci_lower[base] < mean.relativity[base] < mean.ci_upper[base]


def test_no_mean_centered_interval_is_degenerate(centering_model):
    for name in ("region", "age", "dens", "term"):
        ti = centering_model.term_inference(name, centering="mean")
        assert np.all(ti.se_log_relativity > 0.0), name
        assert np.all(ti.ci_upper > ti.ci_lower), name


# ── No level is privileged ─────────────────────────────────────────


def test_mean_centered_se_is_invariant_to_the_base_level():
    """The property the mode exists for, asserted without the C V C' formula.

    Dropping a different dummy reparametrises the same fitted model, so every
    estimable contrast -- which is what a mean-centered log relativity is --
    has to come back identical.  Under the defect the reported errors are the
    against-base errors, and those move with the base.
    """
    X, y, weight = _design(seed=11)
    reports = {}
    for base in ("L0", "L2", "L3"):
        model = SuperGLM(
            features={"region": Categorical(base=base), "dens": Numeric()},
            selection_penalty=0.0,
        )
        model.fit(X, y, sample_weight=weight)
        ti = model.term_inference("region", centering="mean")
        reports[base] = (list(ti.levels), np.asarray(ti.se_log_relativity, dtype=float))

    reference_levels, reference_se = reports["L0"]
    for base, (levels, se) in reports.items():
        assert levels == reference_levels, base
        np.testing.assert_allclose(se, reference_se, rtol=1e-8, atol=1e-12, err_msg=base)


def test_two_level_mean_centered_se_is_half_the_contrast_se():
    """Closed form: for L=2, C V C' has both diagonals at V11/4."""
    rng = np.random.default_rng(3)
    n = 2000
    band = rng.choice(["lo", "hi"], n)
    weight = rng.uniform(0.4, 1.0, n)
    eta = -1.2 + np.where(band == "hi", 0.45, 0.0)
    y = rng.poisson(np.exp(eta) * weight).astype(float)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(features={"band": Categorical(base="first")}, selection_penalty=0.0)
    model.fit(X, y, sample_weight=weight)

    native = model.term_inference("band", centering="native")
    mean = model.term_inference("band", centering="mean")
    contrast_se = float(np.max(native.se_log_relativity))
    np.testing.assert_allclose(
        mean.se_log_relativity,
        np.full(2, contrast_se / 2.0),
        rtol=RTOL,
        atol=ATOL,
    )


# ── Intervals are rebuilt, not translated ──────────────────────────


@pytest.mark.parametrize("name", ["region", "age", "dens", "term"])
def test_mean_centered_interval_is_rebuilt_from_the_centered_se(centering_model, name):
    """Rebuilt from ``sqrt(diag(C V C'))``, not the against-base error.

    Translating the endpoints by ``exp(-shift)`` also satisfies
    ``log(ci) = log_rel -+ z*se_reported``; the reference here is the contrast
    error, so the assertion only holds once the error itself is propagated.
    """
    ti = centering_model.term_inference(name, centering="mean", alpha=0.05)
    z = norm.ppf(0.975)
    se = _contrast_se(centering_model, name)
    np.testing.assert_allclose(
        np.log(ti.ci_lower), ti.log_relativity - z * se, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.log(ti.ci_upper), ti.log_relativity + z * se, rtol=RTOL, atol=ATOL
    )


def test_mean_centered_simultaneous_band_uses_the_centered_se(centering_model):
    ti = centering_model.term_inference("age", centering="mean", simultaneous=True)
    critical = ti.critical_value_simultaneous
    assert critical is not None and critical > norm.ppf(0.975)
    se = _contrast_se(centering_model, "age")
    np.testing.assert_allclose(
        np.log(ti.ci_upper_simultaneous),
        ti.log_relativity + critical * se,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.log(ti.ci_lower_simultaneous),
        ti.log_relativity - critical * se,
        rtol=1e-8,
        atol=1e-10,
    )


# ── The other reporting surface ────────────────────────────────────


@pytest.mark.parametrize("name", ["region", "age", "dens", "term"])
def test_relativities_mean_centered_se_is_the_contrast_covariance(centering_model, name):
    """``relativities()`` centers its own frame, so it owns the same defect."""
    frame = centering_model.relativities(with_se=True, centering="mean")[name]
    np.testing.assert_allclose(
        frame["se_log_relativity"].to_numpy(),
        _contrast_se(centering_model, name),
        rtol=RTOL,
        atol=ATOL,
    )


# ── Native centering is untouched ──────────────────────────────────


@pytest.mark.parametrize("name", ["region", "age", "dens", "term"])
def test_native_centering_still_reports_the_against_base_errors(centering_model, name):
    """Guard: ``native`` asks for no contrast, so the errors stay ``diag(V)``."""
    native = centering_model.term_inference(name, centering="native")
    basis = _report_map(centering_model, name)
    V = basis @ _coefficient_block(centering_model, name) @ basis.T
    np.testing.assert_allclose(
        native.se_log_relativity,
        np.sqrt(np.maximum(np.diag(V), 0.0)),
        rtol=RTOL,
        atol=ATOL,
    )
    default = centering_model.term_inference(name)
    np.testing.assert_array_equal(default.se_log_relativity, native.se_log_relativity)
    np.testing.assert_array_equal(default.ci_lower, native.ci_lower)
    np.testing.assert_array_equal(default.ci_upper, native.ci_upper)


def test_native_categorical_base_level_keeps_its_zero(centering_model):
    """Guard: the against-base report still pins the reference at zero."""
    ti = centering_model.term_inference("region", centering="native")
    base = int(np.argmin(ti.se_log_relativity))
    assert ti.se_log_relativity[base] == 0.0
    assert ti.log_relativity[base] == 0.0


def test_pinned_level_also_stops_being_certain_under_mean_centering():
    """A declared level with no training rows has a zero row of ``V`` too.

    Its coefficient is fixed at zero, so ``native`` gives it the base's exactly
    zero error.  Under centering it is a deviation from an estimated mean, so it
    picks up ``sqrt(p'Vp)`` like the base does -- and the map is a selection
    narrower than the level table, which is the shape the correction has to
    survive.
    """
    rng = np.random.default_rng(17)
    n = 2000
    band = rng.choice(["a", "b", "c"], n)
    weight = rng.uniform(0.4, 1.0, n)
    eta = -1.3 + np.where(band == "b", 0.3, 0.0) + np.where(band == "c", -0.2, 0.0)
    y = rng.poisson(np.exp(eta) * weight).astype(float)
    X = pd.DataFrame({"band": band})

    model = SuperGLM(
        features={"band": Categorical(base="first", levels=["a", "b", "c", "d"])},
        selection_penalty=0.0,
    )
    with pytest.warns(UserWarning, match="pinned to base"):
        model.fit(X, y, sample_weight=weight)

    spec = model._specs["band"]
    assert spec._pinned_levels == ["d"]
    native = model.term_inference("band", centering="native")
    mean = model.term_inference("band", centering="mean")
    positions = {level: i for i, level in enumerate(native.levels)}
    for level in (spec._base_level, "d"):
        assert native.se_log_relativity[positions[level]] == 0.0
        assert mean.se_log_relativity[positions[level]] > 0.0
    np.testing.assert_allclose(
        mean.se_log_relativity, _contrast_se(model, "band"), rtol=RTOL, atol=ATOL
    )


def test_mean_centering_shifts_a_smooth_curve_and_passes_everything_else_through() -> None:
    """``_recenter_term`` shifts a ``SmoothCurve``'s values and copies the rest.

    Called directly, because no in-tree caller reaches this branch.  The only
    term kind that carries a ``smooth_curve`` is ``OrderedCategorical`` with a
    smooth inner basis, and that branch is anchored on its base level and
    returns without recentering -- so the curve arm of ``_recenter_term`` is
    live code on an unexercised path.  Measured rather than assumed: probing
    every call across the 2191 tests that touch centering, term inference,
    relativities, plotting, the editor and each inner basis recorded 991 calls,
    104 of them under ``centering="mean"``, and **0** carrying a curve.  An
    end-to-end fixture would therefore pin nothing at all; this one does.

    What it pins is the ``replace`` spelling, not today's field list.  The
    function names four of ``SmoothCurve``'s seven fields; the other three were
    being copied across by hand, so an eighth field added later would have
    silently taken its DEFAULT instead of the value the curve carries.  The
    check below is therefore written over ``dataclasses.fields`` -- every field
    the function does not name must come back identical -- so a new field is
    covered the day it is added rather than the day someone remembers this test.

    Exact equality throughout, and it is not a tolerance anyone chose: the
    shift is subtracted from, and the factor multiplied into, the very arrays
    the assertions rebuild, so both sides are the same IEEE operation on the
    same operands.  A tolerance here would only hide a genuine difference.
    """
    import dataclasses

    from superglm.inference import SmoothCurve, TermInference, _recenter_term, _safe_exp

    log_rel = np.array([-0.4, -0.1, 0.2, 0.7])
    curve = SmoothCurve(
        x=np.linspace(0.0, 3.0, 7),
        log_relativity=np.linspace(-0.4, 0.7, 7),
        relativity=np.exp(np.linspace(-0.4, 0.7, 7)),
        level_x=np.arange(4.0),
        se_log_relativity=np.full(7, 0.05),
        ci_lower=np.exp(np.linspace(-0.4, 0.7, 7) - 0.098),
        ci_upper=np.exp(np.linspace(-0.4, 0.7, 7) + 0.098),
    )
    ti = TermInference(
        name="band",
        kind="categorical",
        active=True,
        levels=["a", "b", "c", "d"],
        log_relativity=log_rel,
        relativity=np.exp(log_rel),
        smooth_curve=curve,
    )

    centered = _recenter_term(ti, "mean")
    shift = float(np.mean(log_rel))
    factor = _safe_exp(-shift)
    assert centered.centering_shift == shift
    assert centered.centering_mode == "mean"

    new = centered.smooth_curve
    assert new is not None
    # the four the function names: a shift on the log scale, a scaling on the
    # relativity scale, and the interval endpoints carried with it
    np.testing.assert_array_equal(new.log_relativity, np.asarray(curve.log_relativity) - shift)
    np.testing.assert_array_equal(new.relativity, np.asarray(curve.relativity) * factor)
    np.testing.assert_array_equal(new.ci_lower, np.asarray(curve.ci_lower) * factor)
    np.testing.assert_array_equal(new.ci_upper, np.asarray(curve.ci_upper) * factor)
    # everything else, whatever it grows to be
    shifted = {"log_relativity", "relativity", "ci_lower", "ci_upper"}
    passed_through = [f.name for f in dataclasses.fields(SmoothCurve) if f.name not in shifted]
    assert passed_through == ["x", "level_x", "se_log_relativity"], (
        f"SmoothCurve gained or lost a field; check it is shifted or copied: {passed_through}"
    )
    for field in passed_through:
        np.testing.assert_array_equal(
            getattr(new, field), getattr(curve, field), err_msg=f"{field} was not passed through"
        )
