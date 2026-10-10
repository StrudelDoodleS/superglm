"""Fitted outputs reproduce the golden record within tolerance.

Re-recorded 2026-09-02 in tolerance form.  The original byte-identical record
(sha256 of coefficients and covariance, hex-exact objective and lambdas) was
the gate for the package reorganisation and served that purpose once; as a
standing test it asserted bit-identity of floating-point results across BLAS
builds and thread counts, and 0 of its 12 hashes reproduced on the next stack
that ran it.  The record now stores the numbers and the comparison allows
last-bit noise while still catching a numerical regression: coefficients to a
relative 1e-8, the covariance trace and Frobenius norm to 1e-8, the smoothing
objective to the outer loop's acceptance band (``objective_tolerance``, 1e-9,
times 1 + |V|), and the convergence reason exactly.  Lambdas are compared on
the log scale, because a smoothing parameter is a positive quantity and the
record spans 0.079 to 7.9e7: |log(new/old)| <= 1e-6 below the saturation floor,
and such a lambda may not cross that floor. A lambda the loop was still moving
when it stopped, saturated or not (``gaussian:reml`` and ``lognormal:reml``
``scale:z#wiggle`` are both), is held only to drifting the same way as in the
record, since one ridge step can carry it across the floor either way; any
other saturated lambda is held to staying above the floor
(``_SATURATED_LAMBDA``, ``_DRIFTING_LOG_STEP``).

Re-recording the 12 pre-existing entries in tolerance form did move two
numbers, and only two.  Decoding the byte-identical record against this one,
all 8 objectives agree to 4.1e-14 and the 14 unsaturated lambdas to 1.2e-8,
while ``gaussian:reml`` and ``gaussian:reml+newton`` both carry
``scale:z#wiggle`` 79543305.66 there against 79541630.40 here -- |log| 2.1e-5,
on a lambda the REML objective is flat in, and with the objective itself
unmoved at 4.1e-14.  That is the drift the saturated bound is sized for; a
uniform 1e-6 would have made the record reproduce on this stack and fail on
the stack that recorded it.

The record also distinguishes the observed-Hessian path used by families with
expected information from the Fisher path.  The Gaussian and Gamma cases reach
the same optimum by a different iteration path; NB2 and Tweedie do not provide
expected information.  Measured against the Fisher path, the relative change
of the penalised log-likelihood and of the largest coefficient, with total
inner iterations Fisher -> observed, was:

    gaussian:fixed  pll 5.7e-15  largest coefficient 1.4e-7  (8 -> 5)
    gaussian:reml   pll 2.1e-10  largest coefficient 2.2e-9  (36 -> 18)
    gamma:fixed     pll 8.1e-16  largest coefficient 8.9e-8  (9 -> 5)
    gamma:reml      pll 1.7e-10  largest coefficient 6.0e-8  (161 -> 61)

The ``:reml`` cases pin ``outer="efs"`` so they stay on the Fellner--Schall
path they were recorded on; the ``:reml+newton`` cases also exercise the
Newton endgame.

Re-recorded 2026-10-10, when a ``cr`` penalty became the curvature integral
over the covariate in knot intervals, ``hbar**3`` times the integral over the
covariate (``hbar`` the mean knot interval, about 2/7 on ``x`` and 2/5 on
``z``). The cases keep their models by multiplying every lambda and start by
``hbar**-3`` (``_per_knot``), and the record keeps lambdas over the covariate
(``_in_recorded_units``). Against the previous record the 16 smoothing cases'
fitted parameters reproduced to 4.5e-10 relative and their EDF to 1.5e-11,
the fixed cases' to 1.8e-13, the objectives to 4.3e-12 and the unsaturated
lambdas to 1.4e-12 in log. Two lambdas on flat directions moved further,
``gaussian`` ``scale:z#wiggle`` (saturated) by 6.5e-4 and ``nb2``
``theta:z#wiggle`` by 1.8e-6 in log, with the fits unmoved. Coefficients and
covariance are in the SSP basis, whose ridge ``0.1 Omega`` the new scale of
``Omega`` changes, so they moved and were re-recorded with the rest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import (
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogNormalLS,
    NegativeBinomialLS,
    Predictor,
    TweedieLSS,
    TwoPieceLogNormalLSS,
)
from superglm.distributional.kernels.generalized_gamma import log_mean_loading
from superglm.distributional.kernels.two_piece import (
    log_mean_loading as two_piece_log_mean_loading,
)
from superglm.distributional.kernels.two_piece import two_piece_quantile
from superglm.distributional.result import DistributionalEFSConfig
from superglm.features import Categorical, CubicRegressionSpline, Spline
from tests.bound_predictor_fixtures import model_from_templates

GOLDEN = Path(__file__).parent / "fixtures" / "distributional_golden.json"


def _frame(n: int = 900, seed: int = 17):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(np.array(["a", "b", "c"]), size=n)
    mu = np.exp(0.5 + 0.6 * np.sin(np.pi * x) + 0.2 * (g == "b"))
    return pd.DataFrame({"x": x, "z": z, "g": g}), mu, rng


def _cases():
    frame, mu, rng = _frame()
    sigma = np.exp(-0.5 + 0.3 * frame["z"].to_numpy())
    gaussian_y = np.log(mu) + rng.normal(scale=sigma)
    gamma_y = rng.gamma(4.0, mu / 4.0)
    nb_y = rng.negative_binomial(2.0, 2.0 / (mu + 2.0)).astype(float)
    lam = mu**0.5 / (0.8 * 0.5)
    counts = rng.poisson(lam)
    tweedie_y = np.where(
        counts > 0,
        rng.gamma(np.maximum(counts, 1), 0.8 * 0.5 * mu**0.5),
        0.0,
    )
    # exp() of an already-drawn response: consumes no randomness, so every
    # existing case keeps the draws its golden hash was recorded on
    lognormal_y = np.exp(gaussian_y)
    gg_q = 0.6
    gg_sigma = np.exp(-0.5 + 0.3 * frame["z"].to_numpy())
    gg_k = 1.0 / gg_q**2
    gg_w = np.log(rng.gamma(gg_k, 1.0, len(frame)) / gg_k) / gg_q
    gengamma_y = np.exp(
        np.log(mu) - log_mean_loading(gg_sigma, np.full(len(frame), gg_q))[0] + gg_sigma * gg_w
    )
    gpd_shape = 0.35
    gpd_scale = mu / 4.0
    gpd_y = gpd_scale * np.expm1(-gpd_shape * np.log(rng.random(len(frame)))) / gpd_shape
    # drawn last, so every earlier case keeps the draws its golden hash was recorded on
    tp_skew = np.full(len(frame), 0.4)
    tp_sigma = np.exp(-0.5 + 0.3 * frame["z"].to_numpy())
    tp_mu = np.log(mu) - two_piece_log_mean_loading(tp_sigma, tp_skew)[0]
    twopiece_y = np.exp(two_piece_quantile(rng.random(len(frame)), tp_mu, tp_sigma, tp_skew))
    mean = Predictor("mean", {"x": Spline(kind="cr", k=8), "g": Categorical()})
    return {
        "gaussian": (
            GaussianLS(),
            (
                Predictor("location", {"x": Spline(kind="cr", k=8), "g": Categorical()}),
                Predictor("scale", {"z": Spline(kind="cr", k=6)}),
            ),
            gaussian_y,
        ),
        "lognormal": (
            LogNormalLS(),
            (mean, Predictor("scale", {"z": Spline(kind="cr", k=6)})),
            lognormal_y,
        ),
        "gamma": (
            GammaLS(),
            (mean, Predictor("scale", {"z": Spline(kind="cr", k=6)})),
            gamma_y,
        ),
        "nb2": (
            NegativeBinomialLS(),
            (mean, Predictor("theta", {"z": Spline(kind="cr", k=6)})),
            nb_y,
        ),
        "tweedie": (
            TweedieLSS(),
            (
                mean,
                Predictor("dispersion", {"z": Spline(kind="cr", k=6)}),
                Predictor("power", {}),
            ),
            tweedie_y,
        ),
        "gengamma": (
            GeneralizedGammaLSS(),
            (mean, Predictor("scale", {"z": Spline(kind="cr", k=6)}), Predictor("shape", {})),
            gengamma_y,
        ),
        "twopiece": (
            TwoPieceLogNormalLSS(),
            (mean, Predictor("scale", {"z": Spline(kind="cr", k=6)}), Predictor("skew", {})),
            twopiece_y,
        ),
        "gpd": (
            GeneralizedParetoLSS(),
            (
                Predictor("scale", {"x": Spline(kind="cr", k=8), "g": Categorical()}),
                Predictor("shape", {}),
            ),
            gpd_y,
        ),
    }, frame


def _record(model: SuperLSS) -> dict[str, object]:
    fitted = model._require_fitted()
    coef = np.array(list(model.coef_.values()), dtype=np.float64)
    covariance = np.asarray(model.covariance_, dtype=np.float64)
    smoothing = fitted.smoothing
    payload: dict[str, object] = {
        "coefficients": [float(value) for value in coef],
        "covariance_trace": float(np.trace(covariance)),
        "covariance_frobenius": float(np.linalg.norm(covariance)),
    }
    if smoothing is not None:
        payload["objective"] = float(smoothing.objective)
        payload["lambdas"] = {k: float(v) for k, v in smoothing.lambdas.items()}
        payload["reason"] = smoothing.convergence_reason
        payload["drifting"] = _drifting(model)
    return payload


# A lambda at or above this floor has saturated: the REML objective is flat in
# it, so its recorded value is a property of where the outer loop stopped on
# that ridge rather than of the data.  The record's largest unsaturated lambda
# is 4.25e5 and its four saturated ones are gaussian's 7.96e7 and lognormal's
# 1.09e7 (``scale:z#wiggle``, each twice), a gap of 25.7: the floor sits 2.35
# times above the first and 10.9 times below the second.
#
# A saturated lambda is held only to staying saturated. Fellner-Schall advances
# a lambda on that ridge by a constant additive step, so |dlog lambda| decays as
# 1/iteration (``DistributionalEFSConfig``) and the stopping test does not bound
# where it stops: that is set by the rounding path. The 1e-4 bound this replaces
# was one measured drift (2.1e-5) widened, not a derivation; since the penalties
# moved to knot intervals, CI's OpenBLAS kernels spread the pair by 1.2e-4 to
# 8.9e-4 in log while the coefficients agreed to 1e-8 and the objective to 1e-10.
# The coefficients, the covariance and the objective (to the outer loop's
# acceptance band), held here, are what the fit is.
_SATURATED_LAMBDA = 1.0e6
# Measured on this record: unsaturated lambdas reproduce to |log(new/old)| =
# 1.2e-8 across stacks.
_LOG_LAMBDA_TOLERANCE = 1.0e-6
# The same holds below the floor for a lambda the outer loop was still moving
# when it stopped: on its ridge each accepted step is O(1) in log (``nb2:reml``
# ``theta:z#wiggle`` advances 1.327 per iteration to 4.25e5, and its
# ``+newton`` twin stops at 2.16e4 with the same fit), so its value is where the
# loop stopped. Measured on this record, the ridge lambdas' last accepted steps
# are 0.45 to 2.5 in log and every other lambda's at most 6.5e-2; this bar sits
# in that gap. The record stores which way each such lambda was moving; the
# computed run must drift the same way, and its value is not held.
_DRIFTING_LOG_STEP = 0.25
_OBJECTIVE_RESOLUTION = DistributionalEFSConfig().objective_tolerance


def _drifting(model: SuperLSS) -> dict[str, float]:
    """The lambdas whose last accepted outer step still exceeded ``_DRIFTING_LOG_STEP``,
    each with that step's sign."""
    smoothing = model._require_fitted().smoothing
    if smoothing is None:
        return {}
    last: dict[str, float] = {}
    for item in smoothing.history:
        if item.accepted:
            for key in smoothing.lambdas:
                last[key] = math.log(item.lambdas_after[key] / item.lambdas_before[key])
    return {
        key: math.copysign(1.0, step)
        for key, step in last.items()
        if abs(step) > _DRIFTING_LOG_STEP
    }


def _assert_close(name: str, computed: dict[str, object], recorded: dict[str, object]) -> None:
    assert set(computed) == set(recorded), f"{name}: recorded fields differ"
    # Scale near-zero coefficients like the rest instead of imposing a tighter floor.
    computed_coefficients = np.asarray(computed["coefficients"])
    recorded_coefficients = np.asarray(recorded["coefficients"])
    coefficient_error = np.max(
        np.abs(computed_coefficients - recorded_coefficients)
        / (1.0 + np.abs(recorded_coefficients))
    )
    assert coefficient_error <= 1e-8, f"{name}: coefficients moved ({coefficient_error=:.3g})"
    for field in ("covariance_trace", "covariance_frobenius"):
        assert abs(computed[field] - recorded[field]) <= 1e-8 * abs(recorded[field]), (
            f"{name}: {field} moved"
        )
    if "objective" in recorded:
        # The outer loop accepts a step whose objective rises by up to
        # ``objective_tolerance * (1 + |V|)``, so it resolves where it stops only
        # to that band, and a step it rejects there is a rounding decision: ARM64
        # stopped ``twopiece:reml+newton`` 2.1e-10 of |V| from x86. The 1e-10 this
        # replaces was a measured spread (4.1e-14) widened, below that resolution.
        assert abs(computed["objective"] - recorded["objective"]) <= _OBJECTIVE_RESOLUTION * (
            1.0 + abs(recorded["objective"])
        ), f"{name}: objective moved"
        assert set(computed["lambdas"]) == set(recorded["lambdas"]), f"{name}: lambda keys differ"
        # A drifting lambda must drift the same way as in the recording run; its
        # value is where the loop stopped, a step either side of the record's.
        drifting = recorded["drifting"]
        assert computed["drifting"] == drifting, (
            f"{name}: drifting lambdas {computed['drifting']} against recorded {drifting}"
        )
        for key, value in recorded["lambdas"].items():
            other = computed["lambdas"][key]
            assert value > 0.0 and other > 0.0, f"{name}: lambda {key} is not positive"
            if key in drifting:
                continue
            if value >= _SATURATED_LAMBDA:
                assert other >= _SATURATED_LAMBDA, f"{name}: lambda {key} left saturation"
                continue
            assert other < _SATURATED_LAMBDA, f"{name}: lambda {key} saturated"
            assert abs(math.log(other / value)) <= _LOG_LAMBDA_TOLERANCE, (
                f"{name}: lambda {key} moved"
            )
        assert computed["reason"] == recorded["reason"], f"{name}: convergence reason changed"


def _wiggle_names(predictors) -> list[str]:
    names = []
    for predictor in predictors:
        for feature, spec in predictor.features.items():
            if isinstance(spec, CubicRegressionSpline):
                names.append(f"{predictor.name}:{feature}#wiggle")
    return names


def _per_knot(frame, predictors) -> dict[str, float]:
    """``hbar**-3`` for each wiggle penalty, ``hbar`` its spline's mean knot interval.

    The record was taken on penalties integrated over the covariate; they are
    now integrated over the covariate in knot intervals, ``hbar**3`` times as
    large, so a lambda this many times larger fits the same model.
    """
    return {
        f"{predictor.name}:{feature}#wiggle": float(
            (np.ptp(frame[feature]) / (spec.n_knots + 1)) ** -3
        )
        for predictor in predictors
        for feature, spec in predictor.features.items()
        if isinstance(spec, CubicRegressionSpline)
    }


def _in_recorded_units(record: dict[str, object], per_knot: dict[str, float]):
    if "lambdas" in record:
        record["lambdas"] = {
            key: value / per_knot.get(key, 1.0) for key, value in record["lambdas"].items()
        }
    return record


def _compute() -> dict[str, dict[str, object]]:
    cases, frame = _cases()
    out = {}
    for name, (family, predictors, y) in cases.items():
        per_knot = _per_knot(frame, predictors)
        fixed = model_from_templates(family=family, predictors=predictors).fit(
            frame, y, lambdas={key: 1.0 * per_knot[key] for key in _wiggle_names(predictors)}
        )
        out[f"{name}:fixed"] = _record(fixed)
        # The record predates automatic initialization. Keep its numerical
        # configuration fixed; default-start behavior has its own regressions.
        starts = {key: 0.1 * value for key, value in per_knot.items()}
        reml = model_from_templates(family=family, predictors=predictors).fit_reml(
            frame, y, outer="efs", initial_lambda=0.1, lambdas=starts
        )
        out[f"{name}:reml"] = _in_recorded_units(_record(reml), per_knot)
        newton = model_from_templates(family=family, predictors=predictors).fit_reml(
            frame, y, outer="efs+newton", initial_lambda=0.1, lambdas=starts
        )
        out[f"{name}:reml+newton"] = _in_recorded_units(_record(newton), per_knot)
    return out


def test_outputs_reproduce_the_golden_record_within_tolerance(request) -> None:
    computed = _compute()
    if request.config.getoption("--regenerate-golden", default=False):
        GOLDEN.write_text(json.dumps(computed, indent=2, sort_keys=True))
        pytest.skip("golden record regenerated")
    recorded = json.loads(GOLDEN.read_text())
    assert set(computed) == set(recorded)
    # Only the record's flat scale directions drift; the location and mean
    # lambdas, which every case identifies, stay held to the tight bound.
    drifting = [entry.get("drifting", {}) for entry in recorded.values()]
    assert all(key.endswith("z#wiggle") for keys in drifting for key in keys)
    for name in recorded:
        _assert_close(name, computed[name], recorded[name])


def _recorded_entry(name: str) -> dict[str, object]:
    return json.loads(json.dumps(json.loads(GOLDEN.read_text())[name]))


def test_a_drifting_lambda_is_held_to_its_recorded_direction_not_its_value() -> None:
    """``nb2:reml``'s ``theta:z#wiggle`` climbs 1.327 in log per step on its ridge: a run
    that stops a step later (past the floor) or a step earlier passes, and one that
    drifts the other way, or stops drifting, fails."""
    recorded = _recorded_entry("nb2:reml")
    assert recorded["drifting"] == {"theta:z#wiggle": 1.0}
    for factor in (math.exp(1.327), math.exp(-1.327)):
        computed = _recorded_entry("nb2:reml")
        computed["lambdas"]["theta:z#wiggle"] *= factor
        _assert_close("nb2:reml", computed, recorded)
    for drift in ({"theta:z#wiggle": -1.0}, {}):
        computed = _recorded_entry("nb2:reml")
        computed["drifting"] = drift
        with pytest.raises(AssertionError, match="drifting"):
            _assert_close("nb2:reml", computed, recorded)


def test_a_lambda_that_leaves_saturation_is_still_caught() -> None:
    # The Newton endgame stops this saturated lambda without drifting (the plain
    # EFS run's twin is still climbing, and is held only to that).
    recorded = _recorded_entry("gaussian:reml+newton")
    assert recorded["drifting"] == {}
    computed = _recorded_entry("gaussian:reml+newton")
    computed["lambdas"]["scale:z#wiggle"] = 1.0e3
    with pytest.raises(AssertionError, match="scale:z#wiggle"):
        _assert_close("gaussian:reml+newton", computed, recorded)


def test_an_unsaturated_lambda_is_still_held_to_the_tight_bound() -> None:
    recorded = _recorded_entry("gaussian:reml")
    computed = _recorded_entry("gaussian:reml")
    value = recorded["lambdas"]["location:x#wiggle"]
    assert value < _SATURATED_LAMBDA
    computed["lambdas"]["location:x#wiggle"] = value * (1.0 + 1.0e-4)
    with pytest.raises(AssertionError, match="location:x#wiggle"):
        _assert_close("gaussian:reml", computed, recorded)
