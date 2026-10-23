"""A SuperLSS model saved on one machine loads on another.

A load hashes the inputs a model stores and compares them exactly. Values it
recomputes with a transcendental or a BLAS kernel, such as the ``-log y``
likelihood carrier or a penalty face's tolerances, are compared within a
rounding bound or not rebound at all. The simulated tests replace a family
module's ``log`` (or a face's spectral norm, or its projection) at load with
one that rounds one row an ulp the other way, as another C library, SIMD loop
or OpenBLAS kernel may, and run on every platform. The fixtures in
``fixtures/saved_portable_v041`` were saved on real platforms that disagree
(``scripts/make_portable_load_fixtures.py``).
"""

from __future__ import annotations

import lzma
import math
import pickle
import types
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import superglm as sg
from superglm import s
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional.families import gamma as gamma_module
from superglm.distributional.families import gaussian as gaussian_module
from superglm.distributional.families import generalized_gamma as generalized_gamma_module
from superglm.distributional.families import log_normal as log_normal_module
from superglm.distributional.families import negative_binomial as negative_binomial_module
from superglm.distributional.families import two_piece as two_piece_module
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.results import smoothing as result_smoothing
from superglm.distributional.results import solver as result_solver
from superglm.distributional.serialization import DistributionalSerializationError
from superglm.distributional.smoothing import penalty_face as penalty_face_module
from superglm.distributional.smoothing.penalty_face import PenaltyFace

from .test_userspace_040 import _lss_eta_rounding

FIXTURES = Path(__file__).parent / "fixtures" / "saved_portable_v041"
N_ROWS = 120


def _nudged(values: np.ndarray, *, factor: float | None = None) -> np.ndarray:
    """``values`` with its largest-magnitude row moved one ulp, or scaled by ``factor``."""
    if np.ndim(values) == 0:
        return values
    out = np.array(values, dtype=np.float64, copy=True)
    row = int(np.argmax(np.abs(out)))
    out[row] = out[row] * factor if factor is not None else np.nextafter(out[row], np.inf)
    return out


def _rounded_up(values: np.ndarray, *, factor: float | None = None) -> np.ndarray:
    """``values`` with every row one ulp up, or the largest row scaled by ``factor``."""
    if np.ndim(values) == 0:
        return values
    if factor is not None:
        return _nudged(values, factor=factor)
    return np.nextafter(np.asarray(values, dtype=np.float64), np.inf)


class _OtherNumpy(types.ModuleType):
    """``numpy``, except ``log`` rounds every row the other way."""

    def __init__(self, factor: float | None) -> None:
        super().__init__("numpy")
        self._factor = factor

    def __getattr__(self, name: str):
        return getattr(np, name)

    def log(self, values, *args, **kwargs):
        return _rounded_up(np.log(values, *args, **kwargs), factor=self._factor)


class _OtherSpecial(types.ModuleType):
    """``scipy.special``, except ``gammaln`` rounds every row the other way."""

    def __init__(self, factor: float | None) -> None:
        super().__init__("scipy.special")
        self._factor = factor

    def __getattr__(self, name: str):
        return getattr(_REAL_SPECIAL, name)

    def gammaln(self, values, *args, **kwargs):
        real = _REAL_SPECIAL.gammaln(values, *args, **kwargs)
        return _rounded_up(real, factor=self._factor)


_REAL_SPECIAL = negative_binomial_module.special


def _another_platform(monkeypatch: pytest.MonkeyPatch, case: str, factor: float | None = None):
    """Patch ``case``'s carrier transcendental as another platform rounds it."""
    module = _CASES[case]["module"]
    if case == "nb":
        monkeypatch.setattr(module, "special", _OtherSpecial(factor))
    elif case == "two_piece_normal":
        constant = module._HALF_LOG_TWO_PI
        moved = constant * factor if factor is not None else np.nextafter(constant, np.inf)
        monkeypatch.setattr(module, "_HALF_LOG_TWO_PI", float(moved))
    else:
        monkeypatch.setattr(module, "np", _OtherNumpy(factor))


def _data(seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, N_ROWS)
    mean = np.exp(0.4 + 0.1 * x)
    return (
        pd.DataFrame({"x": x}),
        {
            "positive": rng.gamma(3.0, mean / 3.0),
            "real": 1.0 + 0.2 * x + rng.normal(0.0, 0.5, N_ROWS),
            "count": rng.poisson(mean).astype(np.float64),
            "weight": rng.uniform(0.2, 4.0, N_ROWS),
            "replicates": rng.integers(1, 4, N_ROWS).astype(np.float64),
        },
    )


_CASES = {
    "gamma_prior": {"module": gamma_module, "family": sg.GammaLS, "response": "positive"},
    "gamma_frequency": {
        "module": gamma_module,
        "family": sg.GammaLS,
        "response": "positive",
        "weights": "replicates",
        "semantics": "frequency",
    },
    "log_normal": {"module": log_normal_module, "family": sg.LogNormalLS, "response": "positive"},
    "generalized_gamma": {
        "module": generalized_gamma_module,
        "family": sg.GeneralizedGammaLSS,
        "response": "positive",
    },
    "two_piece_log_normal": {
        "module": two_piece_module,
        "family": sg.TwoPieceLogNormalLSS,
        "response": "positive",
    },
    "two_piece_normal": {
        "module": two_piece_module,
        "family": sg.TwoPieceNormalLSS,
        "response": "real",
    },
    "gaussian_prior_weights": {
        "module": gaussian_module,
        "family": sg.GaussianLS,
        "response": "real",
        "weights": "weight",
    },
    "nb": {
        "module": negative_binomial_module,
        "family": sg.NegativeBinomialLS,
        "response": "count",
    },
}


def _fit(case: str) -> tuple[sg.SuperLSS, pd.DataFrame]:
    spec = _CASES[case]
    frame, columns = _data()
    family = spec["family"]()
    first = family.parameters[0].name
    predictors = [getattr(family, first)(s("x", kind="cr", k=5))]
    predictors += [getattr(family, parameter.name)() for parameter in family.parameters[1:]]
    model = sg.SuperLSS(family, *predictors, weight_semantics=spec.get("semantics", "prior"))
    weights = columns[spec["weights"]] if "weights" in spec else None
    model.fit(frame, columns[spec["response"]], weights, lambdas={f"{first}:x#wiggle": 1.0})
    return model, frame


@pytest.fixture(scope="module", params=sorted(_CASES))
def saved_case(request):
    model, frame = _fit(request.param)
    return request.param, model, frame, model.to_bytes()


def _live_carrier(model: sg.SuperLSS) -> np.ndarray:
    fitted = model._require_fitted()
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    return np.asarray(plan.parameter_independent_carrier)


def test_a_model_loads_where_log_rounds_every_row_differently(saved_case, monkeypatch) -> None:
    case, model, frame, saved = saved_case
    expected = _live_carrier(model)
    with monkeypatch.context() as patch:
        _another_platform(patch, case)
        # The perturbation reaches the carrier the load recomputes.
        assert np.any(_live_carrier(model) != expected)
        loaded = sg.SuperLSS.from_bytes(saved)
        refit = sg.SuperLSS.from_bytes(saved).fit(
            frame, *_refit_arguments(case), lambdas=loaded.smoothing_parameters_
        )
    assert dict(loaded.coef_) == dict(model.coef_)
    np.testing.assert_array_equal(
        loaded.predict_link(frame).to_numpy(), model.predict_link(frame).to_numpy()
    )
    assert refit.result_.converged


def _refit_arguments(case: str):
    spec = _CASES[case]
    _, columns = _data()
    weights = columns[spec["weights"]] if "weights" in spec else None
    return columns[spec["response"]], weights


def test_a_carrier_formula_change_is_refused(saved_case, monkeypatch) -> None:
    """A carrier row moved far beyond rounding is a different likelihood, not another libm."""
    case, _, _, saved = saved_case
    with monkeypatch.context() as patch:
        _another_platform(patch, case, factor=1.0 + 1.0e-6)
        with pytest.raises(DistributionalSerializationError, match="does not identify"):
            sg.SuperLSS.from_bytes(saved)


def test_a_saved_carrier_sum_outside_the_bound_is_refused(saved_case) -> None:
    from superglm.distributional.likelihood_identity import rebound_plan_mismatch

    case, model, _, _ = saved_case
    fitted = model._require_fitted()
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    identifier = fitted.fit_state.family_likelihood_plan_identifier
    stored = fitted.result.parameter_independent_carrier
    assert rebound_plan_mismatch(plan, identifier, stored) is None
    magnitude = math.fsum(np.abs(plan.parameter_independent_carrier)) + 1.0
    assert rebound_plan_mismatch(plan, identifier, stored + 1.0e-9 * magnitude) == "carrier"


def test_the_carrier_sum_check_uses_the_bound_on_both_sides(saved_case) -> None:
    """Within the bound loads, beyond it refuses, whichever side the saved sum lies."""
    from superglm.distributional.likelihood_identity import (
        carrier_sum_bound,
        rebound_plan_mismatch,
    )

    _, model, _, _ = saved_case
    fitted = model._require_fitted()
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    identifier = fitted.fit_state.family_likelihood_plan_identifier
    rounding = plan.carrier_rounding()
    live = float(np.sum(rounding.carrier))
    bound = carrier_sum_bound(rounding)
    # The bound's first-order value, (2 kappa + 8) u sum a + (n - 1) u 2 sum |c|;
    # gamma_k and the upward steps add only O((n + k) u) relative to it.
    u = 2.0**-53
    n = len(rounding.carrier)
    rows = 2 * rounding.error_units + 8
    first_order = rows * u * math.fsum(rounding.scale) + (n - 1) * u * 2.0 * math.fsum(
        np.abs(rounding.carrier)
    )
    assert first_order <= bound <= first_order * (1.0 + 4.0 * (n + rows) * u)
    for side in (-1.0, 1.0):
        assert rebound_plan_mismatch(plan, identifier, live + side * 0.5 * bound) is None
        assert rebound_plan_mismatch(plan, identifier, live + side * 2.0 * bound) == "carrier"


def test_a_legacy_identifier_is_accepted_only_for_its_own_family() -> None:
    """``/v1`` identifiers hashed the carrier: never recomputed, matched by family tag."""
    from superglm.distributional.likelihood_identity import rebound_plan_mismatch

    model, _ = _fit("gamma_prior")
    fitted = model._require_fitted()
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    stored = fitted.result.parameter_independent_carrier
    digest = "0123456789abcdef" * 4
    assert rebound_plan_mismatch(plan, f"GammaLS/v1:{digest}", stored) is None
    assert rebound_plan_mismatch(plan, f"LogNormalLS/v1:{digest}", stored) == "identifier"
    assert rebound_plan_mismatch(plan, f"GammaLS/v2:{digest}", stored) == "identifier"
    assert rebound_plan_mismatch(plan, f"GammaLS/v1:{digest[:-1]}", stored) == "identifier"
    assert rebound_plan_mismatch(plan, f"GammaLS/v1:{digest}", stored + 1.0) == "carrier"


@pytest.fixture(scope="module")
def stationary_gamma():
    rng = np.random.default_rng(7351)
    x = rng.uniform(-1.0, 1.0, 320)
    frame = pd.DataFrame({"x": x})
    response = rng.gamma(4.0, np.exp(0.3 + 0.6 * np.sin(2.4 * x)) / 4.0)
    family = sg.GammaLS()
    model = sg.SuperLSS(
        family, family.mean(s("x", kind="cr", k=7)), family.scale(s("x", kind="cr", k=5))
    )
    model.fit_reml(frame, response, outer="efs+newton")
    return model._require_fitted()


def test_corrected_covariance_replays_where_log_rounds_differently(
    stationary_gamma, monkeypatch
) -> None:
    fitted = stationary_gamma
    assert fitted.smoothing.convergence_reason == "stationary"
    assert fitted.smoothing.smoothing_hessian is None  # the correction replays the plan
    with monkeypatch.context() as patch:
        _another_platform(patch, "gamma_prior")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            corrected = posterior_covariance(fitted, kind="corrected")
    unavailable = [
        str(item.message)
        for item in record
        if issubclass(item.category, ConvergenceWarning) and "unavailable" in str(item.message)
    ]
    assert unavailable == []
    assert not np.array_equal(corrected, posterior_covariance(fitted, kind="fixed"))


def test_corrected_covariance_refuses_a_changed_carrier(stationary_gamma, monkeypatch) -> None:
    """The replay checks the carrier as the load does: a formula change is not rounding."""
    with monkeypatch.context() as patch:
        _another_platform(patch, "gamma_prior", factor=1.0 + 1.0e-6)
        with pytest.warns(
            ConvergenceWarning, match="corrected covariance is unavailable"
        ) as record:
            posterior_covariance(stationary_gamma, kind="corrected")
    assert any("likelihood plan" in str(item.message) for item in record)


@pytest.fixture(scope="module")
def face_model():
    """A ``fit_reml`` that revalidates an exact face whose endpoint moves by roundoff."""
    n = 2000
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {"x": rng.uniform(0, 10, n), "z": rng.uniform(-2, 2, n), "w": rng.uniform(0, 5, n)}
    )
    y = 1.0 + 0.3 * frame["x"].to_numpy() + 0.2 * frame["z"].to_numpy() + rng.normal(0, 0.5, n)
    family = sg.GaussianLS()
    model = sg.SuperLSS(
        family,
        family.location(s("x", kind="cr", k=14), s("z", kind="cr", k=8), s("w", kind="cr", k=7)),
        family.scale(s("x", kind="cr", k=6)),
    )
    model.fit_reml(frame, y)
    assert model.exact_face_components_
    return model, frame, model.to_bytes()


def test_a_face_model_loads_where_the_spectral_norm_rounds_differently(
    face_model, monkeypatch
) -> None:
    """Another OpenBLAS kernel moves the face's LAPACK norm by a few ulps."""
    model, frame, saved = face_model
    real_norm = penalty_face_module._spectral_norm

    def other_kernel(values):
        norm = real_norm(values)
        for _ in range(4):
            norm = np.nextafter(norm, np.inf)
        return float(norm)

    with monkeypatch.context() as patch:
        patch.setattr(penalty_face_module, "_spectral_norm", other_kernel)
        loaded = sg.SuperLSS.from_bytes(saved)
    np.testing.assert_array_equal(
        loaded.predict_link(frame).to_numpy(), model.predict_link(frame).to_numpy()
    )


def _stationary_caps(patch: pytest.MonkeyPatch) -> None:
    """Treat every finite cap as numerically stationary.

    A revalidation whose cap is stationary authenticates its endpoint as a
    replay of the face projection of its source. This fit's cap misses that
    by a tolerance, so the load grants it to reach the replay check.
    """
    patch.setattr(result_smoothing, "_assessment_is_numerically_stationary", lambda *_: True)


def test_a_revalidated_endpoint_loads_where_the_projection_rounds_differently(
    face_model, monkeypatch
) -> None:
    model, frame, saved = face_model
    smoothing = model._require_fitted().fit_state.smoothing
    moved = [
        item
        for item in smoothing.history
        if item.revalidated_face_components
        and item.accepted_fit_index is not None
        and not np.array_equal(
            smoothing.coefficient_fits[item.source_fit_index].coefficients,
            smoothing.coefficient_fits[item.accepted_fit_index].coefficients,
        )
    ]
    assert moved  # the load replays a projection that moved its source
    real_project = PenaltyFace.project
    with monkeypatch.context() as patch:
        _stationary_caps(patch)
        patch.setattr(
            PenaltyFace, "project", lambda face, values: _nudged(real_project(face, values))
        )
        loaded = sg.SuperLSS.from_bytes(saved)
    np.testing.assert_array_equal(
        loaded.predict_link(frame).to_numpy(), model.predict_link(frame).to_numpy()
    )


def test_a_revalidated_endpoint_must_still_be_the_projection(face_model, monkeypatch) -> None:
    _, _, saved = face_model
    real_project = PenaltyFace.project
    with monkeypatch.context() as patch:
        _stationary_caps(patch)
        patch.setattr(
            PenaltyFace,
            "project",
            lambda face, values: _nudged(real_project(face, values), factor=1.0 + 1.0e-6),
        )
        with pytest.raises(DistributionalSerializationError, match="canonical endpoint state"):
            sg.SuperLSS.from_bytes(saved)


def _rotated_face(face: PenaltyFace, angle: float) -> PenaltyFace:
    basis = np.array(face.null_basis, copy=True)
    c, s_ = math.cos(angle), math.sin(angle)
    first, second = basis[:, 0].copy(), basis[:, 1].copy()
    basis[:, 0] = c * first - s_ * second
    basis[:, 1] = s_ * first + c * second
    return replace(face, null_basis=basis)


def test_two_bases_of_one_face_match_by_their_projectors(face_model) -> None:
    """Rotated bases give the same projector up to rounding, which no BLAS fixes bit for bit."""
    model, _, _ = face_model
    face = model._require_fitted().result.coefficient_face
    rotated = _rotated_face(face, 0.3)
    assert not np.array_equal(rotated.projector, face.projector)
    assert result_solver._assessment_face_geometry_matches(face, rotated)
    tilted = np.array(face.null_basis, copy=True)
    tilted[:, 0] += 1.0e-9 * face.constraint_basis[:, 0]
    other = types.SimpleNamespace(null_basis=tilted, projector=tilted @ tilted.T)
    assert not result_solver._face_projectors_agree(face, other)


FIXTURE_NAMES = sorted(path.name.removesuffix(".pkl.xz") for path in FIXTURES.glob("*.pkl.xz"))


def test_the_fixture_set_covers_both_carriers_and_both_kernels() -> None:
    assert FIXTURE_NAMES == ["face_haswell", "face_native", "gamma_avx512", "gamma_libm"]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_model_saved_on_another_platform_loads_and_predicts(name: str) -> None:
    """Coefficients come back bit for bit, linear predictors within two evaluations' rounding."""
    with lzma.open(FIXTURES / f"{name}.pkl.xz", "rb") as handle:
        record = pickle.load(handle)
    model = sg.SuperLSS.from_bytes(record["serialized"])
    frame = record["frame"]
    restored = dict(model.coef_)
    assert list(restored) == list(record["coefficients"])
    assert all(
        math.copysign(1.0, restored[key]) == math.copysign(1.0, value) and restored[key] == value
        for key, value in record["coefficients"].items()
    )
    eta = model.predict_link(frame).to_numpy(dtype=np.float64)
    excess = np.abs(eta - record["eta"]) / _lss_eta_rounding(model, frame)
    assert np.all(excess <= 1.0), float(np.max(excess))
