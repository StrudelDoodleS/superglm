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

import json
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
from superglm.distributional import posterior as posterior_module
from superglm.distributional import serialization as serialization_module
from superglm.distributional.families import gamma as gamma_module
from superglm.distributional.families import gaussian as gaussian_module
from superglm.distributional.families import generalized_gamma as generalized_gamma_module
from superglm.distributional.families import log_normal as log_normal_module
from superglm.distributional.families import negative_binomial as negative_binomial_module
from superglm.distributional.families import two_piece as two_piece_module
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.results import solver as result_solver
from superglm.distributional.results.smoothing import DistributionalEFSResult
from superglm.distributional.serialization import (
    DistributionalSerializationError,
    serialize_distributional_model,
)
from superglm.distributional.smoothing import penalty_face as penalty_face_module
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.weights import UnsupportedLikelihoodContractError

from . import test_distributional_endpoint_laml as endpoint_laml_tests
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
    return _rebound_carrier(model._require_fitted())


def _rebound_carrier(fitted) -> np.ndarray:
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


def _legacy_tag(tag: str | None):
    """``GammaLikelihoodPlan.carrier_rounding`` declaring ``tag`` as its legacy tag."""
    real = gamma_module.GammaLikelihoodPlan.carrier_rounding

    def carrier_rounding(self):
        return real(self)._replace(legacy_identifier_tag=tag)

    return carrier_rounding


def test_a_legacy_tag_that_names_the_plans_own_identifiers_is_refused(monkeypatch) -> None:
    """Such a tag would accept any identifier carrying it, on its tag and carrier sum alone."""
    from superglm.distributional.likelihood_identity import rebound_plan_mismatch

    model, _ = _fit("gamma_prior")
    fitted = model._require_fitted()
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    identifier = fitted.fit_state.family_likelihood_plan_identifier
    stored = fitted.result.parameter_independent_carrier
    digest = "0123456789abcdef" * 4
    assert identifier.startswith("GammaLS/v2:") and identifier != f"GammaLS/v2:{digest}"
    with monkeypatch.context() as patch:
        patch.setattr(
            gamma_module.GammaLikelihoodPlan, "carrier_rounding", _legacy_tag("GammaLS/v2")
        )
        assert rebound_plan_mismatch(plan, f"GammaLS/v2:{digest}", stored) == "carrier certificate"
        with pytest.raises(UnsupportedLikelihoodContractError, match="legacy_identifier_tag"):
            _fit("gamma_prior")
    with monkeypatch.context() as patch:
        # A family that never wrote legacy identifiers declares none.
        patch.setattr(gamma_module.GammaLikelihoodPlan, "carrier_rounding", _legacy_tag(None))
        assert rebound_plan_mismatch(plan, identifier, stored) is None
        assert rebound_plan_mismatch(plan, f"GammaLS/v1:{digest}", stored) == "identifier"
        assert dict(_fit("gamma_prior")[0].coef_) == dict(model.coef_)


@pytest.fixture(scope="module")
def gamma_reml():
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


@pytest.fixture(scope="module")
def replayed_gamma(gamma_reml):
    """``gamma_reml`` with no published smoothing Hessian, so the correction replays the plan.

    Whether an endgame publishes its Hessian depends on the path the outer
    loop takes. Without one, every stop replays: a stationary stop against its
    terminal gradient, any other at its own terminal point.
    """
    smoothing = replace(
        gamma_reml.smoothing, smoothing_hessian=None, smoothing_hessian_certificate=None
    )
    return DenseDistributionalModel(
        gamma_reml.family, replace(gamma_reml.fit_state, smoothing=smoothing)
    )


def _counted_replays(patch: pytest.MonkeyPatch) -> list:
    """Record each corrected-covariance replay of the likelihood plan."""
    calls = []
    real = posterior_module._replayed_smoothing_hessian

    def replay(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    patch.setattr(posterior_module, "_replayed_smoothing_hessian", replay)
    return calls


def test_corrected_covariance_replays_where_log_rounds_differently(
    replayed_gamma, monkeypatch
) -> None:
    fitted = replayed_gamma
    expected = _rebound_carrier(fitted)
    with monkeypatch.context() as patch:
        replays = _counted_replays(patch)
        _another_platform(patch, "gamma_prior")
        # The perturbation reaches the carrier the replay rebinds.
        assert np.any(_rebound_carrier(fitted) != expected)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            corrected = posterior_covariance(fitted, kind="corrected")
    assert len(replays) == 1
    unavailable = [
        str(item.message)
        for item in record
        if issubclass(item.category, ConvergenceWarning) and "unavailable" in str(item.message)
    ]
    assert unavailable == []
    assert not np.array_equal(corrected, posterior_covariance(fitted, kind="fixed"))


def test_corrected_covariance_refuses_a_changed_carrier(replayed_gamma, monkeypatch) -> None:
    """The replay checks the carrier as the load does: a formula change is not rounding."""
    with monkeypatch.context() as patch:
        replays = _counted_replays(patch)
        _another_platform(patch, "gamma_prior", factor=1.0 + 1.0e-6)
        with pytest.warns(
            ConvergenceWarning, match="corrected covariance is unavailable"
        ) as record:
            posterior_covariance(replayed_gamma, kind="corrected")
    assert len(replays) == 1
    assert any("likelihood plan" in str(item.message) for item in record)


@pytest.fixture(scope="module")
def face_model():
    """A model on an exact penalty face, saved again by this build.

    It is the committed ``face_native`` artifact rather than a live fit, so the
    face is there on every platform; whether ``fit_reml`` reaches one depends
    on the path it takes.
    """
    with lzma.open(FIXTURES / "face_native.pkl.xz", "rb") as handle:
        record = pickle.load(handle)
    model = sg.SuperLSS.from_bytes(record["serialized"])
    assert model.exact_face_components_
    return model, record["frame"], model.to_bytes()


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


@pytest.fixture(scope="module")
def revalidated_face():
    """An EFS result that revalidates an exact face beside its cap, and that event."""
    smoothing = endpoint_laml_tests._gamma_isolated_cap_face_problem()[-1]
    event = next(
        item
        for item in smoothing.history
        if item.revalidated_face_components and item.accepted_fit_index is not None
    )
    return smoothing, event


def _forged_projection_replay(
    revalidated_face, patch: pytest.MonkeyPatch, *, replay_gap: float
) -> tuple[object, tuple]:
    """The revalidation's fits with a forged endpoint, and a forged replay of its projection.

    The stored endpoint is the source moved by an eighth of the projection
    bound along one coordinate, reached without a step, beside a cap whose
    score is zero, so the load authenticates it as a replay of the source's
    face projection. The load's own replay is the stored endpoint moved by
    ``replay_gap`` times the bound along that coordinate. Every margin is a
    fixed fraction of the bound, far above one rounding of the coordinate, so
    the decision is the same on every BLAS.
    """
    smoothing, event = revalidated_face
    fits = list(smoothing.coefficient_fits)
    cap_index, endpoint_index = event.coefficient_fit_indices[:2]
    source = np.asarray(fits[event.source_fit_index].coefficients)
    endpoint_fit = fits[endpoint_index]
    face = endpoint_fit.coefficient_face
    bound = result_solver._endpoint_revalidation_projection_bound(face, source)
    assert bound is not None and bound > 0.0
    row = int(np.argmax(np.abs(source)))
    assert 2.0 * float(np.spacing(np.abs(source[row]))) < 1.0e-2 * bound
    endpoint = np.array(source, copy=True)
    endpoint[row] += 0.125 * bound
    replay = np.array(endpoint, copy=True)
    replay[row] += replay_gap * bound
    fits[cap_index] = replace(
        fits[cap_index], terminal_score=np.zeros_like(fits[cap_index].terminal_score)
    )
    fits[endpoint_index] = replace(endpoint_fit, coefficients=endpoint, iterations=0)
    real_project = PenaltyFace.project

    def project(face_, values):
        values = np.asarray(values)
        if values.shape == source.shape and np.array_equal(values, source):
            return replay.copy()
        return real_project(face_, values)

    patch.setattr(PenaltyFace, "project", project)
    return smoothing, tuple(fits)


@pytest.mark.parametrize("replay_gap", [0.0, 1.5])
def test_a_revalidated_endpoint_loads_where_the_projection_rounds_differently(
    revalidated_face, monkeypatch, replay_gap: float
) -> None:
    """A stored replay within twice the bound of the load's own replay is accepted.

    Each replay lies within the bound of the exact projection on its own
    machine, so two machines' replays differ by up to twice the bound.
    """
    with monkeypatch.context() as patch:
        smoothing, fits = _forged_projection_replay(revalidated_face, patch, replay_gap=replay_gap)
        replace(smoothing, coefficient_fits=fits)


@pytest.mark.parametrize("replay_gap", [2.5, 1.0e6])
def test_a_revalidated_endpoint_must_still_be_the_projection(
    revalidated_face, monkeypatch, replay_gap: float
) -> None:
    with monkeypatch.context() as patch:
        smoothing, fits = _forged_projection_replay(revalidated_face, patch, replay_gap=replay_gap)
        with pytest.raises(ValueError, match="canonical endpoint state"):
            replace(smoothing, coefficient_fits=fits)


def _quarter_turned_face(face: PenaltyFace) -> PenaltyFace:
    """``face`` with its first two basis columns turned a quarter: an exact change of basis."""
    basis = np.array(face.null_basis, copy=True)
    basis[:, [0, 1]] = basis[:, [1, 0]]
    basis[:, 0] = -basis[:, 0]
    return replace(face, null_basis=basis)


def _load_projector_bound(left: PenaltyFace, right: PenaltyFace) -> np.ndarray:
    """``2 gamma_{2r+4} (|N_L||N_L|^T + |N_R||N_R|^T)``, the gap a load accepts."""
    count = 2 * left.null_basis.shape[1] + 4
    u = 2.0**-53
    magnitude = np.abs(left.null_basis) @ np.abs(left.null_basis).T + (
        np.abs(right.null_basis) @ np.abs(right.null_basis).T
    )
    return 2.0 * (count * u / (1.0 - count * u)) * magnitude


def _forged_projectors(
    patch: pytest.MonkeyPatch, left: PenaltyFace, right: PenaltyFace, gap: np.ndarray
) -> None:
    """Replay ``left``'s product as zero and ``right``'s as ``gap``, so the gap is exact."""
    real = PenaltyFace.projector
    forged = {id(left): np.zeros_like(gap), id(right): gap}
    patch.setattr(
        PenaltyFace, "projector", property(lambda face: forged.get(id(face), real.fget(face)))
    )


def test_two_bases_of_one_face_match_by_their_projectors(face_model, monkeypatch) -> None:
    """Two bases of one face match when their products agree within the load's bound.

    Each machine rounds the two products differently, so they never agree bit
    for bit. The replays are forged at fixed fractions of the bound, which
    decides the same way on every BLAS.
    """
    model, _, _ = face_model
    face = model._require_fitted().result.coefficient_face
    assert face is not None and face.null_basis.shape[1] >= 2
    turned = _quarter_turned_face(face)
    assert not np.array_equal(turned.null_basis, face.null_basis)
    bound = _load_projector_bound(face, turned)
    for fraction, matches in ((0.5, True), (2.0, False)):
        with monkeypatch.context() as patch:
            _forged_projectors(patch, face, turned, fraction * bound)
            assert result_solver._assessment_face_geometry_matches(face, turned) is matches
    # An exact change of basis matches as the products round.
    assert result_solver._assessment_face_geometry_matches(face, turned)
    tilted = np.array(face.null_basis, copy=True)
    tilted[:, 0] += 1.0e-9 * face.constraint_basis[:, 0]
    other = types.SimpleNamespace(null_basis=tilted, projector=tilted @ tilted.T)
    assert not result_solver._face_projectors_agree(face, other)


def test_a_fit_accepts_two_bases_of_one_face_within_half_the_load_bound(
    face_model, monkeypatch
) -> None:
    """A load replays both products, so a fit leaves it that rounding as margin.

    Otherwise a gap a fit accepted near the bound could exceed it after
    another machine's replay, and the saved model would be refused there.
    """
    model, _, _ = face_model
    face = model._require_fitted().result.coefficient_face
    turned = _quarter_turned_face(face)
    bound = _load_projector_bound(face, turned)
    for fraction, at_fit, at_load in ((0.25, True, True), (0.75, False, True)):
        with monkeypatch.context() as patch:
            _forged_projectors(patch, face, turned, fraction * bound)
            with result_solver.fit_time_face_geometry():
                assert result_solver._face_projectors_agree(face, turned) is at_fit
            assert result_solver._face_projectors_agree(face, turned) is at_load
    with result_solver.fit_time_face_geometry():
        assert result_solver._face_projectors_agree(face, turned)


def test_a_fit_validates_its_smoothing_result_with_the_fit_acceptance(
    face_model, monkeypatch
) -> None:
    """The fit builds its smoothing result under the fit's acceptance, and a load under its own."""
    _, _, saved = face_model
    seen: list[bool] = []
    real = DistributionalEFSResult.__post_init__

    def post_init(self) -> None:
        seen.append(result_solver._FIT_TIME_FACE_GEOMETRY.get())
        real(self)

    with monkeypatch.context() as patch:
        patch.setattr(DistributionalEFSResult, "__post_init__", post_init)
        family = sg.GammaLS()
        refit = sg.SuperLSS(family, family.mean(s("x", kind="cr", k=7)), family.scale())
        frame, columns = _data()
        refit.fit_reml(frame, columns["positive"])
        at_fit, seen[:] = list(seen), []
        sg.SuperLSS.from_bytes(saved)
        at_load = list(seen)
    assert at_fit and all(at_fit)
    assert at_load and not any(at_load)


FIXTURE_NAMES = sorted(path.name.removesuffix(".pkl.xz") for path in FIXTURES.glob("*.pkl.xz"))


def test_the_fixture_set_covers_both_carriers_and_both_kernels() -> None:
    assert FIXTURE_NAMES == ["face_haswell", "face_native", "gamma_avx512", "gamma_libm"]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_model_saved_on_another_platform_loads_and_predicts(name: str) -> None:
    """Coefficients come back bit for bit, linear predictors within two evaluations' rounding."""
    with lzma.open(FIXTURES / f"{name}.pkl.xz", "rb") as handle:
        record = pickle.load(handle)
    # Written by builds that wrote schema 11, which this build reads as a
    # previous major; its face manifests still record the derived tolerances.
    assert json.loads(record["serialized"])["schema_version"] == "11.0.0"
    if name.startswith("face_"):
        assert all(
            set(_DERIVED_FACE_KEYS) <= set(face)
            for face in _coefficient_faces(json.loads(record["serialized"])["manifest"])
        )
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


_DERIVED_FACE_KEYS = ("rank_resolution", "null_residual_bound", "orthogonality_bound")


def _coefficient_faces(value) -> list[dict]:
    """Every ``coefficient_face`` mapping in a manifest."""
    if isinstance(value, dict):
        found = (
            [value["coefficient_face"]] if isinstance(value.get("coefficient_face"), dict) else []
        )
        return found + [face for item in value.values() for face in _coefficient_faces(item)]
    if isinstance(value, list):
        return [face for item in value for face in _coefficient_faces(item)]
    return []


def _schema_eleven_reader(patch: pytest.MonkeyPatch) -> None:
    """The reader of a build that wrote schema 11, which also read 8, 9 and 10."""
    patch.setattr(serialization_module, "SCHEMA_VERSION", "11.0.0")
    patch.setattr(serialization_module, "READABLE_PREVIOUS_MAJORS", frozenset({8, 9, 10}))

    def unexpected_unpickle(*args, **kwargs):
        raise AssertionError("the schema-11 reader must refuse before unpickling")

    patch.setattr(serialization_module.pickle, "loads", unexpected_unpickle)


def test_a_reloaded_schema_eleven_face_model_is_saved_as_schema_twelve(monkeypatch) -> None:
    """A face model re-saved by this build is refused by a schema-11 reader at its version gate.

    Schema 12 drops the derived face tolerances from the manifest. Written as
    11.0.0, the re-saved model passed a schema-11 reader's version gate and
    then failed its manifest comparison, the error reserved for tampering.
    """
    with lzma.open(FIXTURES / "face_native.pkl.xz", "rb") as handle:
        record = pickle.load(handle)
    resaved = sg.SuperLSS.from_bytes(record["serialized"]).to_bytes()
    artifact = json.loads(resaved)
    assert artifact["schema_version"] == "12.0.0"
    faces = _coefficient_faces(artifact["manifest"])
    assert faces and not any(set(_DERIVED_FACE_KEYS) & set(face) for face in faces)
    assert dict(sg.SuperLSS.from_bytes(resaved).coef_) == record["coefficients"]

    with monkeypatch.context() as patch:
        _schema_eleven_reader(patch)
        with pytest.raises(DistributionalSerializationError, match="major versions differ"):
            sg.SuperLSS.from_bytes(resaved)


def test_a_new_fit_is_saved_as_schema_twelve_with_v2_identifiers(gamma_reml) -> None:
    artifact = json.loads(serialize_distributional_model(gamma_reml))
    assert artifact["schema_version"] == "12.0.0"
    identifiers = set(_manifest_values(artifact["manifest"], "family_likelihood_plan_identifier"))
    assert identifiers and all(item.startswith("GammaLS/v2:") for item in identifiers)


def test_a_schema_twelve_manifest_that_records_a_derived_face_value_is_refused(
    face_model,
) -> None:
    """Schema 12 never records the derived face tolerances, so one there is not ignored."""
    _, _, saved = face_model
    artifact = json.loads(saved)
    assert artifact["schema_version"] == "12.0.0"
    _coefficient_faces(artifact["manifest"])[0]["rank_resolution"] = 1.0e-12
    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        sg.SuperLSS.from_bytes(json.dumps(artifact).encode())


def _manifest_values(value, key: str) -> list:
    if isinstance(value, dict):
        own = [value[key]] if key in value else []
        return own + [found for item in value.values() for found in _manifest_values(item, key)]
    if isinstance(value, list):
        return [found for item in value for found in _manifest_values(item, key)]
    return []
