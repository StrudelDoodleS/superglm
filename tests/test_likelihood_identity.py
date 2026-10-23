"""The likelihood identity a fit stores and a posterior replay checks.

``rebound_plan_mismatch`` and ``carrier_sum_bound`` decide whether a likelihood
plan rebound from a model's retained rows is the plan the fit used. A fit reads
the legacy identifier tag, and the corrected covariance replays the plan under
the same check. The tests simulate another machine rounding the carrier's
transcendentals differently.
"""

from __future__ import annotations

import math
import types
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm as sg
from superglm import s
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import posterior as posterior_module
from superglm.distributional.families import gamma as gamma_module
from superglm.distributional.families import gaussian as gaussian_module
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.results import solver as result_solver
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.weights import UnsupportedLikelihoodContractError

from . import test_distributional_endpoint_laml as endpoint_laml_tests

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


def _another_gamma_platform(monkeypatch: pytest.MonkeyPatch, factor: float | None = None) -> None:
    """Patch the Gamma carrier's ``log`` as another platform rounds it."""
    monkeypatch.setattr(gamma_module, "np", _OtherNumpy(factor))


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
    "gamma_prior": {"family": sg.GammaLS, "response": "positive"},
    "gamma_frequency": {
        "family": sg.GammaLS,
        "response": "positive",
        "weights": "replicates",
        "semantics": "frequency",
    },
    "log_normal": {"family": sg.LogNormalLS, "response": "positive"},
    "generalized_gamma": {
        "family": sg.GeneralizedGammaLSS,
        "response": "positive",
    },
    "two_piece_log_normal": {
        "family": sg.TwoPieceLogNormalLSS,
        "response": "positive",
    },
    "two_piece_normal": {
        "family": sg.TwoPieceNormalLSS,
        "response": "real",
    },
    "gaussian_prior_weights": {
        "family": sg.GaussianLS,
        "response": "real",
        "weights": "weight",
    },
    "nb": {
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
def fitted_case(request):
    model, _ = _fit(request.param)
    return request.param, model


def _live_carrier(model: sg.SuperLSS) -> np.ndarray:
    return _rebound_carrier(model._require_fitted())


def _rebound_carrier(fitted) -> np.ndarray:
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    return np.asarray(plan.parameter_independent_carrier)


def test_a_stored_carrier_sum_outside_the_bound_is_refused(fitted_case) -> None:
    from superglm.distributional.likelihood_identity import rebound_plan_mismatch

    _, model = fitted_case
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


def test_the_carrier_sum_check_uses_the_bound_on_both_sides(fitted_case) -> None:
    """Within the bound matches, beyond it does not, whichever side the stored sum lies."""
    from superglm.distributional.likelihood_identity import (
        carrier_sum_bound,
        rebound_plan_mismatch,
    )

    _, model = fitted_case
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
        patch.setattr(gamma_module.GammaLikelihoodPlan, "legacy_identifier_tag", "GammaLS/v2")
        assert rebound_plan_mismatch(plan, f"GammaLS/v2:{digest}", stored) == "carrier certificate"
        with pytest.raises(UnsupportedLikelihoodContractError, match="legacy_identifier_tag"):
            _fit("gamma_prior")
    with monkeypatch.context() as patch:
        # A family that never wrote legacy identifiers declares none.
        patch.setattr(gamma_module.GammaLikelihoodPlan, "legacy_identifier_tag", None)
        assert rebound_plan_mismatch(plan, identifier, stored) is None
        assert rebound_plan_mismatch(plan, f"GammaLS/v1:{digest}", stored) == "identifier"
        assert dict(_fit("gamma_prior")[0].coef_) == dict(model.coef_)


def test_a_chunked_fit_checks_the_legacy_tag_without_building_the_carrier(monkeypatch) -> None:
    """A fit reads the legacy tag as a string; only a replay of the plan builds the carrier.

    A chunked ``GaussianLS`` plan keeps no carrier, and binding it evaluates
    each carrier block once for the weight digest. The fit evaluates no other
    block and never calls ``carrier_rounding()``, which concatenates all of them.
    """
    from superglm.distributional import model as model_module
    from superglm.distributional.predictor import Predictor
    from superglm.features import Numeric

    from .bound_predictor_fixtures import model_from_templates

    counts = {"rounding": 0, "blocks": 0, "binding blocks": 0}
    roots = []
    plan_class = gaussian_module.GaussianLikelihoodPlan
    real_rounding, real_block = plan_class.carrier_rounding, gaussian_module._carrier_block
    real_bind, real_fit = sg.GaussianLS.bind_chunked_likelihood, model_module.fit_dense_fixed_lambda

    def counted_rounding(self):
        counts["rounding"] += 1
        return real_rounding(self)

    def counted_block(*args):
        counts["blocks"] += 1
        return real_block(*args)

    def counted_bind(self, *args, **kwargs):
        before = counts["blocks"]
        plan = real_bind(self, *args, **kwargs)
        counts["binding blocks"] += counts["blocks"] - before
        return plan

    def capture(family, layout, response, plan, *args, **kwargs):
        roots.append(plan)
        return real_fit(family, layout, response, plan, *args, **kwargs)

    monkeypatch.setattr(plan_class, "carrier_rounding", counted_rounding)
    monkeypatch.setattr(gaussian_module, "_carrier_block", counted_block)
    monkeypatch.setattr(sg.GaussianLS, "bind_chunked_likelihood", counted_bind)
    monkeypatch.setattr(model_module, "fit_dense_fixed_lambda", capture)
    n = 2 * gaussian_module._CARRIER_BLOCK_ROWS + 1
    rng = np.random.default_rng(314)
    x = np.linspace(-1.0, 1.0, n)
    frame, response, weights = pd.DataFrame({"x": x}), 0.5 * x + rng.normal(size=n), x + 2.0

    def fit():
        model_from_templates(
            family=sg.GaussianLS(),
            predictors=(Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
            discrete=True,
        ).fit(frame, response, weights)

    fit()
    assert len(roots) == 1 and roots[0].parameter_independent_carrier is None
    assert counts == {"rounding": 0, "blocks": 3, "binding blocks": 3}
    # The string check still refuses a legacy tag that the live identifiers carry.
    live_tag = roots[0].plan_identifier.split(":")[0]
    monkeypatch.setattr(plan_class, "legacy_identifier_tag", live_tag)
    with pytest.raises(UnsupportedLikelihoodContractError, match="legacy_identifier_tag"):
        fit()
    assert counts["rounding"] == 0


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
        _another_gamma_platform(patch)
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
    """The replay checks the carrier as a fit does: a formula change is not rounding."""
    with monkeypatch.context() as patch:
        replays = _counted_replays(patch)
        _another_gamma_platform(patch, factor=1.0 + 1.0e-6)
        with pytest.warns(
            ConvergenceWarning, match="corrected covariance is unavailable"
        ) as record:
            posterior_covariance(replayed_gamma, kind="corrected")
    assert len(replays) == 1
    assert any("likelihood plan" in str(item.message) for item in record)


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


@pytest.fixture(scope="module")
def exact_face(revalidated_face) -> PenaltyFace:
    """The exact penalty face the revalidated endpoint sits on."""
    smoothing, event = revalidated_face
    endpoint_index = event.coefficient_fit_indices[1]
    return smoothing.coefficient_fits[endpoint_index].coefficient_face


def _forged_projection_replay(
    revalidated_face, patch: pytest.MonkeyPatch, *, replay_gap: float
) -> tuple[object, tuple]:
    """The revalidation's fits with a forged endpoint, and a forged replay of its projection.

    The stored endpoint is the source moved by an eighth of the projection
    bound along one coordinate, reached without a step, beside a cap whose
    score is zero, so the result authenticates it as a replay of the source's
    face projection. The result's own replay is the stored endpoint moved by
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
def test_a_revalidated_endpoint_is_accepted_where_the_projection_rounds_differently(
    revalidated_face, monkeypatch, replay_gap: float
) -> None:
    """A stored replay within twice the bound of the result's own replay is accepted.

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


def _replay_projector_bound(left: PenaltyFace, right: PenaltyFace) -> np.ndarray:
    """``2 gamma_{2r+4} (|N_L||N_L|^T + |N_R||N_R|^T)``, the gap a result accepts."""
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


def test_two_bases_of_one_face_match_by_their_projectors(exact_face, monkeypatch) -> None:
    """Two bases of one face match when their products agree within the result's bound.

    Each machine rounds the two products differently, so they never agree bit
    for bit. The replays are forged at fixed fractions of the bound, which
    decides the same way on every BLAS.
    """
    face = exact_face
    assert face is not None and face.null_basis.shape[1] >= 2
    turned = _quarter_turned_face(face)
    assert not np.array_equal(turned.null_basis, face.null_basis)
    bound = _replay_projector_bound(face, turned)
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


def test_a_fit_accepts_two_bases_of_one_face_within_half_the_replay_bound(
    exact_face, monkeypatch
) -> None:
    """A rebuilt result replays both products, so a fit leaves it that rounding as margin.

    Otherwise a gap a fit accepted near the bound could exceed it after
    another machine's replay, and the result would be refused there.
    """
    face = exact_face
    turned = _quarter_turned_face(face)
    bound = _replay_projector_bound(face, turned)
    for fraction, at_fit, at_replay in ((0.25, True, True), (0.75, False, True)):
        with monkeypatch.context() as patch:
            _forged_projectors(patch, face, turned, fraction * bound)
            with result_solver.fit_time_face_geometry():
                assert result_solver._face_projectors_agree(face, turned) is at_fit
            assert result_solver._face_projectors_agree(face, turned) is at_replay
    with result_solver.fit_time_face_geometry():
        assert result_solver._face_projectors_agree(face, turned)


def test_a_fit_validates_its_smoothing_result_with_the_fit_acceptance(monkeypatch) -> None:
    """Every smoothing result a fit builds is validated under the fit's face acceptance."""
    from superglm.distributional.results.smoothing import DistributionalEFSResult

    seen: list[bool] = []
    real = DistributionalEFSResult.__post_init__

    def post_init(self) -> None:
        seen.append(result_solver._FIT_TIME_FACE_GEOMETRY.get())
        real(self)

    monkeypatch.setattr(DistributionalEFSResult, "__post_init__", post_init)
    family = sg.GammaLS()
    model = sg.SuperLSS(family, family.mean(s("x", kind="cr", k=7)), family.scale())
    frame, columns = _data()
    model.fit_reml(frame, columns["positive"])
    assert seen and all(seen)
