"""Gaussian chunk preparation removes the retained root carrier only."""

import math
import pickle
import weakref
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional.families import gaussian
from superglm.distributional.families.gaussian import GaussianLikelihoodPlan, GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.predictor import Predictor
from superglm.distributional.solver import solver
from superglm.distributional.solver._likelihood_cache import build_likelihood_cache
from superglm.distributional.weights import (
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric, Spline


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("n", [17, 16389])
def test_derived_root_curvature_matches_eager_and_independent_gaussian(semantics, n):
    from superglm.distributional.smoothing.endpoint_direction import _curvature_packed

    family, response, eager, root = _plans(n, semantics)
    eta = np.column_stack((0.25 * response, np.linspace(-0.5, 0.5, n)))
    links = tuple(parameter.default_link for parameter in family.parameters)
    expected = _curvature_packed(family, response, eta, links, eager)
    actual = _curvature_packed(family, response, eta, links, root)
    np.testing.assert_array_equal(actual, expected)

    # Independent negative eta-Hessian for sigma = floor + exp(eta_scale).
    shift = np.exp(eta[:, 1])
    sigma = family.scale_floor + shift
    residual = response - eta[:, 0]
    weight = eager.weights.values
    precision = weight if semantics == "prior" else np.ones(n)
    mass = np.ones(n) if semantics == "prior" else weight
    a = mass * shift**2 * 3.0 * precision * residual**2 / sigma**4
    b = mass * shift**2 / sigma**2
    c = mass * shift / sigma
    d = mass * shift * precision * residual**2 / sigma**3
    oracle = np.column_stack(
        (
            mass * precision / sigma**2,
            mass * 2.0 * precision * residual * shift / sigma**3,
            a - b + c - d,
        )
    )
    absolute_terms = np.column_stack((np.abs(oracle[:, 0]), np.abs(oracle[:, 1]), a + b + c + d))
    bound = 64 * np.finfo(float).eps * np.maximum(1.0, absolute_terms)
    assert np.all(np.abs(actual - oracle) <= bound)
    assert root.parameter_independent_carrier is None


def test_derived_curvature_dispatch_owns_bounded_children_and_refreshes(monkeypatch):
    from superglm.distributional.smoothing import endpoint_direction

    family, response, eager, root = _plans(23)
    eta = np.column_stack((response * 0.2, np.zeros(23)))
    links = tuple(parameter.default_link for parameter in family.parameters)
    takes, evaluations, children = [], [], []
    original_take = GaussianLikelihoodPlan.take
    original_evaluate = GaussianLS.evaluate_natural

    def take(plan, indices):
        assert all(reference() is None for reference in children)
        assert plan is root
        takes.append(tuple(indices))
        child = original_take(plan, indices)
        children.append(weakref.ref(child))
        return child

    def evaluate(self, y, theta, plan, **kwargs):
        assert plan is not root
        assert plan.parameter_independent_carrier is not None
        evaluations.append(len(y))
        return original_evaluate(self, y, theta, plan, **kwargs)

    monkeypatch.setattr(endpoint_direction, "_CURVATURE_CHUNK_ROWS", 7, raising=False)
    monkeypatch.setattr(GaussianLikelihoodPlan, "take", take)
    monkeypatch.setattr(GaussianLS, "evaluate_natural", evaluate)
    endpoint_direction._curvature_packed(family, response, eta, links, root)
    endpoint_direction._curvature_packed(family, response, eta, links, root)
    assert evaluations == [7, 7, 7, 2] * 2
    assert takes == [tuple(range(start, min(start + 7, 23))) for start in (0, 7, 14, 21)] * 2
    assert all(reference() is None for reference in children)
    # Eager input keeps its single existing family call, without taking children.
    endpoint_direction._curvature_packed(family, response, eta, links, eager)
    assert evaluations[-1] == 23


@pytest.mark.parametrize("changed", ["response", "eta", "weights", "mode", "family_config"])
def test_derived_curvature_does_not_accept_stale_or_mismatched_root(changed):
    from superglm.distributional.smoothing.endpoint_direction import _curvature_packed

    family, response, _, root = _plans(11)
    eta = np.column_stack((response * 0.2, np.zeros(11)))
    links = tuple(parameter.default_link for parameter in family.parameters)
    _curvature_packed(family, response, eta, links, root)
    if changed == "response":
        response = response[:-1]
    elif changed == "eta":
        eta = eta[:-1]
    elif changed == "weights":
        object.__setattr__(root.weights, "values", root.weights.values[:-1])
    elif changed == "mode":
        object.__setattr__(root, "carrier_preparation", "unknown")
    else:
        object.__setattr__(root, "family_config", ("GaussianLS/v1", family.scale_floor + 0.1))
    with pytest.raises((ValueError, UnsupportedLikelihoodContractError)):
        _curvature_packed(family, response, eta, links, root)


def _plans(n=31, semantics="prior", *, extreme=False):
    values = np.linspace(0.25, 4.0, n) if semantics == "prior" else 1 + np.arange(n) % 4
    if extreme and semantics == "prior":
        values[:3] = [np.nextafter(0.0, 1.0), np.finfo(float).tiny, np.finfo(float).max]
    weights = resolve_likelihood_weights(
        values, n_observations=n, contract=WeightContract(semantics)
    )
    response = np.linspace(-0.5, 0.75, n)
    family = GaussianLS()
    eager = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    root = family.bind_chunked_likelihood(response, weights, COMPLETE_OBSERVATION)
    return family, response, eager, root


@pytest.mark.parametrize("n", [37, 113])
def test_public_chunked_fit_does_not_own_a_full_row_carrier(monkeypatch, n):
    import superglm.distributional.model as model_module

    roots = []
    original = model_module.fit_dense_fixed_lambda

    def capture(family, layout, response, plan, *args, **kwargs):
        roots.append(plan)
        return original(family, layout, response, plan, *args, **kwargs)

    monkeypatch.setattr(model_module, "fit_dense_fixed_lambda", capture)
    x = np.linspace(-1.0, 1.0, n)
    response = 0.5 * x + np.random.default_rng(314).normal(size=len(x))
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
        discrete=True,
    )
    model.fit(pd.DataFrame({"x": x}), response)
    assert len(roots) == 1
    carrier = roots[0].parameter_independent_carrier
    assert carrier is None, f"root still owns {carrier.nbytes} carrier bytes for {len(x)} rows"


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("n", [8191, 8192, 8193, 16403])
def test_derived_digest_and_owned_children_match_eager(semantics, n):
    family, response, eager, root = _plans(n, semantics, extreme=True)
    assert root.parameter_independent_carrier is None
    assert root.plan_identifier == eager.plan_identifier
    assert root.carrier_digest == eager.carrier_digest
    indices = np.array([0, 2, n - 1], dtype=np.intp)
    child, reference = root.take(indices), eager.take(indices)
    assert child.plan_identifier == reference.plan_identifier
    assert child.carrier_preparation == "stored"
    np.testing.assert_array_equal(
        child.parameter_independent_carrier, reference.parameter_independent_carrier
    )
    assert child.parameter_independent_carrier.flags.writeable is False
    for name in ("values", "geometry_values", "root_take_map", "input_positions"):
        assert not np.shares_memory(getattr(child.weights, name), getattr(root.weights, name))
    with pytest.raises(UnsupportedLikelihoodContractError, match="owned row children"):
        family.evaluate_natural(response, np.column_stack((response, np.ones(n))), root)
    with pytest.raises(LikelihoodWeightError, match="duplicates"):
        root.take(np.array([0, 0]))


def test_root_digest_blocks_and_child_lifetimes_are_bounded(monkeypatch):
    family, response, eager, _ = _plans(3 * 8192 + 3)
    original = gaussian._carrier_block
    references = []
    sizes = []

    def block(values, semantics):
        assert all(reference() is None for reference in references)
        result = original(values, semantics)
        sizes.append(len(result))
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(gaussian, "_carrier_block", block)
    root = family.bind_chunked_likelihood(response, eager.weights, COMPLETE_OBSERVATION)
    assert sizes == [8192, 8192, 8192, 3]
    assert all(reference() is None for reference in references)
    cache = build_likelihood_cache(family, root)
    assert cache is not None
    child = cache.take(root, np.arange(5), start=0, stop=5)
    assert cache.take(root, np.arange(5), start=0, stop=5) is child
    assert cache.retained_bytes <= cache.byte_budget
    reference = weakref.ref(root)
    del root
    assert reference() is None
    cache.clear()
    assert child.parameter_independent_carrier.shape == (5,)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_owned_child_matches_independent_gaussian_likelihood_derivatives(semantics):
    family, response, _, root = _plans(19, semantics)
    indices = np.array([0, 3, 7, 12, 18])
    child = root.take(indices)
    location = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    scale = np.array([0.75, 1.25, 1.5, 0.5, 2.0])
    actual = family.evaluate_natural(response[indices], np.column_stack((location, scale)), child)
    likelihood, scores, hessians = [], [], []
    likelihood_scales, score_scales, hessian_scales = [], [], []
    for y, mu, sigma, weight in zip(
        response[indices], location, scale, child.weights.values, strict=True
    ):
        residual = float(y - mu)
        mass = 1.0 if semantics == "prior" else float(weight)
        precision = float(weight)
        likelihood.append(
            -mass * (math.log(sigma) + 0.5 * math.log(2 * math.pi))
            - 0.5 * precision * residual**2 / sigma**2
            + (0.5 * math.log(weight) if semantics == "prior" else 0.0)
        )
        scores.append(
            [precision * residual / sigma**2, -mass / sigma + precision * residual**2 / sigma**3]
        )
        hessians.append(
            [
                -precision / sigma**2,
                -2 * precision * residual / sigma**3,
                mass / sigma**2 - 3 * precision * residual**2 / sigma**4,
            ]
        )
        likelihood_scales.append(
            mass * (abs(math.log(sigma)) + 0.5 * math.log(2 * math.pi))
            + 0.5 * precision * residual**2 / sigma**2
            + (0.5 * abs(math.log(weight)) if semantics == "prior" else 0.0)
        )
        score_scales.append([abs(scores[-1][0]), mass / sigma + precision * residual**2 / sigma**3])
        hessian_scales.append(
            [
                abs(hessians[-1][0]),
                abs(hessians[-1][1]),
                mass / sigma**2 + 3 * precision * residual**2 / sigma**4,
            ]
        )
    # Bounded normal-range formulas, with fewer than 32 rounded operations
    # per entry; scale by the term magnitudes rather than cancellation error.
    gamma = 32 * np.finfo(float).eps / (1 - 32 * np.finfo(float).eps)
    for observed, expected, magnitude in (
        (actual.reported_log_likelihood, likelihood, likelihood_scales),
        (actual.score, scores, score_scales),
        (actual.hessian_packed, hessians, hessian_scales),
    ):
        expected = np.asarray(expected)
        assert np.all(np.abs(observed - expected) <= gamma * np.asarray(magnitude))
    assert abs(
        math.fsum(actual.reported_log_likelihood) - math.fsum(likelihood)
    ) <= gamma * math.fsum(likelihood_scales)


@pytest.mark.parametrize("target", ["root", "child"])
@pytest.mark.parametrize("mutation", ["weights", "backing", "digest", "variant"])
def test_derived_cache_revokes_changed_source_or_child(monkeypatch, target, mutation):
    family, _, _, root = _plans()
    cache = build_likelihood_cache(family, root)
    assert cache is not None
    indices = np.arange(7)
    child = cache.take(root, indices, start=0, stop=7)
    changed = root if target == "root" else child
    if mutation in ("weights", "backing"):
        values = changed.weights.values.copy()
        values[0] *= 2
        if mutation == "weights":
            values = np.frombuffer(values.tobytes(), dtype=float)
        else:
            # A read-only view must not authenticate a still-writable owner.
            values = values.view()
            values.flags.writeable = False
        object.__setattr__(changed.weights, "values", values)
    elif mutation == "digest":
        object.__setattr__(changed, "carrier_digest", "forged")
    else:
        object.__setattr__(changed, "carrier_preparation", "unknown")

    def fresh(*args):
        raise RuntimeError("fresh child required")

    monkeypatch.setattr(GaussianLikelihoodPlan, "take", fresh)
    with pytest.raises(RuntimeError, match="fresh child required"):
        cache.take(root, indices, start=0, stop=7)
    assert cache.entry_count == 0


def test_derived_source_certificate_preserves_live_inputs_and_storage_authority():
    from .test_distributional_likelihood_cache_integration import _context
    from .test_distributional_support_predictions import _support_problem

    problem = _support_problem(False)
    context = _context(problem)
    eager = context.likelihood_plan
    root = context.family.bind_chunked_likelihood(
        context.response, eager.weights, COMPLETE_OBSERVATION
    )
    derived = replace(context, likelihood_plan=root)
    certificate = solver._chunk_reuse_data_certificate(derived)
    assert certificate is not None
    assert root.plan_identifier == eager.plan_identifier
    assert certificate != solver._chunk_reuse_data_certificate(context)
    values = root.weights.values.copy()
    values[0] *= 2
    object.__setattr__(root.weights, "values", np.frombuffer(values.tobytes(), dtype=float))
    assert solver._chunk_reuse_data_certificate(derived) != certificate
    object.__setattr__(root, "carrier_preparation", "stored")
    assert solver._chunk_reuse_data_certificate(derived) is None
    assert build_likelihood_cache(context.family, root) is None
    object.__setattr__(root, "carrier_preparation", np.array([1]))
    assert solver._chunk_reuse_data_certificate(derived) is None


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("retain_history_rows", [False, True])
def test_public_derived_fit_matches_eager_binding_and_serialization(
    monkeypatch, semantics, retain_history_rows
):
    from superglm.distributional import fit_state

    x = np.linspace(-1.0, 1.0, 49)
    frame = pd.DataFrame({"x": x})
    response = np.sin(2 * x) + np.random.default_rng(331).normal(size=len(x)) * 0.4
    weights = np.linspace(0.5, 2.0, len(x)) if semantics == "prior" else 1 + np.arange(len(x)) % 3
    weights[3] = 0
    offsets = {"location": 0.03 * x, "scale": np.full(len(x), 0.1)}
    null_references = []
    original_null = fit_state.fit_joint_null_model

    def null(*args, likelihood_plan, **kwargs):
        assert likelihood_plan.parameter_independent_carrier is not None
        null_references.append(weakref.ref(likelihood_plan))
        return original_null(*args, likelihood_plan=likelihood_plan, **kwargs)

    monkeypatch.setattr(fit_state, "fit_joint_null_model", null)

    def fit():
        model = SuperLSS(
            family=GaussianLS(),
            predictors=(Predictor("location", {"x": Spline(n_knots=4)}), Predictor("scale", {})),
            weight_semantics=semantics,
            discrete=True,
        )
        model.fit_reml(
            frame,
            response,
            sample_weight=weights,
            offsets=offsets,
            max_reml_iter=2,
            retain_history_rows=retain_history_rows,
        )
        return model

    actual = fit()
    assert len(null_references) == 1 and null_references[0]() is None
    monkeypatch.setattr(GaussianLS, "bind_chunked_likelihood", GaussianLS.bind_likelihood)
    expected = fit()
    assert actual._model.smoothing.history == expected._model.smoothing.history
    for field in ("coefficients", "terminal_data_curvature", "terminal_penalized_curvature"):
        np.testing.assert_array_equal(
            getattr(actual._model.result, field), getattr(expected._model.result, field)
        )
    np.testing.assert_array_equal(
        actual._model.fit_state.inference.covariance, expected._model.fit_state.inference.covariance
    )
    for field in fields(actual._model.fit_state.null_model):
        left = getattr(actual._model.fit_state.null_model, field.name)
        right = getattr(expected._model.fit_state.null_model, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    restored = SuperLSS.from_bytes(actual.to_bytes())
    np.testing.assert_array_equal(
        restored.predict(frame, offsets=offsets), actual.predict(frame, offsets=offsets)
    )


def test_custom_gaussian_binding_is_not_bypassed(monkeypatch):
    calls = []

    class CustomGaussian(GaussianLS):
        def bind_likelihood(self, *args):
            calls.append(True)
            return super().bind_likelihood(*args)

        def bind_chunked_likelihood(self, *args):
            pytest.fail("inherited chunk binding must not replace a custom binder")

        def to_config(self):
            return GaussianLS().to_config()

    x = np.linspace(-1.0, 1.0, 31)
    model = SuperLSS(
        family=CustomGaussian(),
        predictors=(Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
        discrete=True,
    )
    model.fit(pd.DataFrame({"x": x}), x + np.random.default_rng(991).normal(size=len(x)))
    assert calls == [True]


def test_legacy_eager_plan_missing_private_mode_retains_fresh_evaluation():
    family, response, eager, _ = _plans()
    theta = np.column_stack((response, np.ones(len(response))))
    expected = family.evaluate_natural(response, theta, eager)
    object.__delattr__(eager, "carrier_preparation")
    legacy = pickle.loads(pickle.dumps(eager))
    actual = family.evaluate_natural(response, theta, legacy)
    np.testing.assert_array_equal(actual.reported_log_likelihood, expected.reported_log_likelihood)
    assert legacy.plan_identifier == eager.plan_identifier
    assert build_likelihood_cache(family, legacy) is None
    assert legacy.take(np.arange(3)).carrier_preparation == "stored"


def test_derived_cache_keeps_prefix_after_budget_refuses_tail():
    family, _, _, root = _plans(41)
    probe = build_likelihood_cache(family, root)
    probe.take(root, np.arange(20), start=0, stop=20)
    budget = probe.retained_bytes + 128
    cache = build_likelihood_cache(family, root, byte_budget=budget)
    first = cache.take(root, np.arange(20), start=0, stop=20)
    cache.take(root, np.arange(20, 40), start=20, stop=40)
    tail = cache.take(root, np.arange(40, 41), start=40, stop=41)
    reference = weakref.ref(tail)
    del tail
    assert reference() is None
    assert cache.entry_count == 1
    assert cache.retained_bytes <= budget
    assert cache.take(root, np.arange(20), start=0, stop=20) is first


def test_public_gamma_chunked_binding_remains_eager(monkeypatch):
    import superglm.distributional.model as model_module
    from superglm.distributional.families.gamma import GammaLS

    family = GammaLS()
    original = model_module.fit_dense_fixed_lambda
    roots = []

    def capture(family, layout, response, plan, *args, **kwargs):
        roots.append(plan)
        return original(family, layout, response, plan, *args, **kwargs)

    monkeypatch.setattr(model_module, "fit_dense_fixed_lambda", capture)
    x = np.linspace(-1.0, 1.0, 37)
    response = np.random.default_rng(331).gamma(3.0, np.exp(0.2 * x) / 3.0)
    model = SuperLSS(
        family=family,
        predictors=(
            Predictor(family.parameters[0].name, {"x": Numeric()}),
            Predictor(family.parameters[1].name, {}),
        ),
        discrete=True,
    )
    model.fit(pd.DataFrame({"x": x}), response)
    assert len(roots) == 1
    assert roots[0].parameter_independent_carrier.shape == response.shape
