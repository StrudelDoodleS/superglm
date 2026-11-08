from dataclasses import replace
from importlib import import_module, util

import numpy as np
import pytest

from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import (
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)


def weights(values, semantics="frequency"):
    return resolve_likelihood_weights(
        np.asarray(values), n_observations=len(values), contract=WeightContract(semantics=semantics)
    )


@pytest.fixture
def family():
    name = "superglm.distributional.families.lomax"
    assert util.find_spec(name) is not None, "immutable Lomax family adapter is missing"
    return import_module(name).LomaxLSS()


def test_declared_plan_is_bound_to_law_rows_weights_and_observation(family):
    y = np.array([0.1, 1.0, 3.0])
    w = np.array([1, 2, 5])
    plan = family.bind_likelihood(y, weights(w), COMPLETE_OBSERVATION)
    original = y.copy()
    identifier = plan.plan_identifier
    y[:] = 99
    w[:] = 99
    np.testing.assert_array_equal(plan.exact_response, original)
    assert plan.plan_identifier == identifier
    with pytest.raises(ValueError):
        plan.exact_response.setflags(write=True)
    for changed in (y, original[::-1]):
        with pytest.raises(UnsupportedLikelihoodContractError, match="response"):
            family.evaluate_natural(changed, np.ones((3, 2)), plan)
    with pytest.raises(UnsupportedLikelihoodContractError, match="configuration|law"):
        family.evaluate_natural(original, np.ones((3, 2)), replace(plan, law_key="other/v1"))
    with pytest.raises(UnsupportedLikelihoodContractError, match="configuration|schema"):
        family.evaluate_natural(original, np.ones((3, 2)), replace(plan, schema_version=2))
    other = family.bind_likelihood(original, weights([1, 1, 1]), COMPLETE_OBSERVATION)
    assert other.plan_identifier != identifier
    selected = np.array([2, 0])
    child = plan.take(selected)
    whole = family.evaluate_natural(original, np.ones((3, 2)), plan)
    part = family.evaluate_natural(original[selected], np.ones((2, 2)), child)
    np.testing.assert_array_equal(part.hessian_packed, whole.hessian_packed[selected])
    assert child.plan_identifier != identifier


def test_declared_derivative_order_contract(family):
    y = np.array([0.5, 1.0])
    plan = family.bind_likelihood(y, weights([1, 2]), COMPLETE_OBSERVATION)
    for invalid in (False, True, np.bool_(True), -1, 1.0, 3, 4):
        with pytest.raises(ValueError, match="integer"):
            family.evaluate_natural(y, np.ones((2, 2)), plan, derivative_order=invalid)
    for order in (0, 1, 2):
        out = family.evaluate_natural(y, np.ones((2, 2)), plan, derivative_order=order)
        assert (out.score is None) == (order == 0)
        assert (out.hessian_packed is None) == (order < 2)


def test_declared_spec_rejects_coordinate_and_capability_disagreement(family):
    from dataclasses import dataclass

    from superglm.distributional.families._declared import DeclaredScalarFamily

    @dataclass(frozen=True)
    class WrongOrder(DeclaredScalarFamily):
        spec = replace(family.spec, parameters=family.spec.parameters[::-1])

    with pytest.raises(ValueError, match="parameter order"):
        WrongOrder()

    @dataclass(frozen=True)
    class FalseCapabilities(DeclaredScalarFamily):
        spec = replace(family.spec, supported_operations=())

    with pytest.raises(TypeError, match="capability"):
        FalseCapabilities()
    with pytest.raises(ValueError, match="schema"):
        replace(family.spec, schema_version=2)


def test_pickled_plan_restores_immutable_bounds_and_weight_identity(family):
    import pickle

    from superglm.distributional.family import bounded_observation
    from superglm.distributional.observation import ResponseBounds

    y = np.array([1.0, 2.0])
    bounds = ResponseBounds.from_arrays(
        np.array([0.1, 0.2]), np.array([3.0, 4.0]), np.array([-1, 1])
    )
    plan = family.bind_likelihood(y, weights([2, 5]), bounded_observation(bounds))
    loaded = pickle.loads(pickle.dumps(plan))
    assert loaded.plan_identifier == plan.plan_identifier
    for array in (
        loaded.exact_response,
        loaded.parameter_independent_carrier,
        loaded.weights.values,
        loaded.observation.bounds.lower,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    bad = replace(plan, schema_version=2)
    with pytest.raises(UnsupportedLikelihoodContractError, match="schema"):
        pickle.loads(pickle.dumps(bad))


def test_lomax_frequency_equals_literal_replication(family):
    y, mass = np.array([0.2, 1.0, 5.0]), np.array([1, 2, 5])
    theta = np.array([[2.0, 0.75], [3.0, 1.0], [0.5, 3.0]])
    plan = family.bind_likelihood(y, weights(mass), COMPLETE_OBSERVATION)
    unit = family.bind_likelihood(y, weights(np.ones(3), "prior"), COMPLETE_OBSERVATION)
    k = import_module("superglm.distributional.kernels.lomax")
    freq = k.density_jets(y, theta, plan.multiplier, order=4)
    single = k.density_jets(y, theta, unit.multiplier, order=4)
    for weighted, unweighted in zip(freq.jets, single.jets, strict=True):
        expected = unweighted * (mass if weighted.ndim == 1 else mass[:, None])
        np.testing.assert_allclose(weighted, expected, rtol=32 * 2.0**-53, atol=0)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                weighted * (mass if weighted.ndim == 1 else mass[:, None]), expected
            )
    out = family.evaluate_natural(y, theta, plan)
    duplicate = np.repeat(np.arange(3), mass)
    rep = family.bind_likelihood(
        y[duplicate], weights(np.ones(len(duplicate))), COMPLETE_OBSERVATION
    )
    repeated = family.evaluate_natural(y[duplicate], theta[duplicate], rep)
    np.testing.assert_allclose(
        out.reported_log_likelihood.sum(), repeated.reported_log_likelihood.sum()
    )
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        family.bind_likelihood(y, weights(mass, "prior"), COMPLETE_OBSERVATION)
    with pytest.raises(LikelihoodWeightError, match="integer"):
        weights([1, 1.5, 2])
    resolved = weights([0, 1, 2, 5])
    np.testing.assert_array_equal(resolved.input_positions, [1, 2, 3])


def test_initialization_uses_weighted_median_and_discloses_no_information(family):
    y = np.array([0.1, 1.0, 10.0])
    plan = family.bind_likelihood(y, weights([1, 5, 1]), COMPLETE_OBSERVATION)
    start = family.initialize(y, plan)
    np.testing.assert_allclose(start.theta, np.tile([1 / (np.sqrt(2) - 1), 2.0], (3, 1)))
    y = np.zeros(3)
    plan = family.bind_likelihood(y, weights([1, 1, 1]), COMPLETE_OBSERVATION)
    with pytest.warns(UserWarning, match="identified|identification|uninformative"):
        start = family.initialize(y, plan)
    np.testing.assert_array_equal(start.theta[:, 0], np.ones(3))


def test_carrier_and_differently_named_three_parameter_adapter():
    from dataclasses import dataclass

    from superglm.distributional.families._declared import DeclaredScalarFamily, DeclaredScalarSpec
    from superglm.distributional.family import FamilyCapabilities, ParameterSpec, ParameterSupport
    from superglm.links import IdentityLink, LogLink

    @dataclass(frozen=True)
    class Three(DeclaredScalarFamily):
        spec = DeclaredScalarSpec(
            "test-normal/v1",
            1,
            "tests._declared_scalar_oracles",
            (
                ParameterSpec("origin", IdentityLink(), "location", ParameterSupport(), "observed"),
                ParameterSpec("width", LogLink(), "scale", ParameterSupport(lower=0.0), "observed"),
                ParameterSpec("drift", IdentityLink(), "location", ParameterSupport(), "observed"),
            ),
            -np.inf,
            False,
            FamilyCapabilities(2, False, False, False, False, False),
        )

    f = Three()
    y, mass = np.array([-0.5, 2.0]), np.array([2, 5])
    plan = f.bind_likelihood(y, weights(mass), COMPLETE_OBSERVATION)
    theta = np.array([[0.2, 1.5, 0.1], [0.2, 1.5, 0.1]])
    actual = f.evaluate_natural(y, theta, plan)
    from scipy.stats import norm

    np.testing.assert_allclose(
        actual.reported_log_likelihood, mass * norm.logpdf(y, loc=0.3, scale=1.5)
    )
    assert actual.score.shape == (2, 3) and actual.hessian_packed.shape == (2, 6)
    np.testing.assert_allclose(
        actual.parameter_independent_carrier, -0.5 * np.log(2 * np.pi) * mass
    )
    assert not hasattr(f, "expected_information_natural") and not hasattr(f, "cdf")


def test_generated_carrier_keeps_the_warmed_operand_signature():
    import subprocess
    import sys

    # A fresh interpreter ensures an earlier test's extra specialization
    # cannot hide the first-fit compile this contract is intended to prevent.
    script = """
import numpy as np
from superglm.distributional.kernels import lomax, log_logistic
for kernel in (lomax, log_logistic):
    module=kernel.generated
    y=np.array([.5,2.]); mass=np.ones(2); theta=np.tile([2.,1.5],(2,1))
    kernel.window_jets(y,y+1e-6,theta,order=4)  # same narrow path as warmup
    warmed=tuple(module.carrier.signatures)
    assert len(warmed)==1
    y.setflags(write=False); mass.setflags(write=False)
    for order in range(5):
        out=kernel.density_jets(y,theta,mass,order=order)
        assert out.valid.all()
    assert tuple(module.carrier.signatures)==warmed, (warmed,module.carrier.signatures)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
