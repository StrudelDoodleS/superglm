from dataclasses import replace

import numpy as np
import pytest

import superglm.distributional.families.gamma as gamma_module
import superglm.distributional.families.gaussian as gaussian_module
import superglm.distributional.weights as weight_module
from superglm.distributional.family import COMPLETE_OBSERVATION


def _case(kind):
    if kind == "Gaussian":
        family = gaussian_module.GaussianLS(scale_floor=0.0)
        return gaussian_module, family, np.array([-0.5, 2.0]), np.array([[0.0, 0.7], [1.5, 1.2]])
    family = gamma_module.GammaLS()
    return gamma_module, family, np.array([0.5, 2.0]), np.array([[0.8, 0.7], [1.5, 1.2]])


def _plan(family, y, semantics="prior"):
    weights = weight_module.resolve_likelihood_weights(
        np.ones(len(y)),
        n_observations=len(y),
        contract=weight_module.WeightContract(semantics),
    )
    return family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)


def test_gaussian_public_validation_rejects_foreign_family_configuration():
    _, family, y, theta = _case("Gaussian")
    forged = replace(_plan(family, y))
    object.__setattr__(forged, "family_config", ("GaussianLS/v1", 1.0))
    with pytest.raises(weight_module.UnsupportedLikelihoodContractError):
        family.evaluate_natural(y, theta, forged)


@pytest.mark.parametrize("kind", ["Gaussian", "Gamma"])
def test_public_evaluation_keeps_order_y_theta_plan_precedence(kind):
    _, family, y, theta = _case(kind)
    with pytest.raises(ValueError, match="derivative_order"):
        family.evaluate_natural(np.full_like(y, np.nan), theta[:, :1], object(), derivative_order=3)
    with pytest.raises(ValueError, match="y.*finite"):
        family.evaluate_natural(np.full_like(y, np.nan), theta[:, :1], object())
    with pytest.raises(ValueError, match="theta.*shape"):
        family.evaluate_natural(y, theta[:, :1], object())


def test_gaussian_subclass_capability_metadata_does_not_control_evaluation_order() -> None:
    class OrderZeroGaussian(gaussian_module.GaussianLS):
        capabilities = replace(gaussian_module._CAPABILITIES, max_derivative_order=0)

    family = OrderZeroGaussian(scale_floor=0.0)
    _, _, response, theta = _case("Gaussian")
    evaluation = family.evaluate_natural(
        response,
        theta,
        _plan(family, response),
        derivative_order=1,
    )

    assert evaluation.score is not None
    assert evaluation.hessian_packed is None


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("theta_rows", [1, 3])
def test_family_fisher_cores_reject_both_row_mismatch_directions(semantics, theta_rows):
    for kind in ("Gaussian", "Gamma"):
        _, family, y, _ = _case(kind)
        with pytest.raises(weight_module.UnsupportedLikelihoodContractError, match="rows"):
            family.expected_information_natural(
                np.ones((theta_rows, 2)),
                _plan(family, y, semantics),
            )
