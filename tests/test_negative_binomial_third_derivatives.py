"""Analytic NB2 third derivatives for the LAML endgame.

Held to an independent 120-digit oracle within the bound the kernel
certifies, and to the Richardson difference of the published curvature at
that difference's resolution.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributional.families import NegativeBinomialLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    finite_difference_curvature_direction,
)
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink, LogLink
from tests._distributional_family_kernels import negative_binomial as nb_kernel

_U = 2.0**-53


def _oracle_third(count: int, mean: float, theta: float) -> list[Decimal]:
    """``l_aaa, l_aab, l_abb, l_bbb`` at 120 digits from the theta derivatives of the row."""
    with localcontext() as context:
        context.prec = 120
        y, mu, th = Decimal(count), Decimal(mean), Decimal(theta)
        t = mu + th
        d1 = sum((1 / (th + j) for j in range(count)), Decimal(0))
        d2 = -sum((1 / (th + j) ** 2 for j in range(count)), Decimal(0))
        d3 = 2 * sum((1 / (th + j) ** 3 for j in range(count)), Decimal(0))
        f1 = d1 + (th / t).ln() + (mu - y) / t
        f2 = d2 + 1 / th - 2 / t + (y + th) / t**2
        f3 = d3 - 1 / th**2 + 3 / t**2 - 2 * (y + th) / t**3
        q, p, r = mu / t, th / t, (th - mu) / t
        s = q * p
        return [
            -(y + th) * r * s,
            s * (y * r - 2 * mu * p),
            -(y - mu) * r * s,
            th * f1 + 3 * th**2 * f2 + th**3 * f3,
            s * (abs(y * r) + 2 * mu * p),
        ]


def _oracle_rows(seed: int, n_rows: int):
    rng = np.random.default_rng(seed)
    theta = np.exp(rng.uniform(-8.0, 40.0, n_rows))
    mean = theta * np.exp(rng.uniform(np.log(2.0**-52), np.log(2.0**26), n_rows))
    counts = np.minimum(rng.poisson(np.minimum(mean, 300.0)), 400).astype(np.float64)
    return counts, mean, theta


def test_theta_third_derivative_is_inside_its_certified_bound() -> None:
    """Near Poisson ``l_bbb`` cancels by ``theta^2``; the chosen route still certifies its error."""
    counts, mean, theta = _oracle_rows(7, 600)
    counts[:3], mean[:3], theta[:3] = [3.0, 0.0, 2.0], [0.5, 1e-3, 2.0**-30], [1e15, 1e12, 2.0**20]
    third, error = nb_kernel.negative_binomial_log_third_derivatives(
        counts, mean, theta, np.ones(len(counts)), "prior"
    )
    gamma_11 = 11 * _U / (1 - 11 * _U)
    for row in range(len(counts)):
        oracle = _oracle_third(int(counts[row]), float(mean[row]), float(theta[row]))
        assert abs(Decimal(float(third[row, 3])) - oracle[3]) <= Decimal(float(error[row]))
        scales = (abs(oracle[0]), oracle[4], abs(oracle[2]))
        for channel, scale in enumerate(scales):
            difference = abs(Decimal(float(third[row, channel])) - oracle[channel])
            assert difference <= Decimal(gamma_11) * scale


def test_theta_third_derivative_bound_is_not_vacuous() -> None:
    counts, mean, theta = _oracle_rows(8, 400)
    third, error = nb_kernel.negative_binomial_log_third_derivatives(
        counts, mean, theta, np.ones(len(counts)), "prior"
    )
    resolvable = np.abs(third[:, 3]) > 0.0
    assert np.median(error[resolvable] / np.abs(third[resolvable, 3])) < 1e-12


def _plan(response, weights, semantics):
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(response), contract=WeightContract(semantics)
    )
    return NegativeBinomialLS().bind_likelihood(response, resolved, COMPLETE_OBSERVATION)


@pytest.mark.parametrize("semantics", ["prior", "frequency", "exposure"])
def test_analytic_direction_agrees_with_the_published_curvature(semantics: str) -> None:
    """Consistency with the Richardson difference of the published curvature.

    The difference is a truncation estimate whose certificate is not a bound
    (it can miss by a few times on single rows), so this checks agreement at
    the difference's resolution; the accuracy claim is the oracle test above.
    """
    rng = np.random.default_rng(4)
    n_rows = 3000
    counts = rng.poisson(0.3, n_rows).astype(np.float64)
    weights = None
    response = counts
    if semantics == "frequency":
        weights = rng.integers(1, 4, n_rows).astype(np.float64)
    elif semantics == "exposure":
        weights = rng.uniform(0.1, 1.0, n_rows)
        response = counts / weights
    plan = _plan(response, weights, "frequency" if semantics == "frequency" else "prior")
    eta = np.column_stack(
        (np.log(0.3) + rng.normal(0.0, 0.5, n_rows), rng.normal(0.0, 3.0, n_rows))
    )
    family, links = NegativeBinomialLS(), (LogLink(), LogLink())
    for direction in (np.tile([1.0, 0.0], (n_rows, 1)), rng.normal(size=eta.shape)):
        analytic = family.predictor_curvature_directional_derivative(
            response, eta, direction, links, plan
        )
        numeric = finite_difference_curvature_direction(
            family, response, eta, direction, links, plan, step=10.0 * DEFAULT_STEP
        )
        scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
        relative = np.abs(numeric.values - analytic) / scale
        assert np.max(relative) < 1.0e-8
        assert np.median(relative) < 1.0e-10


def test_laml_derivatives_take_the_analytic_direction() -> None:
    """The endgame differentiates NB2 curvature analytically instead of by stencil."""
    from superglm.distributional.family import PredictorCurvatureDirectionalFamily
    from superglm.distributional.smoothing.derivatives import _analytic_first_derivative

    response = np.array([0.0, 1.0, 2.0, 0.0, 5.0])
    plan = _plan(response, None, "prior")
    eta = np.column_stack((np.full(5, 0.1), np.full(5, 1.5)))
    family = NegativeBinomialLS()
    assert isinstance(family, PredictorCurvatureDirectionalFamily)
    derivative = _analytic_first_derivative(
        family, response, eta, (LogLink(), LogLink()), plan, 5, 2
    )
    assert derivative is not None


def test_analytic_direction_refuses_links_it_does_not_differentiate() -> None:
    response = np.array([0.0, 1.0, 2.0])
    plan = _plan(response, None, "prior")
    eta = np.zeros((3, 2))
    with pytest.raises(UnsupportedLikelihoodContractError):
        NegativeBinomialLS().predictor_curvature_directional_derivative(
            response, eta, eta, (IdentityLink(), LogLink()), plan
        )


def test_certified_direction_bounds_its_theta_channel_against_the_oracle() -> None:
    """Every row's ``bb`` entry is within its certificate plus the closed-form rounding.

    The ``bb`` entry is ``-(l_abb d_a + l_bbb d_b)``: ``l_abb`` is a closed form
    within ``gamma_11`` of its scale, ``l_bbb`` is within its certified ``e``,
    and the two products and the sum add three roundings, so
    ``|bb - exact| <= |d_b| e + gamma_14 (|l_abb d_a| + (|l_bbb| + e) |d_b|)``,
    with ``|d_b| e`` the certificate.  Held at the maximum over rows from deep
    near Poisson (``mean / theta`` down to ``2^-52``) to far from it.
    """
    counts, mean, theta = _oracle_rows(9, 400)
    counts[:3], mean[:3], theta[:3] = [3.0, 0.0, 2.0], [0.5, 1e-3, 2.0**-30], [1e15, 1e12, 2.0**20]
    rng = np.random.default_rng(10)
    direction = rng.normal(size=(len(counts), 2))
    eta = np.column_stack((np.log(mean), np.log(theta)))
    values, certificate = nb_kernel.negative_binomial_certified_predictor_curvature_directional(
        counts, eta, direction, np.ones(len(counts)), "prior"
    )
    _, error = nb_kernel.negative_binomial_log_third_derivatives(
        counts, np.exp(eta[:, 0]), np.exp(eta[:, 1]), np.ones(len(counts)), "prior"
    )
    assert np.all(certificate[:, :2] == 0.0)
    assert np.all(certificate[:, 2] >= np.abs(direction[:, 1]) * error)
    gamma_14 = Decimal(14 * _U / (1 - 14 * _U))
    for row in range(len(counts)):
        oracle = _oracle_third(
            int(counts[row]), float(np.exp(eta[row, 0])), float(np.exp(eta[row, 1]))
        )
        d_a, d_b = Decimal(float(direction[row, 0])), Decimal(float(direction[row, 1]))
        exact = -(oracle[2] * d_a + oracle[3] * d_b)
        rounding = gamma_14 * (
            abs(oracle[2] * d_a) + (abs(oracle[3]) + Decimal(float(error[row]))) * abs(d_b)
        )
        miss = abs(Decimal(float(values[row, 2])) - exact)
        assert miss <= Decimal(float(certificate[row, 2])) + rounding


def test_gradient_certificate_carries_the_theta_third_bound() -> None:
    """The LAML gradient certificate holds the certified ``l_bbb`` error, not zero.

    With the axis direction ``e_theta`` the certificate of the ``bb`` channel is
    ``e_i``, and the gradient certificate's third-derivative term is
    ``1/2 sum_i |eta_k[i, theta]| |G_i[theta, theta]| e_i`` with
    ``eta_k = X beta_k``, ``beta_k = -H^-1 lambda_k S_k beta`` and
    ``G_i = X_i H^-1 X_i'``, formed here independently of the assembly.  At
    7305fe25 the analytic direction published no certificate and this fit's
    gradient certificate was zero.
    """
    import pandas as pd

    from superglm._frame import as_eager_frame
    from superglm.distributional.layout import build_stacked_layout
    from superglm.distributional.predictor import Predictor, compile_predictors
    from superglm.distributional.results.solver import DenseSolverConfig
    from superglm.distributional.smoothing.derivatives import laml_derivatives
    from superglm.distributional.solver import fit_dense_fixed_lambda
    from superglm.distributional.solver.solver import _DenseObservedReuseSession
    from superglm.features import Spline

    rng = np.random.default_rng(5)
    n_rows = 400
    x = rng.uniform(0.0, 1.0, n_rows)
    mean = np.exp(0.2 + 0.9 * np.sin(2.0 * np.pi * x))
    theta = np.exp(5.0 + 0.5 * np.sin(np.pi * x))
    response = rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float64)
    plan = _plan(response, None, "prior")
    family = NegativeBinomialLS()
    frame = as_eager_frame(pd.DataFrame({"x": x}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            plan.weights,
            family.parameters,
            (
                Predictor("mean", {"x": Spline(kind="cr", k=8)}),
                Predictor("theta", {"x": Spline(kind="cr", k=6)}),
            ),
        )
    )
    lambdas = dict.fromkeys(layout.penalty_names, 1.0)
    fit = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        layout.penalty_matrix(lambdas),
        config=DenseSolverConfig(
            max_iterations=200, tolerance=1e-11, coefficient_curvature="observed"
        ),
    )
    assert fit.converged
    dense = _DenseObservedReuseSession().dense_matrices(layout)
    result = laml_derivatives(
        family,
        layout,
        response,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=dense,
        want_hessian=False,
    )
    assert result.third_derivative_authority == "analytic-observed-curvature-direction/v1"

    eta = np.asarray(fit.eta)
    _, error = nb_kernel.negative_binomial_log_third_derivatives(
        plan.exact_count, np.exp(eta[:, 0]), np.exp(eta[:, 1]), np.ones(n_rows), "prior"
    )
    inverse = np.asarray(fit.terminal_pseudo_inverse())
    beta = np.asarray(fit.coefficients)
    block = layout.predictors[1].coefficient_slice
    design = np.asarray(dense[1])
    leverage = np.einsum("ij,jk,ik->i", design, inverse[block, block], design)
    leverage_scale = np.einsum(
        "ij,jk,ik->i", np.abs(design), np.abs(inverse[block, block]), np.abs(design)
    )
    width = len(beta)
    gamma = (4 * width + 8) * _U / (1.0 - (4 * width + 8) * _U)
    third_terms, slacks = [], []
    for name in layout.penalty_names:
        penalty = layout.penalty_matrix(
            {other: (lambdas[other] if other == name else 0.0) for other in layout.penalty_names}
        )
        direction = design @ (-(inverse @ (penalty @ beta)))[block]
        direction_scale = (
            np.abs(design) @ (np.abs(inverse) @ (np.abs(penalty) @ np.abs(beta)))[block]
        )
        third_terms.append(0.5 * float(np.sum(np.abs(direction) * np.abs(leverage) * error)))
        # Both sides round G_i and eta_k[i] from the same inputs, each within
        # gamma_{2p+4} of its absolute-term scale, so they differ by at most
        # twice that (the sums of n nonnegative products add only gamma_n of themselves).
        slacks.append(2.0 * gamma * float(np.sum(direction_scale * leverage_scale * error)))
    expected, slack = np.array(third_terms), np.array(slacks)
    # The theta component's scales sit 3.6e-5 of its term apart: the check still
    # demands all but a thousandth of the term, where 7305fe25 carried none.
    assert np.all(expected > 0.0) and np.all(slack < 1e-3 * expected)
    assert np.all(result.gradient_certificate >= (1.0 - 2.0 * n_rows * _U) * expected - slack)


def test_axis_derivatives_are_evaluated_once(monkeypatch) -> None:
    """K axis derivatives cost K kernel calls: the admission check is the pass itself."""
    from superglm.distributional.smoothing.derivatives import (
        _analytic_first_derivative,
        _first_derivatives,
    )

    response = np.array([0.0, 1.0, 2.0, 0.0, 5.0])
    plan = _plan(response, None, "prior")
    eta = np.column_stack((np.full(5, 0.1), np.full(5, 1.5)))
    family = NegativeBinomialLS()
    calls = []
    original = NegativeBinomialLS.certified_predictor_curvature_directional_derivative

    def counted(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(
        NegativeBinomialLS, "certified_predictor_curvature_directional_derivative", counted
    )
    analytic = _analytic_first_derivative(family, response, eta, (LogLink(), LogLink()), plan, 5, 2)
    assert analytic is not None
    first, certificate, _stencils, evaluations = _first_derivatives(
        lambda values: values, eta, DEFAULT_STEP, analytic_first=analytic
    )
    assert len(calls) == 2 and evaluations == 0
    np.testing.assert_array_equal(first[0], analytic.axes[0][0])
    np.testing.assert_array_equal(certificate[0], analytic.axes[0][1])


class _CertifiedOnlyNegativeBinomial:
    """NB2 exposing only the certified direction, not the plain protocol method."""

    def __init__(self) -> None:
        self.base = NegativeBinomialLS()

    def __getattr__(self, name):
        if name == "predictor_curvature_directional_derivative":
            raise AttributeError(name)
        return getattr(self.base, name)

    def certified_predictor_curvature_directional_derivative(self, *args):
        return self.base.certified_predictor_curvature_directional_derivative(*args)


def test_laml_admits_a_certified_only_family() -> None:
    """A family with only the certified protocol takes the analytic path with its certificate.

    At 9f1495ef the LAML guard admitted only the plain protocol, so such a
    family fell back to the stencil while the endpoint evaluator took it
    analytically.
    """
    from superglm.distributional.family import (
        CertifiedCurvatureDirectionalFamily,
        PredictorCurvatureDirectionalFamily,
    )
    from superglm.distributional.smoothing.derivatives import _analytic_first_derivative

    response = np.array([0.0, 1.0, 2.0, 0.0, 5.0])
    plan = _plan(response, None, "prior")
    eta = np.column_stack((np.full(5, 0.1), np.full(5, 12.0)))
    family = _CertifiedOnlyNegativeBinomial()
    assert isinstance(family, CertifiedCurvatureDirectionalFamily)
    assert not isinstance(family, PredictorCurvatureDirectionalFamily)
    analytic = _analytic_first_derivative(family, response, eta, (LogLink(), LogLink()), plan, 5, 2)
    assert analytic is not None
    expected = NegativeBinomialLS().certified_predictor_curvature_directional_derivative(
        response, eta, np.tile([0.0, 1.0], (5, 1)), (LogLink(), LogLink()), plan
    )
    np.testing.assert_array_equal(analytic.axes[1][0], expected[0])
    np.testing.assert_array_equal(analytic.axes[1][1], expected[1])
    assert np.all(analytic.axes[1][1][:, 2] > 0.0)
