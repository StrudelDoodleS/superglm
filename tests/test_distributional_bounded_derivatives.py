"""Same compiled-design derivatives and no full grouped-design allocations."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional import GammaLS, GaussianLS, Predictor
from superglm.distributional._row_design import bounded_predictor_matrices
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.model import DenseDistributionalModel, fit_dense_distributional
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.smoothing.derivatives import LamlDerivativeWorkspace, laml_derivatives
from superglm.distributional.smoothing.endpoint_direction import _curvature_packed
from superglm.distributional.smoothing.penalty_face import build_penalty_face
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.distributional.solver.solver import _DenseObservedReuseSession, fit_dense_fixed_lambda
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Spline
from superglm.group_matrix import DesignMatrix


def test_chunked_newton_never_requests_full_predictor_design(monkeypatch):
    """Mutation sentinel: the old Newton handoff unconditionally requests dense X."""
    rng = np.random.default_rng(716)
    n = 500
    x = rng.uniform(-1, 1, n)
    frame = pd.DataFrame({"x": x})
    y = 1.4 * np.sin(3 * x) + np.exp(-0.5 + 0.3 * x) * rng.normal(size=n)

    def forbidden(*args, **kwargs):
        pytest.fail("grouped Newton requested a full predictor design")

    monkeypatch.setattr(_DenseObservedReuseSession, "dense_matrices", forbidden)
    fitted = fit_dense_distributional(
        frame,
        y,
        family=GaussianLS(),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", k=7)}),
            Predictor("scale", {"x": Spline(kind="cr", k=5)}),
        ),
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(max_iterations=200, tolerance=1e-10),
        efs_config=DistributionalEFSConfig(max_iterations=100, outer="efs+newton"),
        chunk_size=63,
        retain_rows=True,
    )
    assert fitted.smoothing is not None
    assert fitted.smoothing.terminal_gradient is not None
    # Force the authenticated posterior replay branch, even if Newton published
    # a Hessian on this platform. Its original unconditional dense call is a
    # second mutation sentinel independent of the Newton handoff.
    retained = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        retained.response, retained.likelihood_weights, COMPLETE_OBSERVATION
    )
    dense_derivatives = laml_derivatives(
        fitted.family,
        fitted.layout,
        retained.response,
        plan,
        lambdas=fitted.smoothing.lambdas,
        fit=fitted.smoothing.terminal_fit,
        dense_matrices=dense_predictor_matrices(fitted.layout),
        step=fitted.smoothing.config.derivative_step,
    )
    published = replace(
        fitted.smoothing,
        smoothing_hessian=dense_derivatives.hessian,
        smoothing_hessian_certificate=dense_derivatives.hessian_certificate,
    )
    dense_reference = DenseDistributionalModel(
        family=fitted.family, _fit_state=replace(fitted.fit_state, smoothing=published)
    )
    reference = posterior_covariance(dense_reference, kind="corrected")
    smoothing = replace(
        fitted.smoothing, smoothing_hessian=None, smoothing_hessian_certificate=None
    )
    replayed = DenseDistributionalModel(
        family=fitted.family, _fit_state=replace(fitted.fit_state, smoothing=smoothing)
    )
    monkeypatch.setattr("superglm.distributional.posterior.dense_predictor_matrices", forbidden)
    corrected = posterior_covariance(replayed, kind="corrected")
    scale = max(1.0, np.linalg.norm(reference, ord=2))
    bound = 256 * n * np.finfo(float).eps * scale
    np.testing.assert_allclose(corrected, reference, rtol=0, atol=bound)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("face", [False, True])
@pytest.mark.parametrize("family", [GaussianLS(), GammaLS()])
@pytest.mark.parametrize("weight_kind", ["prior", "frequency"])
def test_bounded_derivatives_match_identical_compiled_design(
    monkeypatch, discrete, face, family, weight_kind
):
    n, chunk = 317, 29
    rng = np.random.default_rng(213)
    y = rng.normal(1.8, 0.4, n) if isinstance(family, GaussianLS) else rng.gamma(4, 0.5, n)
    sample_weight = np.linspace(0.7, 1.4, n) if weight_kind == "prior" else np.resize([1, 2, 3], n)
    weights = resolve_likelihood_weights(
        sample_weight, n_observations=n, contract=WeightContract(weight_kind)
    )
    x = np.linspace(-1.0, 1.0, n)
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": np.sin(4.2 * x)}))
    names = tuple(parameter.name for parameter in family.parameters)
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(names[0], {"x": Spline(kind="cr", k=7, discrete=discrete)}),
                Predictor(names[1], {"z": Spline(kind="cr", k=6, discrete=discrete)}),
            ),
            offsets={names[0]: 0.03 * x, names[1]: 0.02 * np.cos(x)},
            model_discrete=discrete,
            n_bins_config=31,
        )
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    lambdas = dict.fromkeys(layout.penalty_names, 3.0)
    coefficient_face = build_penalty_face(layout, [layout.penalty_names[0]]) if face else None
    penalty = layout.penalty_matrix(lambdas)
    fit = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        plan,
        penalty,
        config=DenseSolverConfig(max_iterations=300, tolerance=1e-11),
        chunk_size=chunk,
        coefficient_face=coefficient_face,
    )
    assert fit.converged
    cross_rows = _curvature_packed(
        family, y, fit.eta, tuple(state.link for state in layout.predictors), plan
    )[:, 1]
    assert np.min(cross_rows) < 0 < np.max(cross_rows)
    matrices = dense_predictor_matrices(layout)
    expected = laml_derivatives(
        family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=matrices
    )
    original_toarray = DesignMatrix.toarray
    seen = []

    def bounded_toarray(self):
        assert self.n <= chunk, "derivatives expanded more than one bounded design chunk"
        seen.append(self.n)
        return original_toarray(self)

    monkeypatch.setattr(DesignMatrix, "toarray", bounded_toarray)
    bounded = bounded_predictor_matrices(layout, chunk_size=chunk)
    original_zeros = np.zeros

    def bounded_zeros(shape, *args, **kwargs):
        assert shape != (n, len(expected.names), 3), "allocated full row/component curvature tensor"
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", bounded_zeros)
    original_stack = np.stack

    def bounded_stack(arrays, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else 0)
        assert not (
            axis == 1 and len(arrays) == 2 and arrays[0].shape == (n, len(expected.names))
        ), "allocated full row/component direction stack"
        return original_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "stack", bounded_stack)
    workspace = LamlDerivativeWorkspace()
    gradient = laml_derivatives(
        family,
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=bounded,
        want_hessian=False,
        reuse=workspace,
    )
    actual = laml_derivatives(
        family,
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=bounded,
        reuse=workspace,
    )
    assert seen and max(seen) == chunk
    assert actual.names == expected.names
    assert actual.provenance == expected.provenance
    assert actual.evaluations + gradient.evaluations == expected.evaluations
    np.testing.assert_array_equal(actual.gradient, gradient.gradient)
    # Only row reduction ordering differs; the coefficient solve and family
    # stencils are identical. Include its conditioning in the roundoff bound.
    eigenvalues = np.linalg.eigvalsh(fit.terminal_pseudo_inverse())
    retained = eigenvalues[-fit.terminal_rank.rank :]
    condition = retained[-1] / retained[0]
    for name in ("gradient", "hessian", "gradient_certificate", "hessian_certificate"):
        reference = getattr(expected, name)
        scale = max(1.0, np.linalg.norm(reference))
        tolerance = 128 * n * np.finfo(float).eps * max(condition, 1.0) * scale
        np.testing.assert_allclose(getattr(actual, name), reference, rtol=0, atol=tolerance)
