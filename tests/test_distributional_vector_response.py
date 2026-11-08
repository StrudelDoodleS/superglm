import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.results.solver import DenseSolverConfig
from superglm.distributional.weights import UnsupportedLikelihoodContractError, WeightContract

from ._dirichlet_oracle import fit_reference
from .test_dirichlet_lss_family import family


@pytest.mark.parametrize("k,regression", [(2, False), (3, False), (3, True), (5, False)])
def test_fixed_and_null_fit_against_independent_stationary_likelihood(k, regression):
    rng = np.random.default_rng(152 + k)
    x = np.linspace(-1, 1, 100)
    design = np.column_stack((np.ones(len(x)), x)) if regression else np.ones((len(x), 1))
    truth = np.linspace(1.2, 3.7, k)[None, :] * np.exp(x[:, None] * np.linspace(-0.4, 0.4, k))
    y = np.array([rng.dirichlet(a) for a in truth])
    weights = rng.integers(1, 4, len(x))
    fam = family(k)
    X = pd.DataFrame({"x": x}, index=np.arange(len(x)) + 70)
    frame = pd.DataFrame(y, columns=fam.components, index=X.index)
    model = SuperLSS(
        fam,
        *(fam.alpha(c, *(("x",) if regression else ())) for c in fam.components),
        weight_semantics="frequency",
    )
    model.fit(X, frame[frame.columns[::-1]], sample_weight=weights)
    beta, objective = fit_reference(y, design, weights)
    fitted = model.result_.coefficients
    # Check the endpoint before comparing it to the reference.
    assert np.linalg.norm(objective(fitted)[1], ord=np.inf) / weights.sum() < 2e-7
    want = np.exp(design @ beta.reshape(k, -1).T)
    actual = model.predict_parameters(X).to_numpy()
    public_mean = model.predict(X)
    assert isinstance(public_mean, pd.DataFrame)
    assert tuple(public_mean.columns) == fam.components
    pd.testing.assert_index_equal(public_mean.index, X.index)
    np.testing.assert_allclose(
        public_mean, want / want.sum(axis=1, keepdims=True), rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(
        actual / actual.sum(axis=1, keepdims=True),
        want / want.sum(axis=1, keepdims=True),
        rtol=2e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(objective(fitted)[0], objective(beta)[0], rtol=1e-10, atol=1e-8)
    # A composition projection uses all cross-component coefficient covariances.
    mu = want[0] / want[0].sum()
    jac = np.concatenate([mu[0] * ((j == 0) - mu[j]) * design[0] for j in range(k)])
    oracle_cov = np.linalg.inv(objective(beta)[2])
    np.testing.assert_allclose(jac @ model.covariance_ @ jac, jac @ oracle_cov @ jac, rtol=3e-6)
    state = model._model.fit_state
    assert state.retained_rows.response.shape == y.shape
    null_beta, null_objective = fit_reference(y, np.ones((len(y), 1)), weights)
    assert (
        np.linalg.norm(null_objective(state.null_model.coefficients)[1], ord=np.inf) / weights.sum()
        < 2e-7
    )
    np.testing.assert_allclose(state.null_model.objective, null_objective(null_beta)[0], rtol=1e-9)


def test_zero_weight_nullable_inputs_keep_original_positions():
    fam = family()
    X = pd.DataFrame({"x": np.arange(8.0)}, index=np.arange(30, 38))
    y = pd.DataFrame(
        [[0.2, 0.3, 0.5], [0.3, 0.25, 0.45], [0.25, 0.35, 0.4], [0.6, 0.2, 0.2]] * 2,
        columns=fam.components,
        index=X.index,
    ).astype("Float64")
    y.iloc[2, :] = pd.NA
    weights = np.ones(8)
    weights[2] = 0
    model = SuperLSS(fam, *(fam.alpha(c) for c in fam.components), weight_semantics="frequency")
    model.fit(X, y, sample_weight=weights)
    rows = model._model.fit_state.retained_rows
    np.testing.assert_array_equal(rows.likelihood_weights.input_positions, [0, 1, 3, 4, 5, 6, 7])
    assert rows.response.shape == (7, 3)
    with pytest.raises(ValueError, match="index"):
        model.fit(X, y.iloc[::-1], sample_weight=weights)
    with pytest.raises(ValueError, match="component|columns"):
        model.fit(
            X, y.rename(columns={fam.components[0]: fam.components[1]}), sample_weight=weights
        )


@pytest.mark.parametrize("discrete,retain", [(False, False), (True, True), (True, False)])
def test_compact_and_chunked_fixed_fit_replay(discrete, retain):
    rng = np.random.default_rng(411)
    y = rng.dirichlet([1.2, 2.3, 3.1], 45)
    X = pd.DataFrame({"x": np.linspace(-1, 1, len(y))})
    fam = family()
    model = SuperLSS(
        fam,
        *(fam.alpha(c, "x") for c in fam.components),
        weight_semantics="frequency",
        discrete=discrete,
    )
    model.fit(X, y, retain_rows=retain)
    design = np.column_stack((np.ones(len(y)), X.x))
    beta, objective = fit_reference(y, design, np.ones(len(y)))
    assert np.linalg.norm(objective(model.result_.coefficients)[1], ord=np.inf) < 2e-5
    np.testing.assert_allclose(
        model.predict_parameters(X).to_numpy(), np.exp(design @ beta.reshape(3, 2).T), rtol=4e-6
    )
    assert (model._model.fit_state.retained_rows is not None) == retain


def test_permuted_component_fit_and_literal_frequency_replication():
    rng = np.random.default_rng(341)
    y = rng.dirichlet([0.8, 2.3, 3.1], 40)
    weights = rng.integers(1, 4, len(y))
    X = pd.DataFrame({"x": np.linspace(-1, 1, len(y))})
    fam = family()
    model = SuperLSS(
        fam, *(fam.alpha(c) for c in fam.components), weight_semantics="frequency"
    ).fit(X, y, sample_weight=weights)
    order = [2, 0, 1]
    alternate = family(components=tuple(fam.components[i] for i in order))
    other = SuperLSS(
        alternate, *(alternate.alpha(c) for c in alternate.components), weight_semantics="frequency"
    ).fit(X, pd.DataFrame(y, columns=fam.components), sample_weight=weights)
    np.testing.assert_allclose(
        other.predict_parameters(X).to_numpy(),
        model.predict_parameters(X).to_numpy()[:, order],
        rtol=2e-7,
    )
    repeat = np.repeat(np.arange(len(y)), weights)
    expanded = SuperLSS(
        fam, *(fam.alpha(c) for c in fam.components), weight_semantics="frequency"
    ).fit(X.iloc[repeat], y[repeat])
    np.testing.assert_allclose(expanded.result_.coefficients, model.result_.coefficients, rtol=2e-7)
    np.testing.assert_allclose(expanded.covariance_, model.covariance_, rtol=3e-7)


def test_explicit_initial_coefficients_and_invalid_range():
    fam = family()
    X = pd.DataFrame({"x": np.arange(30)})
    y = np.random.default_rng(72).dirichlet([1.2, 2.3, 3.4], len(X))
    args = dict(
        family=fam,
        predictors=tuple(fam.alpha(c).template for c in fam.components),
        weight_contract=WeightContract(semantics="frequency"),
        lambdas={},
    )
    model = fit_dense_distributional(X, y, initial=np.log([1.2, 2.3, 3.4]), **args)
    beta, objective = fit_reference(y, np.ones((len(y), 1)), np.ones(len(y)))
    assert np.linalg.norm(objective(model.coefficients)[1], ord=np.inf) < 1e-5
    np.testing.assert_allclose(model.predict_parameters(X)[0], np.exp(beta), rtol=2e-6)
    with pytest.raises(ValueError, match="initial|finite|valid"):
        fit_dense_distributional(X, y, initial=np.full(3, 1000.0), **args)


def test_constant_shares_do_not_claim_finite_concentration_mle():
    fam = family()
    X = pd.DataFrame({"x": np.arange(12)})
    y = np.tile([0.2, 0.3, 0.5], (len(X), 1))
    model = SuperLSS(fam, *(fam.alpha(c) for c in fam.components), weight_semantics="frequency")
    model.fit(X, y, max_inner_iter=5)
    assert not model.result_.converged
    assert np.isfinite(model.predict_parameters(X)).all().all()
    assert any("concentration" in message for message in model.result_.warnings)


def test_scalar_response_contract_stays_strict_and_bounds_refuse_vectors():
    from superglm import GaussianLS

    X = pd.DataFrame({"x": np.arange(4)})
    y = np.tile([0.2, 0.3, 0.5], (len(X), 1))
    fam = family()
    model = SuperLSS(fam, *(fam.alpha(c) for c in fam.components))
    with pytest.raises(UnsupportedLikelihoodContractError, match="vector"):
        model.fit(X, y, truncation=(0.0, 1.0))
    gaussian = GaussianLS()
    scalar = SuperLSS(gaussian, gaussian.location(), gaussian.scale())
    with pytest.raises(ValueError, match="shape"):
        scalar.fit(X, y)
    with pytest.raises(ValueError, match="shape"):
        model.fit(X, y.ravel())


def test_original_row_reconstruction_keeps_vector_shape():
    from dataclasses import replace

    from superglm.distributional.fit_state import _curvature_disclosures

    fam = family()
    X = pd.DataFrame({"x": np.arange(8)})
    y = np.tile([0.2, 0.3, 0.5], (len(X), 1))
    weights = np.ones(len(X))
    weights[2] = 0
    model = fit_dense_distributional(
        X,
        y,
        family=fam,
        predictors=tuple(fam.alpha(c).template for c in fam.components),
        weight_contract=WeightContract(semantics="frequency"),
        sample_weight=weights,
        lambdas={},
        config=DenseSolverConfig(max_iterations=2),
    )
    result = model.result
    forced = replace(
        result,
        terminal_curvature=replace(
            result.terminal_curvature, fallback_count=1, actual_source="fisher"
        ),
    )
    retained = model.fit_state.retained_rows
    from .test_dirichlet_lss_family import bind

    plan = fam.bind_likelihood(
        retained.response, retained.likelihood_weights, bind(fam, retained.response).observation
    )
    messages = _curvature_disclosures(fam, forced, retained.response, plan, model.layout)
    assert any("concentration" in message for message in messages)


@pytest.mark.parametrize("select", [False, True])
def test_empty_scalar_boundaries_still_record_thin_and_weightless_factor_levels(select):
    from superglm import FactorSmooth, LambdaPolicy, interaction, s
    from superglm._frame import as_eager_frame
    from superglm.distributional.predictor import compile_predictors
    from superglm.distributional.separation import record_factor_smooth_levels
    from superglm.distributional.weights import resolve_likelihood_weights

    X = pd.DataFrame(
        {
            "x": [*np.linspace(0, 1, 12), *np.linspace(0, 1, 12), 0.3],
            "g": ["a"] * 12 + ["b"] * 12 + ["thin"],
        }
    )
    y = np.random.default_rng(122).dirichlet([2, 3, 4], len(X))
    fam = family()
    term = FactorSmooth(
        "x",
        group="g",
        basis="sz",
        k=5,
        levels=["a", "b", "thin", "empty"] if select else ["a", "b", "thin"],
        select=select,
        lambda_policy={
            name: LambdaPolicy.fixed(1.0)
            for name in (("wiggle", "null") if select else ("wiggle",))
        },
    )
    predictors = (
        fam.alpha(fam.components[0], s("x", k=5), interaction(term)),
        fam.alpha(fam.components[1]),
        fam.alpha(fam.components[2]),
    )
    weights = resolve_likelihood_weights(
        None, n_observations=len(X), contract=WeightContract(semantics="frequency")
    )
    compiled = compile_predictors(
        as_eager_frame(X),
        weights,
        fam.parameters,
        tuple(p.template for p in predictors),
        separation_boundaries=((), (), ()),
        separation_response=y,
    )
    record_factor_smooth_levels(
        compiled, y, weights.values, boundaries=((), (), ()), policy="ignore", stacklevel=1
    )
    spec = next(iter(compiled[0].compiled.interaction_specs.values()))
    if select:
        assert set(spec._weightless_levels) == {3}
    else:
        assert 2 in spec._unidentified_levels
    assert spec._separated_levels == ()


def test_chunked_fisher_fit_and_dense_observed_fit_have_the_same_stationary_target():
    from superglm.distributional.solver._likelihood_cache import build_likelihood_cache

    from .test_dirichlet_lss_family import bind

    fam = family()
    y = np.random.default_rng(88).dirichlet([0.8, 1.3, 2.4], 42)
    X = pd.DataFrame({"x": np.linspace(-1, 1, len(y))})
    args = dict(
        family=fam,
        predictors=tuple(fam.alpha(c, "x").template for c in fam.components),
        weight_contract=WeightContract(semantics="frequency"),
        lambdas={},
    )
    observed = fit_dense_distributional(X, y, **args)
    fisher = fit_dense_distributional(
        X, y, chunk_size=7, config=DenseSolverConfig(coefficient_curvature="fisher"), **args
    )
    _, objective = fit_reference(y, np.column_stack((np.ones(len(y)), X.x)), np.ones(len(y)))
    assert np.linalg.norm(objective(fisher.coefficients)[1], ord=np.inf) / len(y) < 2e-7
    np.testing.assert_allclose(
        fisher.predict_parameters(X), observed.predict_parameters(X), rtol=3e-6
    )
    assert build_likelihood_cache(fam, bind(fam, y)) is None


def test_canonical_family_response_cannot_hide_invalid_supplied_values():
    from superglm.distributional.solver.chunks import _validate_bound_likelihood

    from .test_dirichlet_lss_family import bind

    class IgnoringValidator(type(family())):
        def validate_likelihood_plan(self, y, plan):
            return plan.exact_response

    fam = IgnoringValidator(("a", "b", "c"))
    y = np.array([[0.2, 0.3, 0.5]])
    plan = bind(fam, y)
    for bad in (np.array([[np.nan, 0.3, 0.5]]), np.array([["0.2", "0.3", "0.5"]])):
        with pytest.raises(UnsupportedLikelihoodContractError):
            _validate_bound_likelihood(fam, plan, bad)
