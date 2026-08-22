"""Exact IRLS parity for the scalar structured backend."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest
import scipy.sparse as sp

import superglm.reml.objective as reml_objective
import superglm.solvers.irls_direct as irls_direct
from superglm.distributions import (
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
)
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedTensorGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import IdentityLink, LogLink
from superglm.profiling.tweedie import generate_tweedie_cpg
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.hessian_factor import HessianFactor
from superglm.solvers.structured import (
    SymmetricBlockOperator,
    materialize_compact_operator,
    resolve_structured_backend,
)
from superglm.types import (
    GroupSlice,
    LinearConstraintSet,
    PenaltyComponent,
)


def _structured_problem(
    response_factory: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray],
):
    rng = np.random.default_rng(701)
    n = 420
    n_levels = 24
    codes = rng.integers(0, n_levels, size=n, dtype=np.intp)
    numeric = rng.normal(size=(n, 2))
    offset = rng.normal(scale=0.08, size=n)
    random_truth = rng.normal(scale=0.22, size=n_levels)
    linear_predictor = -0.25 + numeric @ np.array([0.3, -0.18]) + random_truth[codes] + offset
    y = response_factory(rng, linear_predictor, offset)
    weights = rng.uniform(0.4, 2.2, size=n)
    matrices = [
        DenseGroupMatrix(numeric),
        RandomEffectGroupMatrix(codes, n_levels),
    ]
    groups = [
        GroupSlice(name="numeric", start=0, end=2, penalized=False),
        GroupSlice(name="policy", start=2, end=2 + n_levels, penalized=True),
    ]
    dm = DesignMatrix(matrices, n=n, p=2 + n_levels)
    penalties = [
        PenaltyComponent(
            name="policy",
            group_name="policy",
            group_index=1,
            group_sl=groups[1].sl,
            omega_raw=None,
            penalty_kind="identity",
        )
    ]
    return dm, groups, penalties, y, weights, offset


def _gaussian_response(rng, linear_predictor, offset):
    del offset
    return linear_predictor + rng.normal(scale=0.12, size=len(linear_predictor))


def _poisson_response(rng, linear_predictor, offset):
    del offset
    return rng.poisson(np.exp(linear_predictor)).astype(np.float64)


def _gamma_response(rng, linear_predictor, offset):
    del offset
    mean = np.exp(linear_predictor)
    return rng.gamma(shape=3.0, scale=mean / 3.0)


def _nb2_response(rng, linear_predictor, offset):
    del offset
    mean = np.exp(linear_predictor)
    theta = 3.5
    return rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float64)


def _tweedie_response(rng, linear_predictor, offset):
    del offset
    return generate_tweedie_cpg(
        len(linear_predictor),
        mu=np.exp(linear_predictor),
        phi=0.8,
        p=1.5,
        rng=rng,
    )


@pytest.mark.parametrize(
    ("family", "link", "response_factory"),
    [
        pytest.param(Gaussian(), IdentityLink(), _gaussian_response, id="gaussian"),
        pytest.param(Poisson(), LogLink(), _poisson_response, id="poisson"),
        pytest.param(Gamma(), LogLink(), _gamma_response, id="gamma"),
        pytest.param(
            NegativeBinomial(theta=3.5),
            LogLink(),
            _nb2_response,
            id="negative_binomial",
        ),
        pytest.param(Tweedie(p=1.5), LogLink(), _tweedie_response, id="tweedie"),
    ],
)
def test_forced_structured_exact_irls_matches_dense_oracle(
    family,
    link,
    response_factory,
):
    dm, groups, penalties, y, weights, offset = _structured_problem(response_factory)
    lambdas = {"policy": 2.75}
    dense_profile: dict = {}
    structured_profile: dict = {}
    structured_cache: dict = {}

    dense_result, dense_factor, dense_gram = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        max_iter=100,
        tol=1e-10,
        return_xtwx=True,
        direct_solve="gram",
        reml_penalties=penalties,
        profile=dense_profile,
        weight_semantics="prior",
    )
    structured_result, structured_factor, structured_operator = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        max_iter=100,
        tol=1e-10,
        return_xtwx=True,
        direct_solve="structured",
        reml_penalties=penalties,
        profile=structured_profile,
        cache_out=structured_cache,
        weight_semantics="prior",
    )

    np.testing.assert_allclose(
        structured_result.beta,
        dense_result.beta,
        rtol=2e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        structured_result.intercept,
        dense_result.intercept,
        rtol=2e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        structured_result.deviance,
        dense_result.deviance,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        structured_result.effective_df,
        dense_result.effective_df,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        structured_result.log_det_H,
        dense_result.log_det_H,
        rtol=2e-9,
        atol=2e-9,
    )
    assert structured_result.n_iter == dense_result.n_iter
    assert structured_result.converged == dense_result.converged
    assert isinstance(structured_factor, HessianFactor)
    assert isinstance(structured_operator, SymmetricBlockOperator)
    np.testing.assert_allclose(
        structured_factor.solve(np.eye(dm.p)),
        dense_factor,
        rtol=2e-8,
        atol=2e-9,
    )
    materialized_operator = np.zeros_like(dense_gram)
    small = structured_operator.small_indices
    dominant = structured_operator.structured_indices
    materialized_operator[np.ix_(small, small)] = structured_operator.A
    materialized_operator[np.ix_(dominant, small)] = structured_operator.C
    materialized_operator[np.ix_(small, dominant)] = structured_operator.C.T
    materialized_operator[dominant, dominant] = structured_operator.d
    # The dense centered-system reconstruction can leave cancellation dust in
    # analytically zero off-diagonal cells of the one-hot block.
    np.testing.assert_allclose(materialized_operator, dense_gram, atol=2e-14)

    assert structured_result.direct_backend == "structured"
    assert structured_profile["direct_backend"] == "structured"
    assert "XtWX" not in structured_cache
    assert isinstance(structured_cache["structured_operator"], SymmetricBlockOperator)


def test_forced_structured_avoids_dense_gram_and_penalty_builders(monkeypatch):
    dm, groups, penalties, y, weights, offset = _structured_problem(_gaussian_response)

    def fail_dense_path(*args, **kwargs):
        raise AssertionError("structured IRLS entered a dense p x p builder")

    monkeypatch.setattr(irls_direct, "build_centered_system", fail_dense_path)
    monkeypatch.setattr(irls_direct, "_build_penalty_matrix", fail_dense_path)

    result, factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2={"policy": 2.75},
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        weight_semantics="frequency",
    )

    assert result.converged
    assert isinstance(factor, HessianFactor)


def test_forced_structured_rejects_constrained_coefficients():
    dm, groups, penalties, y, weights, offset = _structured_problem(_gaussian_response)
    groups[0].constraints = LinearConstraintSet(
        A=np.array([[1.0, 0.0]]),
        b=np.zeros(1),
    )

    with pytest.raises(ValueError, match="constraint"):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2={"policy": 2.75},
            offset=offset,
            direct_solve="structured",
            reml_penalties=penalties,
            weight_semantics="frequency",
        )


def test_structured_irls_handles_all_small_group_kernels_and_penalties():
    rng = np.random.default_rng(414)
    n = 260
    offset = rng.normal(scale=0.05, size=n)
    dominant_codes = rng.integers(0, 17, size=n, dtype=np.intp)
    second_codes = rng.integers(0, 5, size=n, dtype=np.intp)
    categorical_codes = rng.integers(-1, 3, size=n, dtype=np.intp)
    numeric = DenseGroupMatrix(rng.normal(size=(n, 2)))
    categorical = CategoricalGroupMatrix(categorical_codes, n_levels=3)

    spline_basis = sp.csr_matrix(rng.normal(size=(n, 3)))
    spline = SparseSSPGroupMatrix(spline_basis, np.eye(3))
    spline_omega = np.diag([0.0, 1.0, 2.0])
    spline.omega = spline_omega

    B1 = rng.normal(size=(3, 2))
    B2 = rng.normal(size=(2, 2))
    idx1 = np.arange(n, dtype=np.intp) % 3
    idx2 = (np.arange(n, dtype=np.intp) // 3) % 2
    pair_idx = idx1 * 2 + idx2
    B_joint = np.vstack([np.kron(B1[i], B2[j]) for i in range(3) for j in range(2)])
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        np.eye(4),
        pair_idx,
        tensor_id=37,
    )
    tensor_omega = np.diag([0.5, 1.0, 1.5, 2.0])
    tensor.omega = tensor_omega

    matrices = [
        RandomEffectGroupMatrix(second_codes, n_levels=5),
        numeric,
        categorical,
        spline,
        tensor,
        RandomEffectGroupMatrix(dominant_codes, n_levels=17),
    ]
    groups: list[GroupSlice] = []
    start = 0
    names = ["branch", "numeric", "category", "spline", "tensor", "policy"]
    for name, matrix in zip(names, matrices, strict=True):
        end = start + matrix.shape[1]
        groups.append(
            GroupSlice(
                name=name,
                start=start,
                end=end,
                penalized=name in {"branch", "spline", "tensor", "policy"},
            )
        )
        start = end
    dm = DesignMatrix(matrices, n=n, p=start)
    penalties = [
        PenaltyComponent(
            name="branch",
            group_name="branch",
            group_index=0,
            group_sl=groups[0].sl,
            omega_raw=None,
            penalty_kind="identity",
        ),
        PenaltyComponent(
            name="spline",
            group_name="spline",
            group_index=3,
            group_sl=groups[3].sl,
            omega_raw=spline_omega,
            omega_ssp=spline_omega,
        ),
        PenaltyComponent(
            name="tensor",
            group_name="tensor",
            group_index=4,
            group_sl=groups[4].sl,
            omega_raw=tensor_omega,
            omega_ssp=tensor_omega,
        ),
        PenaltyComponent(
            name="policy",
            group_name="policy",
            group_index=5,
            group_sl=groups[5].sl,
            omega_raw=None,
            penalty_kind="identity",
        ),
    ]
    lambdas = {
        "branch": 1.2,
        "spline": 0.8,
        "tensor": 1.4,
        "policy": 2.1,
    }
    truth = rng.normal(scale=0.08, size=dm.p)
    y = 0.4 + dm.matvec(truth) + offset + rng.normal(scale=0.1, size=n)
    weights = rng.uniform(0.5, 1.7, size=n)

    dense_result, dense_inverse = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        reml_penalties=penalties,
        tol=1e-11,
        weight_semantics="frequency",
    )
    profile: dict = {}
    structured_result, structured_factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        profile=profile,
        tol=1e-11,
        weight_semantics="frequency",
    )

    np.testing.assert_allclose(structured_result.beta, dense_result.beta, atol=2e-9)
    np.testing.assert_allclose(
        structured_result.intercept,
        dense_result.intercept,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        structured_factor.solve(np.eye(dm.p)),
        dense_inverse,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        structured_result.effective_df,
        dense_result.effective_df,
        atol=2e-9,
    )
    assert profile["structured_dominant_group"] == "policy"

    S = build_penalty_matrix(
        matrices,
        groups,
        lambdas,
        dm.p,
        reml_penalties=penalties,
    )
    override_result, override_factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        S_override=S,
        tol=1e-11,
        weight_semantics="frequency",
    )
    np.testing.assert_allclose(override_result.beta, structured_result.beta, atol=2e-9)
    np.testing.assert_allclose(
        override_factor.solve(np.eye(dm.p)),
        structured_factor.solve(np.eye(dm.p)),
        atol=2e-9,
    )


def test_structured_correlated_override_rejects_dominant_random_effect_block():
    dm, groups, _penalties, y, weights, offset = _structured_problem(_gaussian_response)
    dominant = np.arange(groups[1].start, groups[1].end, dtype=np.intp)
    diagonal_penalty = np.zeros((dm.p, dm.p), dtype=np.float64)
    diagonal_penalty[dominant, dominant] = 1.3
    correlated_penalty = diagonal_penalty.copy()
    correlated_penalty[dominant[0], dominant[1]] = 0.2
    correlated_penalty[dominant[1], dominant[0]] = 0.2

    with pytest.raises(
        ValueError,
        match=r"S_override.*dominant RandomEffect block.*diagonal",
    ):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2={"policy": 1.3},
            offset=offset,
            direct_solve="structured",
            S_override=correlated_penalty,
            tol=1.0e-11,
            weight_semantics="frequency",
        )

    structured, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2={"policy": 1.3},
        offset=offset,
        direct_solve="structured",
        S_override=diagonal_penalty,
        tol=1.0e-11,
        weight_semantics="frequency",
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2={"policy": 1.3},
        offset=offset,
        direct_solve="gram",
        S_override=diagonal_penalty,
        tol=1.0e-11,
        weight_semantics="frequency",
    )
    np.testing.assert_allclose(structured.beta, gram.beta, atol=2.0e-9)


def test_structured_tiny_scaled_correlated_override_is_not_silently_diagonalized():
    dm, groups, _penalties, y, weights, offset = _structured_problem(_gaussian_response)
    dominant = np.arange(groups[1].start, groups[1].end, dtype=np.intp)
    correlated_penalty = np.zeros((dm.p, dm.p), dtype=np.float64)
    correlated_penalty[0, 0] = 1.0
    correlated_penalty[dominant, dominant] = 1.3e-13
    correlated_penalty[dominant[0], dominant[1]] = 0.2e-14
    correlated_penalty[dominant[1], dominant[0]] = 0.2e-14

    with pytest.raises(
        ValueError,
        match=r"S_override.*dominant RandomEffect block.*diagonal",
    ):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2={"policy": 1.3e-13},
            offset=offset,
            direct_solve="structured",
            S_override=correlated_penalty,
            tol=1.0e-11,
            weight_semantics="frequency",
        )


@pytest.mark.parametrize("direct_solve", ["auto", "structured"])
@pytest.mark.parametrize("lambda2", [0.0, {}])
def test_structured_override_is_authoritative_for_zero_lambda_eligibility(
    direct_solve: str,
    lambda2: float | dict[str, float],
):
    base_dm, _base_groups, _penalties, y, weights, offset = _structured_problem(_gaussian_response)
    n_levels = 40
    dm = DesignMatrix(
        [
            base_dm.group_matrices[0],
            RandomEffectGroupMatrix(base_dm.group_matrices[1].codes, n_levels),
        ],
        n=base_dm.n,
        p=2 + n_levels,
    )
    groups = [
        GroupSlice(name="numeric", start=0, end=2, penalized=False),
        GroupSlice(name="policy", start=2, end=2 + n_levels, penalized=True),
    ]
    dominant = np.arange(groups[1].start, groups[1].end, dtype=np.intp)
    diagonal_penalty = np.zeros((dm.p, dm.p), dtype=np.float64)
    diagonal_penalty[dominant, dominant] = 1.3

    structured, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambda2,
        offset=offset,
        direct_solve=direct_solve,
        S_override=diagonal_penalty,
        tol=1.0e-11,
        weight_semantics="frequency",
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambda2,
        offset=offset,
        direct_solve="gram",
        S_override=diagonal_penalty,
        tol=1.0e-11,
        weight_semantics="frequency",
    )

    assert structured.direct_backend == "structured"
    assert structured.direct_fallback_reason is None
    np.testing.assert_allclose(structured.beta, gram.beta, atol=2.0e-9)


@pytest.mark.parametrize("unsupported_geometry", ["dominant_correlation", "cross_block"])
def test_auto_falls_back_for_incompatible_authoritative_override(
    unsupported_geometry: str,
):
    base_dm, _base_groups, _penalties, y, weights, offset = _structured_problem(_gaussian_response)
    n_levels = 40
    dm = DesignMatrix(
        [
            base_dm.group_matrices[0],
            RandomEffectGroupMatrix(base_dm.group_matrices[1].codes, n_levels),
        ],
        n=base_dm.n,
        p=2 + n_levels,
    )
    groups = [
        GroupSlice(name="numeric", start=0, end=2, penalized=False),
        GroupSlice(name="policy", start=2, end=2 + n_levels, penalized=True),
    ]
    dominant = np.arange(groups[1].start, groups[1].end, dtype=np.intp)
    penalty = np.zeros((dm.p, dm.p), dtype=np.float64)
    penalty[dominant, dominant] = 1.3
    if unsupported_geometry == "dominant_correlation":
        penalty[dominant[0], dominant[1]] = 0.2
        penalty[dominant[1], dominant[0]] = 0.2
    else:
        penalty[0, 0] = 1.0e12
        penalty[1, 1] = 1.0
        penalty[dominant[0], 1] = 1.0e-3
        penalty[1, dominant[0]] = 1.0e-3

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=0.0,
        offset=offset,
        direct_solve="auto",
        S_override=penalty,
        tol=1.0e-11,
        weight_semantics="frequency",
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=0.0,
        offset=offset,
        direct_solve="gram",
        S_override=penalty,
        tol=1.0e-11,
        weight_semantics="frequency",
    )

    assert automatic.direct_backend == "gram"
    assert "S_override" in automatic.direct_fallback_reason
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=2.0e-9)


def test_auto_records_dense_fallback_reason_for_constraints():
    dm, groups, penalties, y, weights, offset = _structured_problem(_gaussian_response)
    groups[0].constraints = LinearConstraintSet(
        A=np.array([[1.0, 0.0]]),
        b=np.array([-10.0]),
    )
    profile: dict = {}

    result, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2={"policy": 2.75},
        offset=offset,
        direct_solve="auto",
        reml_penalties=penalties,
        profile=profile,
        weight_semantics="frequency",
    )

    assert result.direct_backend == "gram"
    assert "constraint" in result.direct_fallback_reason.lower()
    assert profile["direct_fallback_reason"] == result.direct_fallback_reason


@pytest.mark.parametrize(
    ("dominant_width", "small_width", "expected_structured"),
    [
        pytest.param(20, 4, False, id="small-total-width-stays-dense"),
        pytest.param(30, 4, True, id="measured-scalar-crossover"),
        # Ratio ((q+1)/(p+1))**2 = 0.26 here.  A real ~67k-row fit measured the
        # structured backend ~1.7x SLOWER end to end at this shape class
        # (issue #343): the factorization the ratio prices is a small minority
        # of per-iteration work beside the shared O(n) moment build, so a wide
        # dense border must stay on the dense path.
        pytest.param(20, 20, False, id="wide-border-stays-dense"),
        pytest.param(4, 28, False, id="insufficient-schur-cost-reduction"),
        # Ratio 0.0009: the dominant block spans nearly the whole width.  This
        # is the measured-win regime (1.5x-3.9x faster on real and synthetic
        # fits) that the recalibrated bound must keep structured.
        pytest.param(300, 8, True, id="dominant-block-spans-width"),
    ],
)
def test_auto_backend_uses_measured_structured_crossover(
    dominant_width: int,
    small_width: int,
    expected_structured: bool,
):
    n = max(80, dominant_width * 2)
    matrices = [
        DenseGroupMatrix(np.ones((n, small_width))),
        RandomEffectGroupMatrix(np.arange(n) % dominant_width, dominant_width),
    ]
    groups = [
        GroupSlice(name="numeric", start=0, end=small_width, penalized=False),
        GroupSlice(
            name="policy",
            start=small_width,
            end=small_width + dominant_width,
            penalized=True,
        ),
    ]

    decision = resolve_structured_backend(
        matrices,
        groups,
        direct_solve="auto",
        coefficient_width=small_width + dominant_width,
    )

    assert decision.use_structured is expected_structured
    if expected_structured:
        assert decision.fallback_reason is None
        assert decision.group_name == "policy"
    else:
        assert "crossover" in decision.fallback_reason
    # Either way the cost model ran, and its prediction must be published for
    # calibration.  For the scalar geometry it has closed form ((q+1)/(p+1))^2.
    expected_ratio = ((small_width + 1) / (small_width + dominant_width + 1)) ** 2
    assert decision.auto_cost_ratio == pytest.approx(expected_ratio)


@pytest.mark.parametrize(
    ("dominant_width", "small_width", "expect_structured"),
    [
        pytest.param(300, 8, True, id="pick-structured"),
        pytest.param(20, 20, False, id="decline-on-cost"),
    ],
)
def test_auto_fit_publishes_predicted_cost_ratio_in_profile(
    dominant_width: int,
    small_width: int,
    expect_structured: bool,
):
    """Issue #343: every automatic cost decision lands in the fit profile.

    The profile pairs the crossover model's prediction with the realized
    per-phase timings recorded by the same fit, which is the record a
    recalibration against real workloads reads.
    """
    rng = np.random.default_rng(343)
    n = 4 * dominant_width
    codes = np.asarray(np.arange(n) % dominant_width, dtype=np.intp)
    numeric = rng.normal(size=(n, small_width))
    y = 0.05 * numeric[:, 0] + rng.normal(scale=0.3, size=n)
    matrices = [
        DenseGroupMatrix(numeric),
        RandomEffectGroupMatrix(codes, dominant_width),
    ]
    groups = [
        GroupSlice(name="numeric", start=0, end=small_width, penalized=False),
        GroupSlice(
            name="policy",
            start=small_width,
            end=small_width + dominant_width,
            penalized=True,
        ),
    ]
    dm = DesignMatrix(matrices, n=n, p=small_width + dominant_width)
    penalties = [
        PenaltyComponent(
            name="policy",
            group_name="policy",
            group_index=1,
            group_sl=groups[1].sl,
            omega_raw=None,
            penalty_kind="identity",
        )
    ]
    profile: dict = {}

    result, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=np.ones(n),
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2={"policy": 1.5},
        direct_solve="auto",
        reml_penalties=penalties,
        profile=profile,
        weight_semantics="frequency",
    )

    expected_backend = "structured" if expect_structured else "gram"
    assert result.direct_backend == expected_backend
    assert profile["structured_auto_selected"] is expect_structured
    expected_ratio = ((small_width + 1) / (small_width + dominant_width + 1)) ** 2
    assert profile["structured_auto_cost_ratio"] == pytest.approx(expected_ratio)


def test_record_auto_backend_decision_logs_only_structured_auto_picks(caplog):
    """One INFO line per automatic structured pick; silence otherwise."""
    from superglm.solvers.structured import (
        StructuredBackendDecision,
        record_auto_backend_decision,
    )

    pick = StructuredBackendDecision(
        use_structured=True,
        group_index=1,
        group_name="policy",
        fallback_reason=None,
        auto_cost_ratio=0.01,
    )
    profile: dict = {}
    with caplog.at_level(logging.INFO, logger="superglm.solvers._structured.selection"):
        record_auto_backend_decision(profile, "auto", pick)
    assert profile == {
        "structured_auto_cost_ratio": 0.01,
        "structured_auto_selected": True,
    }
    assert sum("chose the structured backend" in r.message for r in caplog.records) == 1

    caplog.clear()
    decline = StructuredBackendDecision(
        use_structured=False,
        group_index=1,
        group_name="policy",
        fallback_reason="below the measured structured crossover",
        auto_cost_ratio=0.26,
    )
    quiet_profile: dict = {}
    with caplog.at_level(logging.INFO, logger="superglm.solvers._structured.selection"):
        record_auto_backend_decision(quiet_profile, "auto", decline)
        # Forced backends and eligibility fallbacks carry no prediction and
        # must write nothing.
        record_auto_backend_decision(quiet_profile, "structured", pick)
        no_ratio = StructuredBackendDecision(
            use_structured=False,
            group_index=None,
            group_name=None,
            fallback_reason="no structured term",
        )
        record_auto_backend_decision({}, "auto", no_ratio)
    assert quiet_profile == {
        "structured_auto_cost_ratio": 0.26,
        "structured_auto_selected": False,
    }
    assert not caplog.records


@pytest.mark.parametrize("discrete", [False, True], ids=["exact-driver", "discrete-driver"])
def test_reml_driver_emits_one_info_line_per_structured_auto_pick(caplog, discrete):
    """Issue #343: the fit-owning REML driver emits the INFO line exactly once.

    The inner PIRLS solves re-resolve the same decision many times per fit and
    must stay quiet (``log=False``); the driver-level call is the only INFO
    emission point.  This pins the driver integration end to end -- deleting
    the ``record_auto_backend_decision`` call in either REML driver, or
    promoting the per-solve call to ``log=True``, must fail this test.
    """
    import pandas as pd

    from superglm import Categorical, RandomEffect, SuperGLM
    from superglm.distributions import Gaussian as GaussianFamily

    rng = np.random.default_rng(3430)
    n, n_levels = 900, 300
    codes = rng.integers(0, n_levels, size=n)
    cats = rng.integers(0, 5, size=n)
    frame = pd.DataFrame(
        {
            "grp": np.array([f"L{code:03d}" for code in codes], dtype=object),
            "cat": np.array([f"v{code}" for code in cats], dtype=object),
        }
    )
    y = 0.2 + 0.4 * rng.normal(size=n_levels)[codes] + rng.normal(scale=0.3, size=n)

    model = SuperGLM(
        family=GaussianFamily(),
        selection_penalty=0.0,
        features={"cat": Categorical(), "grp": RandomEffect()},
        direct_solve="auto",
        discrete=discrete,
    )
    with caplog.at_level(logging.INFO, logger="superglm.solvers._structured.selection"):
        model.fit_reml(frame, y, max_reml_iter=3, runtime_validation="skip")

    profile = model.reml_diagnostics()["profile"]
    assert profile["direct_backend"] == "structured"
    assert profile["structured_auto_selected"] is True
    q = 4
    observed_levels = frame["grp"].nunique()
    expected_ratio = ((q + 1) / (observed_levels + q + 1)) ** 2
    assert profile["structured_auto_cost_ratio"] == pytest.approx(expected_ratio)
    picks = [r for r in caplog.records if "chose the structured backend" in r.message]
    assert len(picks) == 1


def test_auto_missing_compact_penalties_falls_back_but_forced_rejects():
    rng = np.random.default_rng(20260727)
    n_levels = 40
    codes = np.repeat(np.arange(n_levels), 5)
    dm = DesignMatrix(
        [RandomEffectGroupMatrix(codes, n_levels)],
        n=len(codes),
        p=n_levels,
    )
    groups = [
        GroupSlice(
            name="policy",
            start=0,
            end=n_levels,
            penalized=True,
        )
    ]
    y = rng.normal(scale=0.2, size=len(codes))
    weights = np.ones(len(codes))

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2={"policy": 1.0},
        direct_solve="auto",
        weight_semantics="frequency",
    )

    assert automatic.direct_backend == "gram"
    assert "compact reml_penalties" in automatic.direct_fallback_reason
    with pytest.raises(
        ValueError,
        match=r"direct_solve='structured'.*compact reml_penalties",
    ):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2={"policy": 1.0},
            direct_solve="structured",
            weight_semantics="frequency",
        )


def test_structured_factor_matches_dense_fixed_weight_reml_derivatives():
    dm, groups, penalties, y, weights, offset = _structured_problem(_poisson_response)
    lambdas = {"policy": 2.75}
    dense_result, dense_inverse = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        reml_penalties=penalties,
        tol=1e-10,
        weight_semantics="frequency",
    )
    structured_result, structured_factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        tol=1e-10,
        weight_semantics="frequency",
    )

    dense_gradient = reml_direct_gradient(
        dm.group_matrices,
        dense_result,
        dense_inverse,
        lambdas,
        reml_penalties=penalties,
    )
    structured_gradient = reml_direct_gradient(
        dm.group_matrices,
        structured_result,
        structured_factor,
        lambdas,
        reml_penalties=penalties,
    )
    np.testing.assert_allclose(structured_gradient, dense_gradient, atol=2e-10)

    dense_hessian = reml_direct_hessian(
        dm.group_matrices,
        Poisson(),
        dense_inverse,
        lambdas,
        gradient=dense_gradient,
        reml_penalties=penalties,
    )
    structured_hessian = reml_direct_hessian(
        dm.group_matrices,
        Poisson(),
        structured_factor,
        lambdas,
        gradient=structured_gradient,
        reml_penalties=penalties,
    )
    np.testing.assert_allclose(structured_hessian, dense_hessian, atol=2e-10)


def test_structured_w_derivatives_match_dense_first_and_second_order():
    dm, groups, penalties, y, weights, offset = _structured_problem(_poisson_response)
    lambdas = {"policy": 2.75}
    dense_result, dense_inverse = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        reml_penalties=penalties,
        tol=1e-10,
        weight_semantics="frequency",
    )
    structured_result, structured_factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        tol=1e-10,
        weight_semantics="frequency",
    )
    dense_correction = reml_w_correction(
        dm,
        LogLink(),
        groups,
        dense_result,
        dense_inverse,
        lambdas,
        sample_weight=weights,
        offset_arr=offset,
        distribution=Poisson(),
        w_correction_order=2,
        reml_penalties=penalties,
    )
    structured_correction = reml_w_correction(
        dm,
        LogLink(),
        groups,
        structured_result,
        structured_factor,
        lambdas,
        sample_weight=weights,
        offset_arr=offset,
        distribution=Poisson(),
        w_correction_order=2,
        reml_penalties=penalties,
    )
    assert dense_correction is not None
    assert structured_correction is not None
    dense_gradient_correction, dense_operators, dense_second = dense_correction
    structured_gradient_correction, structured_operators, structured_second = structured_correction
    np.testing.assert_allclose(
        structured_gradient_correction,
        dense_gradient_correction,
        atol=3e-9,
    )
    for index, dense_operator in dense_operators.items():
        np.testing.assert_allclose(
            materialize_compact_operator(structured_operators[index]),
            dense_operator,
            atol=2e-9,
        )
    np.testing.assert_allclose(structured_second, dense_second, atol=2e-8)

    dense_partial = reml_direct_gradient(
        dm.group_matrices,
        dense_result,
        dense_inverse,
        lambdas,
        reml_penalties=penalties,
    )
    structured_partial = reml_direct_gradient(
        dm.group_matrices,
        structured_result,
        structured_factor,
        lambdas,
        reml_penalties=penalties,
    )
    dense_hessian = reml_direct_hessian(
        dm.group_matrices,
        Poisson(),
        dense_inverse,
        lambdas,
        gradient=dense_partial,
        dH_extra=dense_operators,
        dH2_cross=dense_second,
        reml_penalties=penalties,
    )
    structured_hessian = reml_direct_hessian(
        dm.group_matrices,
        Poisson(),
        structured_factor,
        lambdas,
        gradient=structured_partial,
        dH_extra=structured_operators,
        dH2_cross=structured_second,
        reml_penalties=penalties,
    )
    np.testing.assert_allclose(structured_hessian, dense_hessian, atol=3e-8)


def test_structured_reml_objective_uses_compact_penalty_and_gram(monkeypatch):
    dm, groups, penalties, y, weights, offset = _structured_problem(_poisson_response)
    lambdas = {"policy": 2.75}
    dense_result, _, dense_gram = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        reml_penalties=penalties,
        return_xtwx=True,
        tol=1e-10,
        weight_semantics="frequency",
    )
    structured_result, _, structured_gram = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        return_xtwx=True,
        tol=1e-10,
        weight_semantics="frequency",
    )
    dense_penalty = build_penalty_matrix(
        dm.group_matrices,
        groups,
        lambdas,
        dm.p,
        reml_penalties=penalties,
    )
    dense_value = reml_objective.reml_laml_objective(
        dm,
        Poisson(),
        LogLink(),
        groups,
        y,
        dense_result,
        lambdas,
        weights,
        offset,
        XtWX=dense_gram,
        log_det_H=dense_result.log_det_H,
        S_override=dense_penalty,
        reml_penalties=penalties,
        weight_semantics="frequency",
    )

    def fail_dense_penalty(*args, **kwargs):
        raise AssertionError("structured objective expanded a dense penalty")

    monkeypatch.setattr(
        reml_objective,
        "build_penalty_matrix",
        fail_dense_penalty,
    )
    structured_value = reml_objective.reml_laml_objective(
        dm,
        Poisson(),
        LogLink(),
        groups,
        y,
        structured_result,
        lambdas,
        weights,
        offset,
        XtWX=structured_gram,
        log_det_H=structured_result.log_det_H,
        reml_penalties=penalties,
        weight_semantics="frequency",
    )
    np.testing.assert_allclose(structured_value, dense_value, atol=2e-9)


class TestStructuredPenaltyBuilderComponentLambdas:
    """The legacy (``reml_penalties=None``) branches of the structured penalty
    builders must resolve component-named lambdas for multi-penalty groups,
    matching the dense assembly sites fixed on this branch. These branches are
    defensive today (``fit_irls_direct`` refuses structured dispatch without
    ``reml_penalties``), but the builders are exported and must agree with the
    dense semantics."""

    def _system_with_multipenalty_small_block(self):
        import scipy.sparse as sp

        from superglm.group_matrix import SparseSSPGroupMatrix
        from superglm.solvers.structured import build_structured_system

        rng = np.random.default_rng(11)
        n = 200
        n_levels = 40
        codes = rng.integers(0, n_levels, size=n, dtype=np.intp)
        B = rng.normal(size=(n, 3))
        gm_small = SparseSSPGroupMatrix(sp.csr_matrix(B), np.eye(3))
        U1 = rng.normal(size=(3, 2))
        U2 = rng.normal(size=(3, 1))
        omega_1 = U1 @ U1.T
        omega_2 = U2 @ U2.T
        gm_small.omega = omega_1 + omega_2
        gm_small.omega_components = [("m1", omega_1), ("m2", omega_2)]
        matrices = [gm_small, RandomEffectGroupMatrix(codes, n_levels)]
        groups = [
            GroupSlice(name="s", start=0, end=3, penalized=True),
            GroupSlice(name="policy", start=3, end=3 + n_levels, penalized=True),
        ]
        W = rng.uniform(0.5, 1.5, size=n)
        Wz = rng.normal(size=n)
        system = build_structured_system(matrices, groups, W, Wz, dominant_group_index=1)
        return system, matrices, groups, omega_1, omega_2

    def test_scalar_builder_applies_component_named_lambdas(self):
        from superglm.solvers.structured import build_penalized_scalar_operator

        system, matrices, groups, omega_1, omega_2 = self._system_with_multipenalty_small_block()
        lambdas = {"s:m1": 2.0, "s:m2": 3.0, "policy": 1.5}
        penalized = build_penalized_scalar_operator(system, matrices, groups, lambdas)

        base = np.asarray(system.operator.A, dtype=np.float64)
        small_position = {
            int(idx): pos for pos, idx in enumerate(np.asarray(system.operator.small_indices))
        }
        local = [small_position[i] for i in range(3)]
        expected_block = base[np.ix_(local, local)] + 2.0 * omega_1 + 3.0 * omega_2
        np.testing.assert_allclose(
            np.asarray(penalized.A)[np.ix_(local, local)], expected_block, rtol=1e-12
        )

    def _system_with_fs_dominant(self, factor_basis):
        import scipy.sparse as sp

        from superglm.group_matrix import SparseSSPGroupMatrix
        from superglm.solvers.structured import build_structured_system
        from tests.test_factor_smooth_structured_system import _dominant

        rng = np.random.default_rng(12)
        dominant = _dominant(discrete=False, factor_basis=factor_basis)
        n = dominant.shape[0]
        B = rng.normal(size=(n, 3))
        gm_small = SparseSSPGroupMatrix(sp.csr_matrix(B), np.eye(3))
        U1 = rng.normal(size=(3, 2))
        U2 = rng.normal(size=(3, 1))
        omega_1 = U1 @ U1.T
        omega_2 = U2 @ U2.T
        gm_small.omega = omega_1 + omega_2
        gm_small.omega_components = [("m1", omega_1), ("m2", omega_2)]
        matrices = [gm_small, dominant]
        groups = [
            GroupSlice(name="s", start=0, end=3, penalized=True),
            GroupSlice(name="f", start=3, end=3 + dominant.shape[1], penalized=True),
        ]
        W = rng.uniform(0.5, 1.5, size=n)
        Wz = rng.normal(size=n)
        system = build_structured_system(matrices, groups, W, Wz, dominant_group_index=1)
        return system, matrices, groups, omega_1, omega_2

    def _assert_small_block_penalized(self, system, penalized, omega_1, omega_2):
        base = np.asarray(system.operator.A, dtype=np.float64)
        small_position = {
            int(idx): pos for pos, idx in enumerate(np.asarray(system.operator.small_indices))
        }
        local = [small_position[i] for i in range(3)]
        expected_block = base[np.ix_(local, local)] + 2.0 * omega_1 + 3.0 * omega_2
        np.testing.assert_allclose(
            np.asarray(penalized.A)[np.ix_(local, local)], expected_block, rtol=1e-12
        )

    def test_block_builder_applies_component_named_lambdas(self):
        from superglm.solvers._structured.moments import BlockStructuredSystem
        from superglm.solvers.structured import build_penalized_block_operator

        system, matrices, groups, omega_1, omega_2 = self._system_with_fs_dominant("fs")
        assert isinstance(system, BlockStructuredSystem)
        lambdas = {"s:m1": 2.0, "s:m2": 3.0, "f": 1.5}
        penalized = build_penalized_block_operator(system, matrices, groups, lambdas)
        self._assert_small_block_penalized(system, penalized, omega_1, omega_2)

    def test_sum_to_zero_builder_applies_component_named_lambdas(self):
        from superglm.solvers._structured.moments import SumToZeroBlockStructuredSystem
        from superglm.solvers.structured import build_penalized_sum_to_zero_operator

        system, matrices, groups, omega_1, omega_2 = self._system_with_fs_dominant("sz")
        assert isinstance(system, SumToZeroBlockStructuredSystem)
        lambdas = {"s:m1": 2.0, "s:m2": 3.0, "f": 1.5}
        penalized = build_penalized_sum_to_zero_operator(system, matrices, groups, lambdas)
        self._assert_small_block_penalized(system, penalized, omega_1, omega_2)
