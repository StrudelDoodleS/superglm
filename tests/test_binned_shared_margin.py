"""Shared tensor/level margins: dispatch counts and independent product oracles."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.features import Categorical, Spline
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)


def _model():
    rng = np.random.default_rng(7302)
    n = 480
    frame = pd.DataFrame(
        {"x": rng.uniform(size=n), "y": rng.uniform(size=n), "c": np.resize(list("ABC"), n)}
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        discrete=True,
        n_bins=16,
        features={"x": Spline(k=5), "y": Spline(k=5), "c": Categorical(base="first")},
        interactions=[("x", "y"), ("x", "c"), ("y", "c")],
    )
    model._build_design_matrix(frame, rng.normal(size=n), np.ones(n), None)
    return model


def test_builder_shared_margins_eliminate_weighted_row_channels(monkeypatch):
    model = _model()
    contributions = []
    original = algebra._level_tensor_channels

    def counted(rows, *args):
        contributions.append(len(rows))
        return original(rows, *args)

    monkeypatch.setattr(algebra, "_level_tensor_channels", counted)
    model._dm.execution_plan.moments(np.linspace(0.5, 1.5, model._dm.n))
    assert contributions == [], f"Still accumulated row channels for {sum(contributions)} rows"


def _pair(margin=2, *, certify=True, clipped=False):
    from superglm._group_matrix._binned_codes import certify_binned_codes

    rng = np.random.default_rng(7303)
    n, n1, n2 = 73, 5, 7
    A, B = rng.normal(size=(n1, 2)), rng.normal(size=(n2, 3))
    idx1, idx2 = rng.integers(n1, size=n), rng.integers(n2, size=n)
    pair_idx = idx1 * n2 + idx2
    joint = np.einsum("ai,bj->abij", A, B).reshape(n1 * n2, 6)
    tensor = DiscretizedTensorGroupMatrix(
        A, B, idx1, idx2, joint, np.linalg.qr(rng.normal(size=(6, 4)))[0], pair_idx, 7303
    )
    rows = np.flatnonzero(np.arange(n) % 3 != 0)
    codes = (idx1 if margin == 1 else idx2)[rows].copy()
    bins = n1 if margin == 1 else n2
    if clipped:
        codes = np.minimum(codes, bins - 2)
    category = DiscretizedSplineCategoricalGroupMatrix(
        rng.normal(size=(bins, 3)),
        np.linalg.qr(rng.normal(size=(3, 2)))[0],
        codes,
        rows,
        n_rows=n,
        bin_idx_is_level=True,
    )
    if certify:
        certify_binned_codes([tensor, category], n)
    return tensor, category, rng.uniform(0.1, 2.0, n)


@pytest.mark.parametrize("margin", [1, 2])
def test_shared_margin_cross_against_high_precision_row_oracle(margin):
    import mpmath as mp

    tensor, category, W = _pair(margin)
    profile = {}
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, category, W, profile=profile)
    assert profile["block_cross_tensor_shared_level_calls"] == 1
    # Independent exact-input, 80-digit arithmetic: build each row, then its
    # outer product. Neither histogram nor support contraction is the oracle.
    with mp.workdps(80):
        RT = mp.matrix(tensor.R_inv.tolist())
        RC = mp.matrix(category.R_inv.tolist())
        expected = mp.zeros(4, 2)
        for k, r in enumerate(category.row_idx):
            row_t = (
                mp.matrix(
                    [
                        [
                            mp.mpf(float(a)) * mp.mpf(float(b))
                            for a in tensor.B1_unique_t[tensor.idx1[r]]
                            for b in tensor.B2_unique_t[tensor.idx2[r]]
                        ]
                    ]
                )
                * RT
            )
            row_c = mp.matrix([category.B_unique[category.bin_idx_level[k]].tolist()]) * RC
            expected += mp.mpf(float(W[r])) * row_t.T * row_c
        reference = np.array(expected.tolist(), dtype=float)
    # Higham's absolute-product envelope, including both outer transforms.
    A = np.abs(tensor.B1_unique_t[tensor.idx1[category.row_idx]])
    B = np.abs(tensor.B2_unique_t[tensor.idx2[category.row_idx]])
    T = np.einsum("ri,rj->rij", A, B).reshape(len(A), 6) @ np.abs(tensor.R_inv)
    C = np.abs(category.B_unique[category.bin_idx_level]) @ np.abs(category.R_inv)
    scale = T.T @ (W[category.row_idx, None] * C)
    count = len(W) + tensor.n_bins1 + tensor.n_bins2 + 6 + 3 + 6
    u = np.finfo(float).eps / 2
    gamma = count * u / (1 - count * u)
    assert np.all(np.abs(actual - reference) <= 8 * gamma * scale)


@pytest.mark.parametrize("margin", [1, 2])
def test_shared_margin_predictions_se_edf_and_rank_match_independent_dense_rows(margin):
    tensor, category, W = _pair(margin)
    dm = DesignMatrix([tensor, category], len(W), 6)
    X = dm.toarray()
    A = np.column_stack((np.ones(len(W)), X))
    z = np.sin(np.arange(len(W)))
    moments = dm.execution_plan.moments(W, rhs=(W * z,), include_xtw=True)
    G = np.block([[np.array([[W.sum()]]), moments.xtw[None]], [moments.xtw[:, None], moments.gram]])
    reference_gram = A.T @ (W[:, None] * A)
    penalty = np.diag([0.0, *np.ones(6)])
    covariance = np.linalg.inv(G + penalty)
    reference_covariance = np.linalg.inv(reference_gram + penalty)
    beta = covariance @ np.r_[W @ z, moments.xt_rhs[0]]
    reference_beta = reference_covariance @ (A.T @ (W * z))
    # This well-conditioned fixture tests forward outputs; the tolerance
    # includes accumulation and solve conditioning, not BLAS rounding signs.
    tolerance = 32 * np.finfo(float).eps * len(W) * np.linalg.cond(reference_gram + penalty)
    np.testing.assert_allclose(A @ beta, A @ reference_beta, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        np.sqrt(np.einsum("ij,jk,ik->i", A, covariance, A)),
        np.sqrt(np.einsum("ij,jk,ik->i", A, reference_covariance, A)),
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        np.trace(covariance @ G), np.trace(reference_covariance @ reference_gram), rtol=tolerance
    )
    assert np.linalg.matrix_rank(G) == np.linalg.matrix_rank(A) == 7


@pytest.mark.parametrize("mutation", ["clipped", "local_codes", "rows", "tensor_codes"])
def test_uncertified_or_changed_margin_keeps_channel_path_and_correct_values(monkeypatch, mutation):
    tensor, category, W = _pair(clipped=mutation == "clipped")
    if mutation == "local_codes":
        category.bin_idx_level = (category.bin_idx_level + 1) % category.n_bins
    elif mutation == "rows":
        category.row_idx = category.row_idx[::-1].copy()
    elif mutation == "tensor_codes":
        tensor.idx2 = (tensor.idx2 + 1) % tensor.n_bins2
    calls = []
    original = algebra._level_tensor_channels

    def counted(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(algebra, "_level_tensor_channels", counted)
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, category, W)
    assert calls == [len(category.row_idx)]
    literal_tensor = (
        np.einsum(
            "ri,rj->rij", tensor.B1_unique_t[tensor.idx1], tensor.B2_unique_t[tensor.idx2]
        ).reshape(len(W), 6)
        @ tensor.R_inv
    )
    expected = literal_tensor.T @ (W[:, None] * category.toarray())
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("weights", ["signed", "extreme", "extreme_factors"])
def test_signed_and_exceptional_weights_preserve_original_channel_association(monkeypatch, weights):
    tensor, category, W = _pair()
    if weights == "signed":
        W[::2] *= -1
    elif weights == "extreme":
        W *= 2.0**600
        tensor.B2_unique_t *= 2.0**-600
    else:
        # The old route first multiplies W by the tiny second margin. Moving
        # the huge first margin into H.T @ B1 would overflow before B2 could
        # rescale it, even though the literal row tensor is exactly one.
        W[:] = 1.0
        tensor.B1_unique_t[:] = 2.0**1022
        tensor.B2_unique_t[:] = 2.0**-1022
    calls = []
    original = algebra._level_tensor_channels

    def counted(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(algebra, "_level_tensor_channels", counted)
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, category, W)
    assert calls == [len(category.row_idx)]
    # Removal of the code certificate forces the old path independently of
    # range guards; compare its exceptional association bit for bit.
    category._bin_code_source = None
    reference = algebra._cross_gram_tensor_spline_categorical(tensor, category, W)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, reference)


def test_scalar_histogram_and_canonical_full_codes_do_not_outlive_weights_or_basis(monkeypatch):
    from superglm._group_matrix._binned_codes import certify_binned_codes

    tensor, category, W = _pair(certify=False)
    main1 = DiscretizedSSPGroupMatrix(tensor.B1_unique_t, np.eye(2), tensor.idx1.copy())
    main2 = DiscretizedSSPGroupMatrix(tensor.B2_unique_t, np.eye(3), tensor.idx2.copy())
    certify_binned_codes([main1, main2, tensor, category], len(W))
    calls = []
    original = algebra._disc_disc_2d_hist

    def counted(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(algebra, "_disc_disc_2d_hist", counted)
    for step in range(2):
        cache = algebra._BlockWeightCache()
        H = cache.disc_disc_hist(main1.bin_idx, main2.bin_idx, W, main1.n_bins, main2.n_bins)
        np.testing.assert_array_equal(cache.tensor_w_grid(tensor, W), H)
        assert len(calls) == step + 1  # One producer per assembly; no copied-weight identity key.
        actual = algebra._cross_gram_tensor_spline_categorical(tensor, category, W, cache)
        expected = tensor.toarray().T @ (W[:, None] * category.toarray())
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        W[:] = W[::-1]
        category.R_inv *= 0.75


def test_reordered_subsets_recertify_the_actual_new_code_correspondence():
    from superglm._group_matrix._binned_codes import certify_binned_codes

    tensor, category, W = _pair()
    rows = np.array([31, 8, 31, 29, 5, 17, 32])
    sub_t, sub_c = tensor.row_subset(rows), category.row_subset(rows)
    certify_binned_codes([sub_t, sub_c], len(rows))
    expected = tensor.toarray()[rows].T @ (W[rows, None] * category.toarray()[rows])
    actual = algebra._cross_gram_tensor_spline_categorical(sub_t, sub_c, W[rows])
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_whitening_rebuild_preserves_certification_and_pool_is_deterministic():
    from superglm._group_matrix._block_queue import block_queue_config

    model = _model()
    dm = rebuild_design_matrix_with_lambdas(
        model._dm,
        model._groups,
        {g.name: 2.0 for g in model._groups},
        np.ones(model._dm.n),
        1.0,
    )
    W = np.linspace(0.5, 1.5, dm.n)
    for group in dm.group_matrices:
        if getattr(group, "_bin_code_source", None) is not None:
            with pytest.raises(ValueError):
                group.bin_idx_level.setflags(write=True)
    with block_queue_config(workers=1):
        profile = {}
        serial = dm.execution_plan.moments(W, profile=profile).gram
    with block_queue_config(workers=3, min_cost=0):
        pooled = dm.execution_plan.moments(W).gram
    assert profile["block_cross_tensor_shared_level_calls"] == 4
    np.testing.assert_array_equal(pooled, serial)


def test_shared_histogram_budget_decline_keeps_the_original_route(monkeypatch):
    tensor, category, W = _pair()
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 1)
    profile = {}
    actual = algebra._cross_gram_tensor_spline_categorical(tensor, category, W, profile=profile)
    assert not profile.get("block_cross_tensor_shared_level_calls")
    np.testing.assert_allclose(
        actual, tensor.toarray().T @ (W[:, None] * category.toarray()), rtol=2e-13, atol=2e-13
    )
