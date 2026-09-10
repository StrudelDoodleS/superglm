"""Public SZ system geometry is independent of a uniform curvature unit."""

import numpy as np
import pytest

from superglm.solvers.sum_to_zero import SumToZeroBlockFactor, SumToZeroIdentifiabilityError


@pytest.mark.parametrize("scale", [2.0**-400, 1.0, 2.0**400])
@pytest.mark.parametrize("ordinary", [0, 2])
def test_wide_well_conditioned_public_rank_is_certified_without_a_public_matrix(
    monkeypatch, scale, ordinary
):
    import superglm.solvers.sum_to_zero as sz

    levels, width = 130, 2
    p = ordinary + (levels - 1) * width
    C = np.zeros((levels, width, ordinary))
    if ordinary:
        C[:] = (
            np.arange(levels)[:, None, None]
            / levels
            * np.array([[0.125, 0.0625], [0.03125, -0.125]])
        )
    A = np.eye(ordinary) + np.einsum("kiq,kir->qr", C, C)
    D = np.broadcast_to(np.eye(width), (levels, width, width)).copy()
    small = np.arange(ordinary)
    structured = np.arange(ordinary, p).reshape(levels - 1, width)
    gram = sz.decompose_gram
    zeros = sz.np.zeros

    def only_auxiliary_rank(matrix, **kwargs):
        assert kwargs.get("allow_indefinite") is True
        assert matrix.shape[0] < p
        return gram(matrix, **kwargs)

    def no_public_zeros(shape, *args, **kwargs):
        assert shape != (p, p)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(sz, "decompose_gram", only_auxiliary_rank)
    monkeypatch.setattr(sz.np, "zeros", no_public_zeros)
    factor = SumToZeroBlockFactor(
        A=scale * A,
        C=scale * C,
        D=scale * D,
        small_indices=small,
        structured_indices=structured,
        term_name="wide-identified",
    )
    assert p > factor.max_structured_inverse_block
    assert factor.rank == p
    inverse_cache = factor._border_inverse_cache
    assert inverse_cache is not None
    assert factor._border_inverse() is inverse_cache
    expected = np.ones(p)
    rhs = np.empty(p)
    cross = C[:-1] - C[-1]
    rhs[small] = A @ expected[small] + np.einsum("kiq,ki->q", cross, expected[structured])
    rhs[structured] = (
        expected[structured] + np.sum(expected[structured], axis=0) + cross @ expected[small]
    )
    solution = factor.solve(scale * rhs)
    # The raw Schur complement is I and ||C||_2 < sqrt(levels)/4;
    # the free-coordinate map has squared condition levels.
    gamma = (p + width + ordinary) * np.finfo(float).eps
    np.testing.assert_allclose(solution, expected, rtol=0.0, atol=32 * levels * gamma)


def test_unresolved_small_public_rank_calls_the_shared_gram_authority(monkeypatch):
    import superglm.solvers.sum_to_zero as sz

    gram = sz.decompose_gram
    public_calls = []

    def record_public(matrix, **kwargs):
        if not kwargs.get("allow_indefinite", False):
            public_calls.append(matrix.copy())
        return gram(matrix, **kwargs)

    monkeypatch.setattr(
        SumToZeroBlockFactor, "_compact_public_rank_certificate", lambda self: False
    )
    monkeypatch.setattr(sz, "decompose_gram", record_public)
    factor = SumToZeroBlockFactor(
        A=np.array([[1.0]]),
        C=np.array([[[0.0]], [[0.5]]]),
        D=np.array([[[0.0]], [[1.0]]]),
        small_indices=np.array([0]),
        structured_indices=np.array([[1]]),
        term_name="small-fallback",
        max_structured_inverse_block=2,
    )
    assert factor.rank == 2
    assert len(public_calls) == 1
    np.testing.assert_array_equal(public_calls[0], [[1.0, -0.5], [-0.5, 1.0]])


def test_unresolved_wide_public_rank_refuses_without_a_public_matrix(monkeypatch):
    import superglm.solvers.sum_to_zero as sz

    levels = 260
    tail = 2.0**-80
    weights = np.ones(levels)
    weights[-1] = tail
    gram = sz.decompose_gram
    zeros = sz.np.zeros

    def only_auxiliary_rank(matrix, **kwargs):
        assert kwargs.get("allow_indefinite") is True
        return gram(matrix, **kwargs)

    def no_public_zeros(shape, *args, **kwargs):
        assert shape != (levels, levels)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(sz, "decompose_gram", only_auxiliary_rank)
    monkeypatch.setattr(sz.np, "zeros", no_public_zeros)
    with pytest.raises(SumToZeroIdentifiabilityError, match="public numerical rank is unresolved"):
        SumToZeroBlockFactor(
            A=np.array([[np.sum(weights)]]),
            C=weights[:, None, None],
            D=weights[:, None, None],
            small_indices=np.array([0]),
            structured_indices=np.arange(1, levels)[:, None],
            term_name="wide-tiny-level",
        )


def test_public_rank_bound_checks_the_candidate_inverse_residual():
    factor = SumToZeroBlockFactor(
        A=np.empty((0, 0)),
        C=np.empty((2, 1, 0)),
        D=np.ones((2, 1, 1)),
        small_indices=np.array([], dtype=int),
        structured_indices=np.array([[0]]),
        term_name="inverse-residual",
    )
    assert factor._compact_public_rank_certificate()
    # The local inverse alone is twice the public inverse. Full auxiliary
    # rank and a small local condition do not certify this candidate.
    factor._border_inverse_cache[:] = 0.0
    assert not factor._compact_public_rank_certificate()


@pytest.mark.parametrize("scale", [2.0**-400, 1.0, 2.0**400])
@pytest.mark.parametrize("tail", [2.0**-80, 2.0**-20])
def test_public_rank_cannot_be_replaced_by_the_whitened_border_rank(scale, tail):
    from superglm.solvers.rank import decompose_factor, decompose_gram

    # A global intercept and a sum-to-zero two-level effect become collinear
    # when the second level loses support. Auxiliary multiplier units do not
    # restore a public coefficient direction below the shared rank cutoff.
    root = np.sqrt(scale) * np.array([[1.0, 1.0], [np.sqrt(tail), -np.sqrt(tail)]])
    public = scale * np.array([[1.0 + tail, 1.0 - tail], [1.0 - tail, 1.0 + tail]])
    expected_rank = 1 if tail == 2.0**-80 else 2
    assert decompose_factor(root).rank == expected_rank
    assert decompose_gram(public).rank == expected_rank

    kwargs = dict(
        A=np.array([[scale * (1.0 + tail)]]),
        C=scale * np.array([[[1.0]], [[tail]]]),
        D=scale * np.array([[[1.0]], [[tail]]]),
        small_indices=np.array([0]),
        structured_indices=np.array([[1]]),
        term_name="two-level-intercept",
    )
    if expected_rank == 1:
        with pytest.raises(SumToZeroIdentifiabilityError, match="globally unidentifiable"):
            SumToZeroBlockFactor(**kwargs)
    else:
        factor = SumToZeroBlockFactor(**kwargs)
        assert factor.rank == 2
        solution = factor.solve(public @ np.ones(2))
        allowance = 128 * np.finfo(float).eps * np.linalg.cond(public / scale)
        np.testing.assert_allclose(solution, np.ones(2), rtol=0.0, atol=allowance)


@pytest.mark.parametrize("scale", [1e-20, 1e-12, 1.0, 1e12, 1e20])
def test_scaled_condition_two_public_system_retains_solve_and_determinant(scale):
    factor = SumToZeroBlockFactor(
        A=np.array([[scale]]),
        C=np.zeros((2, 1, 1)),
        D=np.full((2, 1, 1), scale),
        small_indices=np.array([0]),
        structured_indices=np.array([[1]]),
        term_name="scaled",
    )
    rhs = np.array([1.0, 2.0])
    answer = factor.solve(scale * rhs)
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(answer, [1.0, 1.0], rtol=allowance, atol=allowance)
    np.testing.assert_allclose(np.diag([1.0, 2.0]) @ answer, rhs, rtol=allowance)
    np.testing.assert_allclose(
        scale * factor.solve(np.eye(2)), np.diag([1.0, 0.5]), rtol=allowance, atol=allowance
    )
    expected_logdet = 2 * np.log(scale) + np.log(2.0)
    log_allowance = allowance * (2 * abs(np.log(scale)) + abs(np.log(2.0)))
    assert abs(factor.logdet() - expected_logdet) <= log_allowance


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_scaled_globally_unidentifiable_sz_system_still_refuses(scale):
    with pytest.raises(SumToZeroIdentifiabilityError):
        SumToZeroBlockFactor(
            A=np.array([[scale]]),
            C=np.zeros((2, 1, 1)),
            D=np.zeros((2, 1, 1)),
            small_indices=np.array([0]),
            structured_indices=np.array([[1]]),
            term_name="unidentifiable",
        )


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_scaled_negative_local_curvature_still_refuses(scale):
    with pytest.raises(np.linalg.LinAlgError, match="negative local curvature"):
        SumToZeroBlockFactor(
            A=np.array([[scale]]),
            C=np.zeros((2, 1, 1)),
            D=np.full((2, 1, 1), -scale),
            small_indices=np.array([0]),
            structured_indices=np.array([[1]]),
            term_name="indefinite",
        )


@pytest.mark.parametrize("scale", [1e-120, 1e-32, 1e-20, 1.0, 1e20, 1e32, 1e120])
@pytest.mark.parametrize("ordinary", [False, True])
def test_constraint_identifies_a_scaled_zero_local_block(scale, ordinary):
    factor = SumToZeroBlockFactor(
        A=np.array([[scale]]) if ordinary else np.empty((0, 0)),
        C=np.zeros((2, 1, 1 if ordinary else 0)),
        D=np.array([[[0.0]], [[scale]]]),
        small_indices=np.array([0] if ordinary else [], dtype=int),
        structured_indices=np.array([[1]] if ordinary else [[0]]),
        term_name="identified-by-constraint",
    )
    expected = np.array([1.0, 2.0] if ordinary else [2.0])
    answer = factor.solve(scale * expected)
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(answer, expected, rtol=allowance, atol=allowance)
    log_scale = len(expected) * np.log(scale)
    assert abs(factor.logdet() - log_scale) <= allowance * (1 + abs(log_scale))


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("ordinary", [False, True])
def test_material_input_asymmetry_does_not_depend_on_block_units(scale, ordinary):
    malformed = scale * np.array([[1.0, 0.25], [0.0, 1.0]])
    with pytest.raises(ValueError, match="must be symmetric"):
        SumToZeroBlockFactor(
            A=malformed if ordinary else np.array([[scale]]),
            C=np.zeros((2, 1, 2) if ordinary else (2, 2, 1)),
            D=np.full((2, 1, 1), scale) if ordinary else np.stack([malformed] * 2),
            small_indices=np.array([0, 1] if ordinary else [0]),
            structured_indices=np.array([[2]] if ordinary else [[1, 2]]),
            term_name="malformed",
        )


@pytest.mark.parametrize("scale", [1e-120, 1e-32, 1.0, 1e120])
def test_coupled_singular_local_border_preserves_condition_three_public_geometry(scale):
    factor = SumToZeroBlockFactor(
        A=np.array([[scale]]),
        C=np.array([[[0.0]], [[0.5 * scale]]]),
        D=np.array([[[0.0]], [[scale]]]),
        small_indices=np.array([0]),
        structured_indices=np.array([[1]]),
        term_name="coupled-identified",
    )
    public_unit = np.array([[1.0, -0.5], [-0.5, 1.0]])
    expected = np.array([1.0, 2.0])
    answer = factor.solve(scale * (public_unit @ expected))
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(answer, expected, rtol=allowance, atol=allowance)
    expected_logdet = 2 * np.log(scale) + np.log(0.75)
    assert abs(factor.logdet() - expected_logdet) <= allowance * (1 + abs(expected_logdet))
