"""Focused regressions for the legacy fixed-point REML runner."""

from types import SimpleNamespace

import numpy as np


def test_direct_fit_caches_authoritative_centered_gram() -> None:
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.types import GroupSlice

    x = np.linspace(-1.2, 1.5, 41)
    X = (x + 1.0e8)[:, None]
    y = 2.0 + 0.7 * x
    cache: dict[str, np.ndarray | float] = {}

    fit_irls_direct(
        X=X,
        y=y,
        weights=np.ones_like(y),
        family=Gaussian(),
        link=IdentityLink(),
        groups=[GroupSlice(name="x", start=0, end=1, penalized=False)],
        lambda2=0.0,
        S_override=np.array([[0.25]]),
        cache_out=cache,
    )

    centered = X - np.mean(X, axis=0)
    np.testing.assert_allclose(
        cache["centered_XtWX"],
        centered.T @ centered,
        rtol=2e-9,
        atol=2e-9,
    )


def test_cached_direct_gram_uses_authoritative_profiled_geometry() -> None:
    from superglm.reml.runner import _center_cached_direct_gram
    from superglm.solvers.irls_direct import _safe_decompose_H

    x = np.linspace(-1.2, 1.5, 17)
    X = np.column_stack((x + 9.0, x**2 - 4.0))
    weights = np.linspace(0.7, 2.1, len(x))
    sum_w = float(np.sum(weights))
    mean_x = (X.T @ weights) / sum_w
    centered_design = X - mean_x
    authoritative_gram = centered_design.T @ (weights[:, None] * centered_design)
    result = SimpleNamespace(rank_info=SimpleNamespace(sum_w=sum_w, mean_x=mean_x))
    penalty = np.diag([0.8, 1.3])

    centered_gram = _center_cached_direct_gram(authoritative_gram, result)
    cheap_inverse, _, _ = _safe_decompose_H(centered_gram + penalty)

    augmented = np.column_stack((np.ones(len(x)), X))
    full_penalty = np.zeros((3, 3))
    full_penalty[1:, 1:] = penalty
    full_hessian = augmented.T @ (weights[:, None] * augmented) + full_penalty
    expected = np.linalg.inv(full_hessian)[1:, 1:]
    np.testing.assert_allclose(cheap_inverse, expected, rtol=1e-12, atol=1e-12)

    shifted = X + np.array([-1.0e10, 6.0e9])
    shifted_mean = (shifted.T @ weights) / sum_w
    shifted_result = SimpleNamespace(rank_info=SimpleNamespace(sum_w=sum_w, mean_x=shifted_mean))
    np.testing.assert_allclose(
        _center_cached_direct_gram(authoritative_gram, shifted_result),
        centered_gram,
        rtol=1e-12,
        atol=1e-12,
    )
