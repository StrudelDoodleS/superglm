"""Compatibility wrappers for legacy post-fit monotone repair entry points."""

from __future__ import annotations


def monotonize(
    model,
    X,
    sample_weight=None,
    offset=None,
    *,
    n_grid: int = 500,
):
    """Compatibility alias for the generalized postfit shape repair path."""
    from superglm.model import shape_ops

    return shape_ops.apply_shape_postfit(model, X, sample_weight, offset, n_grid=n_grid)


def apply_monotone_postfit(
    model,
    X,
    sample_weight=None,
    offset=None,
    *,
    n_grid: int = 500,
):
    """Compatibility alias for :func:`monotonize`."""
    return monotonize(model, X, sample_weight, offset, n_grid=n_grid)
