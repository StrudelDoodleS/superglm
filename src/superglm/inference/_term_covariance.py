"""Internal covariance, SE, and simultaneous-band helpers for term inference."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.inference._term_helpers import _spline_se, mean_centered_variance
from superglm.inference._term_types import _safe_exp

if TYPE_CHECKING:
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def _active_subgroup_columns(
    feature_name: Hashable,
    feature_groups: list[GroupSlice],
    active_subs: list[GroupSlice],
) -> NDArray:
    return np.concatenate(
        [
            np.arange(group.start, group.end) - feature_groups[0].start
            for group in feature_groups
            if any(
                active_group.feature_name == feature_name and active_group.name == group.name
                for active_group in active_subs
            )
        ]
    )


def _covariance_apply(covariance, vector: NDArray) -> NDArray:
    """``Cov @ vector`` without materialising a compact covariance.

    A structured accessor stores a factorisation, applies the inverse through
    ``solve`` and refuses to hand back a large dense block at all -- which is
    why the random-effect branch below reads a selected diagonal rather than a
    submatrix.  One matvec is all the centering correction needs from it.
    """
    solve = getattr(covariance, "solve", None)
    if solve is not None:
        return np.asarray(solve(vector), dtype=np.float64)
    return cast(NDArray, np.asarray(covariance, dtype=np.float64) @ vector)


def _scatter_centered_variance(
    diagonal: NDArray,
    covariance_block: NDArray,
    row_of_column: NDArray,
    n_rows: int,
) -> NDArray:
    """``diag(C V C')`` where each report row reads at most one coefficient.

    A categorical level table and an ordered-categorical step table are both
    ``V = A Cov A'`` for a selection ``A``; ``row_of_column[j]`` is the row
    coefficient ``j`` lands on, and the rows it never names -- the base level,
    and any pinned level fixed at zero -- carry a zero row of ``V``.  Under
    ``native`` that is why their SE is zero; under centering it is why they
    come back at exactly ``p'Vp``, which is the point of the fix.
    """
    weights = np.full(covariance_block.shape[0], 1.0 / n_rows, dtype=np.float64)
    column = covariance_block @ weights
    cross = np.zeros(n_rows, dtype=np.float64)
    cross[row_of_column] = column
    variance = np.zeros(n_rows, dtype=np.float64)
    variance[row_of_column] = diagonal
    return mean_centered_variance(variance, cross, float(weights @ column))


def feature_se_from_cov(
    name: Hashable,
    Cov_active: NDArray,
    active_groups: list[GroupSlice],
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: Mapping[Any, Any],
    interaction_specs: Mapping[Any, Any],
    n_points: int = 200,
    center: bool = False,
) -> NDArray:
    """Compute feature-level SEs from a precomputed covariance matrix.

    ``center=True`` returns the errors of the MEAN-CENTERED report instead:
    the reported vector is then the estimable contrast ``C b`` with
    ``C = I - 11'/L``, whose covariance is ``C V C'``, so the errors are a
    different quantity rather than the same one shifted.  It is the same
    dispatch because the map from coefficients to report is the same map; only
    the quadratic form it is evaluated in changes.  An inactive term returns
    zeros either way -- ``C 0 C'`` is ``0``.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
    from superglm.features.polynomial import Polynomial
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase

    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == name]
    spec = specs.get(name) or interaction_specs.get(name)

    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            return _spline_se(
                spec,
                name,
                beta,
                feature_groups,
                active_groups,
                Cov_active,
                x_eval=np.array(spec._ordered_levels, dtype=object),
                reference_x=np.array([spec._base_level], dtype=object),
                center=center,
            )
        active_subs = [ag for ag in active_groups if ag.feature_name == name]
        if not active_subs:
            return np.zeros(len(spec._ordered_levels))
        indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
        Cov_g = Cov_active[np.ix_(indices, indices)]
        if spec._R_inv is not None:
            Cov_orig = spec._R_inv @ Cov_g @ spec._R_inv.T
        else:
            Cov_orig = Cov_g
        levels = list(spec._ordered_levels)
        if center:
            rows = np.array([levels.index(lev) for lev in spec._non_base], dtype=np.intp)
            return cast(
                NDArray,
                np.sqrt(
                    _scatter_centered_variance(
                        np.maximum(np.diag(Cov_orig), 0.0), Cov_orig, rows, len(levels)
                    )
                ),
            )
        se_nonbase = np.sqrt(np.maximum(np.diag(Cov_orig), 0.0))
        se_all = np.zeros(len(levels))
        for i, lev in enumerate(levels):
            if lev != spec._base_level:
                idx = spec._non_base.index(lev)
                se_all[i] = se_nonbase[idx]
        return se_all

    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        if isinstance(spec, _SplineBase | Polynomial):
            return np.zeros(n_points)
        if isinstance(spec, Categorical):
            return np.zeros(len(spec._levels))
        if isinstance(spec, RandomEffect):
            return np.zeros(len(spec._levels))
        if isinstance(spec, Piecewise):
            # One SE per KNOT, base included -- the same length as the term's
            # log_relativity vector.  The zeros(1) fallback below would leave a
            # dropped piecewise term's CI arrays a different length from its
            # values, which every downstream renderer zips against.
            return np.zeros(spec._knots.size)
        return np.zeros(1)

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])

    if isinstance(spec, RandomEffect):
        from superglm.inference.covariance import covariance_selected_diagonal

        variance = np.maximum(covariance_selected_diagonal(Cov_active, indices), 0.0)
        if center:
            # One coefficient per level and no base, so the map is a pure
            # selection and ``Vp`` is one matvec against the FULL covariance --
            # the block itself is the one this accessor may refuse to form.
            weights = np.zeros(Cov_active.shape[0], dtype=np.float64)
            weights[indices] = 1.0 / len(indices)
            column = _covariance_apply(Cov_active, weights)
            variance = mean_centered_variance(variance, column[indices], float(weights @ column))
        return cast(NDArray, np.sqrt(variance))

    Cov_g = Cov_active[np.ix_(indices, indices)]

    if isinstance(spec, _SplineBase):
        return _spline_se(
            spec,
            name,
            beta,
            feature_groups,
            active_groups,
            Cov_active,
            n_points=n_points,
            center=center,
        )

    if isinstance(spec, Polynomial):
        x_grid = np.linspace(spec._lo, spec._hi, n_points)
        M = np.asarray(spec.transform(x_grid), dtype=np.float64)
        if center:
            M = M - M.mean(axis=0)
        Q = M @ Cov_g
        return cast(NDArray, np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0)))

    if isinstance(spec, Categorical):
        levels = list(spec._levels)
        # _non_base excludes pinned levels (declared, no effective rows); their
        # coefficient is fixed at zero, so their SE stays 0.0 like the base's.
        position = {lev: j for j, lev in enumerate(spec._non_base)}
        if center:
            rows = np.array([levels.index(lev) for lev in spec._non_base], dtype=np.intp)
            return cast(
                NDArray,
                np.sqrt(
                    _scatter_centered_variance(
                        np.maximum(np.diag(Cov_g), 0.0), Cov_g, rows, len(levels)
                    )
                ),
            )
        se_nonbase = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
        se_all = np.zeros(len(levels))
        for i, lev in enumerate(levels):
            idx = position.get(lev)
            if idx is not None:
                se_all[i] = se_nonbase[idx]
        return se_all

    if isinstance(spec, Piecewise):
        # ``_raw_basis_matrix`` at the knots is the identity, so restricting it
        # to the retained columns is the identity padded with a zero row at the
        # base knot.  Going through the basis rather than indexing the diagonal
        # keeps this branch a plain quadratic form: the base knot's SE is 0
        # because its contrast against itself is, not because it was special-cased.
        M = np.asarray(
            spec._raw_basis_matrix(spec._knots)[:, spec._non_base_indices], dtype=np.float64
        )
        if center:
            M = M - M.mean(axis=0)
        Q = M @ Cov_g
        return cast(NDArray, np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0)))

    if isinstance(spec, Numeric):
        # A single reported value: ``centering="mean"`` declines to shift it,
        # so there is no contrast to propagate and ``center`` cannot apply.
        return np.array([np.sqrt(max(Cov_g[0, 0], 0.0))])

    variance = np.maximum(np.diag(Cov_g), 0.0)
    if center:
        size = variance.size
        rows = np.arange(size, dtype=np.intp)
        variance = _scatter_centered_variance(variance, Cov_g, rows, size)
    return cast(NDArray, np.sqrt(variance))


def piecewise_knot_covariance(
    name: str,
    Cov_active: NDArray,
    active_groups: list[GroupSlice],
    specs: dict[str, Any],
) -> NDArray | None:
    """Covariance of a Piecewise term's per-knot log relativities.

    ``(J+2, J+2)`` with the base row/column identically zero, built through
    the same raw-basis-at-the-knots map as ``feature_se_from_cov``'s SE
    branch, so ``sqrt(diag(V))`` is that SE vector.  Carried on
    ``TermInference.knot_covariance`` because the variance BETWEEN knots is
    the quadratic form of both adjacent hats with their covariance --
    ``var f(x) = h1^2 V11 + 2 h1 h2 V12 + h2^2 V22`` -- which pointwise knot
    SEs cannot reproduce.  Returns ``None`` when no subgroup is active.
    """
    spec = specs[name]
    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        return None
    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]
    M = spec._raw_basis_matrix(spec._knots)[:, spec._non_base_indices]
    return cast(NDArray, M @ Cov_g @ M.T)


def simultaneous_bands(
    feature: str,
    *,
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    covariance_fn,
    alpha: float = 0.05,
    n_sim: int = 10_000,
    n_points: int = 200,
    seed: int = 42,
    center: bool = False,
) -> pd.DataFrame:
    """Simultaneous confidence bands for a spline feature.

    ``center=True`` bands the MEAN-CENTERED curve.  The critical value is the
    quantile of ``max_x |f(x)| / se(x)`` over the whole grid, and centering
    changes both the numerator and the denominator, so it is a different
    number rather than the same one applied to shifted values -- which is why
    it is computed here through the centered map rather than reused.
    """
    from scipy.stats import norm

    from superglm.features.spline import _SplineBase

    spec = specs.get(feature)
    if not isinstance(spec, _SplineBase):
        raise TypeError(
            f"simultaneous_bands() only supports spline features, "
            f"got {type(spec).__name__} for '{feature}'."
        )

    Cov_active, active_groups = covariance_fn()
    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == feature]

    active_subs = [ag for ag in active_groups if ag.feature_name == feature]
    if not active_subs:
        raise ValueError(f"Feature '{feature}' is inactive (all coefficients zeroed).")

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]

    x_grid = np.linspace(spec._lo, spec._hi, n_points)
    M = np.asarray(spec.transform(x_grid), dtype=np.float64)
    active_cols = _active_subgroup_columns(feature, feature_groups, active_subs)
    M = M[:, active_cols]
    if center:
        M = M - M.mean(axis=0)

    Q = M @ Cov_g
    se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

    beta_g = np.concatenate(
        [
            beta[g.sl]
            for g in feature_groups
            if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
        ]
    )
    log_rel = M @ beta_g

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Cov_g + 1e-12 * np.eye(Cov_g.shape[0]))
    beta_sim = rng.standard_normal((n_sim, Cov_g.shape[0])) @ L.T
    f_sim = beta_sim @ M.T

    se_safe = np.maximum(se, 1e-20)
    T_sim = np.max(np.abs(f_sim) / se_safe[np.newaxis, :], axis=1)
    c_sim = float(np.quantile(T_sim, 1.0 - alpha))

    z = norm.ppf(1.0 - alpha / 2.0)

    return pd.DataFrame(
        {
            "x": x_grid,
            "log_relativity": log_rel,
            "relativity": _safe_exp(log_rel),
            "se": se,
            "ci_lower_pointwise": _safe_exp(log_rel - z * se),
            "ci_upper_pointwise": _safe_exp(log_rel + z * se),
            "ci_lower_simultaneous": _safe_exp(log_rel - c_sim * se),
            "ci_upper_simultaneous": _safe_exp(log_rel + c_sim * se),
        }
    )


__all__ = [
    "feature_se_from_cov",
    "piecewise_knot_covariance",
    "simultaneous_bands",
]
