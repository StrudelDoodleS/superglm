"""Private factory helpers for spline feature specs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike

from superglm.types import LambdaPolicy

if TYPE_CHECKING:
    from superglm.features.spline import _SplineBase

_VALID_SPLINE_KINDS = ("bs", "ps", "ns", "cr", "cr_cardinal")


def n_knots_from_k(kind: str, k: int, degree: int = 3) -> int:
    """Convert basis dimension ``k`` to interior knot count."""
    if kind not in _VALID_SPLINE_KINDS:
        raise ValueError(
            f"Unknown spline kind {kind!r}, expected one of {sorted(_VALID_SPLINE_KINDS)}"
        )

    if kind in ("bs", "ps"):
        n_knots = k - degree - 1
        min_k = degree + 2
    elif kind == "cr_cardinal":
        n_knots = k - 2
        min_k = 3
    else:
        n_knots = k - degree + 1
        min_k = degree

    if k < min_k:
        raise ValueError(
            f"k={k} is too small for kind={kind!r} with degree={degree}. Minimum k is {min_k}."
        )

    return n_knots


def Spline(
    kind: str = "ps",
    *,
    k: int | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    knot_strategy: str = "uniform",
    penalty: str = "ssp",
    select: bool = False,
    knots: ArrayLike | None = None,
    discrete: bool | None = None,
    n_bins: int | None = None,
    extrapolation: str = "clip",
    boundary: tuple[float, float] | None = None,
    knot_alpha: float = 0.2,
    monotone: str | None = None,
    monotone_mode: str = "postfit",
    m: int | tuple[int, ...] = 2,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
) -> _SplineBase:
    """Create a spline feature spec."""
    from superglm.features.spline import (
        CardinalCRSpline,
        CubicRegressionSpline,
        NaturalSpline,
        PSpline,
    )

    kind_map = {
        "bs": PSpline,
        "ps": PSpline,
        "ns": NaturalSpline,
        "cr": CubicRegressionSpline,
        "cr_cardinal": CardinalCRSpline,
    }

    if kind not in kind_map:
        raise ValueError(
            f"Unknown spline kind {kind!r}, expected one of {sorted(_VALID_SPLINE_KINDS)}"
        )

    if k is not None and n_knots is not None:
        raise ValueError(
            "Cannot specify both k and n_knots. Use k (public basis size) or n_knots (interior knots), not both."
        )

    if monotone is not None and kind == "ns":
        raise NotImplementedError(
            "monotone is not supported for kind='ns'. "
            "Use kind='cr' or kind='ps' with monotone='increasing' or 'decreasing'."
        )

    if kind == "bs":
        import warnings

        warnings.warn(
            "Spline(kind='bs') currently creates a P-spline (discrete-difference "
            "penalty). Use kind='ps' for this behavior. In a future release, "
            "kind='bs' will create a proper B-spline smooth with "
            "integrated-derivative penalty.",
            FutureWarning,
            stacklevel=3,
        )

    if k is not None:
        if kind in ("cr", "cr_cardinal"):
            resolved_n_knots = n_knots_from_k(kind, k, degree=3)
        else:
            resolved_n_knots = n_knots_from_k(kind, k, degree)
    elif n_knots is not None:
        resolved_n_knots = n_knots
    else:
        resolved_n_knots = 10

    cls = kind_map[kind]

    if kind in ("bs", "ps"):
        return cast(
            "_SplineBase",
            cls(
                n_knots=resolved_n_knots,
                degree=degree,
                knot_strategy=knot_strategy,
                penalty=penalty,
                select=select,
                knots=knots,
                discrete=discrete,
                n_bins=n_bins,
                extrapolation=extrapolation,
                boundary=boundary,
                knot_alpha=knot_alpha,
                monotone=monotone,
                monotone_mode=monotone_mode,
                m=m,
                lambda_policy=lambda_policy,
            ),
        )
    if kind in ("cr", "cr_cardinal"):
        return cast(
            "_SplineBase",
            cls(
                n_knots=resolved_n_knots,
                knot_strategy=knot_strategy,
                penalty=penalty,
                select=select,
                knots=knots,
                discrete=discrete,
                n_bins=n_bins,
                extrapolation=extrapolation,
                boundary=boundary,
                knot_alpha=knot_alpha,
                monotone=monotone,
                monotone_mode=monotone_mode,
                m=m,
                lambda_policy=lambda_policy,
            ),
        )
    return cast(
        "_SplineBase",
        cls(
            n_knots=resolved_n_knots,
            degree=degree,
            knot_strategy=knot_strategy,
            penalty=penalty,
            select=select,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            m=m,
            lambda_policy=lambda_policy,
        ),
    )


__all__ = ["Spline", "n_knots_from_k"]
