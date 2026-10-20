"""Private factory helpers for spline feature specs."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from numpy.typing import ArrayLike

from superglm.types import LambdaPolicy

if TYPE_CHECKING:
    from superglm.features._spline_config import (
        ExtrapolationName,
        KnotStrategyName,
        PenaltyName,
        SplineKindName,
    )
    from superglm.features._spline_ranges import PolynomialRange
    from superglm.features.spline import _SplineBase

_VALID_SPLINE_KINDS = ("bs", "ps", "ns", "cr", "cr_cardinal")

# A spline declared with no kind carries the arguments it was declared with,
# so a model can build it as ``kind="ps"`` exactly as that declaration would
# have built on 0.39. Explicit kinds and specs pickled before 0.41 read None.
KINDLESS_ARGS = "_kindless_args"


def _cubic_cannot_take(degree: int, m: int | tuple[int, ...]) -> str | None:
    """Why the default ``cr`` cannot take this degree or penalty order, or None."""
    from superglm.features.spline import CubicRegressionSpline

    if degree != 3:
        return f"is always cubic, so it cannot take degree={degree}"
    cap = CubicRegressionSpline._max_penalty_order
    orders = (m,) if isinstance(m, int) else tuple(m)
    worst = max((order for order in orders if isinstance(order, int)), default=0)
    if cap is not None and worst > cap:
        return f"takes penalty orders up to {cap}, so it cannot take m={worst}"
    return None


def kindless_ps_message(reason: str) -> str:
    """The warning a kindless spline gives when it builds as ``ps``."""
    return (
        f"This spline names no kind, and the default kind='cr' {reason}. It builds as "
        "kind='ps', the P-spline a spline with no kind gave before 0.40. Pass kind= to "
        "choose the kind and silence this warning; kind='ps' keeps this fit."
    )


def kindless_as_ps(spec: Any) -> _SplineBase | None:
    """The ``ps`` spline a kindless ``spec`` was declared as, or None.

    None when ``spec`` named its kind (or was pickled before kinds were
    recorded), or when ``ps`` cannot take its declaration either, so the
    caller keeps the refusal ``cr`` gave. The result is the spline
    ``Spline(kind="ps", ...)`` builds from the same arguments, and it keeps
    the record that no kind was named.
    """
    declared = getattr(spec, KINDLESS_ARGS, None)
    if declared is None or declared.get("polynomial_ranges"):
        return None
    replacement = Spline(kind="ps", **declared)
    setattr(replacement, KINDLESS_ARGS, declared)
    return replacement


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
    kind: SplineKindName | None = None,
    *,
    k: int | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    knot_strategy: KnotStrategyName = "uniform",
    penalty: PenaltyName = "ssp",
    select: bool = False,
    knots: ArrayLike | None = None,
    discrete: bool | None = None,
    n_bins: int | None = None,
    extrapolation: ExtrapolationName = "clip",
    boundary: tuple[float, float] | None = None,
    knot_alpha: float = 0.2,
    constraint=None,
    m: int | tuple[int, ...] = 2,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    polynomial_ranges: Sequence[PolynomialRange] | None = None,
) -> _SplineBase:
    """Create a spline feature spec.

    With no ``kind`` the spline is a cubic regression spline, ``"cr"``,
    unless ``cr`` cannot take ``degree`` or ``m``: it is then the P-spline
    ``"ps"`` that 0.39 built, with a ``UserWarning`` naming the choice.
    """
    declared = None
    if kind is None:
        declared = {
            "k": k,
            "n_knots": n_knots,
            "degree": degree,
            "knot_strategy": knot_strategy,
            "penalty": penalty,
            "select": select,
            "knots": knots,
            "discrete": discrete,
            "n_bins": n_bins,
            "extrapolation": extrapolation,
            "boundary": boundary,
            "knot_alpha": knot_alpha,
            "constraint": constraint,
            "m": m,
            "lambda_policy": lambda_policy,
            "polynomial_ranges": polynomial_ranges,
        }
        reason = None if polynomial_ranges else _cubic_cannot_take(degree, m)
        kind = "cr" if reason is None else "ps"
        if reason is not None:
            # Two frames up is the caller of the public ``Spline`` wrapper.
            warnings.warn(kindless_ps_message(reason), UserWarning, stacklevel=3)
        spec = Spline(
            kind=kind,
            k=k,
            n_knots=n_knots,
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
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
            polynomial_ranges=polynomial_ranges,
        )
        setattr(spec, KINDLESS_ARGS, declared)
        return spec

    from superglm.features.spline import (
        BSplineSmooth,
        CardinalCRSpline,
        CubicRegressionSpline,
        NaturalSpline,
        PSpline,
    )

    kind_map = {
        "bs": BSplineSmooth,
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

    if polynomial_ranges and kind not in ("bs", "cr"):
        raise ValueError(
            f"polynomial_ranges needs kind='bs' or kind='cr', got kind={kind!r}: range edges "
            "repeat knots, which the equal-spacing difference penalties of 'ps' and 'ns' and "
            "the knot-gap penalty of 'cr_cardinal' cannot take."
        )
    # Only the two supporting classes take the keyword; the others never see it.
    range_kwargs = {"polynomial_ranges": polynomial_ranges} if polynomial_ranges else {}

    if kind in ("cr", "cr_cardinal") and degree != 3:
        raise ValueError(
            f"A cubic regression spline (kind={kind!r}) is always cubic, "
            f"so degree={degree} cannot apply. Pass kind='ps' or kind='bs' for a "
            f"degree-{degree} spline."
        )

    if kind in ("cr", "cr_cardinal"):
        # The cap is the class's own, so the refusal moves with it.
        cap = kind_map[kind]._max_penalty_order
        orders = (m,) if isinstance(m, int) else tuple(m)
        worst = max((order for order in orders if isinstance(order, int)), default=0)
        if worst > cap:
            raise ValueError(
                f"A cubic regression spline (kind={kind!r}) takes penalty "
                f"orders up to {cap}, so m={worst} cannot apply. Pass kind='ps' for a penalty "
                f"of order {worst}."
            )

    if constraint is not None and kind == "ns":
        raise NotImplementedError(
            "constraint is not supported for kind='ns'. "
            "Use kind='cr', kind='ps', or remove the constraint."
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
                constraint=constraint,
                m=m,
                lambda_policy=lambda_policy,
                **range_kwargs,
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
                constraint=constraint,
                m=m,
                lambda_policy=lambda_policy,
                **range_kwargs,
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


__all__ = ["KINDLESS_ARGS", "Spline", "kindless_as_ps", "kindless_ps_message", "n_knots_from_k"]
