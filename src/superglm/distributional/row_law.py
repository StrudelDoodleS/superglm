"""One fitted row's law: its density, its mean excess and the curve behind its plot.

The density is the exponent of the family's own log-likelihood, bound with unit weights and a
complete observation, so it is the row's parent law whatever weights, truncation or censoring
the model was fitted with. Amounts are unit-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    AtomFamily,
    DistributionFunctionFamily,
    MeanExcessFamily,
)
from superglm.distributional.response import response_spec
from superglm.distributional.surfaces import SCHEMA_VERSION, _json_scalar, _json_vector
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

if TYPE_CHECKING:
    from superglm.distributional.model import DenseDistributionalModel

__all__ = ["DistributionCurve", "curve_grid", "row_density", "row_mean_excess"]

KINDS = ("density", "survival", "mean_excess")
#: The probabilities whose quantiles bound a default grid, by kind.
LEVELS = {"density": (1e-3, 0.999), "mean_excess": (1e-3, 0.999), "survival": (0.01, 1 - 1e-6)}
_PRIOR = WeightContract("prior")


@dataclass(frozen=True, eq=False)
class DistributionCurve:
    """One row's law on a grid of amounts: what :meth:`plot_distribution` draws.

    ``piece`` names each point's piece: ``"law"`` for a single law, ``"body"`` or ``"tail"``
    for a body-plus-tail law, whose grid holds the threshold twice (the body's left limit,
    then the tail's right limit). ``amount_scale`` is ``"log"`` for the survival of a law of
    positive amounts and ``"linear"`` otherwise.
    """

    kind: str
    y: NDArray[np.float64]
    values: NDArray[np.float64]
    piece: tuple[str, ...]
    threshold: float | None
    mean: float
    family: str
    parameters: Mapping[str, float]
    amount_scale: str
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        """Return the curve as JSON values; a non-finite number becomes null."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "family": self.family,
            "y": _json_vector(self.y),
            "values": _json_vector(self.values),
            "piece": list(self.piece),
            "threshold": _json_scalar(self.threshold),
            "mean": _json_scalar(self.mean),
            "parameters": {name: _json_scalar(value) for name, value in self.parameters.items()},
            "amount_scale": self.amount_scale,
        }


def _rows(positions: NDArray[np.intp]) -> str:
    from superglm.model.base import _shown_positions

    return f"[{_shown_positions(positions)}]"


def require_row_law(family: object, caller: str, *, mean_excess: bool = False) -> None:
    """Refuse a family with no density of one amount, or no mean excess when it is asked for."""
    what = "mean excess" if mean_excess else "density"
    name = type(family).__name__
    vector = bool(response_spec(family).shape)
    if vector:
        reason = f"{name} describes several values at once, so it has no {what} of one amount"
    elif isinstance(family, AtomFamily):
        reason = f"{name} puts a point mass on an amount, so it has no {what} there"
    elif getattr(family, "response_support", None) is None or (
        mean_excess and not isinstance(family, MeanExcessFamily)
    ):
        reason = f"{name} has no {what}: {caller} reads continuous laws of one amount"
    else:
        return
    remedy = (
        "Use predict_cdf() for the chance of at most an amount, or predict_parameters()."
        if isinstance(family, DistributionFunctionFamily) and not vector
        else "Use predict_parameters() for the fitted parameters."
    )
    raise NotImplementedError(f"{reason}. {remedy}")


def _log_density(family: Any, y: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:
    weights = resolve_likelihood_weights(None, n_observations=len(y), contract=_PRIOR)
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    with np.errstate(all="ignore"):
        evaluation = family.evaluate_natural(y, theta, plan, derivative_order=0)
    valid = np.ones(len(y), bool) if evaluation.valid is None else evaluation.valid
    return evaluation.reported_log_likelihood, valid


def refuse_missing(y: NDArray, caller: str, positions: NDArray[np.intp] | None = None) -> None:
    """Refuse a missing amount, naming its rows (``positions`` in the caller's input)."""
    missing = np.isnan(y)
    if np.any(missing):
        where = np.flatnonzero(missing) if positions is None else positions[missing]
        raise ValueError(
            f"{caller} needs an amount for every row; rows {_rows(where)} have none. "
            "Drop those rows or give them an amount."
        )


def row_log_density(
    family: Any,
    y: NDArray,
    theta: NDArray,
    *,
    caller: str = "predict_pdf",
    positions: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """``log f(y | theta)`` per row under the unit-weight complete law; ``-inf`` off the support.

    A missing amount is refused. An amount inside the support whose density the family cannot
    evaluate in float64 is refused by row; ``positions`` are the rows' positions in the
    caller's input, for the message.
    """
    require_row_law(family, caller)
    y = np.asarray(y, dtype=np.float64)
    where = np.arange(len(y)) if positions is None else positions
    refuse_missing(y, caller, where)
    support = family.response_support
    inside = support.contains(y)
    stand_in = 0.0 if support.lower is None else support.lower + 1.0
    values = np.where(inside, y, stand_in)
    try:
        log_f, valid = _log_density(family, values, theta)
        bad = inside & ~valid
    except (ValueError, ArithmeticError):
        bad = np.zeros(len(y), bool)
        for i in np.flatnonzero(inside):
            try:
                bad[i] = not _log_density(family, values[i : i + 1], theta[i : i + 1])[1][0]
            except (ValueError, ArithmeticError):
                bad[i] = True
        if not np.any(bad):
            raise
    if np.any(bad):
        raise ValueError(
            f"{caller} cannot evaluate the density at rows {_rows(where[bad])} in float64: their "
            "amounts lie too far from their rows' laws. Pass amounts nearer each law's bulk."
        )
    return np.where(inside, log_f, -np.inf)


def row_density(
    family: Any, y: NDArray, theta: NDArray, *, caller: str = "predict_pdf"
) -> NDArray[np.float64]:
    """The density per row at ``y``: ``0`` outside the support, a missing amount refused."""
    with np.errstate(over="ignore"):
        return np.exp(row_log_density(family, y, theta, caller=caller))


def finite_amounts(v: NDArray, caller: str) -> NDArray[np.float64]:
    """``v`` as floats, refusing a missing or infinite amount by row."""
    v = np.asarray(v, dtype=np.float64)
    bad = ~np.isfinite(v)
    if np.any(bad):
        raise ValueError(
            f"{caller} needs a finite amount for every row; rows {_rows(np.flatnonzero(bad))} "
            "are missing or infinite."
        )
    return v


def row_mean_excess(
    family: Any, v: NDArray, theta: NDArray, *, caller: str = "predict_mean_excess"
) -> NDArray[np.float64]:
    """``E[Y - v | Y > v]`` per row: ``E[Y] - v`` at or below the support, ``inf`` without a mean."""
    require_row_law(family, caller, mean_excess=True)
    return np.asarray(family.mean_excess(finite_amounts(v, caller), theta), dtype=np.float64)


def row_survival(family: Any, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """``P(Y > y)`` per row from the family's log upper-window probability."""
    support = family.response_support
    below = (y < support.lower) if support.lower is not None else np.zeros(len(y), bool)
    lower = np.where(below, support.lower if support.lower is not None else 0.0, y)
    log_s = np.asarray(family.log_window_probability(lower, np.full(len(y), np.inf), theta))
    return np.where(below, 1.0, np.exp(log_s))


def check_curve_request(
    n_rows: int, kind: str, y: NDArray | None, n_points: int, caller: str
) -> NDArray[np.float64] | None:
    """Refuse anything but one row, a known kind and a finite increasing grid; return the grid."""
    if n_rows != 1:
        verb = "draws" if caller.startswith("plot") else "reads"
        raise ValueError(
            f"{caller} {verb} one row's law and was given {n_rows} rows. Pass one row, for "
            "example X.iloc[[0]]."
        )
    if kind not in KINDS:
        raise ValueError(f"kind must be 'density', 'survival' or 'mean_excess', not {kind!r}.")
    if y is None:
        if isinstance(n_points, bool) or not isinstance(n_points, int | np.integer) or n_points < 2:
            raise ValueError(f"n_points must be an integer of at least 2, not {n_points!r}.")
        return None
    grid = np.asarray(y, dtype=np.float64)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("y must be at least two finite amounts in increasing order.")
    return grid


def curve_grid(
    kind: str, n_points: int, lower: float, upper: float, *, positive: bool
) -> NDArray[np.float64]:
    """``n_points`` amounts from ``lower`` to ``upper``.

    Geometric for the survival of a law of positive amounts, drawn on log axes; even otherwise.
    """
    if kind == "survival" and positive:
        return np.geomspace(lower, upper, n_points)
    return np.linspace(lower, upper, n_points)


def model_curve(
    model: DenseDistributionalModel,
    row: FrameLike | EagerFrame,
    *,
    kind: str,
    y: NDArray | None,
    n_points: int,
    offsets: Mapping[str, NDArray] | None,
    caller: str,
) -> DistributionCurve:
    """One SuperLSS row's :class:`DistributionCurve`: the default grid spans ``LEVELS[kind]``."""
    family: Any = model.family
    require_row_law(family, caller, mean_excess=kind == "mean_excess")
    grid = check_curve_request(len(as_eager_frame(row)), kind, y, n_points, caller)
    theta, outside = model._predicted_parameters(row, offsets)
    support = family.response_support
    positive = support.lower is not None and support.lower >= 0.0
    if grid is None:
        lower, upper = family.quantile(np.asarray(LEVELS[kind]), np.repeat(theta, 2, axis=0))
        grid = curve_grid(kind, n_points, lower, upper, positive=positive)
    rows = np.repeat(theta, len(grid), axis=0)
    if kind == "density":
        values = row_density(family, grid, rows, caller=caller)
    elif kind == "survival":
        values = row_survival(family, grid, rows)
    else:
        values = row_mean_excess(family, grid, rows, caller=caller)
    # A row with no value of a parameter has no law: every number reads NaN.
    missing = bool(outside.any())
    if missing:
        values = np.full(len(grid), np.nan)
        theta = np.full_like(theta, np.nan)
    mean = np.nan if missing else float(np.asarray(family.default_prediction(theta))[0])
    return DistributionCurve(
        kind=kind,
        y=grid,
        values=values,
        piece=("law",) * len(grid),
        threshold=None,
        mean=mean,
        family=type(family).__name__,
        parameters={p.name: float(value) for p, value in zip(family.parameters, theta[0])},
        amount_scale="log" if kind == "survival" and positive else "linear",
    )
