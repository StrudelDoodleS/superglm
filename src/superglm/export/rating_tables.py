"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.export._ppform import extract_ppform
from superglm.export.summary import SummaryExportPayload, build_summary_export_payload
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase
from superglm.inference._ordered_reference import ordered_reference_intercept
from superglm.inference._term_helpers import _VALID_CENTERING
from superglm.inference._term_types import TermInference
from superglm.links import LogLink

if TYPE_CHECKING:
    from superglm.model import SuperGLM


class RatingTableBaseNotRepresentableError(OverflowError):
    """The exported base relativity is not a usable multiplier.

    ``base_relativity`` multiplies every row of the tariff, so it is the one
    exported number that has no tolerable approximation: clipping it the way
    ``_safe_exp`` clips a confidence bound would hand back a workbook that
    silently rates every risk wrong, which is precisely the failure issue #253
    was.  A base that overflows to ``inf``, underflows to ``0.0``, or lands in
    the subnormal range where float64 has already dropped most of its mantissa
    therefore stops the export instead of being repaired.
    """


# The smallest base the export will emit: the smallest float64 that still
# carries a full 53-bit significand.  Below it lie the subnormals, which are
# finite and positive but progressively less precise, so an exported subnormal
# is not the number the model asked for -- see ``_base_relativity``.
_SMALLEST_EXACT_BASE = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True)
class RatingTableBlock:
    """One one-dimensional rating-table block."""

    name: str
    kind: str
    table: pd.DataFrame
    # Piecewise only: the out-of-range rule, so the workbook note can state it.
    extrapolation: str | None = None
    # The constant ``centering=`` removed from this block's log relativities.
    # The payload's ``base_relativity`` carries the total back, so the
    # workbook's product still reproduces ``model.predict``.  Zero on every
    # block a centering leaves alone -- the offset blocks, an
    # ``OrderedCategorical``, a single-valued ``Numeric`` -- which is why the total is summed from the blocks rather
    # than assumed to run over every exported term.
    centering_shift: float = 0.0


@dataclass(frozen=True)
class InteractionTableBlock:
    """One two-way interaction rating-table block.

    ``kind`` says how the block approximates: ``"cells"`` is a full
    categorical-by-categorical cell table and is exact, ``"grid"`` is a
    continuous surface SAMPLED at ``n_bins`` nodes per axis and is not.  It is
    recorded because ``_interaction_blocks`` is the one place that decides,
    and a second place deciding differently is the shape issue #287 took.
    """

    name: str
    table: pd.DataFrame
    kind: str = "cells"


@dataclass(frozen=True)
class RatingTablePayload:
    """Renderer-independent rating-table export payload."""

    base_relativity: float
    selected_n_bins: int
    main_effects: list[RatingTableBlock]
    interactions: list[InteractionTableBlock]
    discretization_impact: pd.DataFrame
    summary: SummaryExportPayload


_OFFSET_SOURCE_RESERVED_COLUMNS = frozenset({"Relativity", "Weight"})


def _unsupported_structured_export_terms(model: SuperGLM) -> list[str]:
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    unsupported = [
        name for name in model._feature_order if isinstance(model._specs[name], RandomEffect)
    ]
    unsupported.extend(
        name
        for name in model._interaction_order
        if isinstance(model._interaction_specs[name], FactorSmooth)
    )
    return unsupported


def _preflight_rating_table_terms(model: SuperGLM) -> None:
    unsupported = _unsupported_structured_export_terms(model)
    if unsupported:
        raise NotImplementedError(
            "Rating-table export does not yet support conditional or population-only "
            f"RandomEffect/FactorSmooth terms {unsupported!r}; no payload was produced."
        )


def _resolve_format(file_path: str | Path, format: str | None) -> str:
    if format is not None:
        fmt = format.lower().lstrip(".")
    else:
        suffix = Path(file_path).suffix.lower()
        fmt = suffix.lstrip(".")
    if fmt in {"xlsx", "xlsm", "excel"}:
        return "excel"
    raise ValueError(
        f"Unsupported rating table export format: {format or Path(file_path).suffix!r}"
    )


def _continuous_features(model: SuperGLM) -> list[str]:
    return [
        name
        for name in model._feature_order
        if isinstance(model._specs.get(name), _SplineBase | Polynomial)
    ]


def _grid_reconstruction_keys() -> frozenset[str]:
    from superglm.diagnostics.discretize import _GRID_RECONSTRUCTION_KEYS

    return _GRID_RECONSTRUCTION_KEYS


def _gridded_interaction_names(blocks: list[InteractionTableBlock]) -> list[str]:
    """The exported blocks that are sampled surfaces, read off the blocks.

    Derived from what ``_interaction_blocks`` actually BUILT rather than
    re-decided from the specs, so "what the workbook approximates" and "what
    the impact sheet covers" cannot drift apart -- which is the shape issue
    #287 took. Selecting by spec class here would have left any grid-shaped
    spec outside the two built-ins exported approximately and off the sheet,
    because the exporter routes on the reconstruction contract, not the class.
    """
    return [block.name for block in blocks if block.kind == "grid"]


def _format_number(value: float) -> str:
    """Print a float so that reading the string back gives the same float.

    Every number this module prints as a KEY -- a bin boundary, an interaction
    axis value -- is a number the consumer converts back to a float in order to
    decide which row of the table a risk belongs to.  So the only requirement
    that matters is Steele and White's first free-format property: converting
    the printed decimal back must recover the original binary64 value exactly
    (G. L. Steele Jr. and J. L. White, "How to print floating-point numbers
    accurately", PLDI 1990, s2 -- no loss of information, no extra information,
    correct rounding; see also D. M. Gay, "Correctly rounded binary-decimal and
    decimal-binary conversions", AT&T Bell Laboratories NAM 90-10, 1990).

    A fixed-format ``.10g`` does not have it.  Ten significant digits do not
    identify a binary64, which needs seventeen (J. Champagne Gareau and
    D. Lemire, "Converting Binary Floating-Point Numbers to Shortest Decimal
    Strings: An Experimental Review", arXiv:2603.06581, s2), so the printed
    value is a DIFFERENT number from the one the model used.  For an ordinary
    reported quantity that is a rounding; for a bin boundary it is not, because
    the map from value to bin is discontinuous there.  An edge perturbed by
    5e-10 relative moves every row inside that band into the neighbouring bin,
    which changes the factor by a whole bin step -- and under the default
    ``bin_strategy="exposure_quantile"`` the edges ARE data values, so a row
    sits exactly on one by construction.  Measured on the equivalence fixture
    (900 rows, 150 bins, two continuous terms): 302 of 302 printed edges
    differed from the exact ones by up to 4.99e-09, 133 of 900 rows (14.8%)
    took a different factor, and the reconstruction missed the discretised
    predictions by 2.29e-01 relative rather than by round-off.

    ``repr`` is the shortest string with that property (``float(repr(x)) ==
    x``; ``sys.float_repr_style == 'short'`` since Python 3.1), so it costs
    only the digits that are load-bearing.  ``float`` first, because
    ``repr(np.float64(x))`` is ``np.float64(...)`` under NumPy 2 and would not
    parse back at all.
    """
    return repr(float(value))


def _format_interval(left: float, right: float) -> str:
    return f"[{_format_number(left)}, {_format_number(right)})"


def _format_axis_value(value: float) -> str:
    return _format_number(value)


def _continuous_block(name: str, table: pd.DataFrame, centering_shift: float) -> RatingTableBlock:
    """A binned block, moved by the same constant every other block moves by.

    The binned relativities come from ``discretization_impact``, which knows
    nothing about ``centering=`` -- so a spline or polynomial block used to
    come out NATIVE while the categorical, piecewise and numeric blocks beside
    it moved, on one ``centering="mean"`` request.  Measured on a four-term
    Poisson fit: ``region`` and ``term`` shifted by exactly what
    ``term_inference`` reports, ``age`` and ``dens`` by 0.000000 against a
    reported -0.004169 and +0.006522 (issue #293).  One request, two
    behaviours, and nothing in the workbook says which block is in which.

    The shift is applied HERE rather than by re-binning a centered term,
    because the constant has to be the same one ``term_inference`` reports and
    the other blocks subtract -- a re-derivation from the binned values would
    be the weighted mean of the BINS instead, a different number, and the
    blocks would no longer share one origin.

    Multiplicative on the emitted factors and recorded on the block, so the
    payload folds it into ``base_relativity`` exactly as it does for the exact
    blocks: the product of the base and every block is unchanged by centering,
    which is the exactness the whole payload rests on.
    """
    # Same discipline as ``_base_relativity``: an extreme shift is a condition
    # ``_require_usable_relativities_export`` is there to refuse, and it should
    # arrive as that refusal rather than as a ``RuntimeWarning`` from here.
    with np.errstate(over="ignore", under="ignore"):
        factor = float(np.exp(-centering_shift))
    out = pd.DataFrame(
        {
            name: [
                _format_interval(float(row.bin_from), float(row.bin_to))
                for row in table.itertuples(index=False)
            ],
            "Relativity": table["relativity"].astype(float).to_numpy() * factor,
            "Weight": table["sample_weight"].astype(float).to_numpy(),
        }
    )
    return RatingTableBlock(
        name=name, kind="continuous", table=out, centering_shift=float(centering_shift)
    )


# The interval bounds are NOT emitted as their own columns.  They are already in
# the key column beside them, exactly: ``_format_number`` prints via ``repr``, the
# shortest string that reads back as the same binary64, so a consumer parsing
# ``"[18.0, 25.363636363636363)"`` recovers both bounds bit-identically -- measured
# at 0.000e+00 against the exact breaks across every row of a real block.  Two float
# columns restating them would be duplicate state that can only ever disagree.
#
# The local variable is therefore ``u = (x - lower) / (upper - lower)`` with both
# bounds read from the key.  That makes the key load-bearing for arithmetic and not
# only for matching, which is a constraint on ``_format_number``: it must keep the
# round-trip property.  Its docstring already records why, and issue #278 is what
# happens when a printed key is not exact.
_PPFORM_COLUMNS = ("a", "b", "c", "d")

# The tail rows are CONSTANT pieces, so their higher coefficients are exactly
# zero and the local variable never matters.  That is deliberate: a consumer
# that computes ``u`` unconditionally on an unbounded row would divide by an
# infinite width, and a consumer that clamps would not -- with b = c = d = 0
# both arrive at the same number.
_PPFORM_TAIL_COEFFICIENTS = (0.0, 0.0, 0.0)


def _continuous_ppform_block(
    model: SuperGLM,
    name: str,
    centering: str,
    weights: NDArray,
    values: NDArray,
) -> RatingTableBlock:
    """A continuous term as its exact polynomial pieces, plus a lookup fallback.

    Nine columns, in two halves.  The first three are the ordinary binned-block
    columns -- an interval key, a relativity, an exposure weight -- so a
    consumer that cannot evaluate a polynomial still finds this block, reads it,
    and scores it as a step function exactly as it scores a binned block today.
    The remaining six are the exact form.

    That superset is not cosmetic.  The downstream loader locates blocks by a
    header signature at fixed offsets and fails the WHOLE package when a block
    does not match, so a six-column block would force two repositories to ship
    in lockstep and would break every other term's publication in between.  See
    the design's section 4.2a and 7.1.

    ``Relativity`` is the curve's value at ``from``, not the interval's average,
    so the two readings of one row agree at its left edge instead of disagreeing
    everywhere.

    The first and last rows are UNBOUNDED and constant.  Extrapolation is
    carried in the table rather than described beside it: a cubic evaluated past
    its last knot is not merely wrong but unbounded -- measured at 1581x the
    correct factor twenty-one years past the boundary of a real age curve -- and no
    note reliably prevents a consumer from doing it.  With the tails emitted,
    the only operation a consumer performs is "match an interval, evaluate it",
    and that operation is correct everywhere.

    Except under ``extrapolation="error"``, where the tails are omitted and the
    block covers the knot range alone.  That term declines to price outside its
    training range, so a matching failure in the consumer is the model's own
    answer rather than a gap in the table.

    The centering shift is RECORDED and not applied.  ``_continuous_block``
    multiplies its binned relativities by ``exp(-centering_shift)`` because
    ``discretization_impact`` knows nothing about centering; the segments here
    come from ``term_inference(centering=...)``, which has already subtracted
    it.  Applying it a second time would double-count the constant while
    ``base_relativity`` added it back once, so the block would be wrong by
    exactly one shift.  Recording it is still required: that is how the payload
    folds the constant back into the base.

    It is taken from ``segments`` and NOT accepted from the caller.  Under
    ``centering="mean"`` the shift is the mean of the curve over the grid it was
    read on, so the same term yields a different constant at a different
    ``n_points``: a shift sourced from a second inference call is a real number
    for a curve this block does not carry.  Measured on the export fixture, the
    200-point default and this module's 1201-point grid disagree by 1.77e-3 in
    log space -- a uniform 0.177% on every premium, with every relativity RATIO
    on the sheet still correct, which is precisely the shape of error that only
    an absolute comparison against ``predict`` can see.  Sourcing it from the
    curve's own call is what makes that mismatch unexpressible.
    """
    segments = extract_ppform(model, name, centering=centering)

    if segments.extrapolation == "error":
        # No tails at all.  The term refuses to price outside its training
        # range, so the block must refuse too: a value below the first knot or
        # above the last matches no row and the consumer's lookup fails, which
        # is the same answer the model gives.  Emitting bounded tails here would
        # be the export quietly deciding a question the model declined.
        lo = segments.breaks[:-1]
        hi = segments.breaks[1:]
        coefficients = segments.coefficients
    else:
        lo = np.concatenate([[-np.inf], segments.breaks[:-1], [segments.breaks[-1]]])
        hi = np.concatenate([[segments.breaks[0]], segments.breaks[1:], [np.inf]])

        boundary_low = float(segments.evaluate(np.asarray([segments.breaks[0]]))[0])
        boundary_high = float(segments.evaluate(np.asarray([segments.breaks[-1]]))[0])

        coefficients = np.vstack(
            [
                [boundary_low, *_PPFORM_TAIL_COEFFICIENTS],
                segments.coefficients,
                [boundary_high, *_PPFORM_TAIL_COEFFICIENTS],
            ]
        )

    # ``a`` is the log relativity at u = 0, so exp(a) is the factor a step-function
    # reader applies across the row -- the same convention the exact blocks use.
    with np.errstate(over="ignore", under="ignore"):
        relativity = np.exp(coefficients[:, 0])

    weight_by_row = _ppform_row_weights(lo, values, weights)

    # ``strict=True`` against the declared column names, so a change in the
    # number of coefficients has to be a change to ``_PPFORM_COLUMNS`` too
    # rather than a silently mis-headed column.
    exact_form = dict(zip(_PPFORM_COLUMNS, coefficients.T, strict=True))
    table = pd.DataFrame(
        {
            name: [_format_interval(left, right) for left, right in zip(lo, hi, strict=True)],
            "Relativity": relativity,
            "Weight": weight_by_row,
            **exact_form,
        }
    )
    return RatingTableBlock(
        name=name,
        kind="continuous_ppform",
        table=table,
        extrapolation=segments.extrapolation,
        centering_shift=segments.centering_shift,
    )


def _export_weights(X: EagerFrame, sample_weight: NDArray | None) -> NDArray:
    """The per-row exposure a weight column sums, with the unweighted default.

    An unweighted fit still gets a ``Weight`` column, carrying counts -- the
    same convention ``_weights_by_level`` uses for a categorical block, so the
    two block kinds report the same quantity rather than one reporting counts
    and the other nothing.
    """
    if sample_weight is None:
        return np.ones(len(X), dtype=np.float64)
    return np.asarray(sample_weight, dtype=np.float64)


def _ppform_row_weights(lo: NDArray, values: NDArray, weights: NDArray) -> NDArray:
    """Exposure falling in each segment, including the unbounded tails.

    Reported so the block carries the same weight column every other block
    does; a reviewer reads it to see which segments the book actually populates
    and which are shape carried by the penalty alone.

    Keyed on ``lo`` alone because the rows partition the whole line: every
    interval's upper bound is the next one's lower bound, so the count of row
    starts at or below a value IS its row, and reading ``hi`` as well would be a
    second statement of the same fact that could disagree with the first.
    """
    idx = np.clip(np.searchsorted(lo[1:], values, side="right"), 0, len(lo) - 1)
    return np.bincount(idx, weights=weights, minlength=len(lo)).astype(np.float64)


_SUPPORTED_CONTINUOUS_KINDS = frozenset({"binned", "ppform"})


def _require_supported_continuous_kind(continuous_kind: str) -> None:
    if continuous_kind not in _SUPPORTED_CONTINUOUS_KINDS:
        raise ValueError(
            f"continuous_kind must be one of {sorted(_SUPPORTED_CONTINUOUS_KINDS)}, "
            f"got {continuous_kind!r}."
        )


def _ppform_convertible_terms(model: SuperGLM) -> list[str]:
    """The names ``continuous_kind="ppform"`` would actually convert.

    The same two conditions the assembly loop routes on -- continuous enough to
    reach the discretisation path, and a spline rather than a ``Polynomial``,
    whose degree is not bounded by the block's four coefficients.  Derived here
    so a guard cannot come to disagree with the router about which terms it
    covers, which would make it refuse terms that stay binned or wave through
    terms that do not.
    """
    return [
        name for name in _continuous_features(model) if isinstance(model._specs[name], _SplineBase)
    ]


def _require_ppform_exportable(
    model: SuperGLM, names: list[str], *, allow_unbounded_extrapolation: bool
) -> None:
    """Refuse the two cases a ppform block cannot state honestly.

    Both are refusals rather than fallbacks.  Silently binning a term the
    caller asked to export exactly would put an approximation in a workbook
    that claims, block by block, to be exact -- which is the failure mode this
    whole feature exists to remove.
    """
    for name in names:
        spec = model._specs[name]
        # The spec does NOT keep the ConstraintSpec it was constructed with --
        # ``getattr(spec, "constraint")`` is None even when one was passed.  It
        # is unpacked at construction into ``constraint_kind`` /
        # ``constraint_mode`` (mirrored as ``monotone`` / ``monotone_mode``).
        #
        # BOTH are read, and that is load-bearing: ``constraint_mode`` is
        # ``"postfit"`` on an UNCONSTRAINED spline too, because that is the
        # default mode a token would have been given rather than a record that
        # one was.  ``constraint_kind is None`` is what says "no constraint", so
        # keying on the mode alone would refuse every spline ever fitted.
        constraint_kind = getattr(spec, "constraint_kind", None)
        if constraint_kind is not None and getattr(spec, "constraint_mode", None) == "postfit":
            raise ValueError(
                f"Term {name!r} carries a postfit {constraint_kind} constraint, whose "
                "repaired curve has not been verified to be piecewise polynomial. It "
                "cannot be exported with continuous_kind='ppform'; export it with "
                "continuous_kind='binned', or use Constraint.fit instead."
            )
        if getattr(spec, "extrapolation", None) == "extend" and not allow_unbounded_extrapolation:
            raise ValueError(
                f"Term {name!r} uses extrapolation='extend', so the fitted model prices "
                "beyond the training range with an unbounded cubic. The block's tail "
                "rows cannot carry that cubic -- an unbounded interval has no width, so "
                "the normalised u the coefficients are written against does not exist "
                "there -- and are emitted as the constant pieces continuous_kind="
                "'ppform' emits under 'clip'. The exported block therefore clips where "
                "the model extends. Pass allow_unbounded_extrapolation=True to export "
                "it on those terms, or refit the term with extrapolation='clip'."
            )


def _weights_by_level(
    X: EagerFrame,
    name: str,
    levels: list[str],
    sample_weight: NDArray | None,
) -> np.ndarray:
    weights = (
        np.ones(len(X), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    grouped = (
        pd.DataFrame(
            {
                "level": pd.Series(X.column_array(name)).astype(str),
                "weight": weights,
            }
        )
        .groupby("level", sort=False)["weight"]
        .sum()
    )
    level_keys = [str(level) for level in levels]
    return grouped.reindex(level_keys, fill_value=0.0).to_numpy(dtype=np.float64)


def _main_effect_inference(model: SuperGLM, name: str, centering: str) -> TermInference:
    """``model.term_inference`` narrowed to the main-effect half of its return type.

    ``build_rating_table_payload`` dispatches on the FEATURE spec, so every
    name that reaches a block builder is a main effect and the
    ``InteractionInference`` arm of the public signature cannot occur here.
    Narrowing it once, by an isinstance check rather than a cast, is what lets
    the builders read fields that exist on only one arm -- ``centering_shift``
    is the first of them -- and turns a future caller that does hand this an
    interaction name into a failure at the boundary rather than an
    ``AttributeError`` several lines into building a table.
    """
    ti = model.term_inference(name, with_se=False, centering=centering)
    if not isinstance(ti, TermInference):
        raise TypeError(
            f"Rating-table export asked for main-effect inference on {name!r} and got "
            f"{type(ti).__name__}; interactions are exported by _interaction_blocks."
        )
    return ti


def _categorical_block(
    model: SuperGLM,
    X: EagerFrame,
    name: str,
    sample_weight: NDArray | None,
    centering: str,
) -> RatingTableBlock:
    ti = _main_effect_inference(model, name, centering)
    levels = list(ti.levels or [])
    return RatingTableBlock(
        name=name,
        kind="categorical",
        table=pd.DataFrame(
            {
                name: levels,
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": _weights_by_level(X, name, levels, sample_weight),
            }
        ),
        centering_shift=float(ti.centering_shift),
    )


def _piecewise_block(model: SuperGLM, name: str, centering: str) -> RatingTableBlock:
    """One row per knot: the knot value, its relativity, and its log relativity.

    No binning, because there is nothing to bin -- a piecewise-linear function
    *is* a rating table, so the numbers here are the numbers the model fitted.

    Exactly three columns, and that is a constraint rather than a preference:
    the Excel renderer lays main-effect blocks on a fixed three-column stride
    and applies number formats globally by column index, so a fourth column
    would overwrite the neighbouring block.  ``Weight`` is the column that gets
    left out, and it is the right one to lose: a per-knot weight is not a
    rating-table quantity, and per-segment weight is already reported by the
    small-segment warning at ``build()``.

    ``Log relativity`` is the exact column.  The model is linear in log
    relativity between knots, so a consumer who interpolates the ``Relativity``
    column linearly gets numbers close to but not equal to the model -- second
    order in the segment width, worst mid-segment on the steepest segment.
    Emitting both, and stating the rule in the sheet, puts that choice in the
    consumer's hands instead of hiding it.

    Exactness holds in both centerings.  ``centering="mean"`` shifts the whole
    block by a constant so its relativities have geometric mean 1; the block
    reports that constant as ``centering_shift`` and the payload folds the
    total into ``base_relativity``, so the workbook's block-times-base product
    reproduces the model either way.  It did not always: the removed constants
    used to go nowhere, scaling every reconstructed prediction by a uniform
    factor (issue #253).
    """
    ti = _main_effect_inference(model, name, centering)
    return RatingTableBlock(
        name=name,
        kind="piecewise",
        table=pd.DataFrame(
            {
                name: np.asarray(ti.x, dtype=np.float64),
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Log relativity": np.asarray(ti.log_relativity, dtype=np.float64),
            }
        ),
        extrapolation=model._specs[name].extrapolation,
        centering_shift=float(ti.centering_shift),
    )


def _numeric_block(model: SuperGLM, name: str, centering: str) -> RatingTableBlock:
    ti = _main_effect_inference(model, name, centering)
    return RatingTableBlock(
        name=name,
        kind="numeric",
        table=pd.DataFrame(
            {
                name: ["per_unit"],
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": [0.0],
            }
        ),
        # Always 0.0 today -- a single-valued term has no mean to center on,
        # so ``_recenter_term`` returns it untouched.  Read from the term
        # anyway rather than hard-coded: this is the one number that must
        # agree with what the block's values actually carry, and a lie here
        # moves every prediction in the workbook.
        centering_shift=float(ti.centering_shift),
    )


def _fit_used_offset(model: SuperGLM) -> bool:
    return bool(
        getattr(
            model,
            "_fit_used_offset",
            getattr(model, "_fit_offset", None) is not None,
        )
    )


def _require_log_link_offset_export(model: SuperGLM) -> None:
    if not isinstance(model._link, LogLink):
        raise ValueError(
            "Rating-table offset relativities are currently supported only for log-link models."
        )


def _require_log_link_export(model: SuperGLM) -> None:
    """A rating table is multiplicative, which is a statement about the LINK.

    Every block reports ``exp`` of a term's contribution and the payload's
    contract multiplies them into the base.  That reproduces ``model.predict``
    only when the link is ``log``, because only then is the mean a product of
    per-term factors.  Under any other link the same arithmetic still runs and
    still produces a complete-looking workbook, of the wrong quantity:

        gaussian/identity: the product is ``exp(linear_predictor)``, measured
            9.19 maximum relative error against ``model.predict``
        binomial/logit:    the product is the ODDS, measured 3.45 -- a
            reconstructed 2.0201 against a predicted probability of 0.6689

    Neither can be repaired by applying the inverse link to the product,
    because the result would no longer be a table of factors: there is no
    multiplicative tariff for a logit model, so the export refuses rather than
    emit one.  Log-link ``Gamma``, ``Tweedie`` and ``Poisson`` are unaffected,
    which is the whole insurance-ratemaking case this module exists for.

    The offset path has always required this (``_require_log_link_offset_export``,
    which stays for its more specific message); the gap was that a model
    without an offset never reached a check at all.
    """
    if not isinstance(model._link, LogLink):
        raise ValueError(
            "Rating-table export is supported only for log-link models, because the "
            "exported table is multiplicative: base_relativity times one relativity "
            "per block reproduces model.predict only under a log link. This model "
            f"uses {type(model._link).__name__}, for which that product would be "
            "exp(linear predictor) rather than the prediction."
        )


def _require_unclamped_response_export(model: SuperGLM) -> None:
    """``clip_mu`` bounds a binomial mean, and a bound is not a factor.

    ``model.predict`` does not stop at the inverse link: it finishes with
    ``clip_mu``, which for ``Binomial`` clamps the mean into
    ``[1e-7, 1 - 1e-7]``.  A clamp cannot be distributed over the exported
    blocks, so no multiplicative table can express it, and the refusal is by
    FAMILY rather than by frame because the payload is frame-independent -- a
    table that satisfies the clamp on every row of the export frame still
    breaks on a risk rated later.

    The usable domain is what settles it.  Under a log link a binomial mean is
    ``exp(eta)``, so the table agrees with ``predict`` only on
    ``-16.118 <= eta <= -1.0e-7``: that is **20.1%** of the ``[-80, 0]`` band
    ``stabilize_eta`` allows, and both edges are ordinary rather than
    degenerate.  Below it lies any probability under 1e-7, which is a real
    rare-event rate and not a mis-scaled fit; above it lies ``mu > 1``, and
    predicting above one out of sample is the CHARACTERISTIC hazard of
    log-binomial regression -- which is fitted precisely because it yields the
    multiplicative risk ratios someone would then want a rating table of.
    Measured: at eta 0.1 the table returns 1.105 where ``predict`` returns
    0.9999999, 10.5% out at a predictor one ten-thousandth of the way to the
    stabilization bound; and on a three-level fit with one 100%-event level,
    974 of 3000 rows are rewritten by the clamp for 4.40e-01 maximum relative
    error against a documented round-off claim of 7.1e-15.

    Refusing the family is what makes ``_require_unsaturated_predictor_export``
    able to check the predictor alone: with ``Binomial`` gone, ``clip_mu``
    cannot fire at all under a log link.  The positive families are clamped to
    ``[1e-50, 1e50]``, which ``exp`` of the stabilization band -- ``[1.8e-35,
    5.5e34]`` -- sits strictly inside, and ``Gaussian`` is not clamped.
    """
    from superglm.distributions import Binomial

    if isinstance(model._distribution, Binomial):
        raise ValueError(
            "Rating-table export is not supported for Binomial models. The exported "
            "table is a product of per-block factors, and model.predict finishes by "
            "clamping a binomial mean into [1e-7, 1 - 1e-7]; a clamp is not a factor, "
            "so no table can carry it. Under a log link the two agree only for "
            "-16.118 <= eta <= -1e-7 -- 20.1% of the permitted range -- and outside it "
            "the workbook returns a 'probability' above one, or below the clamp, while "
            "the model returns the clamped value."
        )


def _require_unsaturated_predictor_export(
    model: SuperGLM,
    frame: EagerFrame,
    offset: NDArray | None,
) -> None:
    """A saturated row stops the export, because a clip is not a factor either.

    ``stabilize_eta`` clips a log-link predictor to ``[-80, 80]`` before the
    inverse link, so a quasi-separated row is predicted at ``exp(80)`` while the
    workbook, which has no such bound, keeps returning ``exp(eta)``.  Tested
    against ``stabilize_eta`` itself rather than against a re-derived ``80``, so
    the gate cannot drift from the rule if the band moves.

    Measured, with the base relativity representable throughout so that no
    existing guard fires: a Poisson fit whose own frame reaches eta 19.0 exports
    cleanly, and the table then misses ``model.predict`` by 1.78e+08 on rows out
    to eta 99.0.  The ratio is exactly ``exp(eta - 80)``, first breaching the
    round-off claim at eta 80.41.

    Not dead code, which was the assumption worth checking: a fit on
    ``y = exp(eta)`` out to eta 85 lands 293 of its own 400 rows past the clip
    and still exported, with a representable base of 1.269e-203.

    What this CANNOT see is a row the export frame does not contain.  The
    payload is frame-independent by design -- a ``Numeric`` block is one
    per-unit relativity that a consumer raises to whatever value it holds -- so
    a table that passes here still diverges above eta 80 on a risk rated later.
    That is stated in ``build_rating_table_payload``'s contract rather than
    guarded, because there is no row here to inspect.
    """
    from superglm.links import stabilize_eta
    from superglm.model.base import predict_eta_raw_exact

    raw = predict_eta_raw_exact(model, frame, offset)
    stabilized = stabilize_eta(raw, model._link)
    clipped = int(np.count_nonzero(raw != stabilized))
    if not clipped:
        return

    raise ValueError(
        "Rating-table export is refused because model.predict saturates on this "
        f"frame: {clipped} of {len(raw)} rows have a linear predictor outside the "
        f"range model.predict clips a log link to, reaching "
        f"{float(np.max(np.abs(raw))):.4g} against a bound of "
        f"{float(np.max(np.abs(stabilized))):.4g}. The exported table is a product of "
        "per-block factors and a clip is not a factor, so the workbook would keep "
        "returning exp(linear predictor) where the model returns the saturated value "
        "-- a complete-looking sheet that silently disagrees with the model it came "
        "from. A fit that saturates is quasi-separated or mis-scaled; refit or "
        "rescale rather than export it."
    )


def _emitted_relativities(block: RatingTableBlock | InteractionTableBlock) -> NDArray:
    """Every number one exported block asks a consumer to multiply by.

    Two shapes, because the export has two.  A main-effect block is keyed on
    its first column and states one factor per row in ``Relativity``; an
    interaction block is a CELL TABLE whose first column is the row key and
    whose every remaining column is a factor.  Reading the interaction table
    positionally rather than by name is deliberate: its column headers are the
    second parent's level labels, which are data, so there is no name to match
    on -- and slicing from column 1 is exactly how ``_continuous_interaction_
    block`` and ``_interaction_blocks`` build it.

    A main-effect block missing ``Relativity`` is a failure rather than a skip.
    Every builder emits that column today, so this never fires; making it raise
    means a future block kind that names its factor column something else is
    caught by the guard rather than quietly exempted from it.
    """
    if isinstance(block, InteractionTableBlock):
        return np.asarray(block.table.iloc[:, 1:].to_numpy(), dtype=np.float64).ravel()
    if "Relativity" not in block.table:
        raise ValueError(
            f"Rating-table block {block.name!r} (kind {block.kind!r}) has no 'Relativity' "
            "column, so its exported factors cannot be validated. Every block kind must "
            "name its factor column 'Relativity'."
        )
    return np.asarray(block.table["Relativity"], dtype=np.float64)


def _require_usable_relativities_export(
    main_effects: list[RatingTableBlock],
    interactions: list[InteractionTableBlock],
) -> None:
    """No exported factor may be one a consumer cannot multiply by.

    Stated as a property of the EMITTED NUMBER rather than of the routine that
    produced it, because the export has two different ways of producing an
    unusable factor and no single provenance covers both.

    * ``_safe_exp`` clips its argument to +/-500 so a quasi-separated
      CONFIDENCE BOUND comes back finite instead of ``inf``.  Right for a
      bound, wrong for a factor, for the same reason it was wrong for the base
      (see ``_base_relativity``): ``exp(+/-500)`` is representable -- 1.4e+217
      and 7.1e-218 -- so a check for ``inf`` or ``0.0`` never fires on it, and
      only a comparison against the clip endpoints catches it.  This is the
      ``Piecewise`` path in both centerings, and ``Categorical`` under
      ``centering="mean"``.
    * Everything else exponentiates with a plain ``np.exp``, where the failure
      is the opposite: ``inf`` above 709.78, subnormals below -708.4, and
      exactly ``0.0`` below -745.13.  Measured on a fitted interaction --
      ``a`` and ``b`` as ``Polynomial(degree=4)`` on data lying along a
      diagonal band -- the exported grid carried three cells of exactly ``0.0``
      and a maximum of 1.8e+155 (issue #289).

    So both arms are needed and neither is redundant, and the guard is stated
    over the values rather than over the call graph.

    Interaction cells are checked here for the first time.  The blow-up above
    needs no cancellation and no extreme coefficient: the export samples a
    continuous interaction on the parents' BOUNDING BOX, and data that occupies
    a diagonal band leaves the two off-diagonal corners with no exposure at
    all, so the corner cells are pure extrapolation of a tensor surface.  Every
    row's predictor stayed inside +/-3.14 against a saturation bound of 80, the
    base relativity was 9.7e-24 and representable, and every main-effect
    relativity was inside the clip -- so the saturation gate, the base guard
    and the per-block guard were all silent while the workbook shipped a factor
    of zero.

    Exactly ``0.0`` is refused, and that is a decision rather than a fallout
    (issue #291).  It used to be carved out of the floor comparison.  It is the
    one factor that is never usable: it drives every premium for the rows it
    covers to zero while every relativity RATIO in the workbook still reads
    correctly, which is the silent shape of issue #253 and precisely the
    reasoning ``_base_relativity`` already refuses ``0.0`` for.  Negative
    values fall under the same comparison, since no multiplicative tariff has a
    negative factor.

    Checked per block, not on the product, because CANCELLATION is what hides
    it.  Term contributions of ``+800`` and ``-700`` have a perfectly ordinary
    product, ``exp(100)``; clipped they become ``exp(500) * exp(-500) = 1``, so
    the workbook rates every such risk 2.7e+43 low while the predictor -- and
    therefore the base guard, and therefore
    ``_require_unsaturated_predictor_export`` -- stays entirely healthy.  The
    sum is well behaved precisely when the parts are not.
    """
    from superglm.inference._term_types import _MAX_LOG_REL

    ceiling = float(np.exp(_MAX_LOG_REL))
    floor = float(np.exp(-_MAX_LOG_REL))
    blocks: list[RatingTableBlock | InteractionTableBlock] = [*main_effects, *interactions]
    for block in blocks:
        values = _emitted_relativities(block)
        # ``<= floor`` and not ``< floor``: the floor IS a clip endpoint, so a
        # value sitting exactly on it is the stand-in this refuses.  It also
        # sweeps up the subnormals, ``0.0`` and every negative in one
        # comparison.
        bad = ~np.isfinite(values) | (values >= ceiling) | (values <= floor)
        if not np.any(bad):
            continue
        raise ValueError(
            f"Rating-table block {block.name!r} carries {int(np.count_nonzero(bad))} "
            "relativity value(s) that a consumer cannot multiply by: outside "
            f"(exp(-{_MAX_LOG_REL:g}), exp({_MAX_LOG_REL:g})) = "
            f"({floor:.4g}, {ceiling:.4g}), or not finite. Such a value is either a "
            "stand-in _safe_exp clipped to that range or a plain exp that overflowed "
            "to inf, underflowed to a subnormal or to exactly 0.0 -- and a factor of "
            "0.0 zeroes every premium it touches while every relativity ratio on the "
            "sheet still reads correctly. Two blocks whose contributions cancel can "
            "leave the prediction well behaved while their individual factors do not, "
            "so this is checked per block. On a main-effect block the fit is "
            "quasi-separated or mis-scaled: refit or rescale. On an interaction block "
            "the cause is more often the sampled grid, which spans the parents' "
            "bounding box and cannot be narrowed: if the data occupies only part of "
            "that box, the corner cells are extrapolation with no exposure behind "
            "them, and the remedy is to refit the parents at a lower degree or with "
            "fewer knots."
        )


def _resolve_export_offset(
    offset,
    model: SuperGLM,
    X: FrameLike,
) -> NDArray | None:
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=np.float64).ravel()
        if len(offset_arr) != len(X):
            raise ValueError("offset must have the same length as X.")
        return offset_arr

    fit_offset = getattr(model, "_fit_offset", None)
    if X is getattr(model, "_fit_X_ref", None) and fit_offset is not None:
        offset_arr = np.asarray(fit_offset, dtype=np.float64).ravel()
        if len(offset_arr) != len(X):
            raise ValueError(
                "The fitted offset has a different length from X; pass offset= "
                "when exporting a frame other than the original fit frame."
            )
        return offset_arr

    raise ValueError("Pass offset= when exporting a frame other than the original fit frame.")


def _resolve_offset_source(
    offset_source,
    X: EagerFrame,
    *,
    offset_name: str | None,
) -> tuple[pd.Series, str]:
    if isinstance(offset_source, str):
        if offset_source not in X.columns:
            raise ValueError(f"offset_source column {offset_source!r} is not present in X.")
        source = pd.Series(X.column_array(offset_source), name=offset_source)
        name = offset_name if offset_name is not None else offset_source
    elif isinstance(offset_source, pd.Series):
        source = offset_source.reset_index(drop=True)
        if offset_name is not None:
            name = offset_name
        elif offset_source.name is not None:
            name = str(offset_source.name)
        else:
            raise ValueError(
                "offset_name is required when offset_source is an unnamed array-like object."
            )
    else:
        if offset_name is None:
            raise ValueError(
                "offset_name is required when offset_source is an unnamed array-like object."
            )
        source = pd.Series(offset_source)
        name = offset_name

    if len(source) != len(X):
        raise ValueError("offset_source must have the same length as X.")
    if source.isna().any():
        raise ValueError("offset_source cannot contain missing values.")
    if not str(name).strip():
        raise ValueError("offset_name must not be blank.")
    if str(name) in _OFFSET_SOURCE_RESERVED_COLUMNS:
        reserved = ", ".join(sorted(_OFFSET_SOURCE_RESERVED_COLUMNS))
        raise ValueError(f"offset_name cannot be one of the reserved columns: {reserved}.")
    return source.reset_index(drop=True), name


def _weights_array(n_rows: int, sample_weight: NDArray | None) -> NDArray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if len(weights) != n_rows:
        raise ValueError("sample_weight must have the same length as X.")
    return weights


def _significant_digits(values: NDArray, digits: int) -> NDArray:
    """Round to ``digits`` SIGNIFICANT digits, so the coarseness is relative.

    ``np.round(x, 12)`` is twelve decimal PLACES, absolute.  What the offset
    block needs it for is collapsing the float noise of ``exp(log(exposure))``
    into distinct tariff levels, and that is a relative question: near ``1.0``
    twelve places is ~1e-12 relative and does exactly that, but the same
    absolute grid degrades in proportion as the multiplier shrinks -- it merges
    genuinely distinct small multipliers into one printed key, and below ~5e-13
    it collapses the column to a single level keyed ``0.0``, a factor that
    prices every risk at zero (issue #303).

    Twelve significant digits keeps the constant and its intent and makes the
    grid magnitude-invariant.  Scaled through the MANTISSA rather than by
    ``10 ** (digits - 1 - exponent)``, because that multiplier overflows for a
    multiplier near the bottom of float64 while the value it would scale is
    perfectly ordinary; ``10.0 ** exponent`` never does, since it is the
    magnitude of a representable number.  Zeros and non-finite values are left
    exactly as they are -- they have no significant digits to keep, and both
    are refused downstream on their own terms.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.array(values, dtype=np.float64, copy=True)
    scalable = np.isfinite(values) & (values != 0.0)
    if not np.any(scalable):
        return out
    # Normal magnitudes only.  Below ``2.2e-308`` the magnitude is itself
    # subnormal, so ``x / magnitude`` loses mantissa bits and the twelve-digit
    # claim degrades with no signal; nearer the floor ``10.0 ** exponent``
    # underflows to ``0.0`` and would turn the value into a ``nan``, which is a
    # worse answer than the value itself.  Such a multiplier is refused
    # downstream on its own terms.
    scalable &= np.abs(values) >= np.finfo(np.float64).tiny
    if not np.any(scalable):
        return out
    magnitude = 10.0 ** np.floor(np.log10(np.abs(values[scalable])))
    mantissa = values[scalable] / magnitude
    keep = 10.0 ** (digits - 1)
    out[scalable] = np.round(mantissa * keep) / keep * magnitude
    return out


def _offset_multiplier_block(
    offset: NDArray,
    n_rows: int,
    sample_weight: NDArray | None,
    *,
    n_bins: int,
    bin_strategy: str,
) -> RatingTableBlock | None:
    offset_arr = np.asarray(offset, dtype=np.float64).ravel()
    if len(offset_arr) != n_rows:
        raise ValueError("offset must have the same length as X.")

    weights = _weights_array(n_rows, sample_weight)
    multiplier = np.exp(offset_arr)
    exact_multiplier = _significant_digits(multiplier, 12)
    levels, inverse = np.unique(exact_multiplier, return_inverse=True)

    if len(levels) < 20:
        exposure = np.bincount(inverse, weights=weights, minlength=len(levels))
        table = pd.DataFrame(
            {
                "Offset Multiplier": levels.astype(float),
                "Relativity": levels.astype(float),
                "Weight": exposure.astype(float),
            }
        )
        return RatingTableBlock(name="Offset Multiplier", kind="offset", table=table)

    from superglm.diagnostics.discretize import _compute_edges

    edges = _compute_edges(multiplier, weights, n_bins, bin_strategy)
    actual_n_bins = len(edges) - 1
    bin_idx = np.digitize(multiplier, edges, right=False)
    bin_idx = np.clip(bin_idx, 1, actual_n_bins) - 1

    rows: list[dict[str, str | float]] = []
    for b in range(actual_n_bins):
        mask = bin_idx == b
        exposure = float(weights[mask].sum()) if np.any(mask) else 0.0
        # Branching on the WEIGHT rather than on emptiness, because those are
        # not the same condition and the difference is reachable.
        # ``sample_weight`` is only validated non-negative, and ``_compute_edges``
        # builds ``"uniform"`` edges from ``x[sample_weight > 0.0]`` -- so a bin
        # can hold rows that between them carry no weight at all, at which point
        # ``np.average`` raises ``ZeroDivisionError`` from inside NumPy.  A bin
        # with no weight has no weighted mean to report, whether or not it has
        # rows, so both take the same answer.
        if exposure <= 0.0:
            # An empty bin still ships a row, so it still ships a FACTOR, and
            # ``0.0`` is the one value a multiplicative tariff can never carry:
            # it prices every risk that lands in the bin at zero while every
            # relativity ratio on the sheet still reads correctly -- the silent
            # shape of issue #253, one level down.  ``1.0`` is no better; it is
            # a neutral-looking number that is generally not even inside the
            # interval the row is keyed on.
            #
            # This block is the one place where the right answer needs no
            # estimate.  Its "relativity" IS its key: the factor for a risk is
            # that risk's own offset multiplier, and a risk in this bin has a
            # multiplier somewhere in ``[edges[b], edges[b + 1])``.  The
            # midpoint is therefore the representative that minimises the worst
            # absolute error over everything the bin can contain, and it is the
            # same statistic the non-empty branch reports -- the weighted mean
            # multiplier -- under the only distribution an empty bin licenses.
            # Under ``bin_strategy="exposure_quantile"`` this branch is
            # unreachable, because every edge is a positive-weight data value;
            # under ``"uniform"`` on a skewed exposure it is the normal case
            # (issue #291: measured 123 of 150 bins on an 800-row fit).
            avg_multiplier = 0.5 * (float(edges[b]) + float(edges[b + 1]))
        else:
            avg_multiplier = float(np.average(multiplier[mask], weights=weights[mask]))
        rows.append(
            {
                "Offset Multiplier": _format_interval(float(edges[b]), float(edges[b + 1])),
                "Relativity": avg_multiplier,
                "Weight": exposure,
            }
        )
    return RatingTableBlock(
        name="Offset Multiplier",
        kind="offset",
        table=pd.DataFrame(rows),
    )


def _offset_source_block(
    offset: NDArray,
    offset_source,
    X: EagerFrame,
    sample_weight: NDArray | None,
    *,
    offset_name: str | None,
    offset_kind: str,
    offset_max_exact_levels: int,
    offset_mapping_rtol: float,
    offset_mapping_atol: float,
) -> RatingTableBlock:
    if offset_kind not in {"auto", "discrete"}:
        raise ValueError("offset_kind must be 'auto' or 'discrete'.")

    source, source_name = _resolve_offset_source(offset_source, X, offset_name=offset_name)
    n_unique = int(source.nunique(dropna=False))
    if n_unique > offset_max_exact_levels:
        raise ValueError(
            f"Offset source {source_name!r} has {n_unique} distinct values, exceeding "
            f"offset_max_exact_levels={offset_max_exact_levels}. Increase "
            "offset_max_exact_levels explicitly if all values are intended tariff levels."
        )

    offset_arr = np.asarray(offset, dtype=np.float64).ravel()
    weights = _weights_array(len(X), sample_weight)
    df = pd.DataFrame(
        {
            "__offset_source__": source,
            "__offset__": offset_arr,
            "__weight__": weights,
        }
    )

    rows: list[dict[str, object | float]] = []
    for level, group in df.groupby(
        "__offset_source__",
        sort=False,
        dropna=False,
        observed=True,
    ):
        if group.empty:
            continue
        offset_values = group["__offset__"].to_numpy(dtype=np.float64)
        multipliers = np.exp(offset_values)
        group_weights = group["__weight__"].to_numpy(dtype=np.float64)
        weight_sum = float(group_weights.sum())
        if weight_sum > 0.0:
            representative = float(np.exp(np.average(offset_values, weights=group_weights)))
        else:
            representative = float(multipliers[0])
        if not np.allclose(
            multipliers,
            representative,
            rtol=offset_mapping_rtol,
            atol=offset_mapping_atol,
        ):
            raise ValueError(
                f"Offset source {source_name!r} is not a valid discrete lookup: "
                f"level {level!r} maps to multiple offset multipliers. Pass a more "
                "granular offset_source, or keep the offset calculation outside the "
                "rating table."
            )
        rows.append(
            {
                source_name: level,
                "Relativity": representative,
                "Weight": weight_sum,
            }
        )

    return RatingTableBlock(
        name=source_name,
        kind="offset",
        table=pd.DataFrame(rows, columns=[source_name, "Relativity", "Weight"]),
    )


def _interaction_beta(model: SuperGLM, name: str) -> np.ndarray:
    groups = [g for g in model._groups if g.feature_name == name]
    return np.concatenate([model.result.beta[g.sl] for g in groups])


def _reconstruct_interaction(ispec, beta: NDArray, n_bins: int) -> dict:
    # Shared with the impact sweep, so "does this spec take n_points" and
    # "which orientation is this surface" are answered once rather than in two
    # places that can disagree -- which is the shape issue #287 took.
    from superglm.diagnostics.discretize import reconstruct_interaction

    return reconstruct_interaction(ispec, beta, n_bins)


def _continuous_interaction_block(
    name: str,
    raw: dict,
    parent1: str,
    parent2: str,
) -> InteractionTableBlock:
    from superglm.diagnostics.discretize import _ascending_grid, orient_grid_surface

    x1 = np.asarray(raw["x1"], dtype=np.float64)
    x2 = np.asarray(raw["x2"], dtype=np.float64)
    relativity = orient_grid_surface(name, x1, x2, raw["relativity"])
    # Sorted with the same helper the sweep uses, so the block a reader looks
    # up and the surface the sheet measures are in one order.  The set of cells
    # is unchanged either way, but a row exactly midway between two nodes has
    # its tie broken by INDEX -- so on a reconstruction that supplies a
    # descending axis the two would otherwise pick different nodes for that
    # row, and the sheet would describe a factor the workbook does not carry.
    x1, x2, relativity = _ascending_grid(x1, x2, relativity)

    table = pd.DataFrame(relativity, columns=[_format_axis_value(v) for v in x2])
    table.insert(0, parent1, [_format_axis_value(v) for v in x1])
    return InteractionTableBlock(name=name, table=table, kind="grid")


def _interaction_blocks(model: SuperGLM, n_bins: int) -> list[InteractionTableBlock]:
    blocks: list[InteractionTableBlock] = []
    for name in model._interaction_order:
        ispec = model._interaction_specs[name]
        parent1, _ = ispec.parent_names
        raw = _reconstruct_interaction(ispec, _interaction_beta(model, name), n_bins)
        # The sweep's predicate, imported rather than respelled.  Four review
        # rounds found the same failure -- the exporter routes a grid on one
        # rule and the sweep re-decides with a second, and where they disagree
        # the whole payload dies refusing a block that shipped.  This was the
        # last copy of that rule; sharing it makes a fifth divergence
        # inexpressible.
        if _grid_reconstruction_keys() <= set(raw):
            # ...and the same rule for whether a grid is a LOOKUP, so a
            # surface over a categorical parent is declined here rather than
            # exported as an axis a reader cannot find their risk on and then
            # read as float64 by the sweep.
            from superglm.diagnostics.discretize import unpositionable_grid_parent

            offender = unpositionable_grid_parent(ispec, model._specs)
            if offender is not None:
                raise NotImplementedError(
                    f"Interaction {name!r} reconstructs a numeric grid, but its parent "
                    f"{offender!r} is a Categorical whose values have no position on a "
                    "grid axis, so the surface is not a lookup table for it. Use an "
                    "OrderedCategorical parent, whose level scores do have positions, "
                    "or reconstruct the term as a cell table."
                )
            parent2 = ispec.parent_names[1]
            blocks.append(_continuous_interaction_block(name, raw, parent1, parent2))
            continue

        if "pairs" not in raw:
            raise NotImplementedError(
                f"Interaction {name!r} is not yet exportable as a rating table."
            )

        levels1 = raw["levels1"]
        levels2 = raw["levels2"]
        rows = []
        for level1 in levels1:
            row: dict[str, str | float] = {parent1: level1}
            for level2 in levels2:
                key = f"{level1}:{level2}"
                row[level2] = float(raw["relativities"].get(key, 1.0))
            rows.append(row)
        blocks.append(InteractionTableBlock(name=name, table=pd.DataFrame(rows), kind="cells"))
    return blocks


def _total_centering_shift(blocks: list[RatingTableBlock]) -> float:
    """The constant the exported base relativity has to carry back.

    A reporting centering subtracts a per-term constant from that term's log
    relativities.  Left there, the constants are simply gone: a consumer who
    multiplies ``base_relativity`` by one relativity per block -- the whole
    documented use of the workbook -- rates every risk by
    ``exp(-sum_t shift_t)`` of what the model says.  A uniform factor is the
    worst shape that error can take, because every relativity RATIO in the
    workbook is still exactly right, so nothing short of comparing absolute
    predictions against ``model.predict`` reveals it (issue #253).

    Summed over the blocks, from the constant each one recorded, for two
    reasons.  The set of shifted terms is not the set of exported terms:
    ``OrderedCategorical`` is never recentered, a single-valued ``Numeric``
    has nothing to center, and the offset blocks are not relativities of a
    fitted term.  The binned continuous blocks DO carry a shift -- read from
    the same term the exact blocks read theirs from, since the discretisation
    path they are built on is never told the centering (issue #293).  And the constant removed
    from a grouped categorical is the mean over its GROUPED levels, computed
    before the term is expanded back to the original ones -- so even for the
    terms that are shifted, re-deriving the constant from the values on the
    sheet gives a different number and the product stops closing.

    Interactions are absent by the same rule: ``_interaction_blocks``
    reconstructs from beta directly and applies no centering, so it removes
    nothing to give back.
    """
    return float(sum(block.centering_shift for block in blocks))


def _base_relativity(log_base: float) -> float:
    """``exp`` of the exported base, refusing a result that cannot be applied.

    Every relativity the centering constant came out of goes through
    ``_safe_exp``, which clips its argument to +/- 500 so a quasi-separated
    level yields a large finite confidence bound instead of ``inf``.  That is
    the right discipline for a bound and the wrong one here.  This number is a
    MULTIPLIER on every row of the tariff, and the argument it exponentiates is
    now a sum -- the ordered-reference intercept plus the total the blocks
    subtracted -- so it reaches further from zero than any single term does.
    Clipping it would return a workbook that looks complete and rates every
    risk by a factor of ``exp(clip)`` off the model, which is exactly the
    silent uniform error issue #253 was about; clipping is therefore refused in
    favour of failing at the export boundary.

    Both tails are rejected, and the lower one stops at the smallest NORMAL
    float rather than at zero.  ``exp`` overflows to ``inf`` above about
    709.78, but on the way down it does not fall off a cliff: below about
    -708.4 it returns subnormals, which are finite and strictly positive and
    so would pass an ``isfinite``/``> 0`` check while having already lost most
    of their mantissa.  A subnormal has no implicit leading bit, so its
    significand shrinks with its exponent, and the exported base stops being
    the number this function was asked for.  On the round trip:

        exp(-720.0) = 2.03e-313, whose log is off by 3.0e-12
        exp(-740.0) = 4.2e-322,  off by 2.6e-03
        exp(-745.0) = 5e-324,    off by 5.6e-01 -- one significant bit

    The last is a workbook rating every risk by a factor of 1.75, which is the
    same silent uniform error as a clip, reached through the guard meant to
    prevent it.  ``tiny`` is therefore the floor: it is exactly where the
    mantissa becomes full, and it is independently where Excel stops, since
    Excel declines to implement IEEE 754's denormals -- "denormalized numbers
    by their very nature have a variable number of significant digits" -- and
    publishes 2.2251E-308 as its smallest allowed positive number, this same
    value to the five figures it prints.

    ``0.0`` is caught by the same comparison and is no more usable than an
    infinite base: it drives every exported premium to zero while every
    relativity RATIO in the workbook still reads correctly.
    """
    # The overflow is the condition being tested for, so it is detected from
    # the result rather than announced as a warning on the way to the raise.
    with np.errstate(over="ignore", under="ignore"):
        base = float(np.exp(log_base))
    if not np.isfinite(base) or base < _SMALLEST_EXACT_BASE:
        raise RatingTableBaseNotRepresentableError(
            f"Exported base relativity is exp({log_base!r}) = {base!r}, which cannot "
            "multiply a rating table. The base is the ordered-reference intercept plus "
            "the centering constant the exported blocks gave back; a fit whose sum "
            "leaves the range float64 represents exactly (roughly [-708.4, 709.8]) is "
            "quasi-separated or mis-scaled, and the export refuses rather than emit a "
            "clipped, subnormal or infinite multiplier that would silently mis-rate "
            "every row."
        )
    return base


def _empty_impact_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "n_bins",
            "exported",
            "feature",
            "actual_bins",
            "deviance_original",
            "deviance_discretized",
            "deviance_change",
            "deviance_change_pct",
            "mean_abs_prediction_change_pct",
            "max_abs_prediction_change_pct",
            "prediction_correlation",
        ]
    )


def _impact_sweep(
    model: SuperGLM,
    X: EagerFrame,
    y: NDArray,
    sample_weight: NDArray | None,
    *,
    offset: NDArray | None,
    impact_bins: tuple[int, ...],
    bin_strategy: str,
    features: list[str],
    exported_n_bins: int,
) -> pd.DataFrame:
    """One row per approximated block per swept resolution.

    ``exported_n_bins`` is folded into the swept set and marked, because
    without it the sheet can describe every resolution EXCEPT the one the
    reader is holding.  The defaults are ``n_bins=150`` against
    ``impact_bins=(20, 50, 100, 200, 250)``, so without the fold-in a reader
    would take one of the ladder's numbers for their own.  Each rung is an
    INDEPENDENT measurement, not a point on a curve guaranteed to fall: a finer
    grid shrinks the worst-case bound, but successive nearest-node grids are
    not nested, so a nearer node can carry a value further from a given row.
    The ``exported`` column says which row is theirs, and it is the one to
    read.

    An empty ``impact_bins`` still means NO SWEEP, and the fold-in does not
    override it.  That is the documented opt-out for a caller who wants the
    tables without paying for the analysis, and turning it into "one sweep at
    ``n_bins``" would have been a silent cost -- the fold-in exists so the
    ladder describes the shipped table, not so a caller who declined the ladder
    gets one anyway.
    """
    rows: list[dict[str, float | int | str]] = []
    if not features or not impact_bins:
        return _empty_impact_frame()

    for n_bins in sorted(dict.fromkeys((*impact_bins, exported_n_bins))):
        result = model.discretization_impact(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            n_bins=int(n_bins),
            bin_strategy=bin_strategy,
            features=features,
        )
        for feature, table in result.tables.items():
            row: dict[str, float | int | str] = {
                "n_bins": int(n_bins),
                "exported": int(n_bins) == exported_n_bins,
                "feature": feature,
                "actual_bins": int(len(table)),
            }
            row.update(result.metrics)
            rows.append(row)
        # Beside the main-effect rows and in the same columns, because a
        # reader has to be able to see every block the workbook approximates
        # from one place.  ``actual_bins`` counts the block's own table rows,
        # which for a grid is its CELLS, so an interaction swept at 20 reports
        # 400 where a main effect reports 20.
        for interaction, table in result.interaction_tables.items():
            row = {
                "n_bins": int(n_bins),
                "exported": int(n_bins) == exported_n_bins,
                "feature": interaction,
                "actual_bins": int(len(table)),
            }
            row.update(result.metrics)
            rows.append(row)
    if not rows:
        return _empty_impact_frame()
    return pd.DataFrame(rows)


def build_rating_table_payload(
    model: SuperGLM,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    offset: NDArray | None = None,
    offset_source=None,
    offset_name: str | None = None,
    offset_kind: str = "auto",
    offset_max_exact_levels: int = 20,
    offset_mapping_rtol: float = 1e-10,
    offset_mapping_atol: float = 1e-12,
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    centering: str = "native",
    continuous_kind: str = "binned",
    allow_unbounded_extrapolation: bool = False,
) -> RatingTablePayload:
    """Build the renderer-independent rating-table payload.

    The payload's contract is multiplicative and per row:
    ``base_relativity`` times one relativity per main-effect block, times one
    per interaction block, reproduces ``model.predict``.

    That is a statement about the LINK before it is one about the blocks, so
    the export is restricted to log-link models and refuses the rest at the
    boundary.  Every block reports ``exp`` of a term's contribution, and only
    a log link makes the mean a product of those factors; under any other link
    the same arithmetic produces a complete-looking workbook of the wrong
    quantity -- ``exp(linear_predictor)`` for gaussian/identity, the ODDS for
    binomial/logit.  See ``_require_log_link_export``.

    It is also a statement about the RANGE, because ``model.predict`` saturates
    twice and a product of factors does not saturate at all.  A saturation
    cannot be distributed over the blocks, so each is refused rather than
    approximated, and between them they close the range claim:

    * ``clip_mu`` clamps the mean.  It reaches only ``Binomial``, whose usable
      band under a log link is a mere 20.1% of the permitted one, so that family
      is refused outright by ``_require_unclamped_response_export`` -- by family
      and not by frame, because the payload is frame-independent and a table
      that satisfies the clamp on every exported row still breaks on a risk
      rated later.  With ``Binomial`` gone the clamp cannot fire: the positive
      families' ``[1e-50, 1e50]`` strictly contains ``exp(+/-80)``, and
      ``Gaussian`` is unclamped.
    * ``stabilize_eta`` clips a log-link predictor to ``[-80, 80]``.  A frame
      that already saturates is refused
      (``_require_unsaturated_predictor_export``), but the same
      frame-independence applies -- a ``Numeric`` block is one per-unit
      relativity a consumer raises to whatever value it holds -- and no guard
      can inspect a row that is not here.

    So the contract above is exactly: it reproduces ``model.predict`` on every
    row whose predictor stays inside ``[-80, 80]``, and that is now the only
    exception, because the family whose second bound sat 80 units inside it is
    no longer exportable.  Beyond the clip the table returns ``exp(eta)`` and
    the model returns ``exp(80)``, a ratio of ``exp(eta - 80)``: measured on a
    Poisson fit whose own frame reached eta 19.0 and exported cleanly, 1.78e+08
    at eta 99.0, first breaching the round-off claim at eta 80.41.  A tariff
    that rates risks into that range is quasi-separated or mis-scaled rather
    than badly exported.

    Exact to round-off for the exactly tabulable blocks -- ``Categorical``,
    ``OrderedCategorical``, ``Numeric``, ``Piecewise``, and the
    categorical-by-categorical interaction, which is a full cell table.
    Measured on the interaction fixture in
    ``tests/test_rating_table_prediction_equivalence.py``: 4.4e-16 (native) and
    6.2e-16 (mean) maximum relative error against ``model.predict``, and
    5.6e-01 if the interaction block is left out of the product -- so that
    factor is load-bearing rather than decorative.

    One configuration is outside that claim, and the product cannot be formed
    at all rather than being formed imprecisely: an interaction whose
    categorical parent carries a ``grouping=``.  ``_categorical_block`` expands
    that parent's main-effect table back to the ORIGINAL level labels, while
    ``_interaction_blocks`` keys its cells on the GROUPED ones the interaction
    was fitted over, and no block in the payload carries the map between them.
    A consumer holding a raw row therefore has no cell to look up.  Measured on
    the committed fixture, a six-level territory collapsed to four: 821 of 1200
    rows (68.4%), and the same share of weight, key on a label the interaction
    table does not have.  This is
    pre-existing -- the interaction export predates issue #253 and centering
    does not touch it -- and is tracked as issue #286; until it is fixed, the
    product contract above holds for interactions whose parents are ungrouped.

    The OFFSET multiplier block is the other exception, and it is a binning
    rather than a rounding.  ``_offset_multiplier_block`` emits one exact row
    per distinct multiplier only while there are fewer than 20 of them; at 20 or
    more -- the normal case for a continuous exposure -- it bins them like a
    continuous block, keying rows on interval STRINGS and reporting the
    exposure-weighted average multiplier of each bin.  A consumer therefore
    cannot look its own multiplier up at all, and the factor it does find is a
    bin average.  Measured on 800 distinct multipliers over ``[0.1, 2.0]`` with
    ``n_bins=150``: every one of the 800 rows receives a factor that is not its
    own, by up to 8.86e-02 (median 2.9e-03), and the documented reconstruction
    misses ``model.predict`` by 8.86e-02 rather than by round-off.  Pass
    ``offset_source=`` for the exact form: that block is keyed on a raw column
    of the frame and is a lookup, which is why the equivalence tests reconstruct
    through it and treat the binned block as the exposure summary it is.

    A bin of that block with no exposure reports the MIDPOINT of its own
    interval, weight zero.  It used to report ``0.0``, which is not a summary
    of anything -- it is a factor that prices every risk landing in the gap at
    zero while every other number on the sheet stays right (issue #291).  The
    branch is unreachable under the default ``bin_strategy="exposure_quantile"``,
    whose edges are all data values; under ``"uniform"`` on a skewed exposure it
    is the normal case, measured at 123 of 150 bins on an 800-row fit.

    Lossy, by construction, for the approximated FITTED TERMS -- ``Spline``,
    ``Polynomial``, and the continuous-by-continuous interaction grid.  Two
    distinct errors ride on those, and the sheet reports the first for all
    three: the sweep is handed ``_continuous_features`` and
    ``_gridded_interaction_names`` -- the second read off the interaction
    blocks that were actually built, so the sheet names every one of those
    three and its metrics are joint over them.

    "Fitted terms" is the scope, not "every factor the workbook carries": the
    binned OFFSET MULTIPLIER block is approximated too, is measured below at
    8.86e-02, and is not swept, because it is not a fitted term and has no
    prediction-plan entry.  See the paragraph on issue #314 further down.

    The interaction's share is a SAMPLING rather than a binning, and it is the
    larger of the two differences between the sheet's two row kinds.  A binned
    main effect gives a consumer an interval to fall into and the bin's
    geometry-weighted mean inside it; ``_continuous_interaction_block`` gives
    them an ``n_bins``-per-axis grid keyed on axis VALUES, so a raw risk has no
    cell to fall into and the only available lookup is the nearest printed
    value on each axis.  What the reader applies is therefore the surface at a
    grid node, and the sweep measures exactly that.  Measured on an
    ``age``-by-``density`` Poisson fit with two ``Spline(n_knots=6)`` parents,
    600 rows: at ``n_bins=20`` the grid lookup misses the model's own
    interaction factor by up to 14.24% (mean 2.51%) and at ``n_bins=10`` by up
    to 36.93% (mean 5.21%).  With that error outside the sweep the sheet was
    not merely incomplete but WRONG about the workbook it describes -- its
    ``max_abs_prediction_change_pct`` read 29.75% at ``n_bins=20`` while the
    reconstruction from the workbook sat 32.81% from ``model.predict``, and its
    mean read 4.92% against 5.51%.  The two now agree to round-off, which is
    what ``test_the_sheets_prediction_change_is_the_whole_workbooks_error``
    pins (issue #287).

    The sheet's metric columns are JOINT over everything discretised at that
    resolution, so within one ``n_bins`` group only ``feature`` and
    ``actual_bins`` vary: the ``age:density`` row's
    ``max_abs_prediction_change_pct`` is the whole model's number under that
    binning, not the interaction's own.  That is what makes it the bound a
    consumer of the whole table needs, and it is stated because this issue
    exists precisely because one number on this sheet was read as more than it
    was.  ``impact_bins`` sweeps that ladder; an empty one skips the sweep
    entirely, and the worksheet is still created with its column headers and
    no rows -- ``write_rating_table_workbook`` creates it unconditionally.

    The bound is over the TERMS the sheet lists, not over every factor the
    workbook carries, and three limits on it are tracked rather than fixed
    here.  The binned OFFSET MULTIPLIER block -- the normal case for a
    continuous exposure without ``offset_source=``, and measured above at
    8.86e-02 -- is not swept at all, and the sweep passes the exact offset into
    every call, so a categorical-only model with a continuous offset gets an
    empty sheet beside a binned offset block (issue #314).  The sheet's
    discretised prediction goes through ``stabilize_eta`` and the workbook's
    product does not, so an approximation that pushes a row out of the link's
    range is clipped on the sheet and not in the table (issue #313).  And for a
    grid whose parent is a spline-mode ``OrderedCategorical``, the block is a
    DISCLOSURE rather than a lookup: ``_continuous_interaction_block`` prints
    the axis in mapped-score space while a consumer holds the level label, and
    no block carries the map between them -- the same keying gap as issue #286.

    Each swept resolution is also an INDEPENDENT measurement rather than a
    point on a falling curve.  A finer grid shrinks the worst-case bound, but
    successive nearest-node grids are not nested, so a nearer node can carry a
    value further from a given row; the ``exported`` row is the one to read,
    not the smallest number on the sheet.

    * The binning itself, which the ``discretization_impact`` sheet quantifies
      for the main-effect ``Spline`` and ``Polynomial`` blocks.
      It is bias-free in the GEOMETRY measure: each bin's relativity is the
      geometry-weighted mean of the smooth log-relativity over the rows in it,
      so the weighted mean residual is zero by construction in that measure and
      the binning error is spread rather than scale.  Which measure that is
      depends on the family, and the difference is not cosmetic.  For
      non-Tweedie families ``sample_weight`` is frequency mass, so the geometry
      measure IS the exposure weighting and the residual is zero in the same
      weights the exported ``Weight`` column reports.  Tweedie weights are
      prior precision, not case counts, so ``discretize`` deliberately gives
      every physical row unit geometry mass and each bin's value is the
      UNWEIGHTED mean; the identity then holds per physical row, and the
      prior-weighted residual is not zero.  Measured on a 900-row Tweedie(p=1.5)
      fit with 20 bins and prior weights drawn on [0.5, 20]: the residual mean
      is 1.2e-18 per physical row and -8.7e-04 under the prior weights, and per
      bin 1.9e-17 against 6.4e-03.  Those figures come from a probe, not from a
      committed fixture; what the suite pins is the mechanism they follow from,
      in ``test_the_binning_measure_is_physical_rows_for_tweedie_and_the_
      weights_otherwise``.
      So the impact sheet is a true bound on the binned blocks' error, which it
      previously was not.  The sweep bins on the exact ``edges`` array, and the
      exported block used to print its boundaries at ``.10g`` -- ten
      significant digits, where a binary64 needs seventeen -- so a consumer
      applying the printed strings re-binned every row within the rounding band
      of an edge, and under ``bin_strategy="exposure_quantile"`` the edges ARE
      data values, so a row sitting exactly on one flipped whenever its printed
      edge rounded up.  Measured on the equivalence fixture (900 rows, 150
      bins, two continuous terms): 302 of 302 printed edges differed from exact
      by up to 4.99e-09, 133 of 900 rows (14.8%) took a different factor, and
      the reconstruction carried 2.29e-01 maximum relative error against the
      discretised predictions the blocks are meant to reproduce exactly.  The
      boundaries are now printed at round-trip precision (``_format_number``),
      so the printed edge IS the edge the sweep bins on and that second error
      is identically zero (issue #278).

    ``continuous_kind : {"binned", "ppform"}, default "binned"`` is how a
    caller declines that binning, and it is stated here beside the loss it
    removes rather than in a parameter list of its own, because the choice is
    only meaningful against what the default costs.

    ``"binned"`` emits the key/relativity/weight block described above, which a
    consumer applies by pure lookup with no arithmetic.  It is an
    APPROXIMATION: the fitted curve is chopped into intervals carrying one
    exposure-weighted average each, so a row inside an interval receives a
    factor that is not its own.  Measured on a motor book with 81 distinct ages
    at ``n_bins=150``, the worst row was mis-rated by 60%, concentrated in the
    wide intervals the quantile strategy opens in the sparse tails -- which is
    where exposure is thinnest, where it is hardest to notice, and where a
    single large risk absorbs all of it.  ``discretization_impact`` quantifies
    the loss for a given model.

    ``n_bins`` is a BUDGET rather than a target, and a covariate with fewer
    distinct values than that budget is still binned today -- into fewer rows
    than it has values.  Measured on the documentation fixture in
    ``tests/test_rating_table_export.py``: 30 distinct ages under a budget of
    150 export as 29 interval rows.  So staying under the budget is not a route
    to an exact block; ``"ppform"`` is.

    ``"ppform"`` emits the exact piecewise-polynomial form of the fitted curve
    instead: one row per knot interval carrying four coefficients, evaluated as
    ``exp(a + b*u + c*u**2 + d*u**3)`` where ``u = (x - from) / (to - from)``.
    It reproduces the fitted model to machine precision -- 2.4e-15 maximum
    relative error against ``model.predict`` on the book above, against 6.0e-01
    for the binned block it replaces -- and usually in an order of magnitude
    fewer rows.  What it costs is a consumer that can evaluate a polynomial.
    ``u`` is NORMALISED onto ``[0, 1]`` rather than being the raw ``x - from``:
    on a covariate ranging to 1e5 a raw local variable loses enough precision
    in a fixed-scale decimal column to produce a 3.3x relativity error, which
    would be worse than the binning it replaces.

    The block is NINE columns rather than three, and it is a superset rather
    than a new shape.  ``<feature>``, ``Relativity`` and ``Weight`` stay in
    front of it unchanged, and ``from``, ``to``, ``a``, ``b``, ``c``, ``d`` are
    appended behind them, so an un-upgraded loader still reads it as a step
    function: it locates the block by the same header signature, slices the same
    three columns positionally, and gets the same approximate factor it gets
    today, while an upgraded one reads the coefficients and is exact.
    ``Relativity`` is the curve's value at ``from`` rather than the interval's
    average, so the two readings of one row agree at its left edge instead of
    disagreeing everywhere.  A consumer that STORES the coefficients must
    include them in any content digest it fingerprints a published package
    with; otherwise two models differing only in their coefficients fingerprint
    identically and the second is silently deduplicated into the first.

    Extrapolation is carried in the table rather than described beside it,
    because a cubic continued past its last knot is not merely wrong but
    unbounded -- 1581x the correct factor twenty-one years past the boundary of
    a real age curve -- and no note reliably stops a consumer evaluating one.
    Under the default ``extrapolation="clip"`` the block emits a constant
    leading row and a constant trailing row, so "match an interval, evaluate
    it" is correct outside the training range as well as inside.
    ``extrapolation="error"`` emits no unbounded rows at all: a value outside
    the knot range matches nothing and the consumer's lookup fails, which is the
    answer the model itself gives.  ``extrapolation="extend"`` is REFUSED unless
    ``allow_unbounded_extrapolation=True``, because it exports a tariff with no
    upper bound and filing guidance asks specifically about behaviour beyond the
    range of the training data; exported with that acknowledgement, its tails
    CLIP where the model extends -- an unbounded interval has no width, so the
    normalised ``u`` the coefficients are written against does not exist there
    -- and the exactness claim above is then over the knot range alone.

    Terms carrying a ``Constraint.postfit`` repair are refused under
    ``"ppform"``, naming the term.  That path's repaired curve has never been
    verified to be piecewise polynomial on the term's own knots, and silently
    converting an unmeasured path inside a block whose entire value is
    exactness is the failure this mode exists to remove.  ``Constraint.fit``
    constraints convert unchanged: the constraint transform leaves ordinary
    B-spline coefficients over the same knot vector.

    ``Polynomial`` terms stay binned under ``"ppform"``, and so does the
    continuous-by-continuous interaction grid -- the block is fixed at four
    coefficients while a ``Polynomial``'s degree is not, and a tensor patch is a
    materially different consumer contract.  Only ``Spline`` main effects
    convert, so one ``"ppform"`` workbook can carry both kinds of block at once
    and the sweep above still describes the ones that stayed binned.

    ``centering`` is a presentation choice and does not change what the
    payload rates.  ``"native"`` reports each term under the model's own
    identifiability constraint.  ``"mean"`` shifts the terms that have a mean
    to shift -- ``Categorical``, ``Piecewise``, and the binned ``Spline`` and
    ``Polynomial`` blocks -- and ``base_relativity`` absorbs the total so the
    product is unchanged; before issue #253 it did not, and every reconstructed
    prediction came out scaled by ``exp(-sum_t shift_t)``.

    A binned block takes the constant from the SAME term the exact blocks read
    theirs from rather than re-deriving it from its own bins, which would be
    the weighted mean of the bins -- a different number, leaving the workbook's
    blocks without one origin.  It used to take no constant at all: the
    discretisation path it is built on is never told the centering, so one
    ``centering="mean"`` request produced a mean-centred categorical block
    beside a native spline block, with nothing on the sheet saying which was
    which (issue #293).

    Being SHIFTED is therefore not the same as arriving at geometric mean 1,
    and only the exact blocks do both.  The constant is the mean of the log
    relativities over the vector ``term_inference`` reports, which for a
    ``Categorical`` or a ``Piecewise`` IS that block's own rows, so
    ``exp(mean(log(Relativity)))`` over one of those is exactly 1.  A binned
    block's rows are ``discretization_impact`` bin averages while its constant
    is the mean over the reporting grid, so the same quantity comes out at
    ``exp(bin_mean - grid_mean)`` -- equal to 1 only where the covariate is
    uniform over the fitted range.  Measured on the equivalence fixture:
    1.000000 for the categorical and the piecewise blocks, 0.997889 and
    1.077824 for the two binned continuous ones.  One origin across the
    workbook is the trade being made; per-block geometric normalisation is what
    it is traded for, and it is a reader's ratio that has to know.

    That mode is still a PARTIAL centering, and deliberately so.  An
    ``OrderedCategorical`` is already anchored on its base level and is not
    recentered; a ``Numeric`` reports one per-unit relativity with no mean to
    take; and an offset block is not a fitted term's relativity.  Only the
    blocks that were shifted contribute to the transferred constant, which is
    why it is summed from the shift each block recorded rather than assumed to
    run over every exported term.

    Partial is also what the ratemaking literature asks for, so the mixture
    above is the published convention rather than an accident of which term
    types happened to implement a shift.  A GLM needs one aliased level per
    FACTOR, chosen per factor (Anderson et al., *A Practitioner's Guide to
    Generalized Linear Models*, CAS Study Note, 3rd ed. 2007, s1.127), and the
    intercept's value is then a function of that arbitrary per-factor choice
    (s1.129) -- which is exactly why it has to move when the choice does.
    Goldburd, Khare, Tevet and Guller, *Generalized Linear Models for Insurance
    Rating* (CAS Monograph 5, 2nd ed.), s2.4.3, likewise directs the actuary to
    pick each variable's base level independently.  No source requires one
    uniform convention across terms.

    Moving the constant into the base is equally standard.  Werner and Modlin,
    *Basic Ratemaking* (CAS, 4th ed. 2010), ch. 10 p. 173, normalises the
    relativities and then states that "the base loss cost also needs to be
    adjusted to reflect the normalization", worked there as a product over
    factors -- which on the log scale is the sum over blocks taken here.
    Chapter 14 names the adjustment the off-balance factor.

    Raises
    ------
    ValueError
        If ``centering`` is not one of ``("native", "mean")``; if
        ``continuous_kind`` is not one of ``("binned", "ppform")``; if the model's
        link is not ``log``; if the family is ``Binomial``; if
        ``model.predict`` saturates on this frame; or if any emitted
        relativity, on a main-effect block or an interaction cell, is one a
        consumer cannot multiply by -- ``inf``, ``nan``, ``0.0``, negative,
        subnormal, or at or beyond the ``exp(+/-500)`` range ``_safe_exp``
        clips to.  Under ``continuous_kind="ppform"`` also if a convertible term
        carries a ``Constraint.postfit`` repair, or uses
        ``extrapolation="extend"`` without ``allow_unbounded_extrapolation=True``;
        both name the term.
    RatingTableBaseNotRepresentableError
        If the exported base relativity overflows or underflows float64.
    NotImplementedError
        If the model carries a term type the export does not support -- a
        ``RandomEffect`` or ``FactorSmooth`` main effect, or an interaction
        whose reconstruction is neither a cell table nor a grid.
    RuntimeError
        If the model has not been fitted.

    Three exception disciplines, deliberately, and the split is by what went
    wrong rather than by when.  ``ValueError`` means THIS MODEL OR FRAME IS NOT
    EXPORTABLE -- a structural refusal a caller answers by changing the model,
    the frame or the argument.  ``NotImplementedError`` is the third, and it is
    genuinely distinct by that same test: an unsupported term type is not
    something a caller answers by changing anything, it is something they
    answer by waiting for a release.  ``RatingTableBaseNotRepresentableError``
    (an ``OverflowError``) means the export ran and A NUMBER CAME OUT UNUSABLE,
    which is why it is a distinct root-exported class: it is the one outcome a
    caller might reasonably catch and report per-model in a batch, rather than
    fix.  (``RuntimeError`` for an unfitted model is the ordinary object-state
    complaint every method on the class makes, not a discipline of this one.)

    Where they collide, the ``ValueError`` wins.  A model that both carries an
    unusable relativity and has a non-representable base now reports the block,
    because that is the more specific complaint and the one that names where to
    look; the base is checked last.  This is stated because it changed: the
    interaction blocks used to be built as a keyword argument beside
    ``base_relativity=``, and Python evaluates those left to right, so the base
    was formed -- and raised -- first.
    """
    if model._result is None:
        raise RuntimeError("Model must be fitted before exporting rating tables.")
    # Validated here rather than left to the first term that happens to consult
    # it.  ``_main_effect_inference`` is the only thing that checks ``centering``
    # today, and a model whose every term is a ``Spline`` or ``Polynomial``
    # never calls it -- the binned blocks come from the discretisation path --
    # so such a model used to accept ``centering="Mean"`` and silently export
    # native values under a name the caller believed meant something else.
    if centering not in _VALID_CENTERING:
        raise ValueError(f"centering must be one of {_VALID_CENTERING}, got {centering!r}")
    _preflight_rating_table_terms(model)
    # Immediately after the term preflight, so a caller hears about the mode
    # before any curve is swept or any block built.  The kind is checked whatever
    # it is; the per-term refusals only for the mode that would convert them,
    # since a postfit constraint and an unbounded extrapolation are both
    # perfectly exportable as binned blocks.
    _require_supported_continuous_kind(continuous_kind)
    if continuous_kind == "ppform":
        _require_ppform_exportable(
            model,
            _ppform_convertible_terms(model),
            allow_unbounded_extrapolation=allow_unbounded_extrapolation,
        )
    # After the term preflight and before anything is built.  After, because an
    # unsupported term type is the more fundamental complaint -- a model with a
    # ``RandomEffect`` has no rating table under any link, and hearing about the
    # link first would send the caller to change something that was not the
    # problem.  Before the build, because the whole payload is a product of
    # exponentials, which is the prediction only under a log link.
    _require_log_link_export(model)
    # Beside the link gate because it is the same kind of complaint -- a property
    # of the model that no frame can rescue -- and before it would be wrong,
    # since a binomial on a logit link should hear about its link first.
    _require_unclamped_response_export(model)

    native_X = X
    frame = as_eager_frame(X)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(frame) != len(y_arr):
        raise ValueError("X and y must have the same length.")
    if sample_weight is not None and len(sample_weight) != len(frame):
        raise ValueError("sample_weight must have the same length as X.")

    export_offset: NDArray | None = None
    if _fit_used_offset(model):
        _require_log_link_offset_export(model)
        export_offset = _resolve_export_offset(offset, model, native_X)
    elif offset is not None or offset_source is not None:
        raise ValueError("Offset rating-table export requires a model fitted with an offset.")

    # After the offset is resolved, because the offset is part of the predictor
    # it inspects.  The gates before it read the model alone; this one is the
    # first that needs the frame, so it is also the first that can be answered
    # only once there is data to answer it about.
    _require_unsaturated_predictor_export(model, frame, export_offset)

    continuous = _continuous_features(model)
    selected = (
        model.discretization_impact(
            frame,
            y_arr,
            sample_weight=sample_weight,
            offset=export_offset,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            features=continuous,
        )
        if continuous
        else None
    )

    main_effects: list[RatingTableBlock] = []
    for name in model._feature_order:
        spec = model._specs[name]
        # Piecewise is tested FIRST and deliberately does not join
        # ``_continuous_features``.  That list is what gets handed to
        # ``discretization_impact``, whose continuity gate raises for anything
        # that is not a spline or a polynomial; and every name it returns is
        # routed to the BINNED ``_continuous_block`` below, which is the lossy
        # path this feature exists to remove.  A piecewise term therefore also
        # contributes no row to the impact sheet -- correctly, since its export
        # has no discretisation error to measure.
        if isinstance(spec, Piecewise):
            main_effects.append(_piecewise_block(model, name, centering))
        elif selected is not None and name in selected.tables:
            # ``Polynomial`` stays binned under ``"ppform"``: the block is fixed
            # at four coefficients while a polynomial's degree is not, so it is
            # routed on ``_SplineBase`` rather than on "is continuous".
            if continuous_kind == "ppform" and isinstance(spec, _SplineBase):
                # No shift passed in.  The ppform block reads its curve and its
                # shift off ONE ``term_inference`` call, because under
                # ``centering="mean"`` the shift is that call's grid mean; a
                # constant computed here, on a differently sized grid, would be
                # the wrong constant for the curve the block emits.
                main_effects.append(
                    _continuous_ppform_block(
                        model,
                        name,
                        centering,
                        _export_weights(frame, sample_weight),
                        np.asarray(frame.column_array(name), dtype=np.float64),
                    )
                )
            else:
                # Through the same term the exact blocks read their shift from,
                # so every block in the workbook is in the centering the caller
                # asked for and all of them share one origin.  ``with_se=False``,
                # so this is the point estimate alone -- the band's own centering
                # is ``_recenter_term``'s business and is not routed through here.
                centering_shift = float(
                    _main_effect_inference(model, name, centering).centering_shift
                )
                main_effects.append(_continuous_block(name, selected.tables[name], centering_shift))
        elif isinstance(spec, Categorical | OrderedCategorical):
            main_effects.append(_categorical_block(model, frame, name, sample_weight, centering))
        elif isinstance(spec, Numeric):
            main_effects.append(_numeric_block(model, name, centering))

    if export_offset is not None:
        if offset_source is None:
            offset_block = _offset_multiplier_block(
                export_offset,
                len(frame),
                sample_weight,
                n_bins=n_bins,
                bin_strategy=bin_strategy,
            )
        else:
            offset_block = _offset_source_block(
                export_offset,
                offset_source,
                frame,
                sample_weight,
                offset_name=offset_name,
                offset_kind=offset_kind,
                offset_max_exact_levels=offset_max_exact_levels,
                offset_mapping_rtol=offset_mapping_rtol,
                offset_mapping_atol=offset_mapping_atol,
            )
        main_effects.append(offset_block)

    # Built here rather than inline in the constructor call below, so that the
    # guard can see the interaction cells.  Those are the only exported factors
    # that never touch ``_safe_exp`` under any centering, so they are the one
    # place the disciplines this module applies to the base -- reject inf,
    # reject 0.0, reject subnormal -- used to be absent entirely (issue #289).
    # ``_interaction_blocks`` is pure, so the hoist changes nothing else.
    interactions = _interaction_blocks(model, n_bins)

    # After the blocks are built, because it is the emitted values it checks
    # rather than the coefficients they came from.
    _require_usable_relativities_export(main_effects, interactions)

    impact = _impact_sweep(
        model,
        frame,
        y_arr,
        sample_weight,
        offset=export_offset,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        exported_n_bins=int(n_bins),
        # Both lists, because both are approximated, and the second read off
        # the blocks that were BUILT rather than re-derived from the specs.
        # The ``selected`` call above stays main-effect-only: its tables are
        # consumed as blocks keyed by feature name, and an interaction has no
        # such block.
        features=continuous + _gridded_interaction_names(interactions),
    )
    return RatingTablePayload(
        base_relativity=_base_relativity(
            float(
                ordered_reference_intercept(
                    model.result.intercept,
                    model.result.beta,
                    model._feature_order,
                    model._specs,
                    model._groups,
                )
            )
            + _total_centering_shift(main_effects)
        ),
        selected_n_bins=int(n_bins),
        main_effects=main_effects,
        interactions=interactions,
        discretization_impact=impact,
        summary=build_summary_export_payload(model),
    )


def export_rating_tables(
    model: SuperGLM,
    file_path: str | Path,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    offset: NDArray | None = None,
    offset_source=None,
    offset_name: str | None = None,
    offset_kind: str = "auto",
    offset_max_exact_levels: int = 20,
    offset_mapping_rtol: float = 1e-10,
    offset_mapping_atol: float = 1e-12,
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    format: str | None = None,
    sheet_name: str = "Rating Tables",
    summary_sheet_name: str = "Model Summary",
    impact_sheet_name: str = "Discretization Impact",
    centering: str = "native",
    continuous_kind: str = "binned",
    allow_unbounded_extrapolation: bool = False,
) -> Path:
    """Render the rating-table payload to a workbook and return the path.

    A thin renderer over :func:`build_rating_table_payload`, which is where the
    payload's contract lives: what the exported product reproduces, which term
    types are exact and which are binned, what ``centering=`` does and does not
    change, and which errors the export raises.  Read that docstring before
    relying on an exported workbook; this function adds only the file format,
    the sheet names, and the extra rounding a renderer imposes.

    ``continuous_kind`` and ``allow_unbounded_extrapolation`` are forwarded
    verbatim and are not interpreted here.  They select how a continuous term
    is represented and whether an unbounded extrapolation may be exported, both
    of which are properties of the payload rather than of the rendering, so the
    mode's validation and its refusals happen once, where the blocks are built.

    The mode is documented here as well as there, and deliberately: this is the
    entry point most callers reach it through, and its sheet -- not the payload
    -- is what the downstream consumer parses.

    ``continuous_kind : {"binned", "ppform"}, default "binned"`` selects how a
    continuous main-effect term is represented.

    ``"binned"`` writes a key/relativity/weight block a consumer applies by
    pure lookup, with no arithmetic.  It is an APPROXIMATION: the fitted curve
    is chopped into intervals carrying one exposure-weighted average each, so a
    row inside an interval receives a factor that is not its own.  Measured on a
    motor book with 81 distinct ages at ``n_bins=150``, the worst row was
    mis-rated by 60%, concentrated in the wide intervals the quantile strategy
    opens in the sparse tails.  ``n_bins`` is a budget rather than a target, and
    staying under it is not a route to an exact block -- see
    :func:`build_rating_table_payload`, where that is measured.

    ``"ppform"`` writes the exact piecewise-polynomial form of the fitted
    curve: one row per knot interval carrying four coefficients, evaluated as
    ``exp(a + b*u + c*u**2 + d*u**3)`` where ``u = (x - from) / (to - from)``,
    normalised onto ``[0, 1]`` rather than the raw ``x - from``.  It reproduces
    the fitted model to machine precision -- 2.4e-15 against 6.0e-01 for the
    block it replaces -- in usually an order of magnitude fewer rows, and costs
    a consumer that can evaluate a polynomial.

    On the sheet that block is NINE columns rather than three, and a superset
    rather than a new shape: ``<feature>``, ``Relativity`` and ``Weight`` stay
    in front unchanged, with ``from``, ``to``, ``a``, ``b``, ``c``, ``d``
    appended behind them.  An un-upgraded loader still reads it as a step
    function -- it locates the block by the same header signature and slices the
    same three columns positionally -- while an upgraded one reads the
    coefficients and is exact.  A consumer that STORES the coefficients must
    include them in any content digest it fingerprints a published package
    with, or two models differing only in their coefficients fingerprint
    identically and the second is silently deduplicated into the first.

    Blocks are laid out at their own widths, so a nine-column block moves every
    block to its right; a reader keyed on the old fixed three-column stride
    reads the header row instead.

    The unbounded tail bounds do not survive the workbook as numbers.  A
    spreadsheet cell cannot hold an infinity, so an unbounded bound is written
    as a BLANK cell and reads back as null, while the interval key beside it
    still says ``[-inf, 18.0)``.  Those rows are the constant tails --
    ``b = c = d = 0`` -- so a consumer that reads a blank bound as unbounded and
    short-circuits ``u`` there is exact; one that computes ``u`` from a null
    bound gets no number at all.

    Extrapolation is otherwise carried in the table rather than described
    beside it: constant leading and trailing rows under the default
    ``extrapolation="clip"``, no unbounded rows at all under
    ``extrapolation="error"``, and a refusal under ``extrapolation="extend"``
    unless ``allow_unbounded_extrapolation=True`` acknowledges that the model
    prices beyond its training range with an unbounded cubic -- which the
    block's tails cannot carry and therefore clip.

    Terms carrying a ``Constraint.postfit`` repair are refused under
    ``"ppform"``, naming the term; ``Constraint.fit`` constraints convert
    unchanged.  ``Polynomial`` terms stay binned under ``"ppform"``, as does the
    continuous-by-continuous interaction grid, so one workbook can carry both
    kinds of block at once.
    """
    out = Path(file_path)
    fmt = _resolve_format(out, format)
    if fmt != "excel":
        raise ValueError(f"Unsupported rating table export format: {fmt!r}")

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        offset_source=offset_source,
        offset_name=offset_name,
        offset_kind=offset_kind,
        offset_max_exact_levels=offset_max_exact_levels,
        offset_mapping_rtol=offset_mapping_rtol,
        offset_mapping_atol=offset_mapping_atol,
        n_bins=n_bins,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        centering=centering,
        continuous_kind=continuous_kind,
        allow_unbounded_extrapolation=allow_unbounded_extrapolation,
    )

    from superglm.export.excel import write_rating_table_workbook

    write_rating_table_workbook(
        payload,
        out,
        sheet_name=sheet_name,
        summary_sheet_name=summary_sheet_name,
        impact_sheet_name=impact_sheet_name,
    )
    return out
