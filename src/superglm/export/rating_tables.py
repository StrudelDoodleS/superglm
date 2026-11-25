"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.export._ppform import extract_ppform
from superglm.export.summary import SummaryExportPayload, build_summary_export_payload
from superglm.features.categorical import Categorical
from superglm.features.constraint import rules_of
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
    """The exported base relativity is not a usable float64 multiplier.

    ``base_relativity`` multiplies every row of the tariff, so clipping it
    would silently mis-rate every risk. Raised when ``base_relativity``
    overflows to ``inf``, underflows to ``0.0``, or lands in the subnormal
    range where float64 loses precision.
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
    # Categorical only: the factors a separated level or cell leaves undetermined, stated
    # (``separated_block_note``).
    note: str | None = None


@dataclass(frozen=True)
class InteractionTableBlock:
    """One two-way interaction rating-table block.

    ``kind`` records how the block represents the interaction: ``"cells"`` is an
    exact cell table (categorical-by-categorical, or spline-by-categorical on an
    ordinal parent), ``"grid"`` is a continuous surface sampled at ``n_bins``
    nodes per axis, and ``"spline_cells"`` is a spline-by-categorical deviation
    tabulated per covariate bin and categorical level (evaluated at each bin's
    representative, or as the band's weighted mean under
    ``bin_strategy="exact"``). Recording ``kind`` here lets
    ``_gridded_interaction_names`` read the approximation set directly from the
    built blocks.
    """

    name: str
    table: pd.DataFrame
    kind: str = "cells"
    # The cell factors a separated level or cell leaves undetermined (``separated_block_note``).
    note: str | None = None


def separated_block_note(model, name: str, marked: list[str]) -> str | None:
    """The note a block carries for its factors ``marked``, which a separated level or cell moves.

    Marked by the flags ``summary()`` reads (``separated_levels``), a factor fixed at 1
    left out, and named with the separated levels and cells they move with
    (``separated_sources``).  The factors are exported as fitted: where the fit stopped.
    """
    from superglm.model.separated_levels import separated_sources

    if not marked:
        return None
    return (
        f"Not estimates: {', '.join(marked)}. Every response on the rows of separated "
        f"{', '.join(separated_sources(model, name))} is 0, or every one is 1, so the data "
        "give these factors no finite estimate; each is where the fit stopped. Group the "
        "separated levels with similar levels, or fit the term as a RandomEffect, before "
        "rating on them."
    )


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


def _require_excel_format(file_path: Path, format: str | None) -> None:
    """Refuse any export format but Excel, named by ``format`` or the file suffix."""
    fmt = (format if format is not None else file_path.suffix).lower().lstrip(".")
    if fmt in {"xlsx", "xlsm", "excel"}:
        return
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
    """Return the exported interaction blocks that are approximations.

    Includes sampled surfaces (``"grid"``) and binned spline deviations
    (``"spline_cells"``), excluding exact ``"cells"`` tables. Derived from the
    built blocks rather than spec classes so the impact sheet matches the
    exported workbook, including for custom grid-shaped specs.
    """
    return [block.name for block in blocks if block.kind in ("grid", "spline_cells")]


def _binned_continuous_names(blocks: list[RatingTableBlock]) -> list[str]:
    """The exported main effects that are binned curves, read off the blocks.

    The main-effect counterpart of ``_gridded_interaction_names``, and for the
    same reason.  ``_continuous_features`` says which terms COULD be binned; it
    is derived from the specs and knows nothing about ``continuous_kind``, so
    handing it to the sweep made the impact sheet report a binning error for a
    term the workbook carries exactly -- the sheet describing the model rather
    than the artifact, which is a disclosure error on a filing document even
    though it overstates the error rather than understating it.  Reading the
    kind off the block that was BUILT cannot drift from what shipped.
    """
    return [block.name for block in blocks if block.kind == "continuous"]


def _format_number(value: float) -> str:
    """Format a float at shortest round-trip decimal precision.

    Bin boundaries and interaction axis values are lookup keys parsed back to
    float64 by consumers, where the bin map is discontinuous at edges.
    ``repr(float(value))`` emits the shortest decimal string that parses back to
    the exact same binary64 value (Steele & White, PLDI 1990; Gay, 1990;
    Champagne Gareau & Lemire, arXiv:2603.06581), unlike fixed-precision
    formatting which perturbs bin boundaries. Converting to Python ``float``
    first avoids ``np.float64(...)`` repr output under NumPy 2.
    """
    return repr(float(value))


def _format_interval(left: float, right: float, *, closed_right: bool = False) -> str:
    """One half-open interval key, or one of the two CLOSED keys the export emits.

    Every row of every banded block is ``[lo, hi)``, so that each row's upper
    bound is identically the next row's lower bound and no value falls between
    two rows.  One exception is a binned block whose last band holds only the
    largest value, ``[x, x]``, which a half-open key would leave matching
    nothing.  The other is the last row of a ppform block under
    ``extrapolation="error"``, which has no next row to hand the endpoint to:
    the block stops at the boundary knot, and a right-open key there leaves the
    boundary itself matching nothing.  That value is normally the observed
    training maximum, and the model rates it -- so the table would fail to rate
    a row the model rates.  The bracket carries which rule applies, so a
    consumer reads it off the key rather than being told separately.
    """
    bracket = "]" if closed_right else ")"
    return f"[{_format_number(left)}, {_format_number(right)}{bracket}"


def _format_axis_value(value: float) -> str:
    return _format_number(value)


def _interval_keys(bin_from, bin_to) -> list[str]:
    """The printed key of each bin, one rule for every block keyed on intervals."""
    # A last band holding only the largest value has bin_from == bin_to, and
    # its half-open key would match nothing, so it alone is closed. Other
    # zero-width rows (repeated uniform edges) stay half-open and empty.
    last = len(bin_from) - 1
    return [
        _format_interval(
            float(lower),
            float(upper),
            closed_right=index == last and float(lower) == float(upper),
        )
        for index, (lower, upper) in enumerate(zip(bin_from, bin_to, strict=True))
    ]


def _continuous_block(name: str, table: pd.DataFrame, centering_shift: float) -> RatingTableBlock:
    """Build a binned continuous block shifted by ``centering_shift``.

    ``discretization_impact`` produces native-centred bin relativities; applying
    ``exp(-centering_shift)`` here keeps the binned block on the same origin as
    ``term_inference`` and records ``centering_shift`` on the block so
    ``base_relativity`` absorbs the total shift and leaves the workbook product
    invariant to ``centering``.
    """
    # Same discipline as ``_base_relativity``: an extreme shift is a condition
    # ``_require_usable_relativities_export`` is there to refuse, and it should
    # arrive as that refusal rather than as a ``RuntimeWarning`` from here.
    with np.errstate(over="ignore", under="ignore"):
        factor = float(np.exp(-centering_shift))
    out = pd.DataFrame(
        {
            name: _interval_keys(table["bin_from"], table["bin_to"]),
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
# zero.  That does NOT make the polynomial formula evaluate on them, and it is
# worth being exact about why, because the plausible reading is wrong: on the
# leading row ``u = (x - -inf) / (18.0 - -inf)`` is ``inf/inf``, which is
# ``nan``, and IEEE arithmetic propagates it through ``0 * u`` -- so
# ``a + u*(b + u*(c + u*d))`` returns ``nan`` rather than ``a``, however many
# of the coefficients are zero.  Verified by evaluation, not by inspection.
#
# The rule an unbounded row therefore carries is: READ ``Relativity`` AND STOP.
# It is not an extra burden on the consumer.  Those bounds are printed
# ``-inf`` and ``inf``, which no numeric cast accepts, so a consumer already
# has to recognise an unbounded key before it can parse one -- and the branch
# it takes then is the plain lookup the block's first three columns exist for.
# ``Relativity`` on a tail row is ``exp(a)`` exactly, so the two readings agree.
# Stated on the sheet by ``_ppform_evaluation_note``, in the docstrings below,
# and in the guide, because a rule that lives only in a test helper is a rule
# the consumer does not have.
_PPFORM_TAIL_COEFFICIENTS = (0.0, 0.0, 0.0)


def _continuous_ppform_block(
    model: Any,  # a SuperGLM, or any object with its term_inference and predict
    name: str,
    centering: str,
    weights: NDArray,
    values: NDArray,
) -> RatingTableBlock:
    """A continuous term as its exact polynomial pieces, plus a lookup fallback.

    Seven columns, in two halves.  The first three are the ordinary
    binned-block columns -- an interval key, a relativity, an exposure weight --
    so a consumer that cannot evaluate a polynomial still finds this block,
    reads it, and scores it as a step function exactly as it scores a binned
    block today.  The remaining four are the exact form, one coefficient each.

    The interval bounds are NOT among them.  They are already in the key, at
    round-trip precision, so ``u = (x - lower) / (upper - lower)`` is computed
    from the string a consumer has already parsed to match the row.  Restating
    them as ``from``/``to`` floats would be duplicate state that can only ever
    disagree -- and a float column cannot carry a tail row's infinity through a
    spreadsheet at all.

    That superset is not cosmetic.  The downstream loader locates blocks by a
    header signature at fixed offsets and fails the WHOLE package when a block
    does not match, so a block that dropped those three columns for its
    coefficients alone would force two repositories to ship in lockstep and
    would break every other term's publication in between.  See the design's
    section 4.2a and 7.1.

    ``Relativity`` is the curve's value at ``from``, not the interval's average,
    so the two readings of one row agree at its left edge instead of disagreeing
    everywhere.

    The first and last rows are UNBOUNDED and constant.  Extrapolation is
    carried in the table rather than described beside it: a cubic evaluated past
    its last knot is not merely wrong but unbounded -- measured at 1581x the
    correct factor twenty-one years past the boundary of a real age curve -- and no
    note reliably prevents a consumer from doing it.  With the tails emitted,
    there is no code path that reaches the blow-up.

    An unbounded row is READ, not evaluated: its factor is ``Relativity``.
    The polynomial formula does not apply to it and cannot -- ``u`` on an
    infinite width is ``nan``, which the zero coefficients do not absorb (see
    ``_PPFORM_TAIL_COEFFICIENTS``).  So the consumer contract is two lines
    rather than one: match the row, and evaluate the polynomial if the row is
    bounded, else take ``Relativity``.  The second line costs nothing, because
    an ``-inf`` bound already has to be recognised before it can be parsed, and
    the branch it selects is the lookup the first three columns already serve.

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
        # The last row alone is CLOSED on the right.  ``error`` declines to
        # price ABOVE the boundary knot; it does not decline to price AT it,
        # and ``model.predict`` accepts it.  With every key right-open the
        # boundary matches no row, so the block cannot rate a value the model
        # rates -- and because the boundary knot is the observed maximum
        # whenever knots are data-driven, that is a row of the training frame.
        # Measured on a 6-knot fit: boundary 79.94164283974858, accepted by
        # ``predict``, matched by no exported row, one training row sitting on
        # it.  Only under ``error``: every other mode hands the endpoint to a
        # trailing row, so closing it there would make two rows match it.
        closed_right = np.zeros(len(lo), dtype=bool)
        closed_right[-1] = True
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
        # The trailing row runs to ``inf`` and takes every value above the last
        # knot, boundary included, so no key needs closing.
        closed_right = np.zeros(len(lo), dtype=bool)

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
            name: [
                _format_interval(left, right, closed_right=bool(closed))
                for left, right, closed in zip(lo, hi, closed_right, strict=True)
            ],
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


def _export_weights(model: SuperGLM, n_rows: int, sample_weight: NDArray | None) -> NDArray:
    """Validate and return per-row weights for exported ``Weight`` columns.

    Uses the same validator as ``discretization_impact`` at the payload boundary
    so every export path (including ``continuous_kind="ppform"`` when no term is
    binned) enforces the same weight checks and defaults unweighted fits to ones.
    """
    from superglm.diagnostics.discretize import _validated_discretization_weights

    weights, _ = _validated_discretization_weights(model, sample_weight, n_rows)
    return weights


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


_SUPPORTED_OFFSET_KINDS = frozenset({"auto", "discrete", "per_unit", "binned"})


def _require_exact_banding_settings(
    model: SuperGLM, bin_strategy: str, band_se, band_max_error, offset_kind: str
) -> None:
    """Check the band limit settings up front, whatever terms turn out to be binned.

    A binned offset has no fitted curve to band exactly, so that refusal comes
    before any main effect is solved rather than after.
    """
    from superglm.diagnostics.discretize import _positive_finite

    _positive_finite("band_se", band_se)
    _positive_finite("band_max_error", band_max_error)
    if bin_strategy == "exact" and offset_kind == "binned" and _fit_used_offset(model):
        raise ValueError(
            "bin_strategy='exact' places bands on a fitted curve and its standard errors, "
            "and a binned offset has no fitted curve. Choose another offset_kind or bin_strategy."
        )


def _require_supported_offset_kind(offset_kind: str) -> None:
    """Refuse a kind neither offset path recognises, whichever path will run.

    Only ``_offset_source_block`` validated, so on the undeclared path an
    unrecognised string fell through every comparison and was scored as
    ``"auto"``.  That is the wrong direction to fail in: ``"binned"`` is the
    only route back to the pre-``per_unit`` behaviour, so a mistyped kind
    handed the caller the exact default they were opting out of, silently.

    This guard catches values no path supports.  The per-path refusals stay
    where they are and keep their own messages -- ``_offset_source_block``
    rejecting ``"binned"`` is a known kind that is unsupported *there*, which
    is a different complaint and deserves to say so.
    """
    if offset_kind not in _SUPPORTED_OFFSET_KINDS:
        raise ValueError(
            f"offset_kind must be one of {sorted(_SUPPORTED_OFFSET_KINDS)}, got {offset_kind!r}."
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
        # ``rules_of`` reads the spec's rule objects, or the kind and mode a model
        # saved before rule collections holds.  Its rules carry their own mode, so
        # an unconstrained spline (whose ``constraint_mode`` still reads
        # ``"postfit"``) has no rule to refuse.  A postfit repair is always one
        # whole-curve shape.
        postfit = next((rule for rule in rules_of(spec) if rule.mode == "postfit"), None)
        if postfit is not None:
            raise ValueError(
                f"Term {name!r} carries a postfit {postfit.describe()} constraint, whose "
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
    spec: Categorical | OrderedCategorical,
) -> RatingTableBlock:
    ti = _main_effect_inference(model, name, centering)
    levels = list(ti.levels or [])
    grouping = getattr(spec, "_grouping", None)

    def fixed(level) -> bool:
        """A level with no coefficient (the base, a pinned level) reads 1 under native."""
        group = level if grouping is None else grouping.original_to_group.get(level, level)
        return centering == "native" and group not in spec._non_base

    flags = [False] * len(levels) if ti.separated is None else list(ti.separated)
    marked = [str(lv) for lv, flag in zip(levels, flags, strict=True) if flag and not fixed(lv)]
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
        note=separated_block_note(model, name, marked),
    )


def _piecewise_block(model: SuperGLM, name: str, centering: str) -> RatingTableBlock:
    """Build an exact knot table for a ``Piecewise`` term.

    Emits three columns (knot value, ``Relativity``, and ``Log relativity``)
    with no binning. The model is linear in ``Log relativity`` between knots, so
    linear interpolation on ``Log relativity`` followed by ``exp`` reproduces
    the model exactly. Records ``centering_shift`` on the block so
    ``base_relativity`` absorbs the shift under ``centering="mean"``.
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
    """Refuse rating-table export for non-log-link models.

    Every block reports ``exp`` of a term's contribution so the base times the
    per-block factors reproduces ``model.predict`` only under a log link. Under
    identity or logit links the product yields ``exp(eta)`` or odds rather than
    the predicted mean, so non-log links are rejected at the export boundary.
    """
    if not isinstance(model._link, LogLink):
        raise ValueError(
            "Rating-table export is supported only for log-link models, because the "
            "exported table is multiplicative: base_relativity times one relativity "
            "per block reproduces model.predict only under a log link. This model "
            f"uses {type(model._link).__name__}, for which that product would be "
            "exp(linear predictor) rather than the prediction."
        )


def _require_distinct_level_texts(model: SuperGLM) -> None:
    """A spline-by-categorical block keys each level's column by the level's text.

    Two levels with one text (``1`` and ``"1"``) would share a column and price
    one level with the other's curve, on every route the block takes.
    """
    from superglm.diagnostics.discretize import is_spline_categorical

    for name in model._interaction_order:
        ispec = model._interaction_specs[name]
        if not is_spline_categorical(ispec):
            continue
        levels = [str(level) for level in ispec._cat_levels]
        shared = sorted({text for text in levels if levels.count(text) > 1})
        if shared:
            parent = ispec.parent_names[1]
            raise ValueError(
                f"Interaction {name!r} cannot be exported: levels of {parent!r} share the "
                f"text {shared}, and a rating table keys each level by its text. Recode "
                f"{parent!r} so that every level's text is distinct, then refit."
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


def _ppform_reachable_relativities(table: pd.DataFrame) -> NDArray:
    """Every factor a ppform row can produce, not only the one it prints.

    ``Relativity`` on a ppform row is ``exp(a)``, the curve at ``u = 0``.  That
    is one point of a cubic the consumer evaluates across the whole interval,
    so validating the printed column alone validates the row's left edge and
    nothing else: a piece with ordinary endpoints can still cross the usable
    band at an interior extremum, and on rows where another term cancels it the
    total-predictor gate stays silent too -- the same cancellation argument
    ``_require_usable_relativities_export`` already makes for checking per
    block rather than on the product, one level further down.

    Exactly, not by sampling.  A cubic's extrema on ``[0, 1]`` are the roots of
    ``b + 2cu + 3du**2`` inside the open interval, plus the two endpoints, and
    those roots are closed form.  A grid could step over a narrow spike; this
    cannot, and it is cheaper.

    The tail rows come through unchanged: ``b = c = d = 0`` leaves the
    derivative identically zero, so no root is admitted and both endpoints
    evaluate to ``exp(a)``.
    """
    a, b, c, d = (np.asarray(table[col], dtype=np.float64) for col in _PPFORM_COLUMNS)

    # Endpoints always; interior stationary points where they exist and lie
    # strictly inside.  ``u`` outside (0, 1) is another row's business.
    points = [np.zeros_like(a), np.ones_like(a)]
    with np.errstate(invalid="ignore", divide="ignore"):
        # Quadratic 3d*u**2 + 2c*u + b.  Where d == 0 it degenerates to the
        # linear -b/(2c), and where c == 0 too there is no stationary point.
        discriminant = 4.0 * c**2 - 12.0 * b * d
        root = np.sqrt(np.where(discriminant > 0.0, discriminant, np.nan))
        quadratic = [(-2.0 * c + root) / (6.0 * d), (-2.0 * c - root) / (6.0 * d)]
        linear = np.where(c != 0.0, -b / (2.0 * c), np.nan)
        points.extend(np.where(d != 0.0, candidate, linear) for candidate in quadratic)

    values = []
    for u in points:
        inside = np.isfinite(u) & (u > 0.0) & (u < 1.0)
        u_safe = np.where(inside | (u == 0.0) | (u == 1.0), np.nan_to_num(u), 0.0)
        with np.errstate(over="ignore", under="ignore"):
            values.append(np.exp(a + u_safe * (b + u_safe * (c + u_safe * d))))
    return np.concatenate(values)


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
    if block.kind == "continuous_ppform":
        # The printed column is one point of each row's cubic; the consumer
        # evaluates all of it.  Returned ALONGSIDE ``Relativity`` rather than
        # instead of it, so the left edge is still covered by the same
        # comparison and nothing is traded away for the extra coverage.
        return np.concatenate(
            [
                np.asarray(block.table["Relativity"], dtype=np.float64),
                _ppform_reachable_relativities(block.table),
            ]
        )
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
    """Refuse any main-effect or interaction block carrying an unusable factor.

    Rejects every block value that is non-finite or outside the open interval
    ``(exp(-_MAX_LOG_REL), exp(_MAX_LOG_REL))``. This catches both values
    clipped at ``exp(+/-500)`` by ``_safe_exp`` and values from plain ``np.exp``
    that overflowed to ``inf``, underflowed to subnormals or ``0.0``, or are
    negative (including extrapolated corner cells of a continuous interaction
    grid). Checked per block rather than on the product so cancelling extreme
    terms cannot slip through.
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


def _require_usable_offset_multipliers(multiplier: NDArray) -> None:
    """Validate per-row ``exp(offset)`` multipliers before a per-unit block hides them.

    Unlike discrete or binned offset blocks, a per-unit row emits only the
    formula (relativity ``1.0`` or the per-unit rate), so the consumer computes
    ``exp(offset)`` per row and ``_require_usable_relativities_export`` never
    sees those factors. Checking the total linear predictor is also insufficient
    because large offset and term contributions can cancel in the sum while the
    offset factor itself overflows or underflows.

    Every row is checked unweighted against the shared ``(exp(-500), exp(500))``
    band (endpoints excluded to match ``_require_usable_relativities_export``),
    so a single unusable row raises even if its weight is zero.
    """
    from superglm.inference._term_types import _MAX_LOG_REL

    ceiling = float(np.exp(_MAX_LOG_REL))
    floor = float(np.exp(-_MAX_LOG_REL))
    bad = ~np.isfinite(multiplier) | (multiplier >= ceiling) | (multiplier <= floor)
    if not np.any(bad):
        return
    raise ValueError(
        f"A per-unit offset row would hide {int(np.count_nonzero(bad))} offset "
        "multiplier(s) that a consumer cannot multiply by: exp(offset) outside "
        f"(exp(-{_MAX_LOG_REL:g}), exp({_MAX_LOG_REL:g})) = ({floor:.4g}, "
        f"{ceiling:.4g}), or not finite. A per-unit row emits the RELATION rather "
        "than the factors, so these values never reach the workbook's relativity "
        "guard and would first fail in the consumer. The total predictor can stay "
        "healthy while they do not, because other terms cancel them. Rescale the "
        "offset, or keep the offset calculation outside the rating table."
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
    """Round finite normal nonzero values to ``digits`` significant figures.

    Rounding relatively (rather than to fixed decimal places) collapses float
    round-off in ``exp(log(x))`` into distinct offset tariff levels at any
    magnitude without collapsing small positive multipliers to ``0.0``. Scales
    through the base-10 mantissa ``values / 10**floor(log10(|values|))`` so
    intermediate powers of ten never overflow. Zeros, subnormals, and non-finite
    values are left unchanged for downstream validation.
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


def _offset_per_unit_block_from_multiplier(
    multiplier: NDArray,
    sample_weight: NDArray | None,
) -> RatingTableBlock:
    """The undeclared offset as one per-unit row, keyed on the multiplier itself.

    The declared form (``_offset_per_unit_block``) has a scale to derive from a
    named column and to verify on every row.  This one has neither: no column
    was declared, so the only input the block can ask for is the multiplier the
    consumer already computes, and the relation between that input and the
    factor is the identity.  ``Relativity`` is therefore ``1.0`` exactly, and
    there is nothing to check -- the block returns what it was handed.

    That reads as though it says nothing.  It says exactly as much as the binned
    block it replaces, and says it correctly: that block is *also* keyed on the
    multiplier, so a consumer must compute ``exp(offset)`` before it can look
    anything up -- and then receives its bin's exposure-weighted average back
    instead of the number it just computed.  A lossy round-trip of a value the
    consumer already held.  This row asks for the same input and returns it
    unmodified, at every magnitude, including above the largest multiplier the
    book happened to contain.
    """
    _require_usable_offset_multipliers(multiplier)
    weights = _weights_array(len(multiplier), sample_weight)
    return RatingTableBlock(
        name="Offset Multiplier",
        kind="offset_per_unit",
        table=pd.DataFrame(
            {
                "Offset Multiplier": ["per_unit"],
                "Relativity": [1.0],
                "Weight": [float(weights.sum())],
            }
        ),
    )


def _offset_multiplier_block(
    offset: NDArray,
    n_rows: int,
    sample_weight: NDArray | None,
    *,
    n_bins: int,
    bin_strategy: str,
    offset_kind: str,
    offset_max_exact_levels: int,
) -> RatingTableBlock | None:
    """An undeclared offset under the same rule the declared path follows.

    An offset is never estimated -- it is a known function of known inputs, so
    it has no estimation error for binning to trade against.  Cardinality
    therefore does not decide whether it is published exactly: few distinct
    multipliers make a lookup worth printing, many make a per-unit row, and
    neither is an approximation.  Binning survives only where a caller asks for
    it by name, as a summary of the fitted exposure rather than something to
    rate from.
    """
    offset_arr = np.asarray(offset, dtype=np.float64).ravel()
    if len(offset_arr) != n_rows:
        raise ValueError("offset must have the same length as X.")

    weights = _weights_array(n_rows, sample_weight)
    # Overflow is a state this routine HANDLES rather than one it avoids: the
    # per-unit branch refuses on it by name, and the lookup branch emits it as a
    # relativity for the workbook's own guard to refuse.  Warning here would
    # only be noise in front of the error that follows.
    with np.errstate(over="ignore", under="ignore"):
        multiplier = np.exp(offset_arr)
    # The rounding exists to stop the float noise of ``exp(log(x))`` splitting
    # one tariff level into thousands, which would push a genuine lookup onto
    # the per-unit row.  It is relative rather than absolute, so it stays
    # proportionate at any magnitude of multiplier.
    exact_multiplier = _significant_digits(multiplier, 12)
    levels, inverse = np.unique(exact_multiplier, return_inverse=True)

    if offset_kind != "binned":
        if offset_kind != "per_unit" and len(levels) <= offset_max_exact_levels:
            exposure = np.bincount(inverse, weights=weights, minlength=len(levels))
            table = pd.DataFrame(
                {
                    "Offset Multiplier": levels.astype(float),
                    "Relativity": levels.astype(float),
                    "Weight": exposure.astype(float),
                }
            )
            return RatingTableBlock(name="Offset Multiplier", kind="offset", table=table)
        return _offset_per_unit_block_from_multiplier(multiplier, sample_weight)
    return _binned_offset_block(multiplier, weights, n_bins, bin_strategy)[0]


def _binned_offset_block(
    multiplier: NDArray, weights: NDArray, n_bins: int, bin_strategy: str
) -> tuple[RatingTableBlock, NDArray]:
    """The binned offset summary, and the factor its interval lookup gives each row."""
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
    table = pd.DataFrame(rows)
    block = RatingTableBlock(name="Offset Multiplier", kind="offset", table=table)
    return block, table["Relativity"].to_numpy(dtype=np.float64)[bin_idx]


class OffsetRelationError(ValueError):
    """A declared offset source does not carry the offset's own relation."""


def _offset_per_unit_block(
    offset: NDArray,
    source,
    source_name: str,
    sample_weight: NDArray | None,
    *,
    offset_mapping_rtol: float,
) -> RatingTableBlock:
    """A continuous offset as the one relation it is, not as a sample of it.

    Shaped like ``_numeric_block`` and for the same reason: this is a PER UNIT
    factor, not a banded one.  It was routed to the binned path because it is
    continuous, but what decides the shape is that the consumer multiplies by a
    column rather than looking a value up in a table.

    There are no bounds and there is no extrapolation rule, deliberately.  An
    offset's coefficient is 1 by construction -- nothing is estimated, so
    nothing is only known over a training range.  The relation holds at a sum
    insured of 10m on a book that saw 2m for exactly the reason it holds at 10k,
    and the binned block it replaces did not: its top bin capped every large
    risk at that bin's average.

    Derived on the log scale and verified on the multiplier scale, on EVERY row.
    A declared source that does not hold the relation is refused by name; the
    point of the block is that it is exact, so a block that is not exact must
    not be written.
    """
    offset_arr = np.asarray(offset, dtype=np.float64).ravel()
    # Before the relation is derived, because a multiplier this block cannot
    # publish is not made publishable by being proportional to something.
    with np.errstate(over="ignore", under="ignore"):
        _require_usable_offset_multipliers(np.exp(offset_arr))
    try:
        column = np.asarray(source, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        # A bare numpy cast error names neither the column nor the remedy, and
        # this is the branch a high-cardinality categorical source lands in.
        raise OffsetRelationError(
            f"Offset source {source_name!r} is not numeric, so the offset cannot be a "
            "log of it, and it has too many distinct values to publish as a lookup. "
            "Declare the numeric column the offset was computed from, or keep the "
            "offset calculation outside the rating table."
        ) from exc
    weights = _weights_array(len(offset_arr), sample_weight)

    if np.any(~np.isfinite(column)) or np.any(column <= 0.0):
        raise OffsetRelationError(
            f"Offset source {source_name!r} has non-positive or non-finite values, so "
            "the offset cannot be a log of it. Declare a different column, or keep the "
            "offset calculation outside the rating table."
        )

    log_ratio = offset_arr - np.log(column)
    total = float(weights.sum())
    scale = float(
        np.exp(np.average(log_ratio, weights=weights) if total > 0.0 else log_ratio.mean())
    )

    # Purely RELATIVE to the scale, rather than ``np.allclose``'s
    # ``atol + rtol*scale``.  A multiplier is a scale-free quantity, so an
    # absolute floor is a relative tolerance of ``atol/scale``: at
    # ``scale = 1e-6`` the ``1e-12`` default admits a 1e-6 relative departure --
    # four orders looser than the rtol that was asked for -- and it keeps
    # loosening in proportion below that, on the one block whose entire claim
    # is exactness.  Measured: the same 1e-8 departure is refused on
    # ``log(SumInsured)`` and accepted on ``log(SumInsured/1e6)``.  This is the
    # argument ``_significant_digits`` already makes for rounding (issue #303).
    # The discrete path is relative for the same reason, so no absolute
    # tolerance survives anywhere in the offset checks.
    worst = float(np.max(np.abs(np.exp(log_ratio) / scale - 1.0)))
    if not np.isfinite(worst) or worst > offset_mapping_rtol:
        raise OffsetRelationError(
            f"Offset is not proportional to {source_name!r}: the worst row deviates by "
            f"{worst:.3e} relative, outside offset_mapping_rtol={offset_mapping_rtol:g}. "
            "A per-unit offset block "
            "states one relation for every row, so it cannot describe this offset. "
            "Declare the column the offset is actually a function of, or keep the "
            "offset calculation outside the rating table."
        )

    return RatingTableBlock(
        name=source_name,
        kind="offset_per_unit",
        table=pd.DataFrame({source_name: ["per_unit"], "Relativity": [scale], "Weight": [total]}),
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
) -> RatingTableBlock:
    if offset_kind not in {"auto", "discrete", "per_unit"}:
        raise ValueError("offset_kind must be 'auto', 'discrete' or 'per_unit'.")

    source, source_name = _resolve_offset_source(offset_source, X, offset_name=offset_name)
    n_unique = int(source.nunique(dropna=False))

    if offset_kind == "per_unit" or (offset_kind == "auto" and n_unique > offset_max_exact_levels):
        return _offset_per_unit_block(
            offset,
            source,
            source_name,
            sample_weight,
            offset_mapping_rtol=offset_mapping_rtol,
        )

    if n_unique > offset_max_exact_levels:
        raise ValueError(
            f"Offset source {source_name!r} has {n_unique} distinct values, exceeding "
            f"offset_max_exact_levels={offset_max_exact_levels}. Increase "
            "offset_max_exact_levels explicitly if all values are intended tariff "
            "levels, or pass offset_kind='per_unit' to export the relation itself."
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
        # Purely RELATIVE to the level's own representative, for the same
        # reason ``_offset_per_unit_block`` is: ``np.allclose`` tolerates
        # ``atol + rtol*representative``, and an absolute floor on a scale-free
        # quantity IS a relative tolerance of ``atol/representative``.  At a
        # multiplier near the old ``1e-12`` absolute default the floor
        # dominated entirely -- multipliers of 1e-13 and 2e-13 sat within it of
        # their geometric mean, so the level published as exact while
        # mis-rating its rows by 41.4% and 29.3%.  A lookup whose whole claim
        # is that it is exact must not be certified by a tolerance that
        # loosens as the quantity shrinks.
        worst = float(np.max(np.abs(multipliers / representative - 1.0)))
        if not np.isfinite(worst) or worst > offset_mapping_rtol:
            raise ValueError(
                f"Offset source {source_name!r} is not a valid discrete lookup: "
                f"level {level!r} maps to multiple offset multipliers, the worst "
                f"deviating by {worst:.3e} relative from the level's representative, "
                f"outside offset_mapping_rtol={offset_mapping_rtol:g}. Pass a more "
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


def _spline_categorical_block(
    model: SuperGLM,
    name: str,
    ispec,
    frame: EagerFrame | None,
    sample_weight: NDArray | None,
    n_bins: int,
    bin_strategy: str,
    banded: pd.DataFrame | None = None,
) -> InteractionTableBlock:
    """A per-level spline deviation as one relativity per row key and level.

    The keys are the first parent's, as its main block keys them, and the
    columns are the categorical's levels in level order, with the base level's
    all ``1.0`` wherever it falls.  A cell is the level's curve relative to the
    base level's, so the product with the main effects is the prediction.

    A spline parent is binned: the keys are its intervals and a cell is read at
    the bin's representative (``spline_categorical_bins``), so the block is
    not exact between representatives, which the impact sheet measures.  Under
    ``bin_strategy="exact"`` the intervals are the covariate's exact bands and
    a cell is the band's weighted mean of the level's deviation, so that the
    main effect's band factor plus the cell is the level's certified factor;
    both come from the one banding ``discretization_impact`` solved and
    ``banded`` is its table for this term.  An ``OrderedCategorical`` parent is
    exported as a categorical, one row per level, and its cells are the curves
    at the levels' scores: exact.
    """
    from superglm.diagnostics.discretize import (
        _validated_discretization_weights,
        banded_cells,
        has_ordinal_parent,
        ordinal_parent_cells,
        spline_categorical_bins,
    )

    parent1, _ = ispec.parent_names
    if has_ordinal_parent(model, ispec):
        keys, log_deviation = ordinal_parent_cells(model, ispec, _interaction_beta(model, name))
        kind = "cells"
    elif banded is not None:
        edges, log_deviation = banded_cells(banded, len(ispec._non_base))
        keys = _interval_keys(edges[:-1], edges[1:])
        kind = "spline_cells"
    else:
        if frame is None:
            raise TypeError(
                f"Interaction {name!r} bins {parent1!r} at its bin representatives, which "
                "are read from the training frame; pass frame=."
            )
        _, geometry_weight = _validated_discretization_weights(model, sample_weight, len(frame))
        bins = spline_categorical_bins(
            ispec,
            _interaction_beta(model, name),
            frame.column_array(parent1, dtype=np.float64),
            geometry_weight,
            n_bins,
            bin_strategy,
        )
        keys = _interval_keys(bins.edges[:-1], bins.edges[1:])
        log_deviation = bins.log_deviation
        kind = "spline_cells"
    factor = dict(zip(map(str, ispec._non_base), np.exp(log_deviation).T, strict=True))
    ones = np.ones(len(keys))
    levels = [str(level) for level in ispec._cat_levels]
    columns = {_row_key_header(parent1, levels): keys}
    for level in levels:
        columns[level] = factor.get(level, ones)
    return InteractionTableBlock(name=name, table=pd.DataFrame(columns), kind=kind)


def _interaction_blocks(
    model: SuperGLM,
    n_bins: int,
    *,
    frame: EagerFrame | None = None,
    sample_weight: NDArray | None = None,
    bin_strategy: str = "exposure_quantile",
    banded_tables: dict[str, pd.DataFrame] | None = None,
) -> list[InteractionTableBlock]:
    from superglm.diagnostics.discretize import is_spline_categorical

    blocks: list[InteractionTableBlock] = []
    for name in model._interaction_order:
        ispec = model._interaction_specs[name]
        parent1, _ = ispec.parent_names
        # Before the reconstruction below, which would walk every level of the
        # term to learn it is not a surface; the sweep routes it by the same
        # exact-type test.
        if is_spline_categorical(ispec):
            # Under "exact" the cells and the intervals were banded together with
            # the main effect, by the call that gave the main block its bands.
            banded = banded_tables.get(name) if banded_tables is not None else None
            blocks.append(
                _spline_categorical_block(
                    model, name, ispec, frame, sample_weight, n_bins, bin_strategy, banded
                )
            )
            continue
        # Shared with the impact sweep, so "does this spec take n_points" and
        # "which orientation is this surface" are answered in one place.
        from superglm.diagnostics.discretize import reconstruct_interaction

        raw = reconstruct_interaction(ispec, _interaction_beta(model, name), n_bins)
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

        from superglm.features.interaction import interaction_cells

        levels2 = raw["levels2"]
        cells = interaction_cells(raw)
        rows = [
            [level1, *(cells.get((level1, level2), 1.0) for level2 in levels2)]
            for level1 in raw["levels1"]
        ]
        columns = [_row_key_header(parent1, levels2), *levels2]
        table = pd.DataFrame(rows, columns=columns)
        from superglm.model.separated_levels import marked_cells

        marked = marked_cells(model, name)
        blocks.append(
            InteractionTableBlock(
                name=name, table=table, kind="cells", note=separated_block_note(model, name, marked)
            )
        )
    return blocks


def _row_key_header(parent: str, levels) -> str:
    """The header of a cell table's row-key column: the first parent's name.

    A column level named like it would share its header, so the name is suffixed until
    no level carries it.
    """
    names = {str(level) for level in levels}
    header = parent
    while header in names:
        header = f"{header} (axis 1)"
    return header


def _total_centering_shift(blocks: list[RatingTableBlock]) -> float:
    """Sum the log-scale centering constants removed from exported main-effect blocks.

    Adding this sum back into ``base_relativity`` keeps the base-times-factors
    product identical across ``centering="native"`` and ``centering="mean"``.
    Each block records its own shift because not every exported term is
    re-centred (``OrderedCategorical``, ``Numeric``, and offset blocks record
    ``0.0``), grouped categoricals centre before expanding back to original
    levels, and interactions apply no centering.
    """
    return float(sum(block.centering_shift for block in blocks))


def _base_relativity(log_base: float) -> float:
    """Exponentiate ``log_base``, refusing overflow, zero, and subnormal values.

    Because ``base_relativity`` multiplies every tariff row, clipping or
    underflowing into subnormals (where float64 drops significand bits and Excel
    clamps at ``2.2251e-308``) would silently mis-scale every prediction. Any
    result outside ``[_SMALLEST_EXACT_BASE, inf)`` raises
    :class:`RatingTableBaseNotRepresentableError`.
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


def _warn_widened_bands(
    band_diagnostics: dict[str, dict[str, float]],
    n_bins: int,
    shared_with: dict[str, str] | None = None,
) -> None:
    """Say when ``bin_strategy="exact"`` could not hold its limit in ``n_bins`` bands.

    A covariate's main effect and the interactions banded with it (``shared_with``
    maps each interaction to its covariate) are one banding, widened by one factor,
    so they are one warning naming all of them.
    """
    shared_with = shared_with or {}
    groups: dict[str, list[str]] = {}
    for name in band_diagnostics:
        groups.setdefault(shared_with.get(name, name), []).append(name)
    for names in groups.values():
        factor = band_diagnostics[names[0]]["tolerance_factor"]
        if factor > 1.0:
            worst = max(band_diagnostics[name]["worst_error"] for name in names)
            listed = ", ".join(repr(name) for name in names)
            warnings.warn(
                f"bin_strategy='exact' could not keep {listed} within its limit in "
                f"{n_bins} bands, so the limit was widened {factor:.3g}x; the worst band "
                f"error is {worst:.1%}. Raise n_bins to keep the limit.",
                UserWarning,
                stacklevel=3,
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
    binned_offset: bool,
    exported_n_bins: int,
    band_se: float,
    band_max_error: float,
) -> pd.DataFrame:
    """One row per approximated block per swept resolution.

    ``exported_n_bins`` joins the swept set, marked ``exported``: without it
    the sheet can describe every resolution EXCEPT the one the reader holds
    (defaults ``n_bins=150`` against ``impact_bins=(20, 50, 100, 200, 250)``).
    Each rung is an INDEPENDENT measurement: nearest-node grids are not nested,
    so a finer one need not sit closer to a given row.

    An empty ``impact_bins`` leaves the sheet empty, but the exported
    resolution is still evaluated: declining the disclosure cannot make a
    non-finite approximation exportable, and only that rung refuses one; a
    comparison rung records an infinite change and the reason in ``note``.  A
    ``binned_offset`` is binned at each resolution as its block is, and the
    change in its log factor joins the terms' changes under its own row.
    """
    rows: list[dict[str, float | int | str]] = []
    if not features and not binned_offset:
        return _empty_impact_frame()

    for n_bins in sorted(dict.fromkeys((*impact_bins, exported_n_bins))):
        blocks: dict[str, pd.DataFrame] = {}
        offset_delta = None
        if binned_offset:
            offset_arr = np.asarray(offset, dtype=np.float64)
            with np.errstate(over="ignore", under="ignore"):
                multiplier = np.exp(offset_arr)
            block, applied = _binned_offset_block(
                multiplier, _weights_array(len(X), sample_weight), int(n_bins), bin_strategy
            )
            blocks[block.name] = block.table
            offset_delta = np.log(applied) - offset_arr
        try:
            result = model.discretization_impact(
                X,
                y,
                sample_weight=sample_weight,
                offset=offset,
                n_bins=int(n_bins),
                bin_strategy=bin_strategy,
                features=features,
                band_se=band_se,
                band_max_error=band_max_error,
                _offset_delta=offset_delta,
            )
        except OverflowError as error:
            # Only the exported resolution refuses; a comparison rung float64
            # cannot hold is recorded as an unbounded change.
            if int(n_bins) == exported_n_bins:
                raise
            unbounded = {
                "n_bins": int(n_bins),
                "exported": False,
                "mean_abs_prediction_change_pct": np.inf,
                "max_abs_prediction_change_pct": np.inf,
                "note": str(error),
            }
            rows.extend({**unbounded, "feature": name} for name in [*features, *blocks])
            continue
        if not impact_bins:
            return _empty_impact_frame()
        for feature, table in {**result.tables, **blocks}.items():
            row: dict[str, float | int | str] = {
                "n_bins": int(n_bins),
                "exported": int(n_bins) == exported_n_bins,
                "feature": feature,
                "actual_bins": int(len(table)),
            }
            row.update(result.metrics)
            if bin_strategy == "exact":
                diagnostics = result.band_diagnostics.get(feature, {})
                row.update(
                    {f"band_{key}": value for key, value in diagnostics.items() if key != "bands"}
                )
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
            if bin_strategy == "exact":
                diagnostics = result.band_diagnostics.get(interaction, {})
                row.update(
                    {f"band_{key}": value for key, value in diagnostics.items() if key != "bands"}
                )
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
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    band_se: float = 1.0,
    band_max_error: float = 0.10,
    centering: str = "native",
    continuous_kind: str = "binned",
    allow_unbounded_extrapolation: bool = False,
) -> RatingTablePayload:
    """Build the renderer-independent rating-table payload.

    The payload represents a multiplicative tariff for a log-link model:
    ``base_relativity`` times one relativity per main-effect block, times one
    per interaction block, reproduces ``model.predict`` on every row whose
    linear predictor stays inside the log-link stability range ``[-80, 80]``.
    Non-log links, ``Binomial`` models, saturated predictors on ``X``, and
    unusable per-block factors are refused at the boundary.

    Exact blocks
    ------------
    ``Categorical``, ``OrderedCategorical``, ``Numeric``, ``Piecewise``, and
    categorical-by-categorical (or ordinal-parent spline-by-categorical)
    interactions are tabulated without binning and reproduce ``model.predict``
    to float64 round-off. Two interaction limitations apply: interactions whose
    categorical parent uses ``grouping=`` are keyed on grouped level labels while
    the main-effect block expands to raw labels, and continuous grids with a
    spline-mode ``OrderedCategorical`` parent print mapped-score axes rather
    than level labels.

    Offsets
    -------
    ``offset_max_exact_levels`` governs BOTH paths — the declared path via
    ``offset_source=`` and the undeclared path using the offset vector alone —
    selecting a per-level lookup up to that cap and a single exact ``"per_unit"``
    row above it. However, the two paths COUNT DIFFERENT THINGS: the declared
    path counts distinct values of the SOURCE column, whereas the undeclared
    path counts distinct MULTIPLIERS (``exp(offset)`` rounded to 12 significant
    digits). On the undeclared path the per-unit row is keyed on the MULTIPLIER
    itself (with ``Relativity = 1.0``) because no column was declared.

    ``offset_kind`` controls the emitted shape:

    - ``"auto"`` (default): exact lookup up to ``offset_max_exact_levels``,
      otherwise ``"per_unit"``.
    - ``"discrete"``: exact lookup; a declared source raises above the cap,
      while the undeclared path falls through to ``"per_unit"``.
    - ``"per_unit"``: always emits the exact per-unit relation (verified
      against ``offset_mapping_rtol`` relative to block scale).
    - ``"binned"``: exposure-banded summary on the undeclared path only
      (refused when ``offset_source`` is declared). Empty bins report their
      interval midpoint with weight 0, and the block is included in the
      discretization-impact sweep.

    Continuous terms and ``continuous_kind``
    ----------------------------------------
    ``continuous_kind : {"binned", "ppform"}, default "binned"`` selects how
    continuous main-effect terms are represented:

    - ``"binned"`` approximates each spline or polynomial curve by interval
      averages (geometry-weighted under ``"frequency"``, physical-row-weighted
      under ``"prior"``). Measured on a motor book with 81 distinct ages at
      ``n_bins=150``, the worst row was mis-rated by 60% in sparse tail bins.
      ``n_bins`` is a BUDGET, not a target: a covariate with fewer distinct
      values than that budget is still binned. Under ``bin_strategy="exact"``,
      bands are placed to keep every value within
      ``min(band_se * SE, log(1 + band_max_error))`` of the curve (jointly with
      any spline-by-categorical interactions on the same covariate), widening
      with a warning if ``n_bins`` is too small.
    - ``"ppform"`` exports each spline's exact piecewise-polynomial form: one
      row per knot interval with SEVEN columns rather than three (``<feature>``,
      ``Relativity``, ``Weight``, followed by ``a``, ``b``, ``c``, ``d``). A
      consumer reads both bounds back out of the interval key and evaluates
      ``exp(a + b*u + c*u**2 + d*u**3)`` with
      ``u = (x - lower) / (upper - lower)`` on bounded rows, and reads
      ``Relativity`` directly on unbounded tail rows. An un-upgraded loader
      still reads it as a step function from the first three columns; a
      consumer that stores coefficients must include them in any content digest
      so models differing only in polynomial coefficients do not fingerprint
      identically.
    - Under ``"ppform"``, ``extrapolation="clip"`` emits constant unbounded
      tail rows, ``extrapolation="error"`` emits no unbounded rows, and
      ``extrapolation="extend"`` raises unless
      ``allow_unbounded_extrapolation=True`` (which clips tails to the boundary).
      Terms with ``Constraint.postfit`` repairs are refused under ``"ppform"``;
      ``Constraint.fit`` constraints convert unchanged. ``Polynomial`` terms
      stay binned, as do continuous-by-continuous interaction grids.

    Interactions and discretization impact
    --------------------------------------
    Continuous-by-continuous interactions export as ``"grid"`` blocks sampled at
    ``n_bins`` nodes per axis; ``SplineCategorical`` interactions on numeric
    parents export as ``"spline_cells"`` blocks keyed on covariate bins and
    categorical levels. All approximated blocks (binned main effects, sampled
    grids, binned spline-categorical deviations, and ``offset_kind="binned"``)
    are swept across ``impact_bins`` plus the exported ``n_bins`` with joint
    unclipped prediction-change metrics, and boundaries are printed at
    round-trip precision.

    Centering
    ---------
    ``centering="native"`` keeps each term under its fit identifiability
    constraint. ``centering="mean"`` shifts ``Categorical``, ``Piecewise``, and
    binned continuous blocks by their ``term_inference`` centering shift and
    folds the sum into ``base_relativity`` so the workbook product is unchanged
    (Anderson et al., CAS Study Note, 3rd ed. 2007, s1.127–1.129; Goldburd et
    al., CAS Monograph 5, 2nd ed., s2.4.3; Werner & Modlin, *Basic Ratemaking*,
    4th ed. 2010, ch. 10 & 14). ``OrderedCategorical``, ``Numeric``, and offset
    blocks are not re-centred.

    Raises
    ------
    ValueError
        If ``centering``, ``continuous_kind``, or ``offset_kind`` is invalid;
        if the model link is not ``log`` or the family is ``Binomial``; if
        ``model.predict`` saturates on ``X``; if any emitted relativity or
        per-unit offset multiplier is non-finite or outside
        ``(exp(-500), exp(500))``; or under ``continuous_kind="ppform"`` if a
        convertible spline carries a ``Constraint.postfit`` repair or
        ``extrapolation="extend"`` without
        ``allow_unbounded_extrapolation=True``. Checked before
        ``base_relativity`` when both fail.
    RatingTableBaseNotRepresentableError
        If ``base_relativity`` overflows, underflows to zero, or is subnormal.
    NotImplementedError
        If the model contains an unsupported term type (``RandomEffect``,
        ``FactorSmooth``, or an unsupported custom interaction).
    RuntimeError
        If ``model`` has not been fitted.
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
    # Beside the continuous kind and for the same reason: the mode is checked
    # once, up front, whatever the model turns out to carry.  Leaving it to the
    # block builders meant only the declared-source path ever looked, so an
    # unknown kind reached the undeclared path as ``"auto"``.
    _require_supported_offset_kind(offset_kind)
    _require_exact_banding_settings(model, bin_strategy, band_se, band_max_error, offset_kind)
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
    # Also the model alone, and before the banding passes a refused term would waste.
    _require_distinct_level_texts(model)

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

    # Unconditionally, and before any block is built.  Every other consumer of
    # these weights validates them on the way past; this is the one call that
    # does not depend on which block kinds the model happens to contain.
    export_weights = _export_weights(model, len(frame), sample_weight)

    # Only the terms that will actually be BINNED, so the tables are built for
    # the blocks that need them.  Under ``"ppform"`` a spline is exported from
    # its own coefficients and never reads a bin average, so sweeping it here
    # is work whose result is discarded -- and it is what used to make the term
    # continuous in the sweep's eyes further down.
    binned_continuous = [
        name
        for name in _continuous_features(model)
        if not (continuous_kind == "ppform" and isinstance(model._specs[name], _SplineBase))
    ]
    # Under "exact" a spline-by-categorical interaction is banded WITH its
    # covariate's main effect, in this one call, so that the main block and the
    # interaction block carry the same intervals and every curve the workbook
    # approximates holds its limit on them.  The other strategies bin the
    # interaction on their own, from the covariate alone.
    from superglm.diagnostics.discretize import banding_companions

    exact_companions = banding_companions(model) if bin_strategy == "exact" else {}
    companion_names = [name for names in exact_companions.values() for name in names]
    selected = (
        model.discretization_impact(
            frame,
            y_arr,
            sample_weight=sample_weight,
            offset=export_offset,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            features=binned_continuous + companion_names,
            band_se=band_se,
            band_max_error=band_max_error,
        )
        if binned_continuous or companion_names
        else None
    )
    if selected is not None and bin_strategy == "exact":
        _warn_widened_bands(
            selected.band_diagnostics,
            n_bins,
            {name: covariate for covariate, names in exact_companions.items() for name in names},
        )

    main_effects: list[RatingTableBlock] = []
    for name in model._feature_order:
        spec = model._specs[name]
        # Piecewise is tested FIRST and deliberately does not join
        # ``_continuous_features``.  That list is what ``binned_continuous`` is
        # drawn from and what reaches ``discretization_impact``, whose
        # continuity gate raises for anything that is not a spline or a
        # polynomial; and every name that survives into it is routed to the
        # BINNED ``_continuous_block`` below, which is the lossy path this
        # feature exists to remove.  A piecewise term therefore also contributes
        # no row to the impact sheet -- correctly, since its export has no
        # discretisation error to measure, which is the same reason a ppform
        # block contributes none.
        if isinstance(spec, Piecewise):
            main_effects.append(_piecewise_block(model, name, centering))
        elif continuous_kind == "ppform" and isinstance(spec, _SplineBase):
            # Routed on the SPEC, not on ``name in selected.tables``.  The exact
            # block needs no bin average, so the term is no longer in that dict
            # at all; routing on it would drop the term through every branch
            # below and emit no block, silently losing a factor from the
            # product.  ``Polynomial`` stays binned -- the block is fixed at
            # four coefficients while a polynomial's degree is not -- which is
            # why the test is ``_SplineBase`` rather than "is continuous".
            #
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
                    export_weights,
                    np.asarray(frame.column_array(name), dtype=np.float64),
                )
            )
        elif selected is not None and name in selected.tables:
            # Through the same term the exact blocks read their shift from,
            # so every block in the workbook is in the centering the caller
            # asked for and all of them share one origin.  ``with_se=False``,
            # so this is the point estimate alone -- the band's own centering
            # is ``_recenter_term``'s business and is not routed through here.
            centering_shift = float(_main_effect_inference(model, name, centering).centering_shift)
            main_effects.append(_continuous_block(name, selected.tables[name], centering_shift))
        elif isinstance(spec, Categorical | OrderedCategorical):
            main_effects.append(
                _categorical_block(model, frame, name, sample_weight, centering, spec)
            )
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
                offset_kind=offset_kind,
                offset_max_exact_levels=offset_max_exact_levels,
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
            )
        main_effects.append(offset_block)

    # Built here rather than inline in the constructor call below, so that the
    # guard can see the interaction cells.  Those are the only exported factors
    # that never touch ``_safe_exp`` under any centering, so they are the one
    # place the disciplines this module applies to the base -- reject inf,
    # reject 0.0, reject subnormal -- used to be absent entirely (issue #289).
    # ``_interaction_blocks`` is pure, so the hoist changes nothing else.
    interactions = _interaction_blocks(
        model,
        n_bins,
        frame=frame,
        sample_weight=sample_weight,
        bin_strategy=bin_strategy,
        banded_tables=selected.interaction_tables if selected is not None else None,
    )

    # After the blocks are built, because it is the emitted values it checks
    # rather than the coefficients they came from.
    _require_usable_relativities_export(main_effects, interactions)

    # Both lists, because both are approximated, and BOTH read off the
    # blocks that were built rather than re-derived from the specs -- so a
    # term the workbook carries exactly cannot be reported as approximated.
    # The ``selected`` call above names the main effects, and under
    # ``"exact"`` the spline-by-categorical interactions banded with them,
    # whose tables become their blocks; the grids are not in it.
    swept = _binned_continuous_names(main_effects) + _gridded_interaction_names(interactions)
    binned_offset = export_offset is not None and offset_source is None and offset_kind == "binned"
    # With no comparison rung the sweep only re-evaluates the exported one,
    # which ``selected`` already did unless the sweep approximates more.
    if (
        not impact_bins
        and not binned_offset
        and set(swept) == set(binned_continuous + companion_names)
    ):
        impact = _empty_impact_frame()
    else:
        impact = _impact_sweep(
            model,
            frame,
            y_arr,
            sample_weight,
            offset=export_offset,
            impact_bins=impact_bins,
            bin_strategy=bin_strategy,
            binned_offset=binned_offset,
            exported_n_bins=int(n_bins),
            band_se=band_se,
            band_max_error=band_max_error,
            features=swept,
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
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    band_se: float = 1.0,
    band_max_error: float = 0.10,
    format: str | None = None,
    sheet_name: str = "Rating Tables",
    summary_sheet_name: str = "Model Summary",
    impact_sheet_name: str = "Discretization Impact",
    centering: str = "native",
    continuous_kind: str = "binned",
    allow_unbounded_extrapolation: bool = False,
) -> Path:
    """Export fitted rating tables, model summary, and discretization impact to Excel.

    Builds the payload via ``build_rating_table_payload`` (which holds the full
    multiplicative contract, offset rules, and validation checks) and writes it
    to an Excel workbook.

    Parameters
    ----------
    model : SuperGLM
        Fitted log-link model (excluding ``Binomial``).
    file_path : str or Path
        Destination workbook path (``.xlsx``).
    X : pandas or eager Polars DataFrame
        Evaluation data aligned with ``y``, used for weights, binning, and
        discretization-impact checks.
    y : NDArray
        Observed response vector aligned with ``X``.
    sample_weight : NDArray or None, default None
        Optional non-negative observation weights aligned with ``X``.
    offset : NDArray or None, default None
        Known link-scale offset aligned with ``X`` (required when exporting on
        a frame other than the original fit frame of an offset-fitted model).
    offset_source : str, Series, array-like, or None, default None
        Optional source column or array explaining a proportional offset.
    offset_name : str or None, default None
        Display name for the offset block.
    offset_kind : {"auto", "discrete", "per_unit", "binned"}, default "auto"
        Representation of the offset block (``"binned"`` is accepted only when
        ``offset_source`` is ``None``).
    offset_max_exact_levels : int, default 20
        Maximum distinct levels emitted as an exact offset lookup before
        switching to ``"per_unit"``.
    offset_mapping_rtol : float, default 1e-10
        Relative tolerance when verifying an ``offset_source`` relation.
    n_bins : int, default 150
        Bin budget for binned continuous terms, sampled interaction grids, and
        maximum band count under ``bin_strategy="exact"``.
    impact_bins : tuple of int, default (20, 50, 100, 200, 250)
        Bin counts swept on the discretization-impact sheet alongside ``n_bins``.
    bin_strategy : str, default "exposure_quantile"
        Binning rule for binned continuous blocks: ``"exposure_quantile"``,
        ``"uniform"``, ``"winsorized"``, or ``"exact"``. Under ``"exact"``,
        band averages stay within ``min(band_se * SE, log(1 + band_max_error))``
        of the curve unless ``n_bins`` forces a wider limit (which warns).
    band_se : float, default 1.0
        Pointwise standard-error tolerance under ``bin_strategy="exact"``.
    band_max_error : float, default 0.10
        Maximum relative error tolerance under ``bin_strategy="exact"``.
    format : str or None, default None
        Optional explicit output format (Excel ``"xlsx"``).
    sheet_name : str, default "Rating Tables"
        Worksheet name for main-effect and interaction rating tables.
    summary_sheet_name : str, default "Model Summary"
        Worksheet name for the model summary.
    impact_sheet_name : str, default "Discretization Impact"
        Worksheet name for the discretization-impact sweep.
    centering : {"native", "mean"}, default "native"
        Block centering convention; ``base_relativity`` absorbs any shift so
        the product of all blocks reproduces ``model.predict``.
    continuous_kind : {"binned", "ppform"}, default "binned"
        How continuous main-effect terms are represented on the sheet:

        - ``"binned"`` writes a three-column lookup block of interval averages
          (measured on a motor book with 81 distinct ages at ``n_bins=150``,
          the worst row was mis-rated by 60% in sparse tail intervals).
        - ``"ppform"`` writes each spline's exact piecewise-polynomial form as
          SEVEN columns rather than three: ``<feature>``, ``Relativity``, and
          ``Weight`` followed by ``a``, ``b``, ``c``, ``d``. A consumer reads
          both bounds back out of the interval key and evaluates
          ``exp(a + b*u + c*u**2 + d*u**3)`` with
          ``u = (x - lower) / (upper - lower)`` on bounded rows, and reads
          ``Relativity`` directly on unbounded tail rows (which carry text
          bounds such as ``[-inf, 18.0)`` and ``b = c = d = 0``). An
          un-upgraded loader still reads it as a step function from the first
          three columns; a consumer storing coefficients must include them in
          any content digest so models differing only in coefficients do not
          fingerprint identically.
        - Under ``"ppform"``, ``extrapolation="clip"`` emits constant tail
          rows, ``extrapolation="error"`` emits no unbounded rows, and
          ``extrapolation="extend"`` raises unless
          ``allow_unbounded_extrapolation=True`` (which clips tails). Terms
          with ``Constraint.postfit`` repairs are refused under ``"ppform"``;
          ``Polynomial`` terms stay binned, as do continuous interaction grids.
    allow_unbounded_extrapolation : bool, default False
        Allow exporting ``extrapolation="extend"`` splines under
        ``continuous_kind="ppform"`` by clipping their unbounded tails.

    Returns
    -------
    Path
        Path to the written workbook.
    """
    out = Path(file_path)
    _require_excel_format(out, format)

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
        n_bins=n_bins,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        band_se=band_se,
        band_max_error=band_max_error,
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
