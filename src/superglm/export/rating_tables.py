"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
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
    # block a centering leaves alone -- the offset blocks, the binned
    # continuous blocks, an ``OrderedCategorical``, a single-valued
    # ``Numeric`` -- which is why the total is summed from the blocks rather
    # than assumed to run over every exported term.
    centering_shift: float = 0.0


@dataclass(frozen=True)
class InteractionTableBlock:
    """One two-way interaction rating-table block."""

    name: str
    table: pd.DataFrame


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


def _format_interval(left: float, right: float) -> str:
    return f"[{left:.10g}, {right:.10g})"


def _format_axis_value(value: float) -> str:
    return f"{value:.10g}"


def _continuous_block(name: str, table: pd.DataFrame) -> RatingTableBlock:
    out = pd.DataFrame(
        {
            name: [
                _format_interval(float(row.bin_from), float(row.bin_to))
                for row in table.itertuples(index=False)
            ],
            "Relativity": table["relativity"].astype(float).to_numpy(),
            "Weight": table["sample_weight"].astype(float).to_numpy(),
        }
    )
    return RatingTableBlock(name=name, kind="continuous", table=out)


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
    exact_multiplier = np.round(multiplier, 12)
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
        if not np.any(mask):
            avg_multiplier = 0.0
            exposure = 0.0
        else:
            exposure = float(weights[mask].sum())
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
    if "n_points" in signature(ispec.reconstruct).parameters:
        return ispec.reconstruct(beta, n_points=n_bins)
    return ispec.reconstruct(beta)


def _continuous_interaction_block(
    name: str,
    raw: dict,
    parent1: str,
    parent2: str,
) -> InteractionTableBlock:
    x1 = np.asarray(raw["x1"], dtype=np.float64)
    x2 = np.asarray(raw["x2"], dtype=np.float64)
    relativity = np.asarray(raw["relativity"], dtype=np.float64)
    if relativity.shape == (len(x2), len(x1)):
        relativity = relativity.T
    elif relativity.shape != (len(x1), len(x2)):
        raise ValueError(
            f"Interaction {name!r} returned a {relativity.shape} relativity grid, "
            f"expected {(len(x1), len(x2))} or {(len(x2), len(x1))}."
        )

    table = pd.DataFrame(relativity, columns=[_format_axis_value(v) for v in x2])
    table.insert(0, parent1, [_format_axis_value(v) for v in x1])
    return InteractionTableBlock(name=name, table=table)


def _interaction_blocks(model: SuperGLM, n_bins: int) -> list[InteractionTableBlock]:
    blocks: list[InteractionTableBlock] = []
    for name in model._interaction_order:
        ispec = model._interaction_specs[name]
        parent1, _ = ispec.parent_names
        raw = _reconstruct_interaction(ispec, _interaction_beta(model, name), n_bins)
        if {"x1", "x2", "relativity"} <= set(raw):
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
        blocks.append(InteractionTableBlock(name=name, table=pd.DataFrame(rows)))
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
    has nothing to center, the binned continuous blocks come from the
    discretisation path and never see ``centering=`` at all, and the offset
    blocks are not relativities of a fitted term.  And the constant removed
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
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if not features:
        return _empty_impact_frame()

    for n_bins in impact_bins:
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
                "feature": feature,
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
    a six-level territory collapsed to four: 68.7% of rows, and the same share
    of weight, key on a label the interaction table does not have.  This is
    pre-existing -- the interaction export predates issue #253 and centering
    does not touch it -- and is tracked as issue #286; until it is fixed, the
    product contract above holds for interactions whose parents are ungrouped.

    Lossy, by construction, for the binned blocks -- ``Spline``,
    ``Polynomial``, and the continuous-by-continuous interaction grid.  Two
    distinct errors ride on those, and only the first is reported:

    * The binning itself, which the ``discretization_impact`` sheet quantifies.
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
      prior-weighted residual is not zero.  Measured on a Tweedie(p=1.5) fit
      with weights drawn on [0.5, 20]: the residual mean is 1.2e-18 per
      physical row and -8.7e-04 under the prior weights, and per bin 1.9e-17
      against 6.4e-03.
    * Interval-string rounding, which the impact sheet does NOT see.  The
      impact sweep bins on the exact ``edges`` array while the exported block
      prints its bin boundaries through ``_format_interval`` at ``.10g``, so a
      consumer applying the printed strings re-bins the rows that sit within
      the rounding band of an edge -- and with ``bin_strategy=
      "exposure_quantile"`` the edges ARE data values, so a row sitting exactly
      on one flips whenever its printed edge rounds up.  Measured on the
      equivalence fixture (900 rows, 150 bins, two continuous terms): 302 of
      302 printed edges differ from exact by up to 4.99e-09, 133 of 900 rows
      (14.8%) land in a different bin, and the reconstruction carries 2.29e-01
      maximum relative error against the discretised predictions the blocks are
      meant to reproduce exactly -- 6.94e-16 with the exact edges.  Tracked
      separately as issue #278; fixing it changes a public column of the
      payload, not the centering this function fixed.

    ``centering`` is a presentation choice and does not change what the
    payload rates.  ``"native"`` reports each term under the model's own
    identifiability constraint.  ``"mean"`` shifts the terms that have a mean
    to shift -- ``Categorical`` and ``Piecewise`` -- so their relativities have
    geometric mean 1, and ``base_relativity`` absorbs the total so the product
    is unchanged; before issue #253 it did not, and every reconstructed
    prediction came out scaled by ``exp(-sum_t shift_t)``.

    That mode is a PARTIAL centering, and deliberately so.  An
    ``OrderedCategorical`` is already anchored on its base level and is not
    recentered; a ``Numeric`` reports one per-unit relativity with no mean to
    take; and the binned ``Spline``/``Polynomial`` blocks come from the
    discretisation path, which never sees ``centering`` at all.  Only the
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
        If ``centering`` is not one of ``("native", "mean")``.
    RatingTableBaseNotRepresentableError
        If the exported base relativity overflows or underflows float64.
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
    # After the term preflight and before anything is built.  After, because an
    # unsupported term type is the more fundamental complaint -- a model with a
    # ``RandomEffect`` has no rating table under any link, and hearing about the
    # link first would send the caller to change something that was not the
    # problem.  Before the build, because the whole payload is a product of
    # exponentials, which is the prediction only under a log link.
    _require_log_link_export(model)

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
            main_effects.append(_continuous_block(name, selected.tables[name]))
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

    impact = _impact_sweep(
        model,
        frame,
        y_arr,
        sample_weight,
        offset=export_offset,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        features=continuous,
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
        interactions=_interaction_blocks(model, n_bins),
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
) -> Path:
    """Render the rating-table payload to a workbook and return the path.

    A thin renderer over :func:`build_rating_table_payload`, which is where the
    payload's contract lives: what the exported product reproduces, which term
    types are exact and which are binned, what ``centering=`` does and does not
    change, and which errors the export raises.  Read that docstring before
    relying on an exported workbook; this function adds only the file format,
    the sheet names, and the extra rounding a renderer imposes.
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
