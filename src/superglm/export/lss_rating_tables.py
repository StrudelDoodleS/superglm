"""Rating tables for a fitted SuperLSS: one multiplicative table per family parameter.

A parameter whose inverse link is ``c + exp(eta)`` (a log link, ``c = 0``, or a floored log
link, ``c`` the floor) is ``c + exp(intercept + sum of term effects + offset)``, so
``c + base * product of relativities`` reproduces ``predict_parameters`` row by row
(Klein, Denuit, Lang and Kneib 2013 report each parameter's effects on its own score in the same
way; Smyth and Jorgensen 2002 tabulate a mean as a base rate with multipliers).  Each table
multiplies the quantity the editor shows, and ``rule`` says how the model's parameter follows from
the product: ``identity``, ``floor_plus_product`` or ``reciprocal`` (negative-binomial theta, whose
table is the overdispersion ``1 / theta`` so a larger factor always means more variance).

The blocks reuse the SuperGLM export's generic pieces.  A spline is its exact piecewise
polynomial, a categorical is one factor per level read against its reference level, a numeric
term is one factor per unit, a categorical-by-categorical interaction is a full cell table and an
offset is an applied multiplier.  Anything that is not exact is refused by name, for its parameter
only, and the parameters that qualify are still exported.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.offsets import prediction_offsets, sources_of
from superglm.distributional.terms import (
    _coefficient_columns,
    _compiled_predictor,
    _feature_label,
    _prepare_term_effect,
    _reference_design,
    _term_kind,
)
from superglm.export.rating_tables import (
    InteractionTableBlock,
    RatingTableBlock,
    _base_relativity,
    _continuous_ppform_block,
    _offset_multiplier_block,
    _require_excel_format,
    _require_usable_relativities_export,
    _row_key_header,
    _weights_array,
    _weights_by_level,
)
from superglm.features.interaction import CategoricalInteraction, interaction_cells
from superglm.features.spline import _SplineBase
from superglm.inference._term_helpers import _build_spline_metadata
from superglm.links import _LOG_LINK_PROBE, _is_shifted_log_link

_OFFSET_KINDS = ("auto", "discrete", "per_unit")

# What a parameter's table multiplies, where it is not the parameter's own name.  Keyed by the
# family and the parameter; any other parameter is tabulated as itself ("Mean", "Dispersion").
_QUANTITY = {
    ("GammaLS", "scale"): "CV",
    ("LogNormalLS", "scale"): "Sigma",
    ("GaussianLS", "scale"): "Sigma",
    ("NegativeBinomialLS", "theta"): "Overdispersion",
}
_RECIPROCAL = frozenset({("NegativeBinomialLS", "theta")})
# The names the editor gives each parameter (``superglm.editor.lss.display``), which the workbook's
# Summary sheet uses for its Parameter column, the code name beside them.
_PARAMETER_LABELS = {
    "mean": "Mean",
    "scale": "CV",
    "theta": "Overdispersion",
    "dispersion": "Dispersion",
    "power": "Power",
}
_WEIGHT_NOTE = (
    "Weight is the sum of the sample weights of the rows in a bin or level, or the number of rows "
    "when no sample weights were given."
)


def _family_notes(family: str, model: object) -> tuple[str, ...]:
    """What a family's tables need besides the product: a Tweedie's chance of no claim.

    A prior weight enters as the dispersion over ``w``, so it scales the claim rate; a frequency
    weight counts repeated rows, each of which keeps the plain law.
    """
    if family != "TweedieLSS":
        return ()
    if getattr(model, "weight_semantics", "prior") == "frequency":
        rate, weight = (
            "mean^(2 - p) / (dispersion * (2 - p))",
            (" A frequency weight counts repeated rows and leaves each row's chance as it is."),
        )
    else:
        rate, weight = (
            "w * mean^(2 - p) / (dispersion * (2 - p))",
            (" w is the row's prior weight, 1 for a model fitted without weights."),
        )
    return (
        f"Chance of no claim: exp(-lambda), with lambda = {rate}, where mean and dispersion are "
        "the row's values from the Mean and Dispersion tables and p is the model's power."
        f"{weight} It is a Tweedie's point mass at zero.",
    )


def parameter_label(parameter: str) -> str:
    """The editor's name for a family parameter: ``mean`` is Mean, ``scale`` is CV, ``theta`` is Overdispersion."""
    return _PARAMETER_LABELS.get(parameter, parameter.capitalize())


_NOTES = {
    ("GammaLS", "scale"): (
        " CV is the coefficient of variation: the standard deviation divided by the mean."
    ),
    ("LogNormalLS", "scale"): (
        " Sigma is the standard deviation of the log amount; for a log-normal, the coefficient"
        " of variation is CV = sqrt(exp(sigma^2) - 1)."
    ),
    ("NegativeBinomialLS", "theta"): (
        " Variance = mean + mean^2 * overdispersion. Offsets enter as exp(-offset)."
    ),
}


@dataclass(frozen=True)
class ParameterRatingTable:
    """One parameter's table set: a base, one block per term, interaction blocks, the offset block.

    ``rule`` says how the model's parameter follows from ``base_relativity`` times one relativity
    per block: ``"identity"`` (the parameter is the product), ``"floor_plus_product"`` (the
    parameter is ``floor`` plus the product) or ``"reciprocal"`` (the parameter is one over the
    product).  ``floor`` is zero unless the rule is ``"floor_plus_product"``.
    """

    parameter: str
    quantity: str
    link: str
    rule: str
    floor: float
    base_relativity: float
    main_effects: list[RatingTableBlock]
    interactions: list[InteractionTableBlock]
    note: str


@dataclass(frozen=True)
class LSSRatingTablePayload:
    """The table sets of every parameter that has one, and why each other parameter has none."""

    parameters: tuple[ParameterRatingTable, ...]
    omitted: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def blocks(self) -> dict[str, RatingTableBlock | InteractionTableBlock]:
        """Every block keyed ``"<parameter>:<term>"``, such as ``"mean:age"``."""
        return {
            f"{table.parameter}:{block.name}": block
            for table in self.parameters
            for block in (*table.main_effects, *table.interactions)
        }


class _SignedCurves:
    """``extract_ppform``'s view of one parameter's terms: the effects, times ``sign``."""

    def __init__(self, fitted, frame, parameter: str, sign: float) -> None:
        self.fitted, self.frame, self.parameter, self.sign = fitted, frame, parameter, sign

    def term_inference(self, term: str, *, n_points: int, with_se: bool, centering: str):
        prepared = _prepare_term_effect(
            self.fitted, self.frame, self.parameter, term, n_points=n_points, simultaneous=False
        )
        spec = _compiled_predictor(self.fitted, self.parameter).compiled.specs[term]
        return SimpleNamespace(
            spline=_build_spline_metadata(spec),
            x=prepared.x,
            log_relativity=self.sign * prepared.effect,
            centering_shift=0.0,
        )


def _link_name(link) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", type(link).__name__.removesuffix("Link")).lower()


def _a_link(link) -> str:
    name = _link_name(link)
    return f"{'an' if name[0] in 'aeiou' else 'a'} {name} link"


def _log_floor(link) -> float:
    """The floor ``c`` of a link whose inverse is ``c + exp(eta)``; refuse any other link."""
    floor = float(getattr(link, "floor", 0.0))
    if not _is_shifted_log_link(link) or not np.allclose(
        link.inverse(_LOG_LINK_PROBE), floor + np.exp(_LOG_LINK_PROBE), rtol=1e-12, atol=0.0
    ):
        raise ValueError(
            f"it is on {_a_link(link)}, so a term changes it by adding to it on that "
            "link's scale, not by multiplying it; no multiplicative rating table exists for it."
        )
    return floor


def _categorical_block(prepared, name, frame, column, sample_weight, sign) -> RatingTableBlock:
    levels = list(prepared.levels)
    table = pd.DataFrame(
        {
            name: levels,
            "Relativity": np.exp(sign * prepared.effect),
            "Weight": _weights_by_level(frame, column, levels, sample_weight),
        }
    )
    return RatingTableBlock(name=name, kind="categorical", table=table)


def _cell_block(name, ispec, beta, sign) -> InteractionTableBlock:
    raw = ispec.reconstruct(beta)
    # Cells are read by their pair of levels, since the "level1:level2" labels collide once a
    # level holds a colon; the row-key header follows the SuperGLM cell table's rule.
    cells = interaction_cells(raw)
    levels2 = raw["levels2"]
    rows = [
        [level1, *(cells.get((level1, level2), 1.0) ** sign for level2 in levels2)]
        for level1 in raw["levels1"]
    ]
    columns = [_row_key_header(ispec.parent_names[0], levels2), *levels2]
    return InteractionTableBlock(name=name, table=pd.DataFrame(rows, columns=columns), kind="cells")


def _parameter_table(
    fitted, frame, weights, sample_weight, parameter, offset, family, offset_kind, max_levels
) -> ParameterRatingTable:
    state = fitted.layout.predictor(parameter)
    floor = _log_floor(state.link)
    key = (family, parameter)
    sign = -1.0 if key in _RECIPROCAL else 1.0
    if sign < 0 and floor != 0.0:
        raise ValueError("its link has a floor, so one over the parameter is not a product.")
    rule = "reciprocal" if sign < 0 else ("floor_plus_product" if floor else "identity")
    quantity = _QUANTITY.get(key, parameter.capitalize())

    beta = np.asarray(fitted.prediction_coefficients, dtype=np.float64)
    compiled = _compiled_predictor(fitted, parameter).compiled
    curves = _SignedCurves(fitted, frame, parameter, sign)
    log_base = 0.0 if state.intercept_index is None else float(beta[state.intercept_index])
    main_effects: list[RatingTableBlock] = []
    for name in compiled.feature_order:
        spec = compiled.specs[name]
        kind = _term_kind(spec)
        qualified = fitted.layout.term_slices[f"{parameter}:{name}"]
        special = fitted.layout.term_slices.get(f"{parameter}:{name}:special")
        blocks = (qualified,) if special is None else (qualified, special)
        column = _feature_label(fitted, parameter, name)
        if getattr(spec, "_grouping", None) is not None:
            raise ValueError(
                f"its term {name!r} groups levels, and a table keyed on the original levels "
                "cannot be read back from the grouped fit; ungroup it to export."
            )
        if kind == "categorical":
            prepared = _prepare_term_effect(
                fitted, frame, parameter, name, n_points=2, simultaneous=False
            )
            log_base += float(
                _reference_design(fitted, frame, parameter, name, state, *blocks)
                @ beta[_coefficient_columns(*blocks)]
            )
            main_effects.append(
                _categorical_block(prepared, name, frame, column, sample_weight, sign)
            )
        elif isinstance(spec, _SplineBase):
            if _build_spline_metadata(spec).extrapolation == "extend":
                raise ValueError(
                    f"its spline {name!r} continues past its last knot, and a table clips there; "
                    "refit it with extrapolation='clip' to export."
                )
            values = np.asarray(frame.column_array(column), dtype=np.float64)
            main_effects.append(_continuous_ppform_block(curves, name, "native", weights, values))
        elif kind == "numeric":
            prepared = _prepare_term_effect(
                fitted, frame, parameter, name, n_points=2, simultaneous=False
            )
            slope = float(beta[qualified][0])
            # The effect is read against the column's reference value; the per-unit factor
            # applies to the value itself, so what the effect leaves out goes back on the base.
            log_base += float(
                prepared.effect[0]
                + prepared.anchor @ beta[_coefficient_columns(*blocks)]
                - slope * np.asarray(prepared.x, dtype=np.float64)[0]
            )
            table = pd.DataFrame(
                {name: ["per_unit"], "Relativity": [np.exp(sign * slope)], "Weight": [0.0]}
            )
            main_effects.append(RatingTableBlock(name=name, kind="numeric", table=table))
        else:
            raise ValueError(
                f"its term {name!r} is a {type(spec).__name__}, which has no exact rating table "
                "yet; remove it from this parameter to export the rest."
            )

    interactions: list[InteractionTableBlock] = []
    for name in compiled.interaction_order:
        ispec = compiled.interaction_specs[name]
        if not isinstance(ispec, CategoricalInteraction):
            raise ValueError(
                f"its interaction {name!r} is a {type(ispec).__name__}, which is a surface and "
                "has no exact rating table yet; only categorical-by-categorical interactions do."
            )
        if any(getattr(compiled.specs[p], "_grouping", None) for p in ispec.parent_names):
            raise ValueError(
                f"its interaction {name!r} is keyed on grouped levels, which cannot be read "
                "back to the original levels; ungroup the parents to export."
            )
        interactions.append(
            _cell_block(name, ispec, beta[fitted.layout.term_slices[f"{parameter}:{name}"]], sign)
        )

    if offset is not None:
        block = _offset_multiplier_block(
            sign * np.asarray(offset, dtype=np.float64),
            len(frame),
            sample_weight,
            n_bins=1,  # only the binned kind reads it, and that kind is not offered
            bin_strategy="exposure_quantile",
            offset_kind=offset_kind,
            offset_max_exact_levels=max_levels,
        )
        if block is not None:
            main_effects.append(block)
    _require_usable_relativities_export(main_effects, interactions)

    note = {
        "identity": f"{quantity} = base x product of the relativities.",
        "floor_plus_product": (
            f"{quantity} = floor + base x product of the relativities; the floor is {floor:g}."
        ),
        "reciprocal": (
            f"{quantity} = base x product of the relativities; the model's {parameter} is "
            f"1 / {quantity}."
        ),
    }[rule] + _NOTES.get(key, "")
    return ParameterRatingTable(
        parameter=parameter,
        quantity=quantity,
        link=_link_name(state.link),
        rule=rule,
        floor=floor,
        base_relativity=_base_relativity(sign * log_base),
        main_effects=main_effects,
        interactions=interactions,
        note=note,
    )


def _postfit_notes(fitted) -> tuple[str, ...]:
    """A sentence for each ``Constraint.postfit`` rule: the repair its table shows, or none yet."""
    from superglm.distributional.shape_repair import postfit_term_feasible, postfit_terms

    edit = fitted.fit_state.hand_edit
    repairs = () if edit is None else edit.repairs
    notes = [
        f"{record.term} was repaired after the fit to be {record.kind} (Constraint.postfit); its "
        "table is the repaired curve, which moved by at most "
        f"{record.largest_change:.3g} on its parameter's link scale."
        for record in repairs
    ]
    repaired = {record.term for record in repairs}
    coefficients = np.asarray(fitted.coefficients, dtype=np.float64)
    notes.extend(
        f"{term.term} carries Constraint.postfit.{term.kind}, which has not been applied, and its "
        "fitted curve breaks it; this table is the unrepaired curve. Call "
        "apply_shape_postfit() before exporting to tabulate the repaired one."
        for term in postfit_terms(fitted)
        if term.term not in repaired and not postfit_term_feasible(term, coefficients)
    )
    return tuple(notes)


def _constant_sentence(fitted, parameter: str) -> str:
    """What to say of a parameter with no terms and no table: its one value."""
    state = fitted.layout.predictor(parameter)
    beta = np.asarray(fitted.prediction_coefficients, dtype=np.float64)
    value = float(state.link.inverse(beta[[state.intercept_index]])[0])
    return (
        f"Parameter {parameter!r} has no terms: it is {value:.6g} on every row and needs no table."
    )


def build_lss_rating_table_payload(
    model,
    X: FrameLike | EagerFrame,
    sample_weight: NDArray | None = None,
    *,
    offsets: Mapping[str, NDArray] | None = None,
    offset_kind: str = "auto",
    offset_max_exact_levels: int = 20,
) -> LSSRatingTablePayload:
    """Build the rating tables of every parameter of a fitted SuperLSS that has one.

    ``X`` is the book the tables describe: it supplies each level's exposure and the offsets that
    sit in columns.  ``offsets`` is what ``predict_parameters`` takes, a dict of link-scale arrays
    keyed by parameter, and is needed for exactly the offsets ``predict_parameters`` needs it for.

    The contract is per row and per parameter: ``base_relativity`` times one relativity per
    block, interaction and offset blocks included, gives the quantity the table is for, and
    ``rule`` turns that into the model's parameter.  It reproduces ``predict_parameters`` to
    round-off wherever the linear predictor stays inside float64's exponent range.  The tables
    follow the coefficients the model predicts with, so a hand-edited model is tabulated as
    edited, and a term repaired after the fit (``SuperLSS.apply_shape_postfit``) as repaired,
    with a note saying so; a note also names each ``Constraint.postfit`` rule whose curve still
    breaks it.

    A parameter whose link is not a log or a floored log is left out, with a sentence in
    ``omitted`` that gives its value where it has no terms; so is one holding a term that has
    no exact table (a ``Polynomial`` or
    ``Piecewise`` term, a surface interaction, a spline-by-categorical interaction, a random
    effect).  Splines are always the exact piecewise-polynomial form.  A model with no
    parameter that has a table raises ``ValueError`` with every sentence.
    """
    if offset_kind not in _OFFSET_KINDS:
        raise ValueError(f"offset_kind must be one of {_OFFSET_KINDS}, got {offset_kind!r}")
    fitted = model._require_fitted()
    frame = as_eager_frame(X)
    weights = _weights_array(len(frame), sample_weight)
    resolved = prediction_offsets(fitted.fit_state.predictor_templates, frame, offsets)
    family = type(fitted.family).__name__
    tables: list[ParameterRatingTable] = []
    omitted: dict[str, str] = {}
    templates = {p.name: p for p in fitted.fit_state.predictor_templates}
    for parameter in model.parameter_names_:
        template = templates[parameter]
        has_offset = bool(sources_of(template)) or (offsets is not None and parameter in offsets)
        try:
            tables.append(
                _parameter_table(
                    fitted,
                    frame,
                    weights,
                    sample_weight,
                    parameter,
                    resolved[parameter] if has_offset else None,
                    family,
                    offset_kind,
                    offset_max_exact_levels,
                )
            )
        except (ValueError, OverflowError) as error:
            constant = not (template.features or template.interactions or has_offset)
            omitted[parameter] = (
                _constant_sentence(fitted, parameter)
                if constant
                else f"Parameter {parameter!r} was left out: {error}"
            )
    if not tables:
        raise ValueError(" ".join(omitted.values()))
    notes = (_WEIGHT_NOTE, *_family_notes(family, model), *_postfit_notes(fitted))
    return LSSRatingTablePayload(parameters=tuple(tables), omitted=omitted, notes=notes)


def export_lss_rating_tables(
    model,
    file_path: str | Path,
    X: FrameLike | EagerFrame,
    sample_weight: NDArray | None = None,
    *,
    format: str | None = None,
    **kwargs,
) -> Path:
    """Write ``build_lss_rating_table_payload`` as a workbook: a sheet per parameter and a summary."""
    from superglm.export.excel import write_lss_rating_table_workbook

    out = Path(file_path)
    _require_excel_format(out, format)
    payload = build_lss_rating_table_payload(model, X, sample_weight, **kwargs)
    write_lss_rating_table_workbook(payload, out)
    return out
