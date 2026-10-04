"""Structure files: a model's structural decisions, without its coefficients.

A structure records, per feature, the decisions that shape a term rather than
its fitted values: how a categorical's levels are grouped, which level or
group is the reference, where levels unseen at fit go, and the polynomial
ranges of a spline. :meth:`Structure.apply` builds those decisions into another
model's features, ready to fit on new data.

The file is JSON, written with sorted keys so that two exports of one model
are byte-identical and a change to it reads as a diff::

    {
      "features": {
        "VehBrand": {
          "groups": {"Other": ["B13", "B14"]},
          "kind": "categorical",
          "levels": ["B1", "B10", ...],
          "reference": "B12",
          "unseen": "Other"
        },
        "DrivAge": {
          "kind": "spline",
          "ranges": [{"degree": 1, "hi": 45.0, "join": "tangent", "lo": 30.0}]
        }
      },
      "format": "superglm.structure.v1",
      "superglm_version": "0.36.1"
    }

Levels keep their native types, so integer levels stay JSON numbers. Every
refusal is a :class:`StructureError`, a ``ValueError`` whose message is one
fixed sentence naming the feature and, where there is one, the level or range.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from superglm.features._spline_ranges import SHAPE_NAMES, PolynomialRange, RangeError
from superglm.features.categorical import Categorical
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import (
    accepted_levels,
    band_edges,
    clone_with_replaced_features,
    current_ranges,
    merged_ranges,
    pristine_basis,
    rebuilt_categorical,
    rebuilt_ordered_spec,
    shape_unavailable_reason,
    shaped_spline,
)
from superglm.features.spline import _SplineBase
from superglm.model import SuperGLM
from superglm.model.fit_state import configured_lambda2, configured_penalty

FORMAT = "superglm.structure.v1"
KINDS = ("categorical", "ordered", "spline")
_POLICIES = ("error", "base")
# The fields each kind of entry holds, besides "kind".
_FIELDS = {
    "categorical": ("groups", "levels", "reference", "unseen"),
    "ordered": ("groups", "levels", "ranges", "reference", "unseen"),
    "spline": ("ranges",),
}
_RANGE_FIELDS = ("degree", "hi", "join", "lo")
_FILE_FIELDS = ("features", "format", "superglm_version")

# One fixed sentence per refusal.
_NOT_JSON = "The structure is not valid JSON; to read a file, pass its path to read_structure."
_UNKNOWN_FORMAT = (
    "Unknown structure format {fmt!r}: this version of superglm reads {expected!r} files; "
    "export the structure again with it."
)
_MALFORMED_FILE = "The structure's {field!r} field is malformed; export the structure again."
_MALFORMED = (
    "The structure entry for {feature!r} has a malformed {field!r}; export the structure again."
)
_MEMBER = (
    "Group {group!r} of {feature!r} holds {member!r}, which is not one of its levels; add it "
    "to the levels or take it out of the group."
)
_TWO_GROUPS = "Level {member!r} of {feature!r} is in more than one group; keep it in one."
_GROUP_NAME = "Group {group!r} of {feature!r} has the name of a level outside it; rename the group."
_REFERENCE = (
    "The reference {reference!r} of {feature!r} is not a level or group of the term; choose "
    "an ungrouped level or a group."
)
_UNSEEN = (
    "New levels of {feature!r} go to {unseen!r}, which is not a group of the term; name one "
    "of its groups, or use 'error' or 'base'."
)
_ORDERED_UNSEEN = (
    "{feature!r} is an ordered term, which refuses new levels; set its unseen to 'error'."
)
_RANGE = "The spline of {feature!r} refuses the {span}; change or remove that range."
_FITTED_OUT = (
    "The spline of {feature!r} is fitted out past this data to hold the {spans} as written."
)
_UNFITTED = (
    "Structure.from_model needs a fitted model: {feature!r} has no fitted levels yet; "
    "fit the model first."
)
_MODEL_UNFITTED = "Structure.from_model needs a fitted model; fit the model first."
_NOT_A_SUPERGLM = (
    "Structure.{method} takes a SuperGLM model, not a {kind}; structure files do not cover "
    "other models yet."
)
_ABSENT = (
    "The structure names {feature!r}, which is not a feature of this model; remove it from "
    "the structure or apply it to a model that has it."
)
_KIND = (
    "The structure has {feature!r} as a {kind} term, but the model does not; apply it to a "
    "model that declares {feature!r} as a {kind} term."
)
_UNIVERSE = (
    "The levels of {feature!r} in the structure are not the levels the model declares for "
    "it; apply the structure to a model declared with the same levels."
)
_OUTSIDE_DECLARED = (
    "The data holds levels of {feature!r} that the model's levels= leaves out: {levels}; "
    "add them to its levels= or leave those rows out."
)
_NOT_APPLIED = (
    "The structure could not be applied to {feature!r}: the model's declaration of it does "
    "not accept these decisions."
)
_SAME_TEXT = (
    "Levels {first!r} and {second!r} of {feature!r} read as the same text, which is how a "
    "structure file tells levels apart; give the term distinct labels."
)
_NOT_WRITABLE = (
    "{value!r} in {feature!r} cannot be written to a structure file, which holds text, "
    "numbers and booleans; give the term plain labels."
)


class StructureError(ValueError):
    """A structure file, or a step that reads, writes or applies one, refused.

    The message is one fixed sentence naming the feature and, where there is
    one, the level or range, so it can be shown to a user as it stands. When
    the refusal reports an error from the library, that error is its
    ``__cause__``.
    """


@dataclass
class FeatureStructure:
    """The structural decisions recorded for one feature.

    Parameters
    ----------
    kind : {"categorical", "ordered", "spline"}
        The kind of term: a :class:`~superglm.Categorical`, an
        :class:`~superglm.OrderedCategorical`, or a numeric spline.
    levels : list
        The full level universe, in model order, in the levels' native types
        (categorical and ordered terms).
    groups : dict[str, list]
        Each group label and the levels it holds. Levels in no group stand
        alone.
    reference : object
        The reference: an ungrouped level or a group label.
    unseen : str
        Where levels unseen at fit go: ``"error"``, ``"base"`` or a group label.
        An ordered term refuses new levels, so its policy is ``"error"``.
    ranges : list of PolynomialRange
        The spline's polynomial ranges in the feature's own units, or between
        band names on an ordered term.
    """

    kind: str
    levels: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)
    reference: Any = None
    unseen: str = "error"
    ranges: list = field(default_factory=list)


@dataclass
class Structure:
    """The structural decisions of a model's features, without coefficients.

    Build one from a fitted model with :meth:`from_model`, write it with
    :meth:`to_json`, read it back with :func:`read_structure`, and build its
    decisions into a model with :meth:`apply`.

    Parameters
    ----------
    features : dict[str, FeatureStructure]
        The decisions for each feature, by feature name.
    superglm_version : str
        The superglm version that exported the structure.

    Raises
    ------
    StructureError
        If an entry is not self-consistent: a group member outside the
        levels, a reference that is not a level or group, an unseen group that
        does not exist, or a malformed field.
    """

    features: dict[str, FeatureStructure]
    superglm_version: str = ""

    def __post_init__(self) -> None:
        for name, entry in self.features.items():
            if not isinstance(name, str) or not isinstance(entry, FeatureStructure):
                raise StructureError(_MALFORMED_FILE.format(field="features"))
            _check_feature(name, entry)

    @classmethod
    def from_model(cls, model, X=None) -> Structure:
        """The structure of ``model``'s categorical, ordered and spline features.

        Parameters
        ----------
        model : SuperGLM
            A fitted model; the structure is the one in force, with each
            reference as the fit resolved it.
        X : DataFrame, optional
            The data the model was fit on. It is read only to give a grouped
            term's levels their native types: a grouping matches levels as
            text, so without ``X`` or a declared ``levels=`` the fitted model
            knows them only as text.

        Returns
        -------
        Structure
            One entry per categorical, ordered and spline feature. Other
            features carry no structural decisions and are left out.

        Raises
        ------
        StructureError
            If the model is not a fitted SuperGLM, or a level cannot be written
            as JSON.
        """
        import superglm

        _require_superglm(model, "from_model")
        frame = None
        if X is not None:
            from superglm._frame import as_eager_frame

            frame = as_eager_frame(X)
        features = {}
        for name, spec in getattr(model, "_specs", {}).items():
            kind = _kind(spec)
            if kind is None:
                continue
            if not isinstance(name, str):
                raise StructureError(_NOT_WRITABLE.format(value=name, feature=name))
            if kind == "spline":
                features[name] = FeatureStructure(kind=kind, ranges=_spec_ranges(spec))
            else:
                features[name] = _level_structure(name, spec, kind, frame)
        if getattr(model, "_result", None) is None:
            # A model with no categorical or ordered term to name says so here.
            raise StructureError(_MODEL_UNFITTED)
        return cls(features=features, superglm_version=str(superglm.__version__))

    def to_json(self, path=None) -> str:
        """The structure as JSON text, written to ``path`` too when one is given.

        Keys are sorted, the indent is two spaces and the text ends in one
        newline, so two exports of one structure are byte-identical. The file
        is written as UTF-8 bytes, the same on every platform.

        Parameters
        ----------
        path : str or path-like, optional
            Where to write the file.

        Returns
        -------
        str
            The JSON text.
        """
        payload = {
            "features": {name: _entry_json(name, entry) for name, entry in self.features.items()},
            "format": FORMAT,
            "superglm_version": self.superglm_version,
        }
        text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        text += "\n"
        if path is not None:
            Path(path).write_bytes(text.encode("utf-8"))
        return text

    @classmethod
    def from_json(cls, source) -> Structure:
        """Read a structure from JSON text, or from the mapping it parses to.

        Parameters
        ----------
        source : str, bytes or Mapping
            The text :meth:`to_json` returns, or its parsed form. To read a
            file, use :func:`read_structure`.

        Returns
        -------
        Structure

        Raises
        ------
        StructureError
            If the text is not JSON, the format is not
            ``"superglm.structure.v1"``, or an entry is malformed or not
            self-consistent.
        """
        if isinstance(source, Mapping):
            payload = source
        else:
            try:
                payload = json.loads(source, parse_constant=_refuse_constant)
            except (TypeError, ValueError) as exc:
                raise StructureError(_NOT_JSON) from exc
        fmt = payload.get("format") if isinstance(payload, Mapping) else None
        if fmt != FORMAT:
            raise StructureError(_UNKNOWN_FORMAT.format(fmt=fmt, expected=FORMAT))
        for key in payload:
            if key not in _FILE_FIELDS:
                raise StructureError(_MALFORMED_FILE.format(field=key))
        features = payload.get("features")
        if not isinstance(features, Mapping):
            raise StructureError(_MALFORMED_FILE.format(field="features"))
        version = payload.get("superglm_version", "")
        if not isinstance(version, str):
            raise StructureError(_MALFORMED_FILE.format(field="superglm_version"))
        entries = {name: _entry_from_json(name, entry) for name, entry in features.items()}
        return cls(features=entries, superglm_version=version)

    def apply(self, model, X=None):
        """An unfitted copy of ``model`` with these decisions built into its features.

        Nothing is fitted. Each feature the structure names is rebuilt from
        the model's declared spec with the structure's grouping, reference,
        unseen policy and polynomial ranges, by the builders the editor uses;
        every other feature is copied as it is. Groupings are built from the
        structure's level universe, so no data is needed, and a grouped term
        is declared with that universe as ``levels=`` declares one: a group
        or reference whose levels have no rows in the fit is pinned to the
        base, with the library's warning, rather than refused. Ranges on a ``ps``
        or ``ns`` spline rebuild it as a ``bs`` spline with the same knots,
        degree and penalty order, as the editor does. The copy's penalties
        are those ``model`` was declared with: applied to a fitted model, a
        ``selection_penalty="auto"`` is calibrated again at the next fit, and
        smoothing that ``fit_reml`` estimated starts again from the declared
        ``spline_penalty``, as they would on a fresh declaration.

        Parameters
        ----------
        model : SuperGLM
            The model to build into, fitted or not. It is left unchanged.
        X : DataFrame, optional
            The data the model will be fit on. It is read for one thing only:
            levels of a grouped term that the structure does not list. Those
            go where the structure sends new levels, into its ``unseen`` group,
            or each into a level of its own when that is ``"error"`` or
            ``"base"``, with one warning per feature; a model that declares
            its levels with ``levels=`` refuses them. Without ``X`` a grouped
            term covers only the structure's levels, and a fit on data holding
            others refuses them. ``X`` also places the ranges the structure
            gives a spline: a spline whose boundary the model leaves to the
            data is fitted out to hold a range that reaches past ``X``'s
            values, with one warning, and a range the spline refuses on ``X``
            is refused here. Without ``X`` the fit checks the ranges.

        Returns
        -------
        SuperGLM
            An unfitted model, ready to ``fit``.

        Raises
        ------
        StructureError
            If ``model`` is not a SuperGLM, a feature is not in the model or is
            another kind of term, its levels are not those the model declares,
            or its spline refuses a range. Any other error while a feature is
            rebuilt is reported as that feature's refusal, with the error as
            its cause.
        """
        _require_superglm(model, "apply")
        frame = None
        if X is not None:
            from superglm._frame import as_eager_frame

            frame = as_eager_frame(X)
        specs = getattr(model, "_specs", None) or {}
        declared = dict(getattr(getattr(model, "_config", None), "feature_templates", ()))
        replacements = {}
        for name in sorted(self.features):
            entry = self.features[name]
            _check_feature(name, entry)
            if name not in specs:
                raise StructureError(_ABSENT.format(feature=name))
            spec = declared.get(name, specs[name])
            if _kind(spec) != entry.kind:
                raise StructureError(_KIND.format(feature=name, kind=entry.kind))
            try:
                replacements[name] = _rebuilt(model, name, spec, entry, frame)
            except StructureError:
                raise
            except Exception as exc:
                raise StructureError(_NOT_APPLIED.format(feature=name)) from exc
        # The penalties as declared: a calibrated selection_penalty="auto" or
        # smoothing a REML fit estimated belongs to that fit, like its
        # coefficients, so a fitted model and its declaration give one copy.
        return clone_with_replaced_features(
            model,
            replacements,
            lambda1=configured_penalty(model).lambda1,
            lambda2=configured_lambda2(model),
        )


def read_structure(path_or_mapping) -> Structure:
    """Read a structure file.

    Parameters
    ----------
    path_or_mapping : str, path-like or Mapping
        The path of a file :meth:`Structure.to_json` wrote, or its parsed JSON.

    Returns
    -------
    Structure

    Raises
    ------
    StructureError
        If the file is not a valid structure file; see
        :meth:`Structure.from_json`.
    """
    if isinstance(path_or_mapping, Mapping):
        return Structure.from_json(path_or_mapping)
    return Structure.from_json(Path(path_or_mapping).read_bytes())


# -- Reading a model -----------------------------------------------------------


def _require_superglm(model, method: str) -> None:
    """Refuse anything but a SuperGLM: a SuperLSS, or no model, has no structure here."""
    if not isinstance(model, SuperGLM):
        raise StructureError(_NOT_A_SUPERGLM.format(method=method, kind=type(model).__name__))


def _kind(spec) -> str | None:
    """The structure kind of a feature spec, or None for a kind it does not record."""
    if isinstance(spec, OrderedCategorical):
        return "ordered"
    if isinstance(spec, Categorical):
        return "categorical"
    if isinstance(spec, _SplineBase):
        return "spline"
    return None


def _spec_ranges(spec) -> list[PolynomialRange]:
    """The polynomial ranges in force on ``spec``, in axis order."""
    return list(current_ranges(spec))


def _level_structure(name: str, spec, kind: str, frame) -> FeatureStructure:
    """The levels, groups, reference and unseen policy of a fitted categorical or ordered term."""
    if spec._base_level == "" or spec._base_level is None:
        raise StructureError(_UNFITTED.format(feature=name))
    grouping = spec._grouping
    if kind == "ordered":
        # The declared levels, specials last: an ordered term's universe is its
        # declaration, grouped or not.
        universe = list(spec._declared_smooth_levels) + list(spec._special_display)
        unseen = "error"
    else:
        universe = list(spec._levels)
        unseen = spec.unseen
    # A grouped categorical's fitted levels are group labels, which say nothing
    # about its raw levels' types.
    typed = [] if grouping is not None and kind == "categorical" else universe
    native = _native_levels(name, typed, spec, frame)
    groups: dict[str, list] = {}
    if grouping is not None:
        if kind == "categorical":
            # The fitted levels are group labels; the universe is the raw one,
            # in model order: each raw level where its group sits in the fit.
            at = {str(level): i for i, level in enumerate(spec._levels)}
            to_group = grouping.original_to_group
            raws = sorted(
                grouping.all_original_levels, key=lambda raw: at.get(str(to_group[raw]), len(at))
            )
            universe = [native.get(str(level), level) for level in raws]
        for label in grouping.grouped_levels:
            members = [str(member) for member in grouping.group_to_originals[label]]
            if members == [str(label)]:
                continue
            if not isinstance(label, str):
                raise StructureError(_NOT_WRITABLE.format(value=label, feature=name))
            groups[label] = [native.get(member, member) for member in members]
    reference = spec._base_level
    if str(reference) not in groups:
        reference = native.get(str(reference), reference)
    ranges = _spec_ranges(spec) if kind == "ordered" else []
    _require_writable_levels(name, universe)
    return FeatureStructure(
        kind=kind,
        levels=universe,
        groups=groups,
        reference=reference,
        unseen=unseen,
        ranges=ranges,
    )


def _require_writable_levels(name: str, levels: list) -> None:
    """Refuse, by name, levels a file cannot hold: not plain scalars, or two with one text.

    Checked before the entry is built, whose own check could only call them
    malformed and ask for the export that is failing.
    """
    seen: dict[str, Any] = {}
    for level in levels:
        _plain(level, name)
        first = seen.setdefault(str(level), level)
        if first is not level and first != level:
            raise StructureError(_SAME_TEXT.format(first=first, second=level, feature=name))


def _native_levels(name: str, universe: list, spec, frame) -> dict[str, Any]:
    """Each level's native value by its text: the term's own levels, declared, then X's column."""
    values = list(universe) + list(getattr(spec, "_declared_levels", None) or [])
    if frame is not None and name in frame.columns:
        import pandas as pd

        values.extend(pd.unique(np.asarray(frame.column_array(name), dtype=object)).tolist())
    native: dict[str, Any] = {}
    for value in values:
        native.setdefault(str(value), value)
    return native


# -- Applying to a model ---------------------------------------------------------


def _rebuilt(model, name: str, spec, entry: FeatureStructure, frame):
    """A fresh, unfitted spec for ``name``: ``spec`` with ``entry``'s decisions."""
    column = frame.column_array(name) if frame is not None and name in frame.columns else None
    if entry.kind == "spline":
        return _rebuilt_spline(model, name, spec, entry, column)
    if entry.kind == "ordered":
        return _rebuilt_ordered(model, name, spec, entry, column)
    return _rebuilt_categorical_term(name, spec, entry, column)


def _rebuilt_categorical_term(name: str, spec, entry: FeatureStructure, column):
    levels = list(entry.levels)
    groups = {label: list(members) for label, members in entry.groups.items()}
    declared = accepted_levels(spec)
    # A grouping covers its levels at fit, so under one the levels the model
    # declares must be the structure's exactly: a level the grouping missed
    # would be refused, and a member the declaration leaves out admitted.
    if groups and declared is not None and _texts(declared) != _texts(levels):
        raise StructureError(_UNIVERSE.format(feature=name))
    if groups and column is not None:
        levels, groups = _placed_new_levels(name, entry, levels, groups, column, declared)
    grouping = _grouping(levels, groups, order=[str(level) for level in levels])
    # A grouped term is declared with the structure's universe, as levels=
    # declares one: a group or reference whose levels have no rows in the
    # next fit is then pinned to the base with the library's warning, not
    # dropped from the universe. The file lists the levels in model order, so
    # the fitted levels, and the design's columns, come out in the order of
    # the model the structure was exported from.
    universe = list(levels) if grouping is not None and declared is None else None
    # Grouped, the design speaks the grouping's text; ungrouped, the builder
    # gives the reference its native type from the levels.
    base = entry.reference if grouping is None else str(entry.reference)
    return rebuilt_categorical(
        spec,
        spec,
        base=base,
        grouping=grouping,
        data=np.asarray(levels, dtype=object),
        unseen=entry.unseen,
        levels=universe,
    )


def _placed_new_levels(
    name: str, entry: FeatureStructure, levels: list, groups: dict, column, declared
):
    """``levels`` and ``groups`` with the column's unlisted levels placed where new levels go.

    Into the ``unseen`` group when it names one, else each as a level of its
    own; one warning names them and their rows. Missing values are left to
    the fit, which refuses them. A model that ``declared`` its levels refuses
    unlisted ones: placing them would widen its declaration.
    """
    import pandas as pd

    values = np.asarray(column, dtype=object).ravel()
    values = values[~np.asarray(pd.isna(values), dtype=bool)]
    known = {str(level) for level in levels}
    unlisted = ~pd.Series(values, dtype=object).astype(str).isin(known).to_numpy()
    if not unlisted.any():
        return levels, groups
    new = list({str(value): value for value in values[unlisted]}.values())
    new.sort(key=str)
    if declared is not None:
        raise StructureError(_OUTSIDE_DECLARED.format(feature=name, levels=new))
    rows = int(unlisted.sum())
    if entry.unseen in _POLICIES:
        destination = "are fitted as levels of their own"
    else:
        destination = f"go to the group {entry.unseen!r}"
        members = groups.get(entry.unseen)
        if members is None:
            # The policy names an ungrouped level, which becomes a group.
            members = [level for level in levels if str(level) == entry.unseen]
        groups[entry.unseen] = members + new
    warnings.warn(
        f"Levels of {name!r} the structure does not list {destination} "
        f"(unseen={entry.unseen!r}): {new} over {rows} row(s).",
        UserWarning,
        stacklevel=4,
    )
    return levels + new, groups


def _grouping(levels: list, groups: dict, *, order: list[str]):
    """The LevelGrouping ``groups`` makes of ``levels``, or None without groups."""
    if not groups:
        return None
    from superglm.features.grouping import collapse_levels

    return collapse_levels(
        [str(level) for level in levels],
        groups={label: [str(member) for member in members] for label, members in groups.items()},
        order=order,
    )


def _rebuilt_ordered(model, name: str, spec, entry: FeatureStructure, column):
    declared = [str(level) for level in (*spec._declared_smooth_levels, *spec._special_display)]
    if sorted(declared) != sorted(str(level) for level in entry.levels):
        raise StructureError(_UNIVERSE.format(feature=name))
    grouping = _grouping(entry.levels, entry.groups, order=declared)
    base = entry.reference if grouping is None else str(entry.reference)
    data = np.asarray(entry.levels, dtype=object)
    if _same_ranges(entry.ranges, _spec_ranges(spec)):
        return rebuilt_ordered_spec(spec, grouping=grouping, base=base, data=data)
    _require_shapes(model, name, entry)
    source = pristine_basis(spec)
    knots = source._named_knots or source._explicit_knots
    boundary = source._explicit_boundary

    def hosted(ranges, bound=boundary):
        basis = shaped_spline(source, ranges, knots=knots, boundary=bound)
        return rebuilt_ordered_spec(spec, grouping=grouping, base=base, data=data, basis=basis)

    # The term without ranges places each band on the axis the ranges name.
    host = hosted([])
    ranges: list[PolynomialRange] = []
    for r in entry.ranges:
        try:
            bands = isinstance(r.lo, str) and isinstance(r.hi, str)
            lo, hi = band_edges(host, name, r.lo, r.hi) if bands else (r.lo, r.hi)
            new = PolynomialRange(lo, hi, r.degree, r.join)
            _require_shape_fits(source, new)
            ranges = merged_ranges(tuple(ranges), new, host._range_edge_value)
        except ValueError as exc:
            raise StructureError(_range_refusal(name, r.lo, r.hi, r.degree)) from exc
    if column is None or not ranges:
        return hosted(ranges)

    def fits(subset, bound):
        hosted(subset, bound).build(column)

    def extent():
        # Where the data's bands sit on the axis the spline is fitted over.
        probe = hosted([])
        probe.build(column)
        return probe._basis_spline.fitted_boundary

    position = host._range_edge_value
    in_order = sorted(ranges, key=lambda r: position(r.lo))
    return hosted(ranges, _placed_boundary(name, in_order, fits, boundary, extent, position))


def _rebuilt_spline(model, name: str, spec, entry: FeatureStructure, column):
    from superglm.dm_builder import resolve_discrete_n_bins, should_discretize
    from superglm.features._spline_ranges import validate_ranges

    if _same_ranges(entry.ranges, _spec_ranges(spec)):
        return spec
    _require_shapes(model, name, entry)
    ranges: list[PolynomialRange] = []
    for r in entry.ranges:
        try:
            new = PolynomialRange(float(r.lo), float(r.hi), r.degree, r.join)
            _require_shape_fits(spec, new)
            if spec._explicit_boundary is not None:
                validate_ranges([new], spec.degree, *spec._explicit_boundary)
            ranges = merged_ranges(tuple(ranges), new, float)
        except ValueError as exc:
            raise StructureError(_range_refusal(name, r.lo, r.hi, r.degree)) from exc
    ranges.sort(key=lambda r: r.lo)

    def shaped(subset, bound):
        # The declared knots, not a fit's: the model is fit afresh.
        return shaped_spline(spec, subset, knots=spec._explicit_knots, boundary=bound)

    boundary = spec._explicit_boundary
    if column is None or not ranges:
        return shaped(ranges, boundary)
    try:
        x = np.asarray(column, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return shaped(ranges, boundary)  # not a numeric column: the fit says so
    x = x[np.isfinite(x)]
    if not x.size:
        return shaped(ranges, boundary)

    def fits(subset, bound):
        # The fit's own range checks: knot placement on the values it sees.
        probe = shaped(subset, bound)
        binned = should_discretize(probe, model._discrete)
        probe._place_knots(
            x, None, resolve_discrete_n_bins(name, probe, model._n_bins) if binned else None
        )

    def extent():
        return float(x.min()), float(x.max())

    return shaped(ranges, _placed_boundary(name, ranges, fits, boundary, extent, float))


def _placed_boundary(name: str, ranges: list, fits, boundary, extent, position):
    """The boundary that holds ``ranges`` on the data, or the structure's refusal.

    ``fits(subset, boundary)`` builds the term with ``subset`` of ``ranges``
    on the data and raises the library's ``RangeError`` when the spline
    refuses it. When the model leaves the spline's boundary to the data
    (``boundary`` is None) and a range reaches past the data's ``extent()``,
    the spline is fitted out to the ranges' ends, with one warning: a range
    drawn to the end of one year's data then holds as written on the next
    year's narrower data. ``position`` places a range edge on the axis.
    """
    refusal = _refusal(name, ranges, fits, boundary)
    if refusal is None:
        return boundary
    if boundary is None:
        lo, hi = _quiet(extent)
        edges = [position(edge) for r in ranges for edge in (r.lo, r.hi)]
        wider = (min(lo, *edges), max(hi, *edges))
        if wider != (lo, hi):
            refusal = _refusal(name, ranges, fits, wider)
            if refusal is None:
                past = [r for r in ranges if position(r.lo) < lo or position(r.hi) > hi]
                spans = " and the ".join(_range_text(r.lo, r.hi, r.degree) for r in past)
                warnings.warn(
                    _FITTED_OUT.format(feature=name, spans=spans), UserWarning, stacklevel=5
                )
                return wider
    raise refusal


def _refusal(name: str, ranges: list, fits, boundary) -> StructureError | None:
    """The range sentence for the first range the spline refuses on the data, or None.

    The ranges are tried whole and, if refused, in growing prefixes in axis
    order, so the range named is the one the spline refuses alone or beside
    those before it. Data the term refuses for another reason is left to the
    fit, which reports it in its own words.
    """

    def refused(subset) -> RangeError | None:
        try:
            _quiet(lambda: fits(subset, boundary))
        except RangeError as exc:
            return exc
        except Exception:
            return None
        return None

    if refused(ranges) is None:
        return None
    for count in range(1, len(ranges) + 1):
        cause = refused(ranges[:count])
        if cause is not None:
            r = ranges[count - 1]
            refusal = StructureError(_range_refusal(name, r.lo, r.hi, r.degree))
            refusal.__cause__ = cause
            return refusal
    return None


def _quiet(probe):
    """``probe()`` with its warnings silenced: the fit gives them, once."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return probe()


def _same_ranges(wanted, current) -> bool:
    def key(r):
        return (r.lo, r.hi, int(r.degree), r.join)

    return {key(r) for r in wanted} == {key(r) for r in current}


def _require_shapes(model, name: str, entry: FeatureStructure) -> None:
    """Refuse the first range when the term's spline takes no ranges at all."""
    if entry.ranges and shape_unavailable_reason(model, name) is not None:
        first = entry.ranges[0]
        raise StructureError(_range_refusal(name, first.lo, first.hi, first.degree))


def _require_shape_fits(source, r: PolynomialRange) -> None:
    """Refuse a range of higher degree than the spline, or a tangent join on a linear one."""
    if int(r.degree) > source.degree or (r.join == "tangent" and source.degree < 2):
        raise ValueError(f"the spline of degree {source.degree} cannot take {r}")


# -- JSON ------------------------------------------------------------------------


def _refuse_constant(constant: str):
    """Refuse JSON's non-standard NaN and Infinity, which no structure holds."""
    raise ValueError(f"non-finite constant {constant}")


def _plain(value, feature: str):
    """``value`` as a JSON scalar of the same kind: a numpy scalar becomes its Python one."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise StructureError(_NOT_WRITABLE.format(value=value, feature=feature))


def _entry_json(name: str, entry: FeatureStructure) -> dict[str, Any]:
    ranges = [
        {
            "degree": int(r.degree),
            "hi": _plain(r.hi, name),
            "join": r.join,
            "lo": _plain(r.lo, name),
        }
        for r in entry.ranges
    ]
    if entry.kind == "spline":
        return {"kind": "spline", "ranges": ranges}
    payload = {
        "groups": {
            label: [_plain(member, name) for member in members]
            for label, members in entry.groups.items()
        },
        "kind": entry.kind,
        "levels": [_plain(level, name) for level in entry.levels],
        "reference": _plain(entry.reference, name),
        "unseen": entry.unseen,
    }
    if entry.kind == "ordered":
        payload["ranges"] = ranges
    return payload


def _entry_from_json(name: str, entry) -> FeatureStructure:
    """One feature's parsed JSON as a FeatureStructure; its consistency is checked after."""
    if not isinstance(entry, Mapping):
        raise StructureError(_MALFORMED.format(feature=name, field="entry"))
    kind = entry.get("kind")
    if kind not in KINDS:
        raise StructureError(_MALFORMED.format(feature=name, field="kind"))
    for key in entry:
        if key != "kind" and key not in _FIELDS[kind]:
            raise StructureError(_MALFORMED.format(feature=name, field=key))
    ranges = entry.get("ranges", [])
    if not isinstance(ranges, list) or not all(_is_range_json(r) for r in ranges):
        raise StructureError(_MALFORMED.format(feature=name, field="ranges"))
    parsed = [_range_from_json(name, r) for r in ranges]
    if kind == "spline":
        return FeatureStructure(kind=kind, ranges=parsed)
    groups = entry.get("groups", {})
    if not isinstance(groups, Mapping) or not all(isinstance(m, list) for m in groups.values()):
        raise StructureError(_MALFORMED.format(feature=name, field="groups"))
    levels = entry.get("levels")
    return FeatureStructure(
        kind=kind,
        levels=list(levels) if isinstance(levels, list) else levels,
        groups={label: list(members) for label, members in groups.items()},
        reference=entry.get("reference"),
        unseen=entry.get("unseen", "error"),
        ranges=parsed,
    )


def _is_range_json(value) -> bool:
    return isinstance(value, Mapping) and sorted(value) == list(_RANGE_FIELDS)


def _range_from_json(name: str, value: Mapping) -> PolynomialRange:
    try:
        return PolynomialRange(value["lo"], value["hi"], value["degree"], value["join"])
    except ValueError as exc:
        raise StructureError(
            _range_refusal(name, value["lo"], value["hi"], value["degree"])
        ) from exc


# -- Consistency -------------------------------------------------------------------


def _check_feature(name: str, entry: FeatureStructure) -> None:
    """Refuse an entry that is not self-consistent, in its fixed sentence."""
    if entry.kind not in KINDS:
        raise StructureError(_MALFORMED.format(feature=name, field="kind"))
    if not isinstance(entry.ranges, list) or not all(
        isinstance(r, PolynomialRange) for r in entry.ranges
    ):
        raise StructureError(_MALFORMED.format(feature=name, field="ranges"))
    if entry.kind == "spline":
        if entry.levels or entry.groups or entry.reference is not None or entry.unseen != "error":
            raise StructureError(_MALFORMED.format(feature=name, field="levels"))
        for r in entry.ranges:
            _check_numeric_range(name, r)
        return
    if entry.kind == "categorical" and entry.ranges:
        raise StructureError(_MALFORMED.format(feature=name, field="ranges"))
    texts = _check_levels(name, entry.levels)
    names = _check_groups(name, entry.groups, texts)
    if not _is_scalar(entry.reference):
        raise StructureError(_MALFORMED.format(feature=name, field="reference"))
    if str(entry.reference) not in names:
        raise StructureError(_REFERENCE.format(reference=entry.reference, feature=name))
    if not isinstance(entry.unseen, str):
        raise StructureError(_MALFORMED.format(feature=name, field="unseen"))
    if entry.kind == "ordered":
        if entry.unseen != "error":
            raise StructureError(_ORDERED_UNSEEN.format(feature=name))
        for r in entry.ranges:
            if not all(isinstance(edge, str) or _is_finite(edge) for edge in (r.lo, r.hi)):
                raise StructureError(_range_refusal(name, r.lo, r.hi, r.degree))
    elif entry.unseen not in _POLICIES and not (entry.groups and entry.unseen in names):
        raise StructureError(_UNSEEN.format(unseen=entry.unseen, feature=name))


def _check_levels(name: str, levels) -> list[str]:
    """The levels' texts; refused unless they are distinct plain scalars."""
    if not isinstance(levels, list) or not levels or not all(_is_scalar(v) for v in levels):
        raise StructureError(_MALFORMED.format(feature=name, field="levels"))
    texts = [str(level) for level in levels]
    if len(set(texts)) != len(texts):
        raise StructureError(_MALFORMED.format(feature=name, field="levels"))
    return texts


def _check_groups(name: str, groups, texts: list[str]) -> set[str]:
    """The names a reference or unseen policy may use: group labels and ungrouped levels."""
    if not isinstance(groups, Mapping):
        raise StructureError(_MALFORMED.format(feature=name, field="groups"))
    known = set(texts)
    owner: dict[str, str] = {}
    for label, members in groups.items():
        if not isinstance(label, str) or not isinstance(members, list) or not members:
            raise StructureError(_MALFORMED.format(feature=name, field="groups"))
        for member in members:
            if not _is_scalar(member) or str(member) not in known:
                raise StructureError(_MEMBER.format(group=label, feature=name, member=member))
            if str(member) in owner:
                raise StructureError(_TWO_GROUPS.format(member=member, feature=name))
            owner[str(member)] = label
    for label, members in groups.items():
        if label in known and label not in {str(member) for member in members}:
            raise StructureError(_GROUP_NAME.format(group=label, feature=name))
    return set(groups) | {text for text in texts if text not in owner}


def _texts(levels) -> set[str]:
    """Levels as the text a grouping matches them by."""
    return {str(level) for level in levels}


def _check_numeric_range(name: str, r: PolynomialRange) -> None:
    if not (_is_finite(r.lo) and _is_finite(r.hi) and float(r.lo) < float(r.hi)):
        raise StructureError(_range_refusal(name, r.lo, r.hi, r.degree))


def _range_refusal(name: str, lo, hi, degree) -> str:
    """The range sentence: ``The spline of 'age' refuses the Line range 30–45; ...``."""
    return _RANGE.format(feature=name, span=_range_text(lo, hi, degree))


def _range_text(lo, hi, degree) -> str:
    """A range as the sentences name it: ``Line range 30–45``."""
    span = f"{_edge_text(lo)}–{_edge_text(hi)}"
    named = isinstance(degree, Integral) and not isinstance(degree, bool) and 0 <= degree <= 3
    shape = f"{SHAPE_NAMES[int(degree)]} range" if named else "range"
    return f"{shape} {span}"


def _edge_text(edge) -> str:
    if _is_finite(edge):
        return f"{edge:g}"
    if isinstance(edge, Integral) and not isinstance(edge, bool):
        # An integer past float64's range: its digits could run to thousands.
        return "inf" if edge > 0 else "-inf"
    return str(edge)


def _is_scalar(value) -> bool:
    return isinstance(value, str | bool | np.bool_) or _is_finite(value)


def _is_finite(value) -> bool:
    """Whether ``value`` is a number float64 holds as a finite value.

    JSON reads an integer of any length, and one past float64's range is no
    level or edge: ``float`` would raise ``OverflowError`` on it.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
