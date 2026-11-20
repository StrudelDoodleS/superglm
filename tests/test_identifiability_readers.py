"""Every reader of a published fit follows its identifiability record: one table, one walk.

A reader turns the published coefficients or covariance into something a user reads.  The
record (``solvers.identifiability``) says which coefficients' values are estimates (not an
alias's coordinate, not one separation moves) and which have a standard error.  Readers take
one of two rules (``READERS``; the stage's reader table names each one's source):

- **a table** (the summary's and ``metrics()``'s rows and their renderers, the summary's SE
  dict, ``diagnostics()``, ``term_importance``, ``spline_redundancy``, the basis plot, an
  edited model's table): a value that reads a coefficient that is not an estimate is NaN or
  ``---``, an SE, interval or test that reads one the fit does not identify is withheld, and a
  joint test reads only identified coefficients;
- **a term functional** (``relativities``, ``term_inference``, ``feature_se``,
  ``simultaneous_bands``, the rating tables, ``predict``): the value is the published fit's,
  the point is flagged with its kind and its SE is NaN (``diagnostics.estimability``).

Each fixture fixes by construction which numbers lean on what the fit does not determine.  The
separated fit (Poisson): the ordered levels above a break, the numeric knot range above 8, a
reference level and an interaction level carry only zero responses.  The aliased fit (Gamma):
three spline basis functions without data or penalty, ``x2 = 2 x1``, and two ordered terms
whose level index a numeric duplicates.  At ``20905fe16`` both walks fail: the separated fit
printed the walk's stopping point on the levels above the break (-24) and an intercept carrying
a separated reference level, the aliased fit a Wood test on the spline it does not identify
(chi2 10.3) and level values reading held coefficients; the summary's SE dict printed 0.0,
``diagnostics()`` and ``term_importance`` the walk's norms (17708), and an edited model's table
the walk's values again.  The third case forces the Wald solve that a held row turns into a
``LinAlgError`` to return ``b' V^+ b``, as a route whose held rows are not exactly zero does:
the joint test must be withheld because the record says so, not because the solve failed.
The vanished-weight fit (an sz factor smooth whose one-row level ``g000`` sits at offset
``-160``, on both backends) holds two directions not resolved at float64; at ``b175d49f2``
``factor_smooth()`` printed ``g000``'s coefficient norm where they left it (11.80 on auto,
2209.50 on gram).  A ``"partial"`` decision, a proven lower bound, is walked as a control.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import (
    Categorical,
    Numeric,
    OrderedCategorical,
    Piecewise,
    Polynomial,
    Spline,
    SuperGLM,
)
from superglm.diagnostics.estimability import NonEstimablePredictionWarning

LEVELS = [f"Mi{i:03d}" for i in range(10)]
GRADES = [f"G{i}" for i in range(6)]


@dataclasses.dataclass(frozen=True)
class _Case:
    """A fitted fixture and what its construction leaves undetermined.

    ``blank`` rows show no value (nor SE); ``shown`` rows show both.  ``untested`` whole-term
    rows show no test, ``tested`` ones do.  ``unidentified`` are each group's coordinates
    without an SE (the record's mask, checked against the construction first).  ``unnormed``
    groups have no norm, ``unimportant`` ones no importance (their share on the rows reads a
    value that is not an estimate; ``important`` ones keep it).  ``off_support`` maps a term to the points of its
    functional readers that lean on what the fit does not determine; ``rows`` are the rows
    ``predict`` names.  ``basis`` names a spline and its basis functions without a value;
    ``edit`` a term an editor edit shifts.  ``smooth_levels`` maps a factor smooth to the levels
    whose ``factor_smooth()`` coefficient norm has no value, and ``skip`` names the readers the
    case does not walk.
    """

    model: SuperGLM
    X: pd.DataFrame
    y: np.ndarray
    blank: tuple[str, ...]
    shown: tuple[str, ...]
    untested: tuple[str, ...]
    tested: tuple[str, ...]
    unidentified: dict[str, list[int]]
    unnormed: tuple[str, ...]
    normed: tuple[str, ...]
    unimportant: tuple[str, ...]
    important: tuple[str, ...]
    off_support: dict[str, Callable[[pd.DataFrame], np.ndarray]]
    rows: np.ndarray
    basis: tuple[str, tuple[int, ...]] | None
    edit: tuple[str, float, float]
    smooth_levels: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    skip: tuple[str, ...] = ()


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def _separated() -> _Case:
    rng = np.random.default_rng(42)
    n = 4000
    X = pd.DataFrame(
        {
            "band": rng.choice(LEVELS, n),
            "x": rng.uniform(0, 10, n),
            "s": rng.uniform(0, 1, n),
            "c": rng.choice(list("ABC"), n),
            "grade": rng.choice(GRADES, n),
            "u": rng.uniform(0.5, 2.0, n),
            "d": rng.choice(["D0", "D1", "D2"], n),
        }
    )
    position = np.array([LEVELS.index(band) for band in X["band"]])
    y = rng.poisson(np.exp(-1.0 + 0.05 * position + 0.2 * np.sin(6 * X["s"]))).astype(float)
    zero = (
        (position >= 8)
        | (X["x"].to_numpy() > 8.0)
        | X["grade"].isin(["G4", "G5"]).to_numpy()
        | (X["d"] == "D2").to_numpy()
    )
    y[zero] = 0.0
    features = {
        "band": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi007"], degrees=[1, 2])),
        "x": Piecewise(breaks=[4.0, 8.0]),
        "s": Spline(n_knots=6),
        "c": Categorical(),
        # the reference level is separated, so every other level and the intercept read it
        "grade": OrderedCategorical(
            order=GRADES, base="G5", basis=Piecewise(breaks=["G3"], degrees=[1, 1])
        ),
        "u": Polynomial(degree=2),
        "d": Categorical(),
    }
    model = SuperGLM(
        family="poisson", selection_penalty=0.0, features=features, interactions=[("u", "d")]
    )
    _quiet(model.fit, X, y)
    top = f"x[{float(X['x'].max()):.10g}]"
    return _Case(
        model=model,
        X=X,
        y=y,
        blank=(
            "Intercept",
            "band[Mi008]",
            "band[Mi009]",
            "band[slope-change @ Mi007]",
            "band[curvature Mi007..Mi009]",
            top,
            *(f"grade[{grade}]" for grade in GRADES[:5]),
            "grade[slope-change @ G3]",
            "d[D2]",
        ),
        shown=("band[Mi000]", "band[Mi007]", "x[8]", "c[A]", "d[D1]", "u[P1]"),
        untested=("band", "x", "grade", "u:d[D2]"),
        tested=("s", "u:d[D1]"),
        unidentified={"band": [1, 2], "x": [2], "grade": [1], "d": [1], "u:d[D2]": [0, 1]},
        unnormed=("band", "x", "grade", "d", "u:d[D2]"),
        normed=("s", "c", "u", "u:d[D1]"),
        unimportant=("band", "x", "grade", "d", "u:d[D2]"),
        important=("s", "c", "u", "u:d[D1]"),
        off_support={
            "band": lambda frame: frame["level"].isin(["Mi008", "Mi009"]).to_numpy(),
            "x": lambda frame: frame["x"].to_numpy() > 8.0,
            "grade": lambda frame: (frame["level"] != "G5").to_numpy(),
            "d": lambda frame: (frame["level"] == "D2").to_numpy(),
            "c": lambda frame: np.zeros(len(frame), dtype=bool),
            "s": lambda frame: np.zeros(len(frame), dtype=bool),
        },
        rows=np.flatnonzero(zero),
        basis=None,
        edit=("s", 0.2, 0.6),
    )


def _aliased() -> _Case:
    rng = np.random.default_rng(7)
    n = 2000
    spread = rng.uniform(0, 10, n)
    x1 = rng.normal(size=n)
    band, top = rng.choice(LEVELS, n), rng.choice(LEVELS, n)
    X = pd.DataFrame(
        {
            # no data in [4, 7): three basis functions with neither data nor penalty
            "g": np.where((spread > 3) & (spread < 7), spread - 3.0, spread),
            "x1": x1,
            "x2": 2.0 * x1,
            "pos": np.array([LEVELS.index(level) for level in band], dtype=float),
            "band": band,
            "pos2": np.array([LEVELS.index(level) for level in top], dtype=float),
            "top": top,
        }
    )
    y = rng.gamma(2.0, np.exp(0.1 * np.sin(spread) + 0.02 * X["pos"].to_numpy()) / 2.0)
    features = {
        "g": Spline(knots=np.linspace(0.5, 9.5, 19), penalty="none"),
        "x1": Numeric(),
        "x2": Numeric(),
        # each ordered term's level index duplicates the numeric before it
        "pos": Numeric(),
        "band": OrderedCategorical(order=LEVELS, basis=Polynomial(degree=2)),
        "pos2": Numeric(),
        "top": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"], degrees=[1, 1])),
    }
    model = SuperGLM(
        family=superglm.families.gamma(), link="log", features=features, spline_penalty=0.0
    )
    _quiet(model.fit, X, y)

    def levels(term):
        base = str(model._specs[term]._base_level)
        return lambda frame: (frame["level"].astype(str) != base).to_numpy()

    def every(frame):
        return np.ones(len(frame), dtype=bool)

    return _Case(
        model=model,
        X=X,
        y=y,
        blank=(
            "Intercept",
            "x2",
            "band[P1]",
            *(f"band[{level}]" for level in LEVELS if level != model._specs["band"]._base_level),
            "top[slope-change @ Mi004]",
            *(f"top[{level}]" for level in LEVELS if level != model._specs["top"]._base_level),
        ),
        shown=("x1", "pos", "pos2", "band[P2]"),
        untested=("g", "band", "top"),
        tested=(),
        unidentified={"g": [8, 9, 10], "x2": [0], "band": [0], "top": [1]},
        unnormed=("g", "x2", "band", "top"),
        normed=("x1", "pos", "pos2"),
        # no row of the fit's lies in the spline's empty basis functions
        unimportant=("x2", "band", "top"),
        important=("g", "x1", "pos", "pos2"),
        off_support={
            "g": lambda frame: (frame["x"].to_numpy() > 4.0) & (frame["x"].to_numpy() < 7.0),
            "x1": every,
            "x2": every,
            "pos": every,
            "band": levels("band"),
            "top": levels("top"),
        },
        rows=np.zeros(0, dtype=np.intp),
        basis=("g", (12, 13, 14)),
        edit=("g", 1.0, 2.0),
    )


def _vanished(direct_solve: str) -> _Case:
    """The sz fit whose one-row level ``g000`` sits at offset ``-160``: its weight vanished."""
    from tests.test_factor_smooth_sz_thin_and_influence import _model, _signed_aliased_frame

    frame, y, weight = _signed_aliased_frame("one_row")
    row = int(np.flatnonzero(frame["g"].to_numpy() == "g000")[0])
    offset = np.where(np.arange(len(y)) != row, 0.0, -160.0)
    model = _model("gaussian_log", direct_solve, lam=1.29, main_lam=100.0, numerics=("x1", "x10"))
    _quiet(model.fit_reml, frame, y, sample_weight=weight, offset=offset)

    def none(frame):
        return np.zeros(len(frame), dtype=bool)

    return _Case(
        model=model,
        X=frame,
        y=y,
        # the level lines its weight no longer reads are not resolved, and move the intercept
        blank=("Intercept",),
        shown=("x1", "x10", "cat[c00]"),
        untested=(),
        tested=("x",),
        unidentified={},
        # the record counts the held unresolved directions' values as estimates, so the sz
        # group's own norm and importance are not walked
        unnormed=(),
        normed=("x1", "x10", "cat", "x"),
        unimportant=(),
        important=("x1", "x10", "cat", "x"),
        off_support={"x1": none, "x10": none, "cat": none, "x": none},
        rows=np.array([row]),
        basis=("x", ()),
        edit=("x", 0.2, 0.6),
        smooth_levels={"x:g:sz": ("g000",)},
        # the rating tables refuse an sz term; an edited model's record is not held on this route
        skip=("rating tables", "edited model's table"),
    )


def _partial() -> _Case:
    """The separated fixture's zero sets beside a 3-level ordered polynomial whose base is zero."""
    rng = np.random.default_rng(42)
    n = 4000
    X = pd.DataFrame(
        {
            "band": rng.choice(LEVELS, n),
            "x": rng.uniform(0, 10, n),
            "s": rng.uniform(0, 1, n),
            "c": rng.choice(list("ABC"), n),
            "tier": rng.choice(["T0", "T1", "T2"], n, p=[0.45, 0.45, 0.1]),
        }
    )
    position = np.array([LEVELS.index(band) for band in X["band"]])
    y = rng.poisson(np.exp(-1.0 + 0.05 * position + 0.2 * np.sin(6 * X["s"]))).astype(float)
    verified = (position >= 8) | (X["x"].to_numpy() > 8.0)
    y[verified | (X["tier"] == "T2").to_numpy()] = 0.0
    features = {
        "band": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi007"], degrees=[1, 2])),
        "x": Piecewise(breaks=[4.0, 8.0]),
        "s": Spline(n_knots=6),
        "c": Categorical(),
        "tier": OrderedCategorical(order=["T0", "T1", "T2"], base="T2", basis=Polynomial(degree=2)),
    }
    model = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
    _quiet(model.fit, X, y)
    top = f"x[{float(X['x'].max()):.10g}]"

    def none(frame):
        return np.zeros(len(frame), dtype=bool)

    return _Case(
        model=model,
        X=X,
        y=y,
        blank=(
            "band[Mi008]",
            "band[Mi009]",
            "band[slope-change @ Mi007]",
            "band[curvature Mi007..Mi009]",
            top,
        ),
        shown=("band[Mi000]", "band[Mi007]", "x[8]", "c[A]"),
        untested=("band", "x"),
        tested=("s",),
        unidentified={"band": [1, 2], "x": [2]},
        unnormed=("band", "x"),
        normed=("s", "c"),
        unimportant=("band", "x"),
        important=("s", "c"),
        off_support={
            "band": lambda frame: frame["level"].isin(["Mi008", "Mi009"]).to_numpy(),
            "x": lambda frame: frame["x"].to_numpy() > 8.0,
            "c": none,
            "s": none,
        },
        rows=np.flatnonzero(verified),
        basis=None,
        edit=("s", 0.2, 0.6),
    )


def _missing(value) -> bool:
    return value is None or (isinstance(value, float | np.floating) and np.isnan(value))


def _present(value) -> bool:
    return value is not None and bool(np.isfinite(value))


def _rows(case: _Case, rows, reader: str) -> list[str]:
    """A coefficient table's rows against the case: values, SEs, intervals and tests."""
    by_name = {row.name: row for row in rows}
    problems = []
    for name in case.blank:
        row = by_name[name]
        if not all(_missing(v) for v in (row.coef, row.se, row.ci_low, row.ci_high, row.p)):
            problems.append(f"{reader}: {name} shows {row.coef} (SE {row.se}, p {row.p})")
    for name in case.shown:
        row = by_name[name]
        if not (_present(row.coef) and _present(row.se)):
            problems.append(f"{reader}: {name} lost its value {row.coef} or SE {row.se}")
    for name in case.untested:
        row = by_name[name]
        if not all(_missing(v) for v in (row.wald_chi2, row.wald_p, row.ref_df)):
            problems.append(f"{reader}: {name} tests chi2 {row.wald_chi2}, p {row.wald_p}")
    for name in case.tested:
        if not _present(by_name[name].wald_chi2):
            problems.append(f"{reader}: {name} lost its test")
    return problems


def _summary_rows(case):
    return _rows(case, case.model.summary()._coef_rows, "summary()")


def _summary_text(case):
    """The printed rows: ``---`` for each blank value and SE, no ``chi2(`` on a withheld test."""
    lines = str(case.model.summary()).splitlines()
    problems = []
    for name in case.blank:
        (line,) = [line for line in lines if line.startswith(f"║ {name} ")]
        if line.split(name, 1)[1].split()[:2] != ["---", "---"]:
            problems.append(f"summary text: {line.strip()}")
    for name in case.untested:
        (line,) = [line for line in lines if line.startswith(f"║ {name} ") and "params" in line]
        if "chi2(" in line:
            problems.append(f"summary text: {line.strip()}")
    return problems


def _metrics_rows(case):
    table = _quiet(case.model.metrics, case.X, case.y).summary()
    return _rows(case, table._coef_rows, "metrics().summary()")


def _export_rows(case):
    from superglm.export.summary import build_summary_export_payload

    terms = {row.term: row for row in build_summary_export_payload(case.model).terms}
    problems = [
        f"summary export: {name} shows {terms[name].estimate}"
        for name in case.blank
        if terms[name].estimate is not None or terms[name].std_error is not None
    ]
    problems += [
        f"summary export: {name} tests {terms[name].statistic}"
        for name in case.untested
        if terms[name].statistic is not None or terms[name].p_value is not None
    ]
    return problems


def _editor_rows(case):
    from superglm.editor.summaries import _compact_summary_payload

    payload = _compact_summary_payload(case.model.summary(), "fit", [], [])
    rows = {row["name"]: row for row in payload["rows"]}
    problems = [
        f"editor summary: {name} shows {rows[name]['coef']}"
        for name in case.blank
        if rows[name]["coef"] is not None or rows[name]["se"] is not None
    ]
    problems += [
        f"editor summary: {name} tests {rows[name]['stat']}"
        for name in case.untested
        if rows[name]["stat"] is not None or rows[name]["ref_df"] is not None
    ]
    return problems


def _group_errors(case, errors: dict[str, np.ndarray], reader: str) -> list[str]:
    problems = []
    for group in case.model._groups:
        values = np.asarray(errors[group.name], dtype=np.float64)
        blank = np.flatnonzero(np.isnan(values)).tolist()
        if blank != case.unidentified.get(group.name, []):
            problems.append(f"{reader}: {group.name} has no SE at {blank}")
    return problems


def _summary_errors(case):
    errors = case.model.summary()["standard_errors"]
    return _group_errors(case, errors["coefficient_se"], "summary()['standard_errors']") + (
        _group_errors(case, errors["coefficient_se_raw"], "summary() raw SEs")
    )


def _metrics_errors(case):
    metrics = _quiet(case.model.metrics, case.X, case.y)
    return _group_errors(case, metrics.coefficient_se, "metrics().coefficient_se") + (
        _group_errors(case, metrics.coefficient_se_raw, "metrics().coefficient_se_raw")
    )


def _norms(values: dict[str, float], blank, kept, reader: str) -> list[str]:
    problems = [f"{reader}: {g} = {values[g]}" for g in blank if not _missing(values[g])]
    return problems + [f"{reader}: {g} lost {values[g]}" for g in kept if _missing(values[g])]


def _diagnostics(case):
    diagnostics = case.model.diagnostics()
    norms = {g.name: diagnostics[g.name]["group_norm"] for g in case.model._groups}
    return _norms(norms, case.unnormed, case.normed, "diagnostics()")


def _importance(case):
    frame = case.model.term_importance(case.X).set_index("term")
    problems = _norms(
        frame["group_norm"].to_dict(), case.unnormed, case.normed, "term_importance group_norm"
    )
    for column in ("variance_eta", "sd_eta"):
        problems += _norms(
            frame[column].to_dict(), case.unimportant, case.important, f"term_importance {column}"
        )
    return problems


def _redundancy(case):
    reports = case.model.spline_redundancy(case.X)
    name, _ = case.basis or ("s", ())
    unidentified = case.unidentified.get(name, [])
    blank = np.flatnonzero(np.isnan(reports[name].coef_energy_penalized)).tolist()
    return [] if blank == unidentified else [f"spline_redundancy: {name} blank at {blank}"]


def _basis_plot(case):
    from superglm.plotting.data import build_main_effect_plot_data

    name, expected = case.basis or ("s", ())
    payload = build_main_effect_plot_data(
        case.model, [_quiet(case.model.term_inference, name)], show_bases=True
    )
    bases = payload["terms"][0]["bases"]
    blank = sorted(set(bases.loc[bases["coefficient"].isna(), "basis_index"].tolist()))
    return [] if tuple(blank) == expected else [f"basis plot: {name} blank at {blank}"]


def _edited(case):
    """An edited model's table: a value it still holds from the fit follows the record."""
    from superglm.editor import EditorSession

    term, low, high = case.edit
    session = _quiet(EditorSession.from_model, case.model, terms=[term])
    session.select_x(term, low, high)
    session.shift(term, 0.05)
    edited = _quiet(session.to_model)
    rows = {row.name: row for row in edited.summary()._coef_rows}
    # its table has no structural or per-power rows
    blank, shown = ([n for n in names if n in rows] for names in (case.blank, case.shown))
    problems = [
        f"edited summary: {n} shows {rows[n].coef}" for n in blank if not _missing(rows[n].coef)
    ]
    return problems + [
        f"edited summary: {n} lost {rows[n].coef}" for n in shown if _missing(rows[n].coef)
    ]


def _relativities(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = case.model.relativities(with_se=True)
    problems = []
    for term, off in case.off_support.items():
        frame = frames[term]
        se = frame["se_log_relativity"].to_numpy(dtype=np.float64)
        if not np.array_equal(np.isnan(se), off(frame)):
            problems.append(f"relativities: {term} SE NaN at {np.flatnonzero(np.isnan(se))}")
        if not np.all(np.isfinite(frame["log_relativity"].to_numpy(dtype=np.float64))):
            problems.append(f"relativities: {term} dropped a value")
    return problems


def _term_inference(case):
    problems = []
    for term, off in case.off_support.items():
        inference = _quiet(case.model.term_inference, term)
        frame = pd.DataFrame(index=range(np.size(inference.log_relativity)))
        if inference.x is not None:
            frame["x"] = inference.x
        if inference.levels is not None:
            frame["level"] = inference.levels
        expected = off(frame)
        flagged = np.asarray(
            inference.non_estimable if inference.non_estimable is not None else [""] * len(frame)
        )
        if not np.array_equal(flagged != "", expected):
            problems.append(f"term_inference: {term} flags {np.flatnonzero(flagged != '')}")
        for field in ("se_log_relativity", "ci_lower", "ci_upper"):
            values = np.asarray(getattr(inference, field), dtype=np.float64)
            if not np.array_equal(np.isnan(values), expected):
                problems.append(
                    f"term_inference: {term}.{field} NaN at {np.flatnonzero(np.isnan(values))}"
                )
    return problems


def _feature_se(case):
    metrics = _quiet(case.model.metrics, case.X, case.y)
    frames = _quiet(case.model.relativities)
    problems = []
    for term, off in case.off_support.items():
        values = _quiet(metrics.feature_se, term)
        key = "se_coef" if "se_coef" in values else "se_log_relativity"
        se = np.atleast_1d(np.asarray(values[key], dtype=np.float64))
        if "x" in values:
            expected = off(pd.DataFrame({"x": values["x"]}))
        elif "levels" in values:
            expected = off(pd.DataFrame({"level": values["levels"]}))
        else:
            expected = off(frames[term])
        if not np.array_equal(np.isnan(se), expected):
            problems.append(f"feature_se: {term} NaN at {np.flatnonzero(np.isnan(se))}")
    return problems


def _bands(case):
    name = case.basis[0] if case.basis else "s"
    band = _quiet(case.model.simultaneous_bands, name, n_sim=200)
    expected = case.off_support[name](band)
    problems = []
    for column in ("se", "ci_lower_simultaneous", "ci_upper_simultaneous"):
        if not np.array_equal(band[column].isna().to_numpy(), expected):
            problems.append(
                f"simultaneous_bands: {column} NaN at {np.flatnonzero(band[column].isna())}"
            )
    return problems


def _rating_tables(case):
    try:
        payload = _quiet(case.model.rating_table_payload, case.X, case.y)
    except ValueError as refusal:
        # a walk's predictor past the link's clip is no product of factors: nothing exported
        return [] if "saturates" in str(refusal) else [f"rating tables: {refusal}"]
    problems = []
    for block in payload.main_effects:
        off = case.off_support.get(block.name)
        if off is None or block.kind not in ("categorical", "numeric"):
            continue
        frame = block.table.rename(columns={block.name: "level"})
        flagged = np.asarray(
            block.non_estimable if block.non_estimable is not None else [""] * len(frame)
        )
        expected = off(frame)
        if not np.array_equal(flagged != "", expected):
            problems.append(f"rating tables: {block.name} flags {np.flatnonzero(flagged != '')}")
    return problems


def _predict(case):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean = case.model.predict(case.X)
    named = [w.message for w in caught if isinstance(w.message, NonEstimablePredictionWarning)]
    rows = np.zeros(0, dtype=np.intp) if not named else named[0].rows
    problems = (
        [] if np.array_equal(np.sort(rows), case.rows) else [f"predict names {rows.size} rows"]
    )
    return problems + ([] if np.all(np.isfinite(mean)) else ["predict dropped a value"])


def _factor_smooth(case):
    """``factor_smooth()``'s level table: a level's coefficient norm reads only what is determined."""
    from superglm.features.factor_smooth import FactorSmooth

    problems = []
    for name, spec in case.model._interaction_specs.items():
        if not isinstance(spec, FactorSmooth):
            continue
        table = _quiet(case.model.factor_smooth, name).table
        norms = table["coefficient_norm"].to_numpy(dtype=np.float64)
        blank = tuple(table.loc[np.isnan(norms), "level"])
        if blank != case.smooth_levels.get(name, ()):
            problems.append(f"factor_smooth: {name} norm NaN at {blank}")
        problems += [f"factor_smooth: {name} norm {norms}"] if np.any(np.isinf(norms)) else []
    return problems


# The reader table: every reader of a published SuperGLM fit a user calls, by its rule.
READERS: dict[str, Callable[[_Case], list[str]]] = {
    # tables
    "summary rows": _summary_rows,
    "summary text": _summary_text,
    "metrics().summary() rows": _metrics_rows,
    "summary export payload": _export_rows,
    "editor summary payload": _editor_rows,
    "summary SE dict": _summary_errors,
    "metrics() SEs": _metrics_errors,
    "diagnostics() norms": _diagnostics,
    "term_importance": _importance,
    "spline_redundancy": _redundancy,
    "basis plot": _basis_plot,
    "edited model's table": _edited,
    # term functionals
    "relativities": _relativities,
    "term_inference": _term_inference,
    "feature_se": _feature_se,
    "simultaneous_bands": _bands,
    "rating tables": _rating_tables,
    "predict": _predict,
    "factor_smooth": _factor_smooth,
}


def _least_squares_wald(beta, covariance):
    """``b' V^+ b``: the statistic a solve returns where a held row is not exactly zero."""
    from superglm.inference.covariance import BinaryScaledCovariance

    if isinstance(covariance, BinaryScaledCovariance):
        beta = np.ldexp(beta, -(covariance.exponent // 2))
        covariance = covariance.base
    matrix = np.asarray(covariance, dtype=np.float64)
    return float(beta @ np.linalg.lstsq(matrix, beta, rcond=None)[0])


def _walk(case: _Case) -> list[str]:
    """Every reader in ``READERS`` the case walks, once the record decides its construction."""
    record = case.model._fit_inference_info["identifiability"]
    decided = {
        group.name: np.flatnonzero(~record.coefficient_identified()[group.sl]).tolist()
        for group in case.model._groups
    }
    # the fixture's construction is what the record decides; the walk tests the readers
    assert {name: blank for name, blank in decided.items() if blank} == case.unidentified
    walked = (reader for name, reader in READERS.items() if name not in case.skip)
    return [problem for reader in walked for problem in reader(case)]


@pytest.mark.parametrize(
    ("build", "forced"),
    [
        (_separated, False),
        (_aliased, False),
        (_separated, True),
        (functools.partial(_vanished, "auto"), False),
        (functools.partial(_vanished, "gram"), False),
    ],
    ids=[
        "separated",
        "aliased",
        "separated, Wald by least squares",
        "vanished weight, auto",
        "vanished weight, gram",
    ],
)
def test_no_reader_shows_a_number_the_fit_does_not_identify(build, forced, monkeypatch):
    """Every reader in ``READERS`` on the case: tables blank, functionals flag and withhold."""
    case = build()
    if forced:
        import superglm.inference.coef_tables as coef_tables

        monkeypatch.setattr(coef_tables, "covariance_wald_statistic", _least_squares_wald)
    assert _walk(case) == []


@pytest.mark.pins_behaviour
def test_a_partial_separation_withholds_what_its_verified_span_reaches():
    """A ``"partial"`` decision: every reader withholds what its verified span reaches.

    Pins that the readers take a proven lower bound as they take a ``"present"`` decision: the
    coefficients its span moves (``band`` above its break, ``x`` above 8) show no value, SE,
    interval or test, and their points are flagged.  It held at ``b175d49f2``: the record reads
    ``Separation.reached`` whatever the state.  The fixture's third zero set, the base ``T2``
    of a 3-level ordered polynomial, separates too, along a direction moving the intercept and
    ``tier``, but no float64 direction is exactly zero on ``T0``'s and ``T1``'s rows (the
    polynomial's level values are not dyadic), so none verifies: those coefficients keep the
    walk's values and SEs (``tier[P1]`` 2752, the intercept 13,540) beside the footnote's "more
    rows may be", and this test asserts nothing about them.
    """
    case = _partial()
    assert case.model._fit_inference_info["identifiability"].separation.state == "partial"
    assert _walk(case) == []


def test_the_footnote_says_which_rule_an_aliased_partner_follows():
    """``x2 = 2 x1``: the table shows ``x1`` in the reduced model, a relativity flags it.

    ``summary.lm`` prints the kept coefficient of an exact duplicate with its value and SE;
    ``predict.lm`` flags a row that moves it alone, as ``x1``'s relativity per unit does.  The
    footnote says so beside the alias, so a reader of both is told why they differ.  At
    ``b175d49f2`` it named the alias only.
    """
    rng = np.random.default_rng(11)
    x1 = rng.normal(size=500)
    X = pd.DataFrame({"x1": x1, "x2": 2.0 * x1, "c": rng.choice(list("AB"), 500)})
    y = rng.poisson(np.exp(0.2 + 0.3 * x1)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x1": Numeric(), "x2": Numeric(), "c": Categorical()},
    )
    _quiet(model.fit, X, y)
    rows = {row.name: row for row in model.summary()._coef_rows}
    assert _missing(rows["x2"].coef) and _present(rows["x1"].coef) and _present(rows["x1"].se)
    relativity = _quiet(model.relativities, with_se=True)["x1"]
    assert np.isnan(relativity["se_log_relativity"].to_numpy(dtype=np.float64)).all()
    text = " ".join(str(model.summary()).split())
    assert "x2 is aliased with x1" in text
    assert (
        'The coefficients named after "is aliased with" are shown as fitted with the ones above '
        "held at zero (R's summary.lm); a relativity or prediction that moves one of them alone "
        "leaves the rows' span, and is flagged as an alias (R's predict.lm)."
    ) in text
