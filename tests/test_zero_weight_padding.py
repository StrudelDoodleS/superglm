"""A row of weight zero changes no number a user reads, bar the named conventions, and an integer frequency weight is a replicated row.

Every case fits a small model on clean data and on the same data with hostile rows appended at
weight zero, then compares each public output on the padded frame with the clean one: numbers to
``PADDING_RTOL`` of their scale, text, flags and shapes exactly.  SuperLSS drops the rows of
weight zero before it reads anything, so its two fits are bit-identical.  SuperGLM keeps them, so
its sums run over more rows and agree with the clean fit to rounding, and only if both stop at the
same REML iterate: that is asserted, so that a different stop reads as a different stop.  A
frequency weight is compared with the replicated rows to ``FREQUENCY_RTOL``, the fits'
convergence tolerance: the two take different rounding paths to the same optimum.

The differences that remain are real, each pinned where it is read:

* SuperGLM prints its intercept and term curves centred on the unweighted mean of the basis over
  every stored row, so they move when rows are added, and with them the standard errors read off
  the centred curve (the ``centering_*`` strict xfails, which say what each moves);
* ``n_obs`` and ``n_observations`` count the rows handed in (243 against 240) and are left out;
* a label seen only at weight zero is a known level of the padded fit (``zzz``), which its
  tables show and ``test_a_label_seen_only_at_weight_zero_is_a_known_level_pinned_at_one`` pins;
* SuperGLM's rating-table export refuses the padded frame itself, whose zero-weight rows saturate
  the prediction (``test_the_rating_table_export_refuses_a_frame_whose_rows_saturate_the_prediction``).

SuperGLM refuses a non-finite feature, label, response or offset even at weight zero, with a
clear error, and the refusal is asserted.  SuperLSS is padded with every kind, including invalid
censoring flags.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import re
import tempfile
import warnings
from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    GammaLS,
    GaussianLS,
    Numeric,
    PSpline,
    SuperGLM,
    SuperLSS,
    cat,
    lift_chart,
    s,
)
from superglm.editor import edit
from superglm.editor.payloads import session_payload
from superglm.model_selection import cross_validate

pytestmark = pytest.mark.pins_behaviour  # these guards pin behaviour that already held

PADDING_RTOL = 1e-8
FREQUENCY_RTOL = 1e-6  # the default tol of SuperGLM and reml_tol of SuperLSS
N = 240
LEVELS = ["a", "b", "c"]
_VOLATILE = re.compile(r"hash|time|second|revision|epoch|fingerprint|(^|_)id$|^fit_share$")
_ROWS_PASSED = {"n_obs", "n_observations", "storage_rows"}  # the rows handed in, by definition

_CONVENTION = (
    "the printed intercept and term curve are centred on the unweighted mean of the basis over "
    "every stored row, zero-weight rows included and a frequency weight counted once "
    "(features/_spline_build.py training_basis_mean), so adding such rows moves them and the "
    "standard errors read off the centred curve; the owner decides whether the convention "
    "should read weights"
)
#: Per case: what moves under the centring convention, measured on this fixture, first with
#: rows added at weight zero and then under frequency weights (the rating table and workbook run
#: on padding only).
CENTERING = {
    "centering_intercept": "the intercept moves by 0.0025 on 0.79; 0.019 on 0.82 under frequency weights",
    "centering_term_curve": (
        "log relativity moves by 0.0025 on 0.72, its standard error by 0.0013 on 0.17 and the "
        "upper confidence limit by 0.0059 on 1.7; under frequency weights 0.019 on 0.54, "
        "0.0020 on 0.14 and 0.039 on 1.7"
    ),
    "centering_intercept_se": (
        "the intercept's standard error moves by 7.6e-5 on 0.065; 3.2e-4 on 0.047 under "
        "frequency weights"
    ),
    "centering_feature_se": (
        "the term's standard error moves by 0.0014 on 0.17; 0.0019 on 0.14 under frequency weights"
    ),
    "centering_relativities": (
        "relativity moves by 0.0039 on 1.5 and its standard error by 0.0013 on 0.17; 0.030 on 1.6 "
        "and 0.0020 on 0.14 under frequency weights"
    ),
    "centering_reconstruct": (
        "original-scale coefficients move by 0.0025 on 1.0 and relativity by 0.0039 on 1.5; "
        "0.019 on 0.88 and 0.030 on 1.6 under frequency weights"
    ),
    "centering_rating_table": "the base relativity moves by 0.0056 on 2.2, the x relativities by 0.0039",
    "centering_export": (
        "the workbook carries the base relativity (0.0056 on 2.2), the intercept and its standard "
        "error, the term's confidence limits, the row count (243 for 240) and the level zzz"
    ),
}
BINNING = (
    "actual_expected bins the covariate into equal counts of physical rows, so a frequency "
    "weight does not move the bin edges the way a replicated row does"
)
LIFT_BINNING = (
    "lift_chart cuts its equal-exposure bins on row boundaries, so a row of weight 3 sits whole "
    "in one bin where three replicated rows can fall in two (bin exposure shares move by 0.004 "
    "on 0.1, observed rates by 0.09 on 4.9)"
)


# ---- reading outputs --------------------------------------------------------------------------


def _flatten(value, path=""):
    """Every leaf of ``value`` as ``{path: leaf}``: numbers as float arrays, text, flags and None as themselves.

    Timings, ids and hashes are left out by name.  A leaf of a type this does not know raises,
    so that an output cannot drop out of the comparison because nobody taught it to read one.
    """
    found = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not _VOLATILE.search(str(key)) and str(key) not in _ROWS_PASSED:
                found.update(_flatten(item, f"{path}.{key}"))
    elif isinstance(value, pd.DataFrame):
        for column in value.columns:
            if column not in _ROWS_PASSED:
                found.update(_flatten(value[column], f"{path}.{column}"))
    elif isinstance(value, pd.Series):
        found.update(_flatten(value.to_numpy(), path))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            found.update(_flatten(getattr(value, field.name), f"{path}.{field.name}"))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.update(_flatten(item, f"{path}[{index}]"))
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
        found[path] = value.astype(float).ravel()
    elif isinstance(value, np.ndarray) and value.dtype.kind in "bOUS":
        for index, item in enumerate(value.ravel().tolist()):
            found.update(_flatten(item, f"{path}[{index}]"))
    elif isinstance(value, bool | np.bool_ | str | type(None)):
        found[path] = value
    elif isinstance(value, Real):
        found[path] = np.array([float(value)])
    else:
        raise TypeError(
            f"{path or '<output>'}: the comparison cannot read a {type(value).__name__}"
        )
    return found


def _differences(got, want, rtol):
    """What differs between two outputs: shapes, text and flags, and numbers by more than ``rtol`` of their scale."""
    got, want = _flatten(got), _flatten(want)
    problems = [] if want else ["the reference output holds no leaves, so nothing was compared"]
    for key in sorted(got.keys() | want.keys()):
        if key not in got or key not in want:
            problems.append(f"{key}: present in only one of the two outputs")
            continue
        a, b = got[key], want[key]
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) or a != b:
                problems.append(f"{key}: {a!r} against {b!r}")
            continue
        if a.shape != b.shape:
            problems.append(f"{key}: shapes {a.shape} against {b.shape}")
            continue
        same_nan = np.isnan(a) & np.isnan(b)
        gap = np.where(same_nan, 0.0, np.abs(a - b))
        gap = np.where(np.isnan(gap), np.inf, gap)  # a NaN against a number
        scale = max(float(np.max(np.abs(b[~np.isnan(b)]), initial=0.0)), 1e-300)
        if float(np.max(gap, initial=0.0)) > rtol * scale:
            problems.append(f"{key}: largest gap {np.max(gap):.3g} on a scale of {scale:.3g}")
    return problems


@dataclasses.dataclass
class _Report:
    status: str
    value: float


def test_the_comparison_reads_text_flags_and_none_and_refuses_what_it_cannot_read():
    frame = pd.DataFrame({"level": ["a", "b"], "value": [1.0, 2.0]})

    assert _differences(frame, frame, PADDING_RTOL) == []
    assert _differences({"status": "stopped"}, {"status": "converged"}, PADDING_RTOL)
    assert _differences({"flag": np.True_}, {"flag": np.False_}, PADDING_RTOL)
    assert _differences({"bound": None}, {"bound": 1.0}, PADDING_RTOL)
    assert _differences(frame.assign(level=["a", "c"]), frame, PADDING_RTOL)
    assert _differences(pd.Series(["a"]), pd.Series(["b"]), PADDING_RTOL)
    assert _differences(_Report("a", 1.0), _Report("b", 1.0), PADDING_RTOL)
    with pytest.raises(TypeError, match="cannot read a object"):
        _flatten({"result": object()})


class _FeatureFolds:
    """A row's fold is read from its own ``x``: padded, reordered and replicated rows keep theirs."""

    def __init__(self, k=3):
        self.k = k

    def split(self, X, y=None, groups=None):
        x = np.clip(np.nan_to_num(X["x"].to_numpy(dtype=float)), -10.0, 10.0)
        fold = np.floor(x * 1e6).astype(np.int64) % self.k
        for k in range(self.k):
            yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)


# ---- data and fits ----------------------------------------------------------------------------


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"x": rng.uniform(size=n), "z": rng.normal(size=n), "g": rng.choice(LEVELS, n)}
    ), rng


def _hostile_frame(finite_only):
    """Rows to append at weight zero; ``finite_only`` keeps what SuperGLM accepts at any weight."""
    rows = pd.DataFrame(
        {
            "x": [0.4, 1e300, 5.0, np.nan, 0.2],
            "z": [0.0, 1e300, 0.0, np.nan, 0.1],
            "g": ["zzz", "a", "b", None, "c"],
        }
    )
    return rows.iloc[:3] if finite_only else rows


def _data(case, variant):
    """``(X, y, weight, extra)`` of a case; variants are clean, padded, weighted and replicated."""
    glm = case.startswith("glm")
    X, rng = _frame(N, 5)
    if glm:
        y = rng.poisson(np.exp(0.2 + np.sin(3 * X["x"]) + 0.3 * X["z"])).astype(float)
        extra = 0.1 * rng.normal(size=N)  # offset
    elif case == "lss_gamma":
        shape = np.exp(
            1.4 + 0.6 * np.sin(2 * X["z"])
        )  # curved, so no penalty is pushed to its ceiling
        y = rng.gamma(shape, np.exp(0.2 + np.sin(3 * X["x"])) / shape)
        extra = None
    else:
        y = np.sin(3 * X["x"]) + np.exp(-0.5 + 0.6 * np.sin(2 * X["z"])) * rng.normal(size=N)
        extra = None
    y = np.asarray(y, dtype=float)
    weight = rng.uniform(0.5, 2.0, N)
    if variant in ("weighted", "replicated"):
        weight = rng.integers(1, 4, N).astype(float)
    if variant == "replicated":
        index = np.repeat(np.arange(N), weight.astype(int))
        X, y = X.iloc[index].reset_index(drop=True), y[index]
        extra = None if extra is None else extra[index]
        weight = np.ones(len(y))
    if variant == "padded":
        hostile = _hostile_frame(finite_only=glm)
        k = len(hostile)
        X = pd.concat([X, hostile], ignore_index=True)
        weight = np.r_[weight, np.zeros(k)]
        if glm:
            y, extra = np.r_[y, [1.0, 1.0, 1e300][:k]], np.r_[extra, [0.0, 0.0, 1e300][:k]]
        else:
            y = np.r_[y, [1e300, -3.0, np.nan, -1e300, 1.0]]
            flags = np.array([False] * N + ["bogus", np.nan, 7, "x", None], dtype=object)
            extra = flags
    return X, y, weight, extra


def _model(case, discrete, frequency):
    semantics = "frequency" if frequency else "prior"
    if case.startswith("glm"):
        return SuperGLM(
            family="poisson",
            discrete=discrete,
            weight_semantics=semantics,
            features={"x": PSpline(n_knots=6), "z": Numeric(), "g": Categorical()},
        )
    family = GammaLS() if case == "lss_gamma" else GaussianLS()
    first = getattr(family, family.parameters[0].name)
    second = getattr(family, family.parameters[1].name)
    return SuperLSS(
        family,
        first(s("x", kind="cr", k=6), cat("g")),
        second(s("z", kind="cr", k=5)),
        discrete=discrete,
        weight_semantics=semantics,
    )


CASES = {
    "glm_exact": False,
    "glm_discrete": True,
    "lss_gaussian": False,
    "lss_gaussian_binned": True,
    "lss_gamma": False,
}


@functools.cache
def _fitted(case, variant):
    X, y, weight, extra = _data(case, variant)
    model = _model(case, CASES[case], frequency=variant in ("weighted", "replicated"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case.startswith("glm"):
            model.fit_reml(X, y, sample_weight=weight, offset=extra)
        else:
            flags = {"censored": extra} if variant == "padded" else {}
            model.fit_reml(X, y, sample_weight=weight, **flags)
    return model, (X, y, weight, extra)


# ---- outputs: each returns what one public call hands the user ---------------------------------

_AT = pd.DataFrame({"x": [0.1, 0.5, 0.9], "z": [0.0, 1.0, -1.0], "g": LEVELS})
_RETAINED = N  # rows the clean frame has; a padded one has them first


def _by_level(table, column="level"):
    """The rows of known levels: the level only the padded fit has is pinned in its own test."""
    return table[table[column].isin(LEVELS)] if column in table else table


def _known_levels(value):
    """``value`` without the level ``zzz``, wherever a mapping or list names it."""
    if isinstance(value, Mapping):
        return {key: _known_levels(item) for key, item in value.items() if key != "zzz"}
    if isinstance(value, list | tuple):
        return [
            _known_levels(item) for item in value if not (isinstance(item, str) and item == "zzz")
        ]
    return value


def _glm_levels(model, term):
    return _by_level(model.term_inference(term).to_dataframe())


def _glm_metrics(model, data):
    metrics = model.metrics(*data)
    scalars = (
        "aic aicc bic deviance effective_df explained_deviance log_likelihood "
        "null_deviance null_log_likelihood pearson_chi2 phi coefficient_se"
    ).split()
    found = {name: getattr(metrics, name) for name in scalars}
    found["ebic"] = metrics.ebic()
    found.update({f"feature_se_{term}": metrics.feature_se(term) for term in ("z", "g")})
    return found


def _glm_residuals(model, data):
    metrics = model.metrics(*data)
    return {
        kind: metrics.residuals(kind)[:_RETAINED] for kind in ("deviance", "pearson", "quantile")
    }


def _glm_diagnostics(model, data):
    report = dict(model.diagnostics())
    report["_model"] = {k: v for k, v in report["_model"].items() if k != "intercept"}
    return report  # the intercept is read in centering_intercept


def _glm_lift(model, data):
    X, y, weight, offset = data
    return lift_chart(y, model.predict(X, offset), sample_weight=weight).bins


def _glm_relativities(model, with_se=False):
    tables = model.relativities(with_se=with_se)
    return {term: _by_level(table) for term, table in tables.items() if term != "x"}


def _rating_table(model, data):
    """The payload on the retained rows: the padded fit's own frame saturates the prediction."""
    X, y, weight, offset = data
    rows = slice(0, _RETAINED)
    return model.rating_table_payload(
        X.iloc[rows], y[rows], sample_weight=weight[rows], offset=offset[rows]
    )


def _rating_table_blocks(model, data):
    blocks = {block.name: block.table for block in _rating_table(model, data).main_effects}
    blocks["g"] = _by_level(blocks["g"], "g")
    return {name: table for name, table in blocks.items() if name != "x"}


def _rating_table_x(model, data):
    payload = _rating_table(model, data)
    return [payload.base_relativity, {b.name: b.table for b in payload.main_effects}["x"]]


def _workbook(model, data):
    X, y, weight, offset = data
    rows = slice(0, _RETAINED)
    with tempfile.TemporaryDirectory() as tmp:
        path = model.export_rating_tables(
            os.path.join(tmp, "tables.xlsx"),
            X.iloc[rows],
            y[rows],
            sample_weight=weight[rows],
            offset=offset[rows],
        )
        return pd.read_excel(path, sheet_name=None)


def _glm_metric_rows(model, data):
    metrics = model.metrics(*data)
    rows = ("leverage", "cooks_distance", "std_deviance_residuals", "std_pearson_residuals", "eta")
    return {name: np.asarray(getattr(metrics, name))[:_RETAINED] for name in rows}


def _glm_cv(model, data):
    X, y, weight, offset = data
    result = cross_validate(
        model,
        X,
        y,
        cv=_FeatureFolds(),
        sample_weight=weight,
        offset=offset,
        fit_mode="fit_reml",
        error_score="raise",
    )
    return _cv_numbers(result)


def _cv_numbers(result):
    fold = result.fold_scores
    counts = ("n_train", "n_test", "n_iter", "n_reml_iter")  # sizes and the path each fit took
    keep = [c for c in fold.columns if not c.endswith("_s") and c not in counts]
    return [result.mean_scores, result.pooled_scores, result.std_scores, fold[keep]]


def _glm_editor(model, data):
    X, y, weight, offset = data
    payload = session_payload(edit(model, train_data=(X, y, weight, offset)))
    return payload["z"]  # the numeric term: the level and centring conventions are tested apart


GLM_OUTPUTS = {
    "coefficients": lambda m, d: [m.result.beta, m.result.effective_df, m.result.deviance],
    "smoothing": lambda m, d: [m.term_inference("x").smoothing_lambda, m.term_inference("x").edf],
    "predictions": lambda m, d: [m.predict(_AT), m.predict(_AT, np.array([0.1, 0.2, 0.3]))],
    "term_standard_errors": lambda m, d: [_glm_levels(m, "z"), _glm_levels(m, "g")],
    "relativities": lambda m, d: [_glm_relativities(m), _glm_relativities(m, with_se=True)],
    "reconstruct_feature": lambda m, d: [
        m.reconstruct_feature("z"),
        _known_levels(m.reconstruct_feature("g")),
    ],
    "rating_table": _rating_table_blocks,
    "summary": lambda m, d: _known_levels(m.summary().to_dict()),
    "metrics": _glm_metrics,
    "metric_rows": _glm_metric_rows,
    "residuals": _glm_residuals,
    "lift_chart": _glm_lift,
    "diagnostics": _glm_diagnostics,
    "drop1": lambda m, d: m.drop1(d[0], d[1], sample_weight=d[2], offset=d[3]),
    "cross_validation": _glm_cv,
    "editor_payload": _glm_editor,
    "centering_intercept": lambda m, d: m.result.intercept,
    "centering_term_curve": lambda m, d: m.term_inference("x").to_dataframe(),
    "centering_intercept_se": lambda m, d: m.metrics(*d).intercept_se,
    "centering_feature_se": lambda m, d: m.metrics(*d).feature_se("x"),
    "centering_relativities": lambda m, d: m.relativities(with_se=True)["x"],
    "centering_reconstruct": lambda m, d: m.reconstruct_feature("x"),
    "centering_rating_table": _rating_table_x,
    "centering_export": _workbook,
}


def _lss_cv(model, data):
    X, y, weight, _ = data
    result = cross_validate(
        model,
        X,
        y,
        cv=_FeatureFolds(),
        sample_weight=weight,
        fit_mode="fit_reml",
        error_score="raise",
    )
    return _cv_numbers(result)


def _lss_rows(call, data, *, flags):
    """A per-row output on the retained rows, and the rows of weight zero it must leave empty."""
    X, y, weight, extra = data
    out = call(X, y, sample_weight=weight, **({"censored": extra} if flags else {}))
    values = np.asarray(out)
    return values[:_RETAINED], values[_RETAINED:]


def _lss_scores(model, data):
    retained, dropped = _lss_rows(lambda *a, **k: model.scores(*a, **k), data, flags=True)
    return [retained, float(np.all(np.isnan(dropped)))]  # 1.0: every dropped row is empty


def _lss_residuals(model, data):
    X, y, weight, extra = data
    flags = {"censored": extra} if extra is not None else {}
    return model.residuals(X, y, sample_weight=weight, **flags)[:_RETAINED]


def _lss_editor(model, data):
    X, y, weight, _ = data
    return session_payload(edit(model, train_data=(X, y, weight)))


def _lss_term(model, parameter):
    return [
        model.term_inference(parameter, "x").to_json(),
        model.term_test(parameter, "x").to_json(),
    ]


def _first(model):
    return model.parameter_names_[0]


LSS_OUTPUTS = {
    "coefficients": lambda m, d: [
        dict(m.coef_),
        m.covariance_,
        m.result_.total_effective_df,
        m.result_.log_likelihood,
        m.result_.penalized_log_likelihood,
    ],
    "smoothing": lambda m, d: dict(m.smoothing_parameters_),
    "predictions": lambda m, d: [m.predict_parameters(_AT), m.predict_link(_AT), m.predict(_AT)],
    "standard_errors": lambda m, d: [
        m.posterior_bounds(_AT, "mean"),
        _lss_term(m, _first(m)),
        m.summary(),
    ],
    "scores": _lss_scores,
    "residuals": _lss_residuals,
    "calibration": lambda m, d: m.calibration(d[0], d[1], sample_weight=d[2]).to_json(),
    "actual_expected": lambda m, d: m.actual_expected(
        d[0], d[1], "x", sample_weight=d[2]
    ).to_json(),
    "diagnose": lambda m, d: m.diagnose().to_dict(),
    "cross_validation": _lss_cv,
    "editor_payload": _lss_editor,
}
# Outputs whose size is the number of rows handed in say nothing about replication.
NOT_REPLICATED = {
    "metric_rows",
    "scores",
    "residuals",
    "calibration",
    "editor_payload",
    "rating_table",  # the binned tables and the workbook are quantiles of the rows handed in
    "centering_rating_table",
    "centering_export",
}


def _output_cases(frequency):
    return list(_cases(frequency))


def _cases(frequency):
    for case in CASES:
        outputs = GLM_OUTPUTS if case.startswith("glm") else LSS_OUTPUTS
        for name, call in outputs.items():
            if frequency and name in NOT_REPLICATED:
                continue
            if name == "editor_payload" and case.startswith("lss_gaussian"):
                continue  # the editor opens GammaLS, LogNormalLS and NegativeBinomialLS models
            marks = []
            if name in CENTERING:
                reason = f"{_CONVENTION}: {CENTERING[name]}"
                marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
            if frequency and name in ("actual_expected", "lift_chart"):
                reason = BINNING if name == "actual_expected" else LIFT_BINNING
                marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
            yield pytest.param(case, name, call, marks=marks, id=f"{case}-{name}")


def _padding_rtol(case, name):
    """A SuperLSS fold is a separate fit that stops on the tolerance, at its own iterate."""
    return FREQUENCY_RTOL if name == "cross_validation" and case.startswith("lss") else PADDING_RTOL


@pytest.mark.parametrize(("case", "name", "call"), _output_cases(frequency=False))
def test_rows_of_weight_zero_change_no_output(case, name, call):
    clean, clean_data = _fitted(case, "clean")
    padded, padded_data = _fitted(case, "padded")

    got, want = call(padded, padded_data), call(clean, clean_data)

    assert _differences(got, want, _padding_rtol(case, name)) == []


@pytest.mark.parametrize(("case", "name", "call"), _output_cases(frequency=True))
def test_integer_frequency_weights_equal_replicated_rows(case, name, call):
    weighted, weighted_data = _fitted(case, "weighted")
    replicated, replicated_data = _fitted(case, "replicated")

    got, want = call(weighted, weighted_data), call(replicated, replicated_data)

    assert _differences(got, want, FREQUENCY_RTOL) == []


# ---- what SuperGLM refuses, and what it keeps -------------------------------------------------


@pytest.mark.parametrize("case", ["glm_exact", "glm_discrete"])
def test_a_padded_superglm_fit_makes_the_same_smoothing_iterations_as_the_clean_one(case):
    # SuperGLM sums over the zero-weight rows too, so the fits agree to rounding, and only when
    # both stop at the same REML iterate: a different stop must read as one, not as noise.
    clean, _ = _fitted(case, "clean")
    padded, _ = _fitted(case, "padded")

    assert padded.reml_diagnostics()["n_reml_iter"] == clean.reml_diagnostics()["n_reml_iter"]


@pytest.mark.parametrize("case", ["glm_exact", "glm_discrete"])
def test_the_rating_table_export_refuses_a_frame_whose_rows_saturate_the_prediction(case):
    # The padded fit's own frame holds rows (x and z of 1e300) at weight zero whose linear
    # predictor overflows, so the export refuses it where the clean frame gets a table.
    padded, (X, y, weight, offset) = _fitted(case, "padded")

    with pytest.raises(ValueError, match="model.predict saturates on this frame"):
        padded.rating_table_payload(X, y, sample_weight=weight, offset=offset)


REFUSED = {
    "a missing feature": ({"x": np.nan}, "must contain only finite values"),
    "a missing label": ({"g": None}, "contains missing values"),
    "a missing response": ({"y": np.nan}, "must contain only finite values"),
    "a negative count": ({"y": -3.0}, "requires nonnegative y"),
    "a missing offset": ({"offset": np.nan}, "offset must contain only finite values"),
}


@pytest.mark.parametrize("bad", REFUSED)
def test_superglm_refuses_a_non_finite_input_even_at_weight_zero(bad):
    changes, message = REFUSED[bad]
    X, y, weight, offset = _data("glm_exact", "clean")
    row = {"x": 0.4, "z": 0.0, "g": "a", "y": 1.0, "offset": 0.0} | changes
    X = pd.concat([X, pd.DataFrame([{k: row[k] for k in "xzg"}])], ignore_index=True)
    model = _model("glm_exact", False, frequency=False)

    with pytest.raises(ValueError, match=message):
        model.fit_reml(
            X,
            np.r_[y, row["y"]],
            sample_weight=np.r_[weight, 0.0],
            offset=np.r_[offset, row["offset"]],
        )


@pytest.mark.parametrize("case", ["glm_exact", "glm_discrete"])
def test_a_label_seen_only_at_weight_zero_is_a_known_level_pinned_at_one(case):
    clean, _ = _fitted(case, "clean")
    padded, _ = _fitted(case, "padded")
    table = padded.term_inference("g").to_dataframe().set_index("level")

    assert list(table.index) == [*clean.term_inference("g").to_dataframe()["level"], "zzz"]
    # The convention of the fit: a level with no effective rows is pinned to the base level, and
    # its standard error reads 0.0 ("known exactly") where nothing is known about it.  Changing
    # that, say to mark the level unestimated, changes this assertion.
    assert table.loc["zzz", ["log_relativity", "se_log_relativity"]].tolist() == [0.0, 0.0]
