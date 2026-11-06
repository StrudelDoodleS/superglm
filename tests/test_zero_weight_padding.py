"""A row of weight zero changes no number a user reads, and an integer frequency weight is a replicated row.

Every case fits a small model on clean data and on the same data with hostile rows appended at
weight zero, then compares each public output on the padded frame with the clean one, to the
rounding the two fits can differ by (they take the same path, so ``PADDING_RTOL``). A frequency
weight is compared with the replicated rows at the fit's own convergence tolerance
(``FREQUENCY_RTOL``): the two take different rounding paths to the same optimum.

SuperGLM refuses a non-finite feature, label, response or offset even at weight zero, with a
clear error, and the refusal is asserted. SuperLSS drops the rows of weight zero before it reads
anything, so it is padded with every kind, including invalid censoring flags.
"""

from __future__ import annotations

import functools
import re
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
    s,
)
from superglm.editor import edit
from superglm.editor.payloads import session_payload
from superglm.model_selection import cross_validate

PADDING_RTOL = 1e-8
FREQUENCY_RTOL = 1e-4
N = 240
LEVELS = ["a", "b", "c"]
_VOLATILE = re.compile(r"hash|time|second|revision|epoch|fingerprint|(^|_)id$|^profile$")
_ROWS_PASSED = {"n_obs", "storage_rows"}  # the number of rows handed in, by definition

CENTERING = (
    "the printed intercept and term curve are shifted by the unweighted mean of the basis over "
    "every stored row, zero-weight rows included and a frequency weight counted once "
    "(features/_spline_build.py training_basis_mean), so they move when such rows are added; "
    "the owner decides whether that convention should read weights"
)
BINNING = (
    "actual_expected bins the covariate into equal counts of physical rows, so a frequency "
    "weight does not move the bin edges the way a replicated row does"
)


# ---- reading outputs --------------------------------------------------------------------------


def _flatten(value, path=""):
    """Every number in ``value`` as ``{path: float array}``; timings, ids and hashes are left out."""
    found = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not _VOLATILE.search(str(key)) and str(key) not in _ROWS_PASSED:
                found.update(_flatten(item, f"{path}.{key}"))
    elif isinstance(value, pd.DataFrame):
        for column in value.columns:
            if pd.api.types.is_numeric_dtype(value[column]) and column not in _ROWS_PASSED:
                found[f"{path}.{column}"] = value[column].to_numpy(dtype=float)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.update(_flatten(item, f"{path}[{index}]"))
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
        found[path] = value.astype(float).ravel()
    elif isinstance(value, Real) and not isinstance(value, bool):
        found[path] = np.array([float(value)])
    return found


def _differences(got, want, rtol):
    """What differs between two outputs: shapes, and numbers by more than ``rtol`` of their scale."""
    got, want = _flatten(got), _flatten(want)
    problems = [] if want else ["the reference output holds no numbers, so nothing was compared"]
    for key in sorted(got.keys() | want.keys()):
        if key not in got or key not in want or got[key].shape != want[key].shape:
            shapes = (got.get(key, np.empty(0)).shape, want.get(key, np.empty(0)).shape)
            problems.append(f"{key}: shapes {shapes[0]} against {shapes[1]}")
            continue
        a, b = got[key], want[key]
        same_nan = np.isnan(a) & np.isnan(b)
        gap = np.where(same_nan, 0.0, np.abs(a - b))
        scale = max(float(np.max(np.abs(b[~np.isnan(b)]), initial=0.0)), 1e-300)
        if np.any(np.isnan(gap)) or float(np.max(gap, initial=0.0)) > rtol * scale:
            problems.append(f"{key}: largest gap {np.nanmax(gap):.3g} on a scale of {scale:.3g}")
    return problems


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
        y = rng.gamma(4.0, np.exp(0.2 + np.sin(3 * X["x"])) / 4.0)
        extra = None
    else:
        y = np.sin(3 * X["x"]) + np.exp(-0.5 + 0.3 * X["z"]) * rng.normal(size=N)
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


def _glm_levels(model, term):
    table = model.term_inference(term).to_dataframe()
    return table[table["level"].isin(LEVELS)] if "level" in table else table


def _glm_metrics(model, data):
    metrics = model.metrics(*data)
    scalars = (
        "aic aicc bic ebic deviance effective_df explained_deviance log_likelihood "
        "null_deviance null_log_likelihood pearson_chi2 phi coefficient_se feature_se"
    ).split()
    return {name: getattr(metrics, name) for name in scalars}


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
    keep = [c for c in fold.columns if not c.endswith("_s") and c not in ("n_train", "n_test")]
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
    "summary": lambda m, d: m.summary().to_dict(),
    "metrics": _glm_metrics,
    "metric_rows": _glm_metric_rows,
    "drop1": lambda m, d: m.drop1(d[0], d[1], sample_weight=d[2], offset=d[3]),
    "cross_validation": _glm_cv,
    "editor_payload": _glm_editor,
    "native_centering": lambda m, d: [
        m.result.intercept,
        m.term_inference("x").to_dataframe(),
        m.metrics(*d).intercept_se,
    ],
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
NOT_REPLICATED = {"metric_rows", "scores", "residuals", "calibration", "editor_payload"}


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
            if name == "native_centering":
                marks = [pytest.mark.xfail(strict=True, reason=CENTERING)]
            if frequency and name == "actual_expected":
                marks = [pytest.mark.xfail(strict=True, reason=BINNING)]
            yield pytest.param(case, name, call, marks=marks, id=f"{case}-{name}")


@pytest.mark.parametrize(("case", "name", "call"), _output_cases(frequency=False))
def test_rows_of_weight_zero_change_no_output(case, name, call):
    clean, clean_data = _fitted(case, "clean")
    padded, padded_data = _fitted(case, "padded")

    got, want = call(padded, padded_data), call(clean, clean_data)

    assert _differences(got, want, PADDING_RTOL) == []


@pytest.mark.parametrize(("case", "name", "call"), _output_cases(frequency=True))
def test_integer_frequency_weights_equal_replicated_rows(case, name, call):
    weighted, weighted_data = _fitted(case, "weighted")
    replicated, replicated_data = _fitted(case, "replicated")

    got, want = call(weighted, weighted_data), call(replicated, replicated_data)

    assert _differences(got, want, FREQUENCY_RTOL) == []


# ---- what SuperGLM refuses, and what it keeps -------------------------------------------------

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
    assert table.loc["zzz", ["log_relativity", "se_log_relativity"]].tolist() == [0.0, 0.0]
