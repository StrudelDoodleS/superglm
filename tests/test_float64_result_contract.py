"""Every floating value a fit publishes is IEEE binary64.

AGENTS.md, "Numerical policy": production numerics target float64 on every
platform. Each case below fits a small model on one representative route and
walks what the fit publishes -- the public result, the REML result and profile,
``diagnostics()`` and ``predict()`` -- failing on any floating array, sparse
matrix or NumPy scalar that is not float64 (complex values must be
complex128). Integer and boolean arrays are index and mask data and are
allowed; Python floats are binary64 by definition. No floating exception is
needed today.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import Constraint, PSpline, RandomEffect, Spline, SuperGLM, Tweedie

_BINARY64 = (np.dtype(np.float64), np.dtype(np.complex128))


def _floating_leaves(value, path="", seen=None, found=None):
    """Return ``(path, dtype)`` for every floating leaf reachable from ``value``."""
    seen = set() if seen is None else seen
    found = [] if found is None else found
    scalar = value is None or isinstance(value, str | bytes | bool | int | float | complex)
    if scalar or callable(value) or id(value) in seen:
        return found
    seen.add(id(value))
    children = ()
    if isinstance(value, np.ndarray | np.generic) or sp.issparse(value):
        if value.dtype.kind in "fc":
            found.append((path, value.dtype))
        if value.dtype.kind == "O":
            children = ((f"[{index}]", item) for index, item in enumerate(value.flat))
    elif isinstance(value, pd.DataFrame | pd.Series):
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        found.extend(
            (f"{path}[{key!r}]", kind) for key, kind in frame.dtypes.items() if kind.kind in "fc"
        )
    elif isinstance(value, Mapping):
        children = ((f"[{key!r}]", item) for key, item in value.items())
    elif isinstance(value, list | tuple | set | frozenset):
        children = ((f"[{index}]", item) for index, item in enumerate(value))
    else:
        names = [item.name for item in fields(value)] if is_dataclass(value) else []
        names += [name for name in getattr(value, "__dict__", {}) if name not in names]
        children = ((f".{name}", getattr(value, name)) for name in names)
    for suffix, item in children:
        _floating_leaves(item, path + suffix, seen, found)
    return found


def _frame(seed: int, n: int = 200):
    rng = np.random.default_rng(seed)
    return rng, rng.uniform(0.0, 1.0, n), rng.integers(0, 6, n)


def _gaussian_smooth():
    rng, x, _ = _frame(0)
    model = SuperGLM(family="gaussian", selection_penalty=0, features={"x": Spline(n_knots=6)})
    return model, pd.DataFrame({"x": x}), np.sin(2 * np.pi * x) + rng.normal(0, 0.3, x.size)


def _poisson_random_effect():
    rng, x, group = _frame(1)
    features = {"x": Spline(n_knots=6), "g": RandomEffect()}
    model = SuperGLM(family="poisson", selection_penalty=0, features=features)
    X = pd.DataFrame({"x": x, "g": pd.Categorical(group.astype(str))})
    return model, X, rng.poisson(np.exp(0.3 * x + rng.normal(0, 0.3, 6)[group]))


def _tweedie_smooth():
    rng, x, _ = _frame(2)
    counts = rng.poisson(0.8 * np.exp(0.5 * x))
    y = np.array([rng.gamma(2.0, 0.6, count).sum() for count in counts])
    model = SuperGLM(family=Tweedie(p=1.5), selection_penalty=0, features={"x": Spline(n_knots=6)})
    return model, pd.DataFrame({"x": x}), y


def _gamma_discrete():
    rng, x, _ = _frame(3)
    features = {"x": Spline(n_knots=6, penalty="ssp")}
    model = SuperGLM(family="gamma", selection_penalty=0, discrete=True, features=features)
    return model, pd.DataFrame({"x": x}), rng.gamma(3.0, np.exp(0.4 * x) / 3.0)


def _gaussian_increasing():
    rng, x, _ = _frame(4)
    features = {"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)}
    model = SuperGLM(family="gaussian", selection_penalty=0, features=features)
    return model, pd.DataFrame({"x": x}), 2 * x + rng.normal(0, 0.2, x.size)


def _gaussian_float32_inputs():
    model, X, y = _gaussian_smooth()
    return model, X.astype(np.float32), y.astype(np.float32)


CASES = {
    "gaussian_smooth": _gaussian_smooth,
    "poisson_random_effect": _poisson_random_effect,
    "tweedie_smooth": _tweedie_smooth,
    "gamma_discrete": _gamma_discrete,
    "gaussian_increasing_scop": _gaussian_increasing,
    "gaussian_float32_inputs": _gaussian_float32_inputs,
}


@pytest.mark.parametrize("case", CASES)
def test_fit_publishes_only_binary64_floating_values(case):
    model, X, y = CASES[case]()
    model.fit_reml(X, y)
    published = {
        "result": model.result,
        "reml_result": model._reml_result,
        "profile": model._reml_profile,
        "diagnostics": model.diagnostics(),
        "predict": model.predict(X),
    }
    leaves = _floating_leaves(published)
    paths = {path for path, _ in leaves}
    # The walk must reach the coefficients inside the result dataclass.
    assert "['result'].beta" in paths and "['predict']" in paths
    offenders = [(path, str(dtype)) for path, dtype in leaves if dtype not in _BINARY64]
    assert not offenders, f"{case} published non-binary64 floating values: {offenders}"


@dataclass
class _Holder:
    value: object


@pytest.mark.parametrize(
    "value",
    [
        _Holder(np.zeros(2, dtype=np.float32)),
        {"key": [np.float32(1.0)]},
        np.array([None, np.zeros(1, dtype=np.float16)], dtype=object),
        sp.csr_array(np.eye(2, dtype=np.float32)),
        pd.DataFrame({"a": np.zeros(2, dtype=np.float32)}),
    ],
    ids=["dataclass", "mapping_scalar", "object_array", "sparse", "frame"],
)
def test_walk_reports_a_reduced_precision_leaf(value):
    leaves = _floating_leaves({"root": value})
    assert [dtype for _, dtype in leaves if dtype not in _BINARY64]
