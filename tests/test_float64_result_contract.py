"""Every floating value these SuperGLM fits publish is IEEE binary64.

AGENTS.md, "Numerical policy": production numerics target float64 on every
platform. Each case below fits a small model on one representative route:
``fit_reml`` on seven routes, two of them the structured engine for random
effects and factor smooths, and ``fit()`` with an active selection penalty,
which runs ``fit_pirls``. Every case runs twice, once as written and once with
float32 features, response, ``sample_weight`` and ``offset``. The test walks
what the fit publishes -- the public result, the REML result and profile,
the retained linear-system and reporting-support state, ``diagnostics()``,
``iteration_diagnostics()`` where it was recorded, and ``predict()`` -- and
fails on any floating array, sparse matrix, pandas column or index, or NumPy
scalar that is not float64 (complex values must be complex128). It also fails
on any object it cannot look inside, so a new published type cannot hide a
leaf. Integer and boolean arrays are index and mask data and are allowed;
Python floats are binary64 by definition. SuperLSS fits are not covered here.
No floating exception is needed today.
"""

import types
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import (
    Constraint,
    FactorSmooth,
    PSpline,
    RandomEffect,
    Spline,
    SuperGLM,
    Tweedie,
)
from superglm.solvers._structured.state import StructuredLinearSystemState

_BINARY64 = (np.dtype(np.float64), np.dtype(np.complex128))
# Code, not data: nothing a fit publishes is stored inside these.
_CODE = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    type,
)


def _slot_names(value) -> list[str]:
    names: list[str] = []
    for klass in type(value).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        names += [slots] if isinstance(slots, str) else list(slots)
    return [name for name in names if name not in ("__dict__", "__weakref__")]


def _children(value):
    """Yield ``(suffix, child)`` for an object the walk can see inside, else None."""
    if isinstance(value, pd.DataFrame):
        return [(".index", value.index), (".columns", value.columns)] + [
            (f"[{key!r}]", column) for key, column in value.items()
        ]
    if isinstance(value, pd.Series | pd.Index):
        pairs = [(".index", value.index)] if isinstance(value, pd.Series) else []
        if isinstance(value.dtype, pd.CategoricalDtype):
            return [*pairs, (".categories", value.dtype.categories)]
        if value.dtype.kind == "O":
            return [*pairs, *((f"[{index}]", item) for index, item in enumerate(value))]
        return pairs
    if isinstance(value, np.ndarray | np.generic) or sp.issparse(value):
        if value.dtype.kind != "O":
            return []
        return [(f"[{index}]", item) for index, item in enumerate(value.flat)]
    if isinstance(value, Mapping):
        return [(f"[{key!r}]", item) for key, item in value.items()]
    if isinstance(value, slice):
        return [(".start", value.start), (".stop", value.stop), (".step", value.step)]
    if isinstance(value, list | tuple | set | frozenset):
        return [(f"[{index}]", item) for index, item in enumerate(value)]
    names = [item.name for item in fields(value)] if is_dataclass(value) else []
    names += [name for name in getattr(value, "__dict__", {}) if name not in names]
    names += [name for name in _slot_names(value) if name not in names]
    if not names and not hasattr(value, "__dict__"):
        return None
    return [(f".{name}", getattr(value, name)) for name in names if hasattr(value, name)]


def _floating_leaves(value, path="", seen=None, found=None, opaque=None):
    """Return ``(path, dtype)`` for every floating leaf reachable from ``value``.

    Objects the walk cannot see inside are appended to ``opaque``; when it is
    not supplied, meeting one is an error, so the walk fails closed.
    """
    # Visited objects stay referenced: a temporary child (a Series from
    # DataFrame.items(), a lazily computed mapping value) is freed once its
    # parent is done, and a later object may then reuse its id().
    seen = {} if seen is None else seen
    found = [] if found is None else found
    scalar = value is None or isinstance(value, str | bytes | bool | int | float | complex)
    if scalar or isinstance(value, _CODE) or id(value) in seen:
        return found
    seen[id(value)] = value
    dtype = getattr(value, "dtype", None)
    if not isinstance(value, pd.DataFrame) and getattr(dtype, "kind", None) in ("f", "c"):
        found.append((path, dtype))
    children = _children(value)
    if children is None:
        if opaque is None:
            raise TypeError(f"cannot inspect {type(value).__qualname__} at {path or 'root'}")
        opaque.append((path, type(value).__qualname__))
        return found
    for suffix, item in children:
        _floating_leaves(item, path + suffix, seen, found, opaque)
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


def _poisson_random_effect_structured():
    # Too narrow for direct_solve="auto" to choose the structured engine.
    _, X, y = _poisson_random_effect()
    features = {"x": Spline(n_knots=6), "g": RandomEffect()}
    model = SuperGLM(
        family="poisson", selection_penalty=0, features=features, direct_solve="structured"
    )
    return model, X, y


def _gaussian_factor_smooth_structured():
    rng, x, group = _frame(5)
    X = pd.DataFrame({"x": x, "g": pd.Categorical(group.astype(str))})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        interactions=[FactorSmooth("x", group="g", basis="fs", k=5)],
        direct_solve="structured",
    )
    return model, X, np.sin(2 * np.pi * x) + 0.2 * group + rng.normal(0, 0.3, x.size)


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


def _gaussian_selection():
    # A positive selection penalty sends fit() through fit_pirls, not fit_reml.
    _, X, y = _gaussian_smooth()
    model = SuperGLM(family="gaussian", selection_penalty=0.1, features={"x": Spline(n_knots=6)})
    return model, X, y


CASES = {
    "gaussian_smooth": _gaussian_smooth,
    "poisson_random_effect": _poisson_random_effect,
    "poisson_random_effect_structured": _poisson_random_effect_structured,
    "gaussian_factor_smooth_structured": _gaussian_factor_smooth_structured,
    "tweedie_smooth": _tweedie_smooth,
    "gamma_discrete": _gamma_discrete,
    "gaussian_increasing_scop": _gaussian_increasing,
    "gaussian_selection_fit": _gaussian_selection,
}


def _float32_inputs(X, y):
    rng = np.random.default_rng(7)
    floats = X.select_dtypes("float").columns
    narrowed = X.astype(dict.fromkeys(floats, np.float32))
    weight = rng.uniform(0.5, 2.0, len(y)).astype(np.float32)
    offset = rng.normal(0.0, 0.05, len(y)).astype(np.float32)
    return narrowed, np.asarray(y, dtype=np.float32), {"sample_weight": weight, "offset": offset}


@pytest.mark.parametrize("inputs", ["float64", "float32"])
@pytest.mark.parametrize("case", CASES)
def test_fit_publishes_only_binary64_floating_values(case, inputs):
    model, X, y = CASES[case]()
    extra = {}
    if inputs == "float32":
        X, y, extra = _float32_inputs(X, y)
    selection = case == "gaussian_selection_fit"
    if selection:
        model.fit(X, y, record_diagnostics=True, **extra)
    else:
        model.fit_reml(X, y, **extra)
    published = {
        "result": model.result,
        "reml_result": model._reml_result,
        "profile": model._reml_profile,
        "linear_system": model._linear_system_state,
        "reporting_support": model._reporting_support_state,
        "diagnostics": model.diagnostics(),
        "predict": model.predict(X, offset=extra.get("offset")),
    }
    if selection:
        published["iteration_diagnostics"] = model.iteration_diagnostics()
    opaque: list[tuple[str, str]] = []
    leaves = _floating_leaves(published, opaque=opaque)
    assert not opaque, f"{case} published objects the walk cannot inspect: {opaque}"
    paths = {path for path, _ in leaves}
    # The walk must reach the coefficients inside the result dataclass, and the
    # selection case must reach the fit_pirls rank metadata.
    assert "['result'].beta" in paths and "['predict']" in paths
    if case.endswith("_structured"):
        assert isinstance(model._linear_system_state, StructuredLinearSystemState)
        assert any(path.startswith("['linear_system']") for path in paths)
    if selection:
        assert model.selection_penalty > 0 and model._reml_result is None
        assert "['result'].rank_info.feature_edf" in paths
    offenders = [(path, str(dtype)) for path, dtype in leaves if dtype not in _BINARY64]
    assert not offenders, f"{case} published non-binary64 floating values: {offenders}"


@dataclass
class _Holder:
    value: object


class _SlottedHolder:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _CallableHolder:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


_FLOAT32 = np.zeros(2, dtype=np.float32)


@pytest.mark.parametrize(
    "value",
    [
        _Holder(_FLOAT32),
        {"key": [np.float32(1.0)]},
        np.array([None, np.zeros(1, dtype=np.float16)], dtype=object),
        sp.csr_array(np.eye(2, dtype=np.float32)),
        pd.DataFrame({"a": _FLOAT32}),
        pd.DataFrame({"a": pd.Series([_FLOAT32, None], dtype=object)}),
        pd.DataFrame({"a": [1, 2]}, index=pd.Index(_FLOAT32)),
        pd.Series(pd.Categorical(_FLOAT32)),
        _SlottedHolder(_FLOAT32),
        _CallableHolder(_FLOAT32),
    ],
    ids=[
        "dataclass",
        "mapping_scalar",
        "object_array",
        "sparse",
        "frame",
        "frame_object_column",
        "frame_index",
        "categorical",
        "slots",
        "callable_instance",
    ],
)
def test_walk_reports_a_reduced_precision_leaf(value):
    leaves = _floating_leaves({"root": value})
    assert [dtype for _, dtype in leaves if dtype not in _BINARY64]


class _FreshValues(Mapping):
    """A mapping that builds a new holder on every lookup, as a lazy view would."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, key):
        return _Holder(np.zeros(2, dtype=self.dtype))

    def __iter__(self):
        return iter(["value"])

    def __len__(self):
        return 1


def test_walk_keeps_temporaries_alive_so_a_reused_id_hides_no_leaf():
    # With ids alone, the float32 holder reused the freed float64 holder's id
    # and the walk skipped it.
    value = {"a": _FreshValues(np.float64), "b": _FreshValues(np.float32)}
    leaves = _floating_leaves(value)
    assert ("['b']['value'].value", np.dtype(np.float32)) in leaves


def test_walk_fails_closed_on_an_object_it_cannot_inspect():
    opaque: list[tuple[str, str]] = []
    _floating_leaves({"root": [object()]}, opaque=opaque)
    assert opaque == [("['root'][0]", "object")]
    with pytest.raises(TypeError, match="cannot inspect"):
        _floating_leaves({"root": object()})
