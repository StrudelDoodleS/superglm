"""The fit-method lifecycle (``superglm.model.fit_method``): its table, its moves and its owner.

The table is walked three ways: every method has a row with the traits its readers
need; every declared move is made on a real model (a successful ``fit``,
``fit_path``, ``fit_reml`` or ``fit_ncv`` from each prior state, a failed fit that
leaves the state, an ``estimate_theta`` publication refit); and a source scan finds no writer
of ``_last_fit_meta`` and no comparison of the recorded method outside the owner.
"""

from __future__ import annotations

import ast
import copy
import pickle
import warnings
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, NegativeBinomial, Spline, SuperGLM
from superglm.editor.terms import resolve_refit_method
from superglm.model.profile_ops import _resolve_profile_fit_mode
from superglm.model.report_ops import rules_enforced

SOURCE = Path(__file__).parents[1] / "src" / "superglm"
OWNER = "model/fit_method.py"
ALL_METHODS = ("fit", "fit_path", "fit_reml", "fit_ncv")
FITTED_METHODS = ALL_METHODS


def _owner():
    """The owner module, imported per test so the saved-model control collects without it."""
    from superglm.model import fit_method

    return fit_method


def _data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "c": rng.choice(list("abc"), n)})
    y = rng.poisson(np.exp(0.3 + np.sin(3.0 * X["x"].to_numpy()))).astype(float)
    return X, y


def _model() -> SuperGLM:
    return SuperGLM(family="poisson", features={"x": Spline(n_knots=6), "c": Categorical()})


def _fit(model: SuperGLM, method: str, X, y) -> SuperGLM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y)
    return model


@pytest.fixture(scope="module")
def fitted() -> dict[str | None, SuperGLM]:
    """One model in each state: never fitted, and fitted by each method."""
    X, y = _data()
    return {None: _model(), **{m: _fit(_model(), m, X, y) for m in FITTED_METHODS}}


# ── the table ────────────────────────────────────────────────────


def test_every_method_has_a_row_with_its_traits():
    fit_method = _owner()
    assert get_args(fit_method.FitMethod) == ALL_METHODS
    rows = {
        method: (t.summary_label, t.criterion, t.profiles_scale, t.refit_as, t.enforces_caps)
        for method, t in fit_method.FIT_METHODS.items()
    }
    assert rows == {
        "fit": ("ML", None, False, "fit", True),
        "fit_path": ("ML", None, False, "fit", False),
        "fit_reml": ("REML", "reml", True, "fit_reml", True),
        "fit_ncv": ("NCV", "ncv", False, "fit", True),
    }
    for traits in fit_method.FIT_METHODS.values():
        assert traits.refit_as in fit_method.FIT_METHODS


def test_moves_lead_every_state_to_the_fit_that_succeeded_and_a_failure_nowhere():
    fit_method = _owner()
    states = (None, *ALL_METHODS)
    assert tuple(fit_method.MOVES) == states
    for state in states:
        assert dict(fit_method.MOVES[state]) == {
            **{method: method for method in ALL_METHODS},
            fit_method.FAILED: state,
        }


def test_record_refuses_a_method_without_a_row():
    fit_method = _owner()
    model = _model()
    model._discrete = False
    with pytest.raises(ValueError, match="unknown fit method 'fit_bogus'"):
        fit_method.record(model, "fit_bogus")
    assert model._last_fit_meta is None


def test_record_keeps_the_saved_shape():
    fit_method = _owner()
    model = _model()
    model._discrete = True
    fit_method.record(model, "fit_reml", direct_backend="dense", lambda_strategy="fixed")
    assert list(model._last_fit_meta.items()) == [
        ("method", "fit_reml"),
        ("discrete", True),
        ("direct_backend", "dense"),
        ("lambda_strategy", "fixed"),
    ]
    fit_method.annotate(model, initial_precision={"x": 1})
    assert model._last_fit_meta["method"] == "fit_reml"
    assert model._last_fit_meta["initial_precision"] == {"x": 1}


# ── the moves, on real models ───────────────────────────────────


def _readers(model) -> dict:
    return {
        "rules_enforced": rules_enforced(model),
        "editor_auto": resolve_refit_method(model, "auto"),
        "profile_inherit": _resolve_profile_fit_mode(model, "inherit"),
        "telemetry_method": model.training_telemetry()["fit"]["method"],
    }


@pytest.mark.parametrize("method", FITTED_METHODS)
@pytest.mark.parametrize("prior", (None, *FITTED_METHODS), ids=lambda s: f"from-{s}")
def test_a_successful_fit_moves_to_its_method(fitted, prior, method):
    fit_method = _owner()
    model = copy.deepcopy(fitted[prior])
    _fit(model, method, *_data())
    assert fit_method.method_of(model) == fit_method.MOVES[prior][method] == method
    traits = fit_method.FIT_METHODS[method]
    assert _readers(model) == {
        "rules_enforced": traits.enforces_caps,
        "editor_auto": traits.refit_as,
        "profile_inherit": traits.refit_as,
        "telemetry_method": method,
    }
    assert model.summary()._info["method"] == traits.summary_label
    assert model._last_fit_meta["discrete"] is model._discrete


@pytest.mark.parametrize("method", FITTED_METHODS)
@pytest.mark.parametrize("prior", FITTED_METHODS, ids=lambda s: f"from-{s}")
def test_a_failed_fit_leaves_the_state_and_the_earlier_fit(fitted, prior, method, monkeypatch):
    fit_method = _owner()
    model = copy.deepcopy(fitted[prior])
    X, y = _data()
    before = model.predict(X)
    meta_before = dict(model._last_fit_meta)

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure at publication")

    monkeypatch.setattr(fit_method, "record", fail)
    with pytest.raises(RuntimeError, match="forced failure at publication"):
        _fit(model, method, X, y)
    monkeypatch.undo()

    assert fit_method.method_of(model) == fit_method.MOVES[prior][fit_method.FAILED]
    assert model._last_fit_meta == meta_before
    np.testing.assert_array_equal(model.predict(X), before)


@pytest.mark.parametrize("prior", ("fit", "fit_reml"))
def test_an_estimate_theta_publication_refit_is_the_fit_it_inherits(prior):
    fit_method = _owner()
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 300)})
    mu = np.exp(0.4 + np.sin(2.0 * X["x"].to_numpy()))
    y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
    model = SuperGLM(family=NegativeBinomial(theta=1.0), features={"x": Spline(n_knots=6)})
    _fit(model, prior, X, y)
    refit_as = fit_method.traits_of(model).refit_as
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.estimate_theta(X, y, fit_mode="inherit")
    assert fit_method.method_of(model) == fit_method.MOVES[prior][refit_as] == prior


def test_a_reml_fit_with_nothing_to_estimate_has_no_result_to_check():
    fit_method = _owner()
    X, y = _data()
    model = _fit(SuperGLM(family="poisson", features={"c": Categorical()}), "fit_reml", X, y)
    assert model._reml_result is None
    assert fit_method.method_of(model) == "fit_reml"
    fit_method.check_search_result(model)


def test_a_result_of_another_criterion_is_refused(fitted):
    fit_method = _owner()
    model = copy.deepcopy(fitted["fit_reml"])
    fit_method.check_search_result(model)
    model._reml_result.criterion = "ncv"
    with pytest.raises(RuntimeError, match="criterion 'ncv', not the one fit_reml searches by"):
        fit_method.check_search_result(model)


# ── models saved before fits recorded their method ──────────────


def _saved_without_meta(model: SuperGLM) -> SuperGLM:
    state = dict(pickle.loads(pickle.dumps(model)).__dict__)
    del state["_last_fit_meta"]
    old = SuperGLM.__new__(SuperGLM)
    old.__setstate__(state)
    return old


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("method", FITTED_METHODS)
def test_a_model_saved_without_fit_metadata_reads_as_before(fitted, method):
    """Pins how a model saved without ``_last_fit_meta`` reads at the method's readers.

    Whatever method made it, it enforces caps, an "auto" editor refit and an
    "inherit" profile refit use ``fit``, telemetry reports no method, REML
    diagnostics are enabled exactly when a search result was saved, and it predicts
    as saved. No method is recovered from the saved result.
    """
    X, _ = _data()
    old = _saved_without_meta(fitted[method])
    assert _readers(old) == {
        "rules_enforced": True,
        "editor_auto": "fit",
        "profile_inherit": "fit",
        "telemetry_method": None,
    }
    assert old.reml_diagnostics()["enabled"] is (fitted[method]._reml_result is not None)
    np.testing.assert_array_equal(old.predict(X), fitted[method].predict(X))


def test_a_model_saved_without_fit_metadata_is_not_checked_against_its_result(fitted):
    fit_method = _owner()
    old = _saved_without_meta(fitted["fit_reml"])
    assert fit_method.method_of(old) is None
    assert fit_method.traits_of(old) is fit_method.FIT_METHODS["fit"]
    fit_method.check_search_result(old)


# ── the owner is the only writer and the only reader of the tag ─
#
# Scope: provenance is the method recorded in ``_last_fit_meta``. The scan follows
# only values read from that attribute (directly, or through a name assigned from
# an expression that reads it) and calls to ``method_of``. Argument dispatch
# (``fit_mode == "fit_reml"`` in ``model_selection.py`` and ``profiling/``, the
# editor's explicit ``method``), ``SuperLSS``'s ``_fit_settings["method"]`` and the
# REML debug recorder's metadata use the same strings on other values and are out of
# scope. Moving the whole fit state between a workspace and its model
# (``fit_state.py``'s attribute lists) assigns no value of its own and is not a write.

_MUTATORS = frozenset({"update", "setdefault", "pop", "popitem", "clear", "__setitem__"})
# The unfitted state, declared with the model's other attributes.
_DECLARATION = "model/base.py"


def _is_meta(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "_last_fit_meta"


def _reads_meta(node: ast.AST) -> bool:
    return any(
        _is_meta(sub) or (isinstance(sub, ast.Constant) and sub.value == "_last_fit_meta")
        for sub in ast.walk(node)
    )


def _writes(tree: ast.Module, path: str) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    hit = _is_meta(sub) or (isinstance(sub, ast.Subscript) and _is_meta(sub.value))
                    declared = (
                        path == _DECLARATION
                        and isinstance(node, ast.AnnAssign)
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is None
                    )
                    if hit and not declared:
                        lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _MUTATORS and _is_meta(func.value):
                lines.append(node.lineno)
            if (
                isinstance(func, ast.Name)
                and func.id in {"setattr", "delattr"}
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "_last_fit_meta"
            ):
                lines.append(node.lineno)
    return lines


def _method_reads(tree: ast.Module) -> list[int]:
    meta_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _reads_meta(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def holds_meta(node: ast.AST) -> bool:
        return _reads_meta(node) or (isinstance(node, ast.Name) and node.id in meta_names)

    def is_method_key(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value == "method"

    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and is_method_key(node.args[0])
            and holds_meta(node.func.value)
        ):
            lines.append(node.lineno)
        if isinstance(node, ast.Subscript) and is_method_key(node.slice) and holds_meta(node.value):
            lines.append(node.lineno)
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            names_method = any(
                isinstance(op, ast.Call)
                and getattr(op.func, "attr", getattr(op.func, "id", None)) == "method_of"
                for op in operands
            )
            constant_method = any(
                isinstance(sub, ast.Constant) and sub.value in ALL_METHODS
                for op in operands
                for sub in ast.walk(op)
            )
            if names_method and constant_method:
                lines.append(node.lineno)
    return lines


def _scan(check) -> dict[str, list[int]]:
    found = {}
    for file in sorted(SOURCE.rglob("*.py")):
        path = file.relative_to(SOURCE).as_posix()
        if path == OWNER:
            continue
        tree = ast.parse(file.read_text(encoding="utf-8"))
        lines = check(tree, path) if check is _writes else check(tree)
        if lines:
            found[path] = sorted(set(lines))
    return found


def test_no_module_but_the_owner_writes_the_fit_record():
    forbidden = ast.parse(
        "model._last_fit_meta = {'method': 'fit'}\n"
        "model._last_fit_meta['initial_precision'] = p\n"
        "model._last_fit_meta.update(x=1)\n"
        "setattr(model, '_last_fit_meta', {})\n"
        "model._last_fit_meta: dict | None = None\n"
    )
    assert sorted(set(_writes(forbidden, "model/x.py"))) == [1, 2, 3, 4, 5]
    assert _scan(_writes) == {}


def test_no_module_but_the_owner_reads_or_compares_the_recorded_method():
    forbidden = ast.parse(
        "a = (getattr(model, '_last_fit_meta', None) or {}).get('method') != 'fit_path'\n"
        "meta = model._last_fit_meta or {}\n"
        "b = meta.get('method') == 'fit_reml'\n"
        "c = model._last_fit_meta['method']\n"
        "d = fit_method.method_of(model) == 'fit_reml'\n"
        "e = fit_mode == 'fit_reml'\n"
        "f = settings['method']\n"
    )
    assert sorted(set(_method_reads(forbidden))) == [1, 3, 4, 5]
    assert _scan(_method_reads) == {}
