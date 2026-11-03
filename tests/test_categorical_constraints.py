"""Relativity caps on plain categoricals: rows ``log lo <= beta_j <= log hi`` per level."""

from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, Spline, SuperGLM, collapse_levels
from superglm.features.constraint import ConstraintRules, rules_of

C = Constraint.fit
_U = 2.0**-53
_LEVELS = ["A", "B", "C", "D", "E"]
# The most exposed level, A, is the base; B sits above the cap and C below it.
_EFFECT = {"A": 0.0, "B": np.log(2.5), "C": np.log(0.5), "D": 0.1, "E": -0.1}


def _region_rows(n=20_000, seed=7, levels=_LEVELS, effect=_EFFECT, x=False):
    rng = np.random.default_rng(seed)
    p = np.array([0.4, 0.15, 0.15, 0.15, 0.15])[: len(levels)]
    region = np.asarray(levels, dtype=object)[rng.choice(len(levels), n, p=p / p.sum())]
    X = pd.DataFrame({"region": region})
    eta = -1.0 + np.array([effect[r] for r in region])
    if x:
        X["x"] = rng.uniform(0.0, 1.0, n)
        eta = eta + np.sin(2.0 * np.pi * X["x"].to_numpy())
    return X, rng.poisson(np.exp(eta)).astype(float)


def _printed(model, name="region") -> dict:
    """The level relativities ``term_inference`` prints, against the base."""
    ti = model.term_inference(name, with_se=False)
    return {str(level): float(rel) for level, rel in zip(ti.levels, ti.relativity, strict=True)}


def _fit(X, y, spec, *, method="fit", family="poisson", **features):
    model = SuperGLM(family=family, selection_penalty=0.0, features={"region": spec, **features})
    return getattr(model, method)(X, y)


def _assert_in_cap(printed, lo, hi):
    values = np.array(list(printed.values()))
    assert np.min(values) >= lo * (1.0 - 1e3 * _U)
    assert np.max(values) <= hi * (1.0 + 1e3 * _U)


def test_categorical_cap_binds():
    """Printed relativities lie in the cap, the extremes at its ends, and predict() agrees.

    The intercept is free, so the Poisson score equation for it still balances
    the fitted and observed totals. Mutation: without the cap rows B prints at
    about 2.5 and C at about 0.5.
    """
    X, y = _region_rows()
    model = _fit(X, y, Categorical(constraint=C.cap(0.7, 1.5)))
    printed = _printed(model)
    assert printed["A"] == 1.0
    _assert_in_cap(printed, 0.7, 1.5)
    assert printed["B"] == pytest.approx(1.5, rel=1e-6)
    assert printed["C"] == pytest.approx(0.7, rel=1e-6)
    mu = model.predict(X)
    assert np.sum(mu) == pytest.approx(np.sum(y), rel=1e-6)
    # The printed relativity is the predicted ratio against a base row.
    rows = pd.DataFrame({"region": np.asarray(_LEVELS, dtype=object)})
    ratio = model.predict(rows) / model.predict(rows.iloc[[0]]).item()
    np.testing.assert_allclose(ratio, [printed[level] for level in _LEVELS], rtol=1e3 * _U)
    free = _printed(_fit(X, y, Categorical()))
    assert free["B"] > 2.0 and free["C"] < 0.6


@pytest.mark.parametrize(
    "rule",
    [
        C.increasing,
        C.decreasing,
        C.convex,
        C.concave,
        C.flat,
        C.max_step(0.1),
        C.increasing & C.cap(0.7, 1.5),
        Constraint.postfit.increasing,
    ],
    ids=["increasing", "decreasing", "convex", "concave", "flat", "max_step", "and", "postfit"],
)
def test_categorical_shape_refused(rule):
    """A shape rule on an unordered categorical is refused, naming it and OrderedCategorical."""
    rules = rule if isinstance(rule, ConstraintRules) else ConstraintRules((rule,))
    named = next(r for r in rules if r.kind != "cap")
    with pytest.raises(ValueError, match=rf"{re.escape(named.describe())}.*OrderedCategorical"):
        Categorical(constraint=rule)


def test_categorical_cap_with_selection_penalty_refused():
    """A capped categorical refuses a selection penalty, naming the feature."""
    X, y = _region_rows(n=2_000)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.1,
        features={"region": Categorical(constraint=C.cap(0.7, 1.5))},
    )
    with pytest.raises(NotImplementedError, match=r"'region'.*selection_penalty"):
        model.fit(X, y)


def test_categorical_cap_refused_under_identity_link():
    """A cap is refused under the identity link, naming the feature."""
    X, y = _region_rows(n=2_000)
    with pytest.raises(ValueError, match=r"Feature 'region': cap\(0\.7, 1\.5\).*log link"):
        _fit(X, y, Categorical(constraint=C.cap(0.7, 1.5)), family="gaussian")


def test_categorical_cap_under_reml():
    """REML restores the categorical's rows after its passthrough; the fit meets the cap.

    The categorical's design has no SSP map (``R_inv``), so its raw rows are
    installed as they are. Mutation: leaving the stripped rows off prints B
    near 2.5.
    """
    X, y = _region_rows(x=True)
    model = _fit(
        X, y, Categorical(constraint=C.cap(0.7, 1.5)), method="fit_reml", x=Spline(kind="ps", k=8)
    )
    assert len(model._reml_result.lambdas) > 0
    printed = _printed(model)
    _assert_in_cap(printed, 0.7, 1.5)
    assert printed["B"] == pytest.approx(1.5, rel=1e-6)
    group = next(g for g in model._groups if g.feature_name == "region")
    assert group.monotone_engine == "qp" and group.constraints is not None


def test_categorical_grouping_and_unseen():
    """Under a grouping the cap binds on the group columns; unseen rows price at the base.

    B and C are grouped and both sit above the cap. Mutation: without the cap
    rows the group prints near 2.5.
    """
    effect = {"A": 0.0, "B": np.log(2.5), "C": np.log(2.5), "D": np.log(0.5), "E": -0.1}
    X, y = _region_rows(effect=effect)
    grouping = collapse_levels(X["region"], groups={"BC": ["B", "C"]})
    spec = Categorical(grouping=grouping, unseen="base", constraint=C.cap(0.7, 1.5))
    model = _fit(X, y, spec)
    printed = _printed(model)
    _assert_in_cap(printed, 0.7, 1.5)
    assert printed["B"] == printed["C"] == pytest.approx(1.5, rel=1e-6)
    assert printed["D"] == pytest.approx(0.7, rel=1e-6)
    rows = pd.DataFrame({"region": np.asarray(["A", "Z"], dtype=object)})
    with pytest.warns(UserWarning, match="Z"):
        mu = model.predict(rows)
    assert mu[1] == mu[0]


def test_categorical_cap_survives_rebuild():
    """An editor or structure rebuild of a capped categorical keeps the cap."""
    from superglm.features.rebuild import rebuilt_categorical

    spec = Categorical(constraint=C.cap(0.7, 1.5))
    rebuilt = rebuilt_categorical(spec, spec, base="A", grouping=None, data=np.array(_LEVELS))
    assert rebuilt.constraint_rules == spec.constraint_rules


def test_categorical_without_constraint_attribute_loads():
    """A categorical pickled before caps existed loads with no rules and predicts as saved.

    An uncapped build carries no rows and no engine, so it fits as before.
    """
    X, y = _region_rows(n=2_000)
    model = _fit(X, y, Categorical())
    expected = model.predict(X)
    spec = model._specs["region"]
    assert spec.build(X["region"].to_numpy()).constraints is None
    spec.__dict__.pop("_constraint_rules")
    restored = pickle.loads(pickle.dumps(model))
    old = restored._specs["region"]
    assert "_constraint_rules" not in old.__dict__
    assert not rules_of(old) and old.constraint_kind is None
    np.testing.assert_array_equal(restored.predict(X), expected)
    info = old.build(X["region"].to_numpy())
    assert info.constraints is None and info.monotone_engine is None
