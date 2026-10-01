"""Which grouping factors nest (one-engine design §3.13, decision 11).

The level-code pattern decides the elimination tree, as the pattern of ``Z``
decides a sparse Cholesky's symbolic analysis.  ``RandomEffect(nested_in=)``
declares a hierarchy: it is validated on the training rows, a violating row is
an error naming the row and both parent levels, and a declared hierarchy is
always used.  An undeclared pair that nests on all but a few rows is named in
a warning that never changes the route: one broken row makes the pair crossed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import NegativeBinomial, Numeric, RandomEffect, SuperGLM, Tweedie
from superglm.features.random_effect import near_nesting_notes, validate_declared_nesting
from superglm.group_matrix import DenseGroupMatrix, RandomEffectGroupMatrix
from superglm.solvers.structured import resolve_structured_backend
from superglm.types import GroupSlice


def _frame(seed: int = 0, n: int = 1200, leaves: int = 60, per_parent: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    leaf = rng.integers(0, leaves, n)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "region": [f"r{code:02d}" for code in leaf],
            "country": [f"c{code // per_parent}" for code in leaf],
        }
    )


def _response(frame: pd.DataFrame, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    region = frame["region"].str[1:].astype(int).to_numpy()
    effect = rng.normal(0.0, 0.3, region.max() + 1)[region]
    return rng.poisson(np.exp(0.2 * frame["x"].to_numpy() + effect)).astype(float)


def _model(**region) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={
            "x": Numeric(),
            "country": RandomEffect(),
            "region": RandomEffect(**region),
        },
        selection_penalty=0,
        direct_solve="auto",
    )


def test_a_declared_nesting_is_checked_on_the_training_rows() -> None:
    """A row that breaks ``nested_in`` is an error naming the row and both parent levels.

    Mutation: without ``validate_declared_nesting`` the fit runs (the pair is
    then crossed).
    """
    frame = _frame()
    frame.loc[7, "country"] = "c9"  # region frame.region[7] also occurs under another country
    region, first = frame.loc[7, "region"], frame.index[frame["region"] == frame.loc[7, "region"]]
    home = frame.loc[first[first != 7][0], "country"]
    with pytest.raises(ValueError) as broken:
        _model(nested_in="country").fit_reml(frame, _response(frame))
    message = str(broken.value)
    assert "nested_in='country'" in message
    assert repr(region) in message and "'c9'" in message and repr(home) in message
    assert "(row 7)" in message


_PROFILED_ENTRY_POINTS = pytest.mark.parametrize(
    ("family", "entry"),
    [(NegativeBinomial(theta=5.0), "estimate_theta"), (Tweedie(p=1.5), "estimate_p")],
)


def _profiled_model(family, **region) -> SuperGLM:
    model = _model(**region)
    model.family = family
    return model


@_PROFILED_ENTRY_POINTS
def test_every_reml_entry_point_checks_a_declared_nesting(family, entry) -> None:
    """``estimate_theta`` and ``estimate_p`` check ``nested_in`` at their entry, as ``fit_reml`` does.

    Mutation: without the check in ``estimate_theta`` the broken row is
    accepted and the pair is fitted as crossed (``estimate_p`` raised it from
    its search's first candidate fit before).
    """
    frame = _frame()
    frame.loc[7, "country"] = "c9" if frame.loc[7, "country"] != "c9" else "c8"
    model = _profiled_model(family, nested_in="country")
    with pytest.raises(ValueError, match=r"nested_in='country'.*\(row 7\)"):
        getattr(model, entry)(frame, _response(frame), fit_mode="reml")


@_PROFILED_ENTRY_POINTS
def test_every_reml_entry_point_names_a_near_nested_pair_once(family, entry) -> None:
    """The near-nesting warning is raised once per call, by every REML entry point.

    Mutation: ``estimate_theta`` without the check raises none; ``estimate_p``
    without its search's candidates marked as checked raises one per candidate.
    """
    frame = _frame()
    frame.loc[11, "country"] = "c9" if frame.loc[11, "country"] != "c9" else "c8"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # a coarse power search: the warning count does not depend on its length
        search = {"xatol": 0.1} if entry == "estimate_p" else {}
        getattr(_profiled_model(family), entry)(frame, _response(frame), fit_mode="reml", **search)
    notes = [str(item.message) for item in caught if "is nested in 'country'" in str(item.message)]
    assert len(notes) == 1 and "row 11" in notes[0]


def test_a_declared_parent_must_be_another_random_effect() -> None:
    frame = _frame()
    for parent in ("x", "region", "missing"):
        with pytest.raises(ValueError, match="not another RandomEffect feature"):
            _model(nested_in=parent).fit_reml(frame, _response(frame))
    with pytest.raises(TypeError, match="nested_in"):
        RandomEffect(nested_in=["country"])


def test_a_valid_declaration_fits_the_chain_the_pattern_finds() -> None:
    """Declared or detected, a strict hierarchy is eliminated as one chain, with no warning."""
    frame = _frame()
    y = _response(frame)
    fits = {}
    for declared in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model = _model(**({"nested_in": "country"} if declared else {})).fit_reml(frame, y)
        assert model._reml_profile["structured_chain"] == ("country", "region")
        fits[declared] = model
    np.testing.assert_array_equal(fits[True].result.beta, fits[False].result.beta)


def _two_parent_case(*, declared: bool):
    """A leaf nested in two crossed coarsenings; the one with more levels is Rule B's."""
    rng = np.random.default_rng(3)
    n, leaves = 900, 60
    leaf = rng.integers(0, leaves, n)
    small = leaf // 10  # 6 levels
    large = leaf % 20  # 20 levels, crossed with ``small``
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        RandomEffectGroupMatrix(small, 6),
        RandomEffectGroupMatrix(large, 20),
        RandomEffectGroupMatrix(leaf, leaves, nested_in="small" if declared else None),
    ]
    names = ["x", "small", "large", "leaf"]
    groups, start = [], 0
    for matrix, name in zip(matrices, names, strict=True):
        groups.append(GroupSlice(name, start, start + matrix.shape[1], penalized=name != "x"))
        start += matrix.shape[1]
    return matrices, groups


@pytest.mark.parametrize("direct_solve", ["auto", "structured"])
def test_a_declared_hierarchy_is_always_used(direct_solve: str) -> None:
    """Rule B takes the heaviest chain; a declared parent is in the chain whatever it weighs.

    Mutation: a chain rule that ignores the declaration keeps ``large``.
    """
    lambdas = {"small": 1.0, "large": 1.0, "leaf": 1.0}
    chains = {}
    for declared in (False, True):
        matrices, groups = _two_parent_case(declared=declared)
        decision = resolve_structured_backend(
            matrices,
            groups,
            direct_solve=direct_solve,
            coefficient_width=groups[-1].end,
            lambda2=lambdas,
        )
        chains[declared] = tuple(groups[index].name for index in decision.chain_group_indices)
    assert chains == {False: ("large", "leaf"), True: ("small", "leaf")}


def test_one_broken_row_makes_the_pair_crossed_and_is_named() -> None:
    """T1 pin: with one row that breaks nesting the decision is the crossed one, and a
    warning names the row; the warning never changes the route.

    Mutation: without ``near_nesting_notes`` no warning is raised.
    """
    frame = _frame()
    y = _response(frame)
    broken = frame.copy()
    broken.loc[11, "country"] = "c9" if broken.loc[11, "country"] != "c9" else "c8"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crossed = _model().fit_reml(broken, y)
    notes = [str(item.message) for item in caught if "is nested in 'country'" in str(item.message)]
    assert len(notes) == 1
    assert "except for 1 row(s)" in notes[0] and "row 11" in notes[0]
    assert "nested_in='country'" in notes[0]
    assert crossed._reml_profile["structured_chain"] == ("region",)
    # the warning decides nothing: the same fit with the warning silenced
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = _model().fit_reml(broken, y)
    np.testing.assert_array_equal(again.result.beta, crossed.result.beta)


def test_a_crossed_pair_and_a_nested_pair_are_not_named() -> None:
    """Crossed factors break nesting on most rows; strictly nested ones on none."""
    frame = _frame()
    rng = np.random.default_rng(9)
    frame["brand"] = [f"b{code}" for code in rng.integers(0, 8, len(frame))]
    specs = {"country": RandomEffect(), "region": RandomEffect(), "brand": RandomEffect()}
    assert near_nesting_notes(specs, lambda name: frame[name].to_numpy()) == []
    validate_declared_nesting(
        {"country": RandomEffect(), "region": RandomEffect(nested_in="country")},
        lambda name: frame[name].to_numpy(),
    )


def test_the_re_helper_declares_nesting() -> None:
    """``re(column, nested_in=)`` declares the hierarchy the terms API validates on its rows.

    Fails on the helper without ``nested_in`` (``TypeError``).
    """
    from superglm.terms import normalize_terms, re

    features = normalize_terms((re("country"), re("region", nested_in="country"))).features
    assert features["region"].nested_in == "country"
    frame = _frame()
    validate_declared_nesting(features, lambda name: frame[name].to_numpy())
    frame.loc[7, "country"] = "c9" if frame.loc[7, "country"] != "c9" else "c8"
    with pytest.raises(ValueError, match=r"nested_in='country'.*\(row 7\)"):
        validate_declared_nesting(features, lambda name: frame[name].to_numpy())
