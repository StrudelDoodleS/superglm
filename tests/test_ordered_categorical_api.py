"""API surface tests for ``OrderedCategorical`` after the 0.24.0 removal.

``basis=Spline(...)`` is the one configuration channel. The five scalar
shortcuts (``kind``/``n_knots``/``degree``/``select``/``penalty``) and the
legacy ``basis="spline"``/``basis="step"`` strings are gone; these tests pin
that the removed surface fails loudly, that the implicit default is exactly
the historical P-spline, and that specs restored from before the removal
either keep working (spline mode) or refuse loudly (step mode).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.editor.collapse import _ordered_spec_with_grouping
from superglm.features.spline import PSpline

LEVELS = [f"L{i}" for i in range(8)]


def test_omitted_basis_is_quiet_and_preserves_default_pspline() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = OrderedCategorical(order=LEVELS)

    assert spec.basis == "spline"
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == 5
    assert spec._spline.degree == 3
    assert spec._spline.select is False
    assert spec._spline.penalty == "ssp"


def test_canonical_rewrite_shape_reproduces_the_removed_shortcut_defaults() -> None:
    """``Spline(kind="ps", n_knots=N)`` is the documented migration for the
    removed ``n_knots=N`` shortcut. That claim is only true while the Spline
    factory's remaining defaults (degree, penalty, select) equal the ones the
    shortcut path hard-coded, so pin them together."""
    spec = OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", n_knots=3))

    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == 3
    assert spec._spline.degree == 3
    assert spec._spline.select is False
    assert spec._spline.penalty == "ssp"


@pytest.mark.parametrize("legacy", ["spline", "step"])
def test_removed_basis_string_raises_and_names_the_replacement(legacy) -> None:
    with pytest.raises(ValueError, match=r"basis=Spline\(\.\.\.\)") as excinfo:
        OrderedCategorical(order=LEVELS, basis=legacy)

    message = str(excinfo.value)
    assert "removed" in message
    assert "Categorical" in message


@pytest.mark.parametrize(
    ("shortcut", "value"),
    [("kind", "ps"), ("n_knots", 4), ("degree", 2), ("select", True), ("penalty", "ssp")],
)
def test_removed_shortcut_raises_type_error(shortcut, value) -> None:
    """The parameters are gone from ``__init__``, so Python itself names the
    stray keyword — no warning path, no silent build."""
    with pytest.raises(TypeError, match=shortcut):
        OrderedCategorical(order=LEVELS, **{shortcut: value})


def test_removed_shortcuts_raise_even_beside_a_spline_basis() -> None:
    """Before removal, shortcuts next to ``basis=Spline(...)`` were ignored
    with a warning; they must not become silently accepted now."""
    with pytest.raises(TypeError):
        OrderedCategorical(
            order=LEVELS,
            basis=Spline(kind="ps", k=7),
            kind="cr",
            n_knots=2,
            degree=1,
            select=False,
            penalty="none",
        )


def test_spline_object_is_the_quiet_canonical_api() -> None:
    basis = Spline(kind="ps", k=7)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = OrderedCategorical(order=LEVELS, basis=basis)

    assert spec.basis == "spline"
    assert spec._spline_obj is basis
    assert spec._spline is not basis
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == basis.n_knots


def test_spline_object_wrapper_metadata_matches_canonical_basis() -> None:
    basis = Spline(kind="cr", k=6, penalty="none", select=True)
    spec = OrderedCategorical(order=LEVELS, basis=basis)

    assert spec.kind == "cr"
    assert spec.select is basis.select
    assert spec.penalty == basis.penalty
    assert spec.degree == basis.degree
    assert spec.n_knots == basis.n_knots


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("kind", "cr"), ("n_knots", 9), ("degree", 1), ("select", True), ("penalty", "none")],
)
def test_derived_spline_metadata_is_read_only(attribute, value) -> None:
    """The five removed constructor parameters survive only as derived views
    of the inner spline; assigning them must fail rather than silently
    diverge from the basis."""
    spec = OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", k=7))

    with pytest.raises(AttributeError):
        setattr(spec, attribute, value)


@pytest.mark.parametrize("basis_select", [True, False])
def test_summary_selection_label_uses_canonical_basis_metadata(basis_select) -> None:
    spec = OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", k=7, select=basis_select))
    X = pd.DataFrame({"band": np.tile(LEVELS, 25)})
    y = np.tile(np.linspace(-0.3, 0.4, len(LEVELS)), 25)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": spec},
    )
    model.fit(X, y)

    assert ("SEL" in model.summary()._info["penalty"]) is basis_select


def test_editor_clone_of_canonical_spline_stays_quiet() -> None:
    spec = OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", k=7))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replacement = _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(LEVELS, dtype=object),
        )

    assert replacement.basis == "spline"
    assert replacement._spline_obj is not None
    assert replacement._spline.n_knots == spec._spline.n_knots


def test_editor_clone_of_quiet_implicit_default_stays_quiet() -> None:
    spec = OrderedCategorical(order=LEVELS)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replacement = _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(LEVELS, dtype=object),
        )

    assert replacement.basis == "spline"
    assert replacement._spline_obj is not None
    assert replacement._spline.n_knots == 5


@pytest.mark.parametrize(
    ("levels", "basis"),
    [
        (["A", "B", "C", "D"], Spline(kind="ps", k=6)),
        (["low", "medium", "high"], Spline(kind="ps", k=6)),
        (["A", "B", "C"], Spline(kind="cr", k=4)),
    ],
)
def test_documented_canonical_examples_do_not_warn_or_clamp(levels, basis) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = OrderedCategorical(order=levels, basis=basis)

    assert spec._spline.n_knots == basis.n_knots


# ── Pre-0.24 pickles ──────────────────────────────────────────────


def _restored_step_spec() -> OrderedCategorical:
    """What a pre-0.24 step-mode pickle restores.

    Unpickling never runs ``__init__`` — it updates the instance dict
    directly — so building the dict by hand and round-tripping it through
    pickle reproduces exactly what loading an old artifact does.
    """
    spec = OrderedCategorical.__new__(OrderedCategorical)
    spec.__dict__.update(
        {
            "basis": "step",
            "_spline": None,
            "_spline_obj": None,
            # The five removed parameters were plain attributes then; the
            # class properties shadow these restored entries.
            "kind": "ps",
            "n_knots": 4,
            "degree": 3,
            "select": False,
            "penalty": "ssp",
            "base": "first",
            "_specials": [],
            "_special_raw": [],
            "_special_display": [],
            "_smooth_levels": ["A", "B", "C"],
            "_ordered_levels": ["A", "B", "C"],
            "_level_to_value": {"A": 0.0, "B": 0.5, "C": 1.0},
            "_grouping": None,
            "_original_level_to_value": None,
            "_known_levels": {"A", "B", "C"},
            "_n_levels": 3,
            "_base_level": "A",
            "_non_base": ["B", "C"],
            "_R_inv": np.eye(2),
        }
    )
    return pickle.loads(pickle.dumps(spec))


def _restored_shortcut_spline_spec() -> OrderedCategorical:
    """What a pre-0.24 SPLINE-mode pickle built from the removed shortcuts
    restores: a built inner spline, ``_spline_obj`` None, and the five
    then-plain attributes carrying their construction-time values."""
    spec = OrderedCategorical(order=["A", "B", "C"], basis=Spline(kind="ps", n_knots=2))
    spec.build(np.array(["A", "B", "C", "A", "B", "C"]))
    spec.__dict__["_spline_obj"] = None
    spec.__dict__.update({"kind": "ps", "n_knots": 4, "degree": 3, "select": False})
    return pickle.loads(pickle.dumps(spec))


def test_step_mode_pickle_fails_loudly_at_every_numeric_path() -> None:
    """A spec pickled with the removed step mode must refuse to run, not score
    silently wrong. Every numeric entry point funnels through
    ``_basis_spline``, which names the removal and the migration."""
    spec = _restored_step_spec()
    x = np.array(["A", "B", "C"])

    for attempt in (
        lambda: spec.build(x),
        lambda: spec.transform(x),
        lambda: spec.score(x, np.zeros(2)),
        lambda: spec.reconstruct(np.zeros(2)),
        lambda: spec.set_reparametrisation(np.eye(2)),
    ):
        with pytest.raises(AttributeError, match=r"[Ss]tep mode was removed"):
            attempt()


def test_step_mode_pickle_repr_does_not_raise() -> None:
    """repr is exactly what someone debugging an old artifact prints first."""
    assert "step" in repr(_restored_step_spec())


def test_step_mode_pickle_cannot_parent_an_interaction() -> None:
    from superglm.features.ordered_categorical import resolve_interaction_parent

    spec = _restored_step_spec()
    with pytest.raises(AttributeError, match=r"[Ss]tep mode was removed"):
        resolve_interaction_parent(spec, np.array(["A", "B", "C"]))


def test_step_mode_pickle_is_refused_by_the_editor_clone() -> None:
    """The collapse clone must not silently rebuild a step spec onto the
    default P-spline; the ``_basis_spline`` read refuses it first."""
    spec = _restored_step_spec()
    with pytest.raises(AttributeError, match=r"[Ss]tep mode was removed"):
        _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(["A", "B", "C"], dtype=object),
        )


def test_spline_mode_shortcut_pickle_still_transforms_and_clones() -> None:
    """A pre-0.24 spline-mode spec (built from the removed shortcuts, so
    ``_spline_obj`` is None) is a legitimate smooth and must keep working:
    the derived attributes read the inner spline, and the editor clone falls
    back to it rather than to the removed five-attribute rebuild."""
    spec = _restored_shortcut_spline_spec()

    assert spec.kind == "ps"
    assert spec.n_knots == 2  # the property shadows the stale pickled 4
    out = spec.transform(np.array(["A", "B", "C"]))
    assert out.shape[0] == 3

    replacement = _ordered_spec_with_grouping(
        spec,
        grouping=None,
        selected_levels=[],
        base="first",
        data=np.asarray(["A", "B", "C"], dtype=object),
    )
    assert isinstance(replacement._spline, PSpline)
    assert replacement._spline.n_knots == 2
