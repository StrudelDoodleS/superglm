"""API compatibility and deprecation tests for ``OrderedCategorical``."""

from __future__ import annotations

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
        warnings.simplefilter("error", FutureWarning)
        spec = OrderedCategorical(order=LEVELS)

    assert spec.basis == "spline"
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == 5
    assert spec._spline.degree == 3
    assert spec._spline.select is False
    assert spec._spline.penalty == "ssp"


def test_explicit_spline_string_warns_once_and_still_works() -> None:
    with pytest.warns(FutureWarning, match=r"basis=.spline.*basis=Spline") as caught:
        spec = OrderedCategorical(order=LEVELS, basis="spline")

    assert len(caught) == 1
    assert spec.basis == "spline"
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == 5


def test_explicit_legacy_shortcut_warns_even_at_legacy_default() -> None:
    with pytest.warns(FutureWarning, match=r"spline shortcut.*basis=Spline") as caught:
        spec = OrderedCategorical(order=LEVELS, kind="ps")

    assert len(caught) == 1
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == 5


def test_multiple_legacy_shortcuts_warn_once_and_still_work() -> None:
    with pytest.warns(FutureWarning, match=r"spline shortcuts.*basis=Spline") as caught:
        spec = OrderedCategorical(
            order=LEVELS,
            n_knots=4,
            degree=2,
            select=True,
            penalty="ssp",
        )

    assert len(caught) == 1
    assert spec._spline.n_knots == 4
    assert spec._spline.degree == 2
    assert spec._spline.select is True
    assert spec._spline.penalty == "ssp"


def test_spline_object_is_the_quiet_canonical_api() -> None:
    basis = Spline(kind="ps", k=7)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        spec = OrderedCategorical(order=LEVELS, basis=basis)

    assert spec.basis == "spline"
    assert spec._spline_obj is basis
    assert spec._spline is not basis
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == basis.n_knots


def test_legacy_shortcuts_with_spline_object_warn_once_that_they_are_ignored() -> None:
    basis = Spline(kind="ps", k=7, degree=2, select=True)

    with pytest.warns(FutureWarning, match=r"ignored.*basis is a Spline object") as caught:
        spec = OrderedCategorical(
            order=LEVELS,
            basis=basis,
            kind="cr",
            n_knots=2,
            degree=1,
            select=False,
            penalty="none",
        )

    assert len(caught) == 1
    assert spec._spline.n_knots == basis.n_knots
    assert spec._spline.degree == basis.degree
    assert spec._spline.select is basis.select
    assert spec._spline.penalty == basis.penalty
    assert spec.degree == basis.degree
    assert spec.select is basis.select
    assert spec.penalty == basis.penalty
    assert spec.n_knots == basis.n_knots


def test_spline_object_wrapper_metadata_matches_canonical_basis() -> None:
    basis = Spline(kind="cr", k=6, penalty="none", select=True)
    spec = OrderedCategorical(order=LEVELS, basis=basis)

    assert spec.kind == "cr"
    assert spec.select is basis.select
    assert spec.penalty == basis.penalty
    assert spec.degree == basis.degree
    assert spec.n_knots == basis.n_knots


@pytest.mark.parametrize(
    ("basis_select", "ignored_select"),
    [(True, False), (False, True)],
)
def test_summary_selection_label_uses_canonical_basis_metadata(
    basis_select, ignored_select
) -> None:
    with pytest.warns(FutureWarning, match="ignored"):
        spec = OrderedCategorical(
            order=LEVELS,
            basis=Spline(kind="ps", k=7, select=basis_select),
            select=ignored_select,
        )
    X = pd.DataFrame({"band": np.tile(LEVELS, 25)})
    y = np.tile(np.linspace(-0.3, 0.4, len(LEVELS)), 25)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": spec},
    )
    model.fit(X, y)

    assert ("SEL" in model.summary()._info["penalty"]) is basis_select


def test_step_basis_warns_once_with_migration_choices_and_still_works() -> None:
    with pytest.warns(FutureWarning, match=r"step smoothing.*removed") as caught:
        spec = OrderedCategorical(order=LEVELS, basis="step")

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "basis=Spline" in message
    assert "Categorical" in message
    assert spec.basis == "step"
    assert spec._spline is None


def test_editor_clone_of_canonical_spline_does_not_warn_about_ignored_shortcuts() -> None:
    spec = OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", k=7))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
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
        warnings.simplefilter("error", FutureWarning)
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


def test_editor_clone_of_deprecated_step_does_not_repeat_user_warning() -> None:
    with pytest.warns(FutureWarning, match="step smoothing"):
        spec = OrderedCategorical(order=LEVELS, basis="step")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        replacement = _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(LEVELS, dtype=object),
        )

    assert replacement.basis == "step"


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
