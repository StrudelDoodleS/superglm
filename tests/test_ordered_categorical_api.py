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


def _ordered_frame(levels: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """A small ordinal frame with a monotone signal across the levels."""
    levels = LEVELS if levels is None else levels
    rng = np.random.default_rng(20260810)
    n = 400
    X = pd.DataFrame({"band": rng.choice(levels, n)})
    position = {level: index / (len(levels) - 1) for index, level in enumerate(levels)}
    y = 0.6 * np.array([position[band] for band in X["band"]]) + rng.normal(0.0, 0.1, n)
    return X, y


def _fit_ordered(spec: OrderedCategorical, X: pd.DataFrame, y: np.ndarray) -> SuperGLM:
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    return model


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


def test_canonical_rewrite_fits_bit_identically_to_the_default_path() -> None:
    """The shape pin above compares constructor parameters only, so it cannot
    observe a behaviour change: a rewrite that agreed on all five parameters
    and still built a different design would pass it unchanged. Fit the same
    data both ways -- the omitted-basis default against the explicit
    ``Spline(kind="ps", n_knots=5)`` migration of the removed ``n_knots=5``
    shortcut -- and require agreement to the BIT rather than to a tolerance.
    A tolerance would absorb exactly the kind of design difference the shape
    pin already cannot see."""
    X, y = _ordered_frame()

    default_path = _fit_ordered(OrderedCategorical(order=LEVELS), X, y)
    rewritten = _fit_ordered(
        OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", n_knots=5)), X, y
    )

    beta_default = np.asarray(default_path._result.beta)
    beta_rewritten = np.asarray(rewritten._result.beta)
    assert beta_default.size > 0
    assert np.max(np.abs(beta_default - beta_rewritten)) == 0.0
    assert float(default_path._result.intercept) == float(rewritten._result.intercept)

    predicted_default = np.asarray(default_path.predict(X), dtype=np.float64)
    predicted_rewritten = np.asarray(rewritten.predict(X), dtype=np.float64)
    assert np.max(np.abs(predicted_default - predicted_rewritten)) == 0.0


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
    # `_spline_obj` is a COPY of the declaration, not the caller's object; see
    # test_mutating_the_caller_s_spline_cannot_change_a_built_spec.
    assert spec._spline_obj is not basis
    assert type(spec._spline_obj) is type(basis)
    assert spec._spline_obj.n_knots == basis.n_knots
    assert spec._spline_obj.degree == basis.degree
    assert spec._spline_obj.penalty == basis.penalty
    assert spec._spline_obj.select == basis.select
    assert spec._spline is not basis
    assert isinstance(spec._spline, PSpline)
    assert spec._spline.n_knots == basis.n_knots


def test_mutating_the_caller_s_spline_cannot_change_a_built_spec() -> None:
    """The caller keeps a reference to the Spline they passed. `_spline_obj` is
    the pristine declaration the editor clone rebuilds from when a grouping is
    undone, so aliasing it let a post-construction mutation change what a later
    collapse produced -- while the fitted `_spline`, already its own copy, kept
    the original geometry. Nothing warned; the two simply disagreed."""
    basis = Spline(kind="ps", n_knots=4)
    spec = OrderedCategorical(order=LEVELS, basis=basis)

    basis.n_knots = 20

    assert spec._spline_obj.n_knots == 4
    assert spec._spline.n_knots == 4

    clone = _ordered_spec_with_grouping(
        spec,
        grouping=None,
        selected_levels=[],
        base="first",
        data=np.asarray(LEVELS, dtype=object),
    )
    assert clone._spline.n_knots == 4


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


def _restored_step_spec(levels: list[str] | None = None) -> OrderedCategorical:
    """What a pre-0.24 step-mode pickle restores.

    Unpickling never runs ``__init__`` — it updates the instance dict
    directly — so building the dict by hand and round-tripping it through
    pickle reproduces exactly what loading an old artifact does.
    """
    levels = ["A", "B", "C"] if levels is None else list(levels)
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
            "_smooth_levels": list(levels),
            "_ordered_levels": list(levels),
            "_level_to_value": {
                level: index / (len(levels) - 1) for index, level in enumerate(levels)
            },
            "_grouping": None,
            "_original_level_to_value": None,
            "_known_levels": set(levels),
            "_n_levels": len(levels),
            "_base_level": levels[0],
            "_non_base": list(levels[1:]),
            "_R_inv": np.eye(len(levels) - 1),
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


def test_step_mode_pickle_is_refused_by_the_editor_apply_path(monkeypatch) -> None:
    """Editing a restored step-mode term must be refused at DISPATCH, before
    any coefficient is written.

    ``_apply_term_edit`` used to route every non-spline OrderedCategorical to a
    one-hot patcher, which writes ``len(_non_base)`` coefficients into the
    fitted block and therefore succeeds whenever the two widths coincide -- as
    they do here by construction: eight levels give seven non-base effects, and
    the fitted P-spline block (``n_knots`` clamped to 4, degree 3) is also seven
    wide. So the edit landed on a P-spline block through the geometry of a basis
    that no longer exists.

    An ``AttributeError`` alone does not pin this: the removed path also
    "raised", but only later and elsewhere, when ``_refresh_fit_statistics``
    scored the edited copy and ``score`` hit ``_basis_spline`` -- by which point
    the coefficients had already been rewritten. Spy on the write itself and
    require that it never happened, and require the message to be the editor's
    own refusal rather than the downstream scoring one.
    """
    from superglm.editor import apply as apply_module
    from superglm.editor._types import EditableTerm
    from superglm.editor.apply import apply_edits_to_model_copy

    writes: list[tuple] = []
    monkeypatch.setattr(
        apply_module, "_patch_beta_block", lambda *args, **kwargs: writes.append(args)
    )
    monkeypatch.setattr(
        apply_module, "_adjust_intercept", lambda *args, **kwargs: writes.append(args)
    )

    X, y = _ordered_frame()
    model = _fit_ordered(OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", n_knots=4)), X, y)
    block = next(group for group in model._groups if group.feature_name == "band")
    # The precondition that let the removed path complete instead of erroring
    # on a width mismatch.
    assert block.size == len(LEVELS) - 1

    # Restoring a pre-0.24 artifact: the fitted state is intact and only the
    # spec is the one step mode left behind.
    model._specs["band"] = _restored_step_spec(LEVELS)
    beta_before = np.asarray(model._result.beta).copy()

    effect = np.linspace(0.0, 0.7, len(LEVELS))
    term = EditableTerm(
        name="band",
        kind="categorical",
        original_log_effect=effect,
        edited_log_effect=effect + 0.25,
        levels=list(LEVELS),
        metadata={"term_type": "ordered categorical"},
    )

    with pytest.raises(AttributeError, match=r"[Ss]tep mode was removed") as excinfo:
        apply_edits_to_model_copy(model, {"band": term})

    assert writes == [], "the edit was applied before the refusal surfaced"
    assert "Editable term 'band'" in str(excinfo.value)
    assert np.array_equal(np.asarray(model._result.beta), beta_before)


def test_step_mode_pickle_is_refused_by_the_summary_row_names() -> None:
    """``_canonical_level_row_names`` kept a step arm that dropped the base
    level -- step geometry. The 0.24.0 removal took its coverage with the mode,
    so a restored artifact silently exported a wrong-SHAPED row set (four names
    where the model has five rows) rather than refusing."""
    from superglm.export.summary import _canonical_level_row_names

    X, y = _ordered_frame()
    model = _fit_ordered(OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", n_knots=4)), X, y)
    assert len(_canonical_level_row_names(model)) == len(LEVELS)

    model._specs["band"] = _restored_step_spec(LEVELS)
    with pytest.raises(AttributeError, match=r"[Ss]tep mode was removed"):
        _canonical_level_row_names(model)


def test_default_path_clamp_warning_states_the_remedy_it_can_name() -> None:
    """With ``basis=`` omitted there is no ``Spline`` in the caller's source, so
    the warning must not read as though they wrote one -- it has to say where
    the number came from and give the declaration that silences it."""
    with pytest.warns(UserWarning, match="clamped") as record:
        OrderedCategorical(order=["low", "medium", "high"])
    message = str(record[0].message)

    assert "No basis= was given" in message
    assert "basis=Spline(kind='ps', n_knots=2)" in message
    assert "clamped to 2" in message


def test_explicit_basis_clamp_warning_names_the_caller_s_own_kind() -> None:
    with pytest.warns(UserWarning, match="clamped") as record:
        OrderedCategorical(order=["low", "medium", "high"], basis=Spline(kind="cr", n_knots=6))
    message = str(record[0].message)

    assert "No basis= was given" not in message
    assert "basis=Spline(kind='cr', n_knots=2)" in message


def test_collapse_clone_does_not_repeat_the_construction_clamp_warning() -> None:
    """Collapsing merges levels, so the caller's own pristine ``n_knots``
    routinely exceeds the collapsed ``n_levels - 1`` and the clone's
    construction clamps it. That is the editor re-fitting the caller's declared
    basis to the levels they just asked to merge, not a configuration mistake
    they can act on, and the user-facing construction already warned if the
    declaration over-specified. Suppress it on the internal clone only -- the
    same treatment the step clone had before 0.24.0 removed it."""
    with pytest.warns(UserWarning, match="clamped"):
        spec = OrderedCategorical(order=["A", "B", "C"], basis=Spline(kind="ps", n_knots=3))
    data = np.asarray(["A", "B", "C"], dtype=object)
    spec.build(np.tile(data, 4))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clone = _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=data,
        )

    # Quiet, not unclamped: the suppression hides the message, not the clamp,
    # and the clone still carries the caller's pristine declaration to re-clamp
    # against its own level count next time.
    assert clone._spline.n_knots == 2
    assert clone._spline_obj.n_knots == 3


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


def test_pickle_without_a_spline_obj_key_still_names_the_migration() -> None:
    """A pickle old enough to predate ``_spline_obj`` restores a ``__dict__``
    with no such key at all. An attribute-style read only falls back when the
    key EXISTS and is None, so that spec raised a bare ``AttributeError``
    naming ``_spline_obj`` -- loud, but silent about the migration, and it
    never reached ``_basis_spline`` where the sentence lives."""
    spec = _restored_step_spec()
    del spec.__dict__["_spline_obj"]
    assert "_spline_obj" not in spec.__dict__

    with pytest.raises(AttributeError, match=r"[Ss]tep mode was removed"):
        _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(["A", "B", "C"], dtype=object),
        )


def test_shortcut_pickle_ungroup_recovers_the_requested_knot_count() -> None:
    """Ungrouping a shortcut-era pickle must restore the REQUESTED knot count.

    A current-era spec keeps ``_spline_obj``, the caller's pristine
    declaration, so re-clamping against a larger level count is automatic. A
    pre-0.24 shortcut pickle has no such declaration -- only the inner spline,
    whose ``n_knots`` was already clamped to the level count it was BUILT
    against. Cloning that alone silently keeps the reduced basis, which is
    wrong in exactly the direction ungrouping goes: back to MORE levels.

    The removed shortcut path rebuilt from the then-plain ``n_knots``
    attribute, i.e. the count the caller asked for. That entry survives in the
    pickled ``__dict__`` -- the class property only shadows it -- so it can be
    recovered.
    """
    original = [f"B{index}" for index in range(8)]
    collapsed = ["B0+B1+B2", "B3+B4", "B5+B6+B7"]

    # The collapsed model: the caller asked for 7 knots, three levels clamp it
    # to two.
    with pytest.warns(UserWarning, match="clamped"):
        spec = OrderedCategorical(order=collapsed, basis=Spline(kind="ps", n_knots=7))
    assert spec._spline.n_knots == 2

    # Make it a shortcut-era pickle OF THAT COLLAPSED MODEL: no pristine
    # declaration survives, the then-plain attribute carries the requested 7,
    # and the original eight levels are what ungrouping restores.
    spec.__dict__["_spline_obj"] = None
    spec.__dict__["n_knots"] = 7
    spec.__dict__["_original_level_to_value"] = {
        level: index / (len(original) - 1) for index, level in enumerate(original)
    }
    spec = pickle.loads(pickle.dumps(spec))
    assert spec.n_knots == 2  # the property still reports the clamped inner value

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(original, dtype=object),
        )

    assert sorted(restored._known_levels) == sorted(original)
    # 7 comes back and clamps only as the NEW level count dictates -- eight
    # levels allow seven -- instead of staying stuck at the collapsed two.
    assert restored._spline.n_knots == 7
    assert restored._spline_obj.n_knots == 7


def test_shortcut_pickle_clone_still_clamps_to_the_new_level_count() -> None:
    """Recovering the requested count must not bypass the clamp: rebuilding
    for FEWER levels than the request still clamps, and stays quiet because it
    is an internal clone."""
    spec = OrderedCategorical(order=[f"B{index}" for index in range(8)], basis=Spline(n_knots=7))
    spec.__dict__["_spline_obj"] = None
    spec.__dict__["n_knots"] = 7
    spec.__dict__["_original_level_to_value"] = {"B0": 0.0, "B1": 0.5, "B2": 1.0}
    spec = pickle.loads(pickle.dumps(spec))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = _ordered_spec_with_grouping(
            spec,
            grouping=None,
            selected_levels=[],
            base="first",
            data=np.asarray(["B0", "B1", "B2"], dtype=object),
        )

    assert restored._spline.n_knots == 2  # three levels, so n_levels - 1
