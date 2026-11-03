"""Every reader of a term's shape rules sees the whole collection, not one string.

Rules join with ``&``. A reader that still expects one constraint string drops the
rest, or prints a mangled command. These tests pin the readers that print the mask
line (summary, term inference, coefficient rows), refuse by rule (the editor, the
rating table, a basis that cannot fit rules) and load a model pickled before
collections existed. SuperLSS warns by rule in test_distributional_shape_constraint_warning.
"""

from __future__ import annotations

import copy
import html
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.export._ppform import extract_ppform
from superglm.export.rating_tables import build_rating_table_payload
from superglm.features._spline_config import SCOPRetiredWarning
from superglm.features.constraint import ConstraintSpec, rules_of
from superglm.features.rebuild import shape_unavailable_reason
from tests.test_editor_knots import BANDS
from tests.test_editor_knots import _book as _editor_book
from tests.test_editor_knots import _declared as _editor_declared
from tests.test_saved_fs_models import assert_predicts_as_saved

C = Constraint.fit
_BANDS = [f"B{i}" for i in range(6)]
_FIXTURES = Path(__file__).parent / "fixtures"
_CANONICAL = ("constraint_rules", "constraint_kind", "constraint_mode")
_LEGACY = ("monotone", "monotone_mode")


def _rows(n: int = 2500, seed: int = 11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 100.0, n)
    band = rng.choice(_BANDS, n)
    region = rng.choice(["A", "B", "C"], n)
    index = np.array([_BANDS.index(b) for b in band])
    eta = -1.0 + 0.012 * x + 0.15 * index + 0.1 * (region == "B")
    frame = pd.DataFrame({"x": x, "band": band, "region": region})
    return frame, rng.poisson(np.exp(eta)).astype(float)


def _fit(features):
    X, y = _rows()
    return SuperGLM(family="poisson", selection_penalty=0.0, features=features).fit(X, y), X, y


def _as_pickled(model, keep: str, term: str = "x"):
    """A copy of ``model`` whose ``term`` carries only the ``keep`` spelling of its rules.

    A model pickled before collections existed holds the legacy ``monotone`` pair
    (``keep="legacy"``); the legacy pair is also the spelling a later release may drop
    (``keep="canonical"``).
    """
    copied = copy.deepcopy(model)
    for attr in _LEGACY if keep == "canonical" else _CANONICAL:
        copied._specs[term].__dict__.pop(attr, None)
    return copied


@pytest.fixture(scope="module")
def collection_model():
    return _fit({"x": Spline(kind="ps", k=7, constraint=C.increasing & C.concave)})


@pytest.fixture(scope="module")
def single_model():
    return _fit({"x": Spline(kind="ps", k=7, constraint=C.increasing)})


def _x_row(model, X, y):
    return next(row for row in model.metrics(X, y)._build_coef_rows() if row.name == "x")


# -- The mask line --------------------------------------------------------------------


def test_summary_prints_mask_line(collection_model):
    model, X, y = collection_model
    assert "mono=increasing & concave" in str(model.summary())
    assert model.term_inference("x", with_se=False).monotone == "increasing & concave"
    assert _x_row(model, X, y).monotone == "increasing & concave"


@pytest.mark.parametrize(
    ("keep", "model_name", "line"),
    [
        ("canonical", "collection_model", "increasing & concave"),
        ("legacy", "single_model", "increasing"),
    ],
)
def test_term_inference_and_summary_read_the_rules_in_either_spelling(
    keep, model_name, line, request
):
    """Mutation: reading the legacy ``monotone`` field prints no line once it is dropped."""
    model, X, y = request.getfixturevalue(model_name)
    model = _as_pickled(model, keep)
    assert model.term_inference("x", with_se=False).monotone == line
    assert f"mono={line}" in str(model.summary())
    assert _x_row(model, X, y).monotone == line


@pytest.mark.parametrize(
    "rules", [C.increasing, C.increasing & C.cap(0.5, 2.5)], ids=["single", "collection"]
)
def test_coef_tables_ordered_and_spline_agree(rules):
    """An ordered term and a spline with the same rules print the same line.

    The spline's legacy field is dropped, so the row must come from its rules.
    """
    model, X, y = _fit(
        {
            "x": Spline(kind="ps", k=7, constraint=rules),
            "band": OrderedCategorical(
                order=_BANDS, basis=Spline(kind="ps", k=5, constraint=rules)
            ),
        }
    )
    model = _as_pickled(model, "canonical")
    lines = {
        row.name: row.monotone
        for row in model.metrics(X, y)._build_coef_rows()
        if row.name in {"x", "band"}
    }
    assert lines == {"x": rules.describe(), "band": rules.describe()}


def _rule_lines(model, term: str, line: str) -> None:
    """``term``'s rule line is in its term inference, the console summary and the HTML summary."""
    assert model.term_inference(term, with_se=False).monotone == line
    summary = model.summary()
    assert f"mono={line}" in str(summary)
    assert html.escape(f"mono={line}") in summary._repr_html_()


@pytest.mark.parametrize(
    "rules", [C.increasing & C.cap(0.8, 1.25), C.increasing], ids=["collection", "single"]
)
def test_summary_shows_the_rules_of_an_ordered_term(rules):
    """An ordered term hosts its rules on a basis; its summary row and inference still say them."""
    basis = Spline(kind="ps", k=5, constraint=rules)
    model, _, _ = _fit({"band": OrderedCategorical(order=_BANDS, basis=basis)})
    _rule_lines(model, "band", rules.describe())


def test_summary_shows_the_cap_of_a_categorical():
    """A capped categorical has no spline row: its rule line sits under the term's heading."""
    model, X, y = _fit({"region": Categorical(constraint=C.cap(0.8, 1.25))})
    _rule_lines(model, "region", "cap(0.8, 1.25)")
    lines = str(model.summary()).splitlines()
    (note,) = [i for i, text in enumerate(lines) if "mono=cap(0.8, 1.25) (qp)" in text]
    heading = next(i for i, text in enumerate(lines) if " region " in text and "╡" in text)
    first_level = next(i for i, text in enumerate(lines) if text.lstrip("║ ").startswith("region["))
    assert heading < note < first_level
    rows = [row for row in model.metrics(X, y)._build_coef_rows() if row.group == "region"]
    assert rows and {row.monotone for row in rows} == {"cap(0.8, 1.25)"}


def test_terms_without_rules_print_no_rule_line():
    """A term with no rules, or an uncapped categorical, shows no line, row field or engine."""
    model, X, y = _fit(
        {
            "x": Spline(kind="ps", k=7),
            "band": OrderedCategorical(order=_BANDS, basis=Spline(kind="ps", k=5)),
            "region": Categorical(),
        }
    )
    for term in ("x", "band", "region"):
        assert model.term_inference(term, with_se=False).monotone is None
    rows = model.metrics(X, y)._build_coef_rows()
    assert {(row.monotone, row.monotone_engine) for row in rows} == {(None, None)}
    summary = model.summary()
    assert "mono=" not in str(summary) and "mono=" not in summary._repr_html_()


# -- The editor -----------------------------------------------------------------------

_NO_NS = (
    "A natural spline takes no shape constraint, and {term!r} has one; choose another kind, "
    "or remove the constraint in code."
)
_NO_SHRINK = (
    "Shrinkage cannot be combined with a shape constraint the fit enforces, which {term!r} "
    "has; leave Shrink off, or apply the constraint after the fit (Constraint.postfit) in code."
)
_NO_SHRINK_RULES = (
    "Shrinkage cannot be combined with the shape rules the fit enforces, which {term!r} has "
    "({rules}); leave Shrink off. Constraint.postfit repairs one whole-curve shape and cannot "
    "take them."
)


def _refused(session, term: str, params) -> str:
    with pytest.raises(EditorValueError) as refused:
        session.stage_structural("basis", term, params)
    return str(refused.value)


def _editor_term(term: str, rules):
    """The opened editor book fitted with ``rules`` on its ``age`` spline or ``band`` basis."""
    X, y, w = _editor_book(n=3000)
    if term == "age":
        declared = _editor_declared(age=Spline(kind="cr", n_knots=4, constraint=rules))
    else:
        basis = Spline(kind="ps", n_knots=3, constraint=rules)
        declared = _editor_declared(band=OrderedCategorical(order=BANDS, basis=basis))
    model = declared.fit(X, y, sample_weight=w)
    return model, EditorSession.from_model(model, train_data=(X, y, w))


@pytest.mark.parametrize(
    ("term", "one", "many"),
    [
        ("age", C.decreasing, C.decreasing & C.convex.above(50)),
        ("band", C.increasing, C.increasing & C.cap(0.8, 1.25)),
    ],
)
def test_editor_refusals_with_collections(term, one, many):
    """A collection is refused Shrink and a natural spline as one rule is, with advice that holds."""
    cases = (
        (one, _NO_SHRINK.format(term=term)),
        (many, _NO_SHRINK_RULES.format(term=term, rules=many.describe())),
    )
    for rules, shrink in cases:
        model, session = _editor_term(term, rules)
        assert _refused(session, term, {"kind": "ns"}) == _NO_NS.format(term=term)
        assert _refused(session, term, {"select": True}) == shrink
        assert shape_unavailable_reason(model, term) == (
            "Remove the term's shape constraint to add shaped ranges."
        )


def test_editor_basis_change_keeps_every_rule():
    X, y, w = _editor_book(n=3000)
    rules = C.decreasing & C.convex.above(50)
    model = _editor_declared(age=Spline(kind="cr", n_knots=4, constraint=rules)).fit(
        X, y, sample_weight=w
    )
    session = EditorSession.from_model(model, train_data=(X, y, w))
    session.replace_with_basis("age", {"kind": "bs"})
    assert rules_of(session.model._specs["age"]) == rules


# -- Old pickles ----------------------------------------------------------------------

_SINGLE = ConstraintSpec("fit", "increasing")
_POSTFIT = ConstraintSpec("postfit", "increasing")
# Every fixture holding a constrained term, with the one rule each term was saved with.
_CONSTRAINED = {
    "saved_31544462/editor_halved_spline.pkl": {"x": _POSTFIT},
    "saved_master_71b133b2/pspline_explicit_scop.pkl": {"x": _SINGLE},
    "saved_v0_35_0/editor_halved_spline.pkl": {"x": _POSTFIT},
    "saved_v0_40_0/glm_poisson_qp.pkl": {"x": _SINGLE},
    "saved_v0_40_0/glm_poisson_scop.pkl": {"x": _SINGLE},
    "saved_v0_41_0/kindless_beside_scop.pkl": {"x": _SINGLE, "z": _SINGLE},
    "saved_v0_41_0/ordered_kindless_beside_scop.pkl": {"band": _SINGLE, "z": _SINGLE},
    "saved_v0_41_0/ordered_ps_on_scop.pkl": {"x": _SINGLE, "band": _SINGLE},
}


def _load(relative: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(_FIXTURES / relative, "rb") as handle:
            return pickle.load(handle)


@pytest.mark.parametrize("relative", sorted(_CONSTRAINED))
def test_old_pickles_unchanged(relative):
    """A model saved by an earlier release predicts as saved and holds its one rule."""
    record = _load(relative)
    model = record["model"]
    for term, rule in _CONSTRAINED[relative].items():
        assert tuple(rules_of(model._specs[term])) == (rule,), term
    with warnings.catch_warnings():
        warnings.simplefilter("error", SCOPRetiredWarning)  # reading a saved model never warns
        assert_predicts_as_saved(model, record["frame"], record["prediction"])
        model.summary()


def test_every_constrained_fixture_is_listed():
    found = {}
    for path in sorted(_FIXTURES.glob("saved_*/*.pkl")):
        relative = str(path.relative_to(_FIXTURES))
        record = _load(relative)
        model = record.get("model") if isinstance(record, dict) else record
        for term, spec in (getattr(model, "_specs", None) or {}).items():
            if rules_of(spec):
                found.setdefault(relative, []).append(term)
    assert {k: sorted(v) for k, v in found.items()} == {
        k: sorted(v) for k, v in _CONSTRAINED.items()
    }


# -- The rating table -----------------------------------------------------------------


def _age_rows(n: int = 2000, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(18.0, 80.0, n)
    y = rng.poisson(np.exp(-1.0 + 0.03 * (x - 18.0))).astype(float)
    return pd.DataFrame({"age": x}), y


@pytest.mark.parametrize("keep", ["canonical", "legacy"])
def test_ppform_refusal_follows_the_rules_in_either_spelling(keep):
    """A postfit term is refused whichever spelling its rules are saved in."""
    X, y = _age_rows()
    spec = Spline(kind="ps", n_knots=8, constraint=Constraint.postfit.increasing)
    model = SuperGLM(family="poisson", features={"age": spec}).fit(X, y)
    saved = _as_pickled(model, keep, term="age")
    with pytest.raises(ValueError, match="'age' carries a postfit increasing constraint"):
        build_rating_table_payload(saved, X, y, continuous_kind="ppform")


@pytest.mark.parametrize(
    ("kind", "rules"),
    [
        ("ps", C.increasing & C.concave),
        ("cr", C.decreasing.below(40) & C.flat.within(40, 60)),
        ("bs", C.increasing & C.cap(0.5, 2.5)),
    ],
    ids=["mask", "flat", "cap"],
)
@pytest.mark.filterwarnings("ignore:.*whole knot spans it touches")
def test_ppform_converts_fit_time_rule_collections(kind, rules):
    """The segments are the fitted curve, so a fit-time collection exports exactly."""
    X, y = _age_rows()
    spec = Spline(kind=kind, n_knots=8, constraint=rules)
    model = SuperGLM(family="poisson", features={"age": spec}).fit(X, y)
    build_rating_table_payload(model, X, y, continuous_kind="ppform")
    curve = model.term_inference("age", n_points=401, with_se=False)
    evaluated = extract_ppform(model, "age").evaluate(np.asarray(curve.x))
    assert np.abs(evaluated - np.asarray(curve.log_relativity)).max() < 1e-12


# -- A basis that cannot fit rules ----------------------------------------------------


def test_unsupported_fit_time_rules_advise_what_the_basis_can_take():
    """Advice names Constraint.postfit only for the one shape it can repair."""
    X, y = _age_rows()

    def message(rules):
        spec = Spline(kind="cr_cardinal", n_knots=6, constraint=rules)
        with pytest.raises(NotImplementedError) as refused:
            SuperGLM(family="poisson", features={"age": spec}).fit(X, y)
        return str(refused.value)

    assert message(C.increasing) == (
        "Fit-time increasing constraints are not implemented yet. "
        "Use Constraint.postfit.increasing instead."
    )
    for rules in (C.increasing & C.concave, C.cap(0.5, 2.0)):
        text = message(rules)
        assert f"Fit-time {rules.describe()} constraints are not implemented" in text
        assert "Constraint.postfit." not in text, text
        assert "kind='cr'" in text and "kind='ps'" in text and "kind='bs'" in text


def _refusal_text(where: str, rules) -> str:
    """The NotImplementedError a spline or ordered term meets for ``rules`` at ``where``."""
    with pytest.raises(NotImplementedError) as refused:
        if where == "degree":
            Spline(kind="ps", n_knots=8, degree=4, constraint=rules)
        elif where == "spline-select":
            X, y = _age_rows()
            spec = Spline(kind="ps", n_knots=8, select=True, constraint=rules)
            SuperGLM(family="poisson", features={"age": spec}).fit(X, y)
        else:
            frame, counts = _rows()
            basis = Spline(kind="ps", k=5, select=True, constraint=rules)
            band = OrderedCategorical(order=_BANDS, basis=basis)
            SuperGLM(family="poisson", features={"band": band}).fit(frame, counts)
    return str(refused.value)


@pytest.mark.parametrize(
    ("where", "single", "many"),
    [
        ("spline-select", C.increasing, C.increasing & C.concave),
        ("ordered-select", C.increasing, C.increasing & C.cap(0.8, 1.25)),
        ("degree", C.convex, C.increasing & C.convex),
    ],
)
def test_postfit_advice_is_given_only_where_it_is_true(where, single, many):
    """Constraint.postfit is offered for one whole-curve shape, not for a combination or a cap."""
    refused = _refusal_text(where, many)
    assert "Constraint.postfit." not in refused, refused
    assert f"cannot take {many.describe()}" in refused, refused
    assert f"Constraint.postfit.{single.kind}" in _refusal_text(where, single)
