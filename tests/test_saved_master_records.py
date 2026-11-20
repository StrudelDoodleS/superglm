"""Models master ``a4a7a64cd`` saved, before the identifiability record had a lifecycle.

The fixtures in ``fixtures/saved_master_a4a7a64cd`` were written by master ``a4a7a64cd`` with
``scripts/make_saved_master_a4a7a64cd_fixtures.py``: each case's rows, an off-support frame,
four pickles (kept fit state or compact, saved before or after ``summary()`` and
``metrics()``) and what master read from each.  A model that kept its fit state is decided
again at its saved coefficients on its first inference (``model.saved_record``); a compact
one keeps the inference it was saved with and says so.  Nothing a model predicts moves.
"""

from __future__ import annotations

import dataclasses
import lzma
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from .test_saved_fs_models import _gamma, _se_tolerance, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_master_a4a7a64cd"
CASES = [
    "dense_full",
    "duplicate",
    "reference_level",
    "near_duplicate",
    "sz_one_row",
    "random_effect",
]
VARIANTS = ["warm_cold", "warm_read", "compact_cold", "compact_read"]
U = 2.0**-53
FROZEN = "Standard errors as saved by an earlier superglm"


def _record(case: str) -> dict:
    with lzma.open(FIXTURES / f"{case}.pkl.xz", "rb") as handle:
        return pickle.load(handle)


def _loads(blob: bytes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pickle.loads(blob)


def _quiet(call, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args)


def _summary(model) -> tuple[list[str], np.ndarray, str]:
    """The summary's coefficient rows (names, SEs with NaN for a blank) and its footnotes."""
    summary = _quiet(model.summary)
    rows = [r for r in summary._coef_rows if not r.is_spline and r.structured_kind is None]
    se = np.array([np.nan if r.se is None else float(r.se) for r in rows])
    notes = "\n".join(summary._info.get("identifiability_notes") or ())
    return [str(r.name) for r in rows], se, notes


def _metrics_se(model, X, y) -> dict[str, np.ndarray]:
    metrics = _quiet(model.metrics, X, y)
    se = {k: np.asarray(v, dtype=float) for k, v in metrics.coefficient_se_raw.items()}
    se["Intercept"] = np.array([float(metrics.intercept_se_raw)])
    return se


def _master_summary(output) -> tuple[list[str], np.ndarray]:
    keep = ~np.asarray(output["summary_spline"])
    names = [n for n, k in zip(output["summary_names"], keep, strict=True) if k]
    return names, np.asarray(output["summary_se"])[keep]


def _rows(model, X, y):
    """The fit's own rows when it kept them (``metrics`` reads its own geometry), else ``X, y``."""
    retained = getattr(model, "_fit_X_ref", None)
    return (X, y) if retained is None else (retained, model._fit_y_ref)


# ------------------------------------------------------------------ the record's lifecycle
def test_the_record_lifecycle_walks_its_table_and_a_saved_model_takes_its_moves():
    """``RECORD_MOVES`` is the only source of the record's tag: every listed move lands where
    the table says and every other event is refused.  The events themselves: a fit publishes
    (``decided``), an edit revises (``stale``), a refit publishes again; master's warm pickle
    loads ``legacy`` and its first inference decides it, so ``x2`` (held at zero by master's
    truncation) is the one blank and ``x1`` has the standard error of the fit with ``x2``
    held; master's compact pickle loads ``frozen``.  At ``253c6d353`` the warm model kept
    master's data-only flags and blanked ``x1`` too."""
    from superglm.model import state_ops

    class Holder:
        pass

    events = sorted({event for moves in state_ops.RECORD_MOVES.values() for event in moves})
    for tag, moves in state_ops.RECORD_MOVES.items():
        for event in events:
            holder = Holder()
            holder.__dict__["_result"] = object()
            holder.__dict__["_record_tag"] = tag
            if tag == "saved":
                del holder.__dict__["_record_tag"]
            if event in moves:
                assert state_ops.move_record(holder, event) == moves[event]
                assert holder.__dict__["_record_tag"] == moves[event]
            else:
                with pytest.raises(RuntimeError, match="is not a move"):
                    state_ops.move_record(holder, event)
    # every tag but the two a model starts in is reached by a move
    assert {target for moves in state_ops.RECORD_MOVES.values() for target in moves.values()} == {
        tag for tag in state_ops.RECORD_MOVES if tag not in ("saved", "unfitted")
    }

    record = _record("duplicate")
    warm = _loads(record["models"]["warm_cold"])
    assert state_ops.record_tag(warm) == "legacy"
    names, se, notes = _summary(warm)
    assert state_ops.record_tag(warm) == "decided"
    blank = [name for name, value in zip(names, se, strict=True) if np.isnan(value)]
    assert blank == ["x2"]
    assert "x2 held at zero" in notes
    compact = _loads(record["models"]["compact_cold"])
    assert state_ops.record_tag(compact) == "frozen"

    from superglm import Numeric, Spline, SuperGLM
    from superglm.editor import EditorSession

    X, y = record["frame"], record["y"]
    features = {"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}
    model = SuperGLM(family="poisson", features=features, selection_penalty=0.0)
    assert state_ops.record_tag(model) == "unfitted"
    _quiet(model.fit, X, y)
    assert state_ops.record_tag(model) == "decided"
    session = EditorSession.from_model(model, terms=["x"])
    session.select_x("x", 0.2, 0.6)
    session.shift("x", 0.35)
    edited = _quiet(session.to_model)
    assert state_ops.record_tag(edited) == "stale" and edited._editor_inference_stale
    _quiet(edited.fit, X, y)
    assert state_ops.record_tag(edited) == "decided"


# ------------------------------------------------------------------ what a saved model predicts
@pytest.mark.pins_behaviour
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES)
def test_a_saved_model_keeps_its_coefficients_and_predictions(case, variant):
    """Pins what loading leaves alone: the published coefficients bit for bit, and the
    predictions on the training rows and off them within two evaluations' rounding
    (``assert_predicts_as_saved``: ``2 gamma_(p+1)`` of each row's ``|b0| + sum |x_j beta_j|``,
    carried through the inverse link).  At ``253c6d353`` loading moved nothing either; the
    readers, which do not move them now (``test_both_readers_agree_in_either_order...``),
    are tested there."""
    record = _record(case)
    output = record["outputs"][variant]
    model = _loads(record["models"][variant])
    _assert_as_saved(model, record, output)


def _assert_as_saved(model, record, output) -> None:
    assert float(model.result.intercept) == output["intercept"]
    np.testing.assert_array_equal(model.result.beta, output["beta"])
    assert float(model.result.effective_df) == output["effective_df"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_predicts_as_saved(model, record["frame"], output["predict"])
        assert_predicts_as_saved(model, record["off"], output["predict_off"])


# ------------------------------------------------------------------ both readers, either order
def _labelled(model, se: dict[str, np.ndarray]) -> dict[str, float]:
    """``metrics``' SEs under the summary's row names, for the terms it gives a row each."""
    specs = model._specs
    named = {"Intercept": float(se["Intercept"][0])}
    for group in model._groups:
        levels = getattr(specs.get(group.feature_name), "_non_base", None)
        values = se.get(group.name)
        if values is None:
            continue
        if group.name == group.feature_name and levels is not None and len(levels) == group.size:
            named.update({f"{group.name}[{level}]": float(v) for level, v in zip(levels, values)})
        elif group.size == 1:
            named[group.name] = float(values[0])
    return named


def _readers(model, record, *, metrics_first: bool):
    rows = _rows(model, record["frame"], record["y"])
    if metrics_first:
        metrics = _metrics_se(model, *rows)
        summary = _summary(model)
    else:
        summary = _summary(model)
        metrics = _metrics_se(model, *rows)
    return summary, metrics


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES)
def test_both_readers_agree_in_either_order_and_after_a_repickle(case, variant):
    """``summary()`` and ``metrics()`` blank the same coefficients, whichever runs first, and a
    model pickled again after reading them (its tag with it) reads the same numbers; nothing
    either reads moves a coefficient or a prediction.  A control against master, whose own
    code read its pickles consistently: it pins that this branch's migration keeps that.  At
    ``253c6d353`` the ``sz`` model's saved covariance accessor raised in both readers
    (``AttributeError: ... 'gauge'``) on the compact pickles and on the warm one saved after
    its readers."""
    record = _record(case)
    blob = record["models"][variant]
    first, second = _loads(blob), _loads(blob)
    (names, summary_se, notes), metrics = _readers(first, record, metrics_first=False)
    (names_b, summary_b, notes_b), metrics_b = _readers(second, record, metrics_first=True)
    assert names == names_b and notes == notes_b
    np.testing.assert_array_equal(summary_se, summary_b)
    assert metrics.keys() == metrics_b.keys()
    for term in metrics:
        np.testing.assert_array_equal(metrics[term], metrics_b[term])
    labelled = _labelled(first, metrics)
    for name, value in zip(names, summary_se, strict=True):
        if name in labelled:
            assert np.isnan(value) == np.isnan(labelled[name]), name
    again = _loads(pickle.dumps(first))
    assert again.__dict__.get("_record_tag") == first.__dict__.get("_record_tag")
    (names_c, summary_c, notes_c), metrics_c = _readers(again, record, metrics_first=True)
    assert names_c == names and notes_c == notes
    np.testing.assert_array_equal(summary_c, summary_se)
    for term in metrics:
        np.testing.assert_array_equal(metrics_c[term], metrics[term])
    for model in (first, second, again):
        _assert_as_saved(model, record, record["outputs"][variant])


# ------------------------------------------------------------------ a warm model, decided again
def _poisson_hessian(model, X, withheld=None) -> tuple[np.ndarray, np.ndarray]:
    """``[1, X]' W [1, X] + S`` at the saved coefficients (Poisson, log link: ``W = mu``), dense.

    ``mu`` is the model's prediction on its training rows; ``withheld`` rows weigh zero.  Also
    returns the design, whose columns are ``[intercept, slopes]`` in fitted coordinates.
    """
    from .test_penalised_existence import _design_and_penalty

    design, penalty = _design_and_penalty(model)
    mu = np.asarray(_quiet(model.predict, X), dtype=np.float64)
    if withheld is not None:
        mu = np.where(withheld, 0.0, mu)
    return design.T @ (mu[:, None] * design) + penalty, design


def _reduced_se(hessian: np.ndarray, kept: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """``sqrt(diag H_kk^-1)`` on the coordinates ``kept``, and ``n u kappa`` of the Jacobi-scaled
    ``H_kk`` (``_se_tolerance``'s bound on two routes' agreement)."""
    block = hessian[np.ix_(kept, kept)]
    scale = 1.0 / np.sqrt(np.diag(block))
    kappa = np.linalg.cond(scale[:, None] * block * scale[None, :])
    return np.sqrt(np.diag(np.linalg.inv(block))), float(_gamma(n) * kappa)


def _column(model, name: str) -> int:
    """The fitted coordinate (``[intercept, slopes]`` index) of a one-column term."""
    group = next(g for g in model._groups if g.name == name)
    return 1 + group.start


@pytest.mark.parametrize("variant", ["warm_cold", "warm_read"])
def test_a_saved_duplicate_column_reads_the_gauge_its_coefficients_satisfy(variant):
    """``x2 = x1`` exactly: master's truncation held ``x2`` at zero, the constraint the fit now
    publishes too, so the record reads it.  ``x2`` alone is blank; ``x1``'s SE is the fit's with
    ``x2`` held, the dense ``H`` without ``x2`` inverted at the saved coefficients (to ``n u
    kappa``), and the footnote names ``x2``.  At ``253c6d353`` the warm model kept master's
    data-only flags: ``x1`` was blank too."""
    record = _record("duplicate")
    model = _loads(record["models"][variant])
    names, se, notes = _summary(model)
    assert [n for n, v in zip(names, se, strict=True) if np.isnan(v)] == ["x2"]
    assert "x2 is aliased with x1" in notes and "(x2 held at zero)" in notes
    hessian, design = _poisson_hessian(model, record["frame"])
    kept = np.setdiff1d(np.arange(design.shape[1]), [_column(model, "x2")])
    oracle, tolerance = _reduced_se(hessian, kept, design.shape[0])
    expected = oracle[list(kept).index(_column(model, "x1"))]
    assert abs(se[names.index("x1")] / expected - 1.0) <= tolerance
    _assert_as_saved(model, record, record["outputs"][variant])


@pytest.mark.parametrize("variant", ["warm_cold", "warm_read"])
def test_a_saved_near_duplicate_withholds_what_its_coefficients_never_fitted(variant):
    """``x2 = x1 + 1e-8 h``: the rows identify the pair's difference, which master's solver
    truncated, so the saved coefficients were never fitted along it.  ``x1`` and ``x2`` have no
    SE (the rows' would be about 2e6, beside a truncated coefficient), the footnote says why,
    and the EDF is the saved one.  At ``253c6d353`` the summary blanked them with no word of
    why (master's data-only flags)."""
    record = _record("near_duplicate")
    model = _loads(record["models"][variant])
    names, se, notes = _summary(model)
    assert [n for n, v in zip(names, se, strict=True) if np.isnan(v)] == ["x1", "x2"]
    assert "never fitted along it: x1, x2 have no standard error" in notes
    metrics = _metrics_se(model, *_rows(model, record["frame"], record["y"]))
    assert np.isnan(metrics["x1"]).all() and np.isnan(metrics["x2"]).all()
    assert np.isfinite(metrics["z"]).all() and np.isfinite(metrics["Intercept"]).all()
    assert float(model.result.effective_df) == record["outputs"][variant]["effective_df"]


@pytest.mark.parametrize("variant", ["warm_cold", "warm_read"])
def test_a_saved_separated_reference_level_reads_its_kept_rows_at_the_saved_coefficients(variant):
    """C3: reference level E has every response zero.  Existence is decided from the saved
    rows: the intercept and every level of ``f`` have no SE, and ``x``'s SEs are those of the
    kept rows' Hessian at the saved coefficients (E's rows at weight zero; ``f[A]`` dropped,
    which leaves ``x``'s variances unchanged since no separated direction reaches ``x``), to
    ``n u kappa``.  The coefficients stay master's walk, and the footnote says so with the kept
    rows' Newton decrement.  At ``253c6d353`` the summary printed master's walked SE 1.2e5 on
    the intercept and on each level."""
    record = _record("reference_level")
    model = _loads(record["models"][variant])
    names, se, notes = _summary(model)
    assert [n for n, v in zip(names, se, strict=True) if np.isnan(v)] == [
        "Intercept",
        "f[A]",
        "f[B]",
        "f[C]",
        "f[D]",
    ]
    assert "level E of f" in notes and "Newton decrement" in notes
    metrics = _metrics_se(model, *_rows(model, record["frame"], record["y"]))
    withheld = np.asarray(record["frame"]["f"] == "E")
    hessian, design = _poisson_hessian(model, record["frame"], withheld)
    level_a = next(g for g in model._groups if g.name == "f").start + 1
    kept = np.setdiff1d(np.arange(design.shape[1]), [level_a])
    oracle, tolerance = _reduced_se(hessian, kept, design.shape[0])
    spline = next(g for g in model._groups if g.name == "x")
    expected = oracle[[list(kept).index(1 + c) for c in range(spline.start, spline.end)]]
    np.testing.assert_allclose(metrics["x"], expected, rtol=tolerance, atol=0.0)
    _assert_as_saved(model, record, record["outputs"][variant])


@pytest.mark.parametrize("variant", ["warm_cold", "warm_read"])
def test_a_saved_sz_fit_blanks_what_its_aliases_reach_in_no_reproducible_gauge(variant):
    """Three one-row ``sz`` levels leave two directions of ``null(H)``.  Master's coefficients
    satisfy neither the constraints the fit now publishes nor the retained factor's own
    minimum-norm split (``|beta - H^+ H beta|`` is 4e-2 of ``|beta|`` in its Jacobi scale,
    under master's code as under this one), so every coordinate the directions reach has no
    SE: 22 of ``x:g:sz``'s and the intercept, whose SE master printed (6.44) depends on the
    split; ``x``'s are given.  At ``253c6d353`` the intercept had that SE, and the warm model
    saved after its readers raised on master's covariance accessor."""
    record = _record("sz_one_row")
    model = _loads(record["models"][variant])
    names, se, notes = _summary(model)
    assert np.isnan(se[names.index("Intercept")])
    assert "split an aliased direction" in notes and "The intercept has no standard error" in notes
    metrics = _metrics_se(model, *_rows(model, record["frame"], record["y"]))
    assert np.count_nonzero(np.isnan(metrics["x:g:sz"])) == 22
    assert np.isfinite(metrics["x"]).all() and np.isnan(metrics["Intercept"]).all()
    _assert_as_saved(model, record, record["outputs"][variant])


# ------------------------------------------------------------------ a compact model, frozen
def _range_tolerance(model) -> float:
    """``n u kappa`` of the Jacobi-scaled penalised Hessian on its range, from the dense rows.

    ``H = [1, X]' W [1, X] + S`` at the fit's working weights; its exact null directions (an
    alias) sit at the formation bound, ``n u`` of the largest eigenvalue, and are left out.
    """
    from superglm.model.state_ops import _solver_space_working_weights

    from .test_penalised_existence import _design_and_penalty

    design, penalty = _design_and_penalty(model)
    weights = _solver_space_working_weights(model)
    hessian = design.T @ (weights[:, None] * design) + penalty
    scale = 1.0 / np.sqrt(np.diag(hessian))
    values = np.linalg.eigvalsh(scale[:, None] * hessian * scale[None, :])
    n = design.shape[0]
    kept = values[values > _gamma(n) * values[-1]]
    return float(_gamma(n) * kept[-1] / kept[0])


@pytest.mark.parametrize("variant", ["compact_cold", "compact_read"])
@pytest.mark.parametrize("case", CASES)
def test_a_compact_saved_model_reads_its_saved_inference_and_says_so(case, variant):
    """A compact pickle has no rows to decide the record from: ``summary()`` and ``metrics()``
    on its training rows read the inference master saved, blank what master blanked, and the
    footnote says how to have the checks.  Each SE is the saved covariance's diagonal read
    again: elementwise on a dense one (bit for bit), through the retained factor's solves on a
    structured one (``n u kappa`` of the Jacobi-scaled ``H`` on its range, ``_range_tolerance``).
    At ``253c6d353`` the ``sz`` model's saved accessor raised, and no model said its SEs were
    the saved ones."""
    record = _record(case)
    output = record["outputs"][variant]
    model = _loads(record["models"][variant])
    names, se, notes = _summary(model)
    assert FROZEN in notes and "retain_fit_state=True" in notes
    master_names, master_se = _master_summary(output)
    shown = [i for i, name in enumerate(master_names) if name in names]
    master_se = master_se[shown]
    assert [master_names[i] for i in shown] == names
    np.testing.assert_array_equal(np.isnan(se), np.isnan(master_se))
    accessor = model._fit_inference_info["XtWX_inv_aug"]
    factor = getattr(accessor, "factor", None)
    if factor is None:
        np.testing.assert_array_equal(se, master_se)
    else:
        tolerance = _range_tolerance(_loads(record["models"]["warm_cold"]))
        finite = np.isfinite(se)
        np.testing.assert_allclose(se[finite], master_se[finite], rtol=tolerance, atol=0.0)
    metrics = _metrics_se(model, record["frame"], record["y"])
    for term, saved in output["metrics_se_raw"].items():
        np.testing.assert_array_equal(np.isnan(metrics[term]), np.isnan(saved), err_msg=term)


# ------------------------------------------------------------------ the gauge, forced
def _with_coefficients(model, beta: np.ndarray):
    """``model`` holding ``beta`` as its saved solver and public slopes (a forced saved state)."""
    assert np.array_equal(model._result.beta, model._solver_result.beta)
    reml = getattr(model, "_reml_result", None)
    copies: dict[int, object] = {}
    for name in ("_result", "_solver_result"):
        result = getattr(model, name)
        copies.setdefault(id(result), dataclasses.replace(result, beta=beta.copy()))
        setattr(model, name, copies[id(result)])
    if reml is not None and getattr(reml, "pirls_result", None) is not None:
        reml.pirls_result = copies.get(id(reml.pirls_result), reml.pirls_result)
    return model


def test_saved_zeros_that_fit_several_gauges_blank_every_coordinate_the_alias_reaches():
    """The w5m2 counterexample, forced: ``x2 = x1`` with both saved coefficients zero.  Each of
    ``x1 = 0`` and ``x2 = 0`` is a gauge the coefficients satisfy, and they report SEs on
    different coefficients, so neither is read: both are blank and the footnote says why.  At
    ``253c6d353`` both were blank from master's flags, with no word of why; without the rule
    the fit's own gauge (``x2`` held) would print ``x1``'s SE."""
    record = _record("duplicate")
    model = _loads(record["models"]["warm_cold"])
    beta = np.array(model.result.beta, dtype=np.float64)
    beta[_column(model, "x1") - 1] = 0.0
    names, se, notes = _summary(_with_coefficients(model, beta))
    assert [n for n, v in zip(names, se, strict=True) if np.isnan(v)] == ["x1", "x2"]
    assert "several gauges fit the saved coefficients" in notes


def test_a_saved_truncation_the_fit_would_not_choose_is_read_from_its_zero():
    """Another solver's truncation, forced: ``x2 = x1`` with the saved slope on ``x2`` and
    ``x1`` at zero (the same fitted values: the columns are equal).  The fit's own constraint
    (``x2 = 0``) does not hold, the one zero does, so ``x1`` is blank and ``x2`` has the SE of
    the fit with ``x1`` held (the dense ``H`` without ``x1``, to ``n u kappa``).  At
    ``253c6d353`` both were blank (master's flags)."""
    record = _record("duplicate")
    model = _loads(record["models"]["warm_cold"])
    one, two = _column(model, "x1") - 1, _column(model, "x2") - 1
    beta = np.array(model.result.beta, dtype=np.float64)
    beta[two], beta[one] = beta[one] + beta[two], 0.0
    model = _with_coefficients(model, beta)
    names, se, notes = _summary(model)
    assert [n for n, v in zip(names, se, strict=True) if np.isnan(v)] == ["x1"]
    assert "hold x1 at zero" in notes
    hessian, design = _poisson_hessian(model, record["frame"])
    kept = np.setdiff1d(np.arange(design.shape[1]), [one + 1])
    oracle, tolerance = _reduced_se(hessian, kept, design.shape[0])
    assert abs(se[names.index("x2")] / oracle[list(kept).index(two + 1)] - 1.0) <= tolerance


# ------------------------------------------------------------------ full rank: master's numbers
@pytest.mark.parametrize("variant", ["warm_cold", "warm_read"])
@pytest.mark.parametrize("case", ["dense_full", "random_effect"])
def test_a_saved_full_rank_fit_keeps_masters_standard_errors(case, variant):
    """Deciding a full-rank warm model again moves no SE master printed: each is master's
    within ``n u kappa`` (``_se_tolerance``), and every coefficient, the random effect's levels
    included, has the SE of the dense ``H`` at the saved coefficients.  At ``253c6d353`` the
    random effect saved after its readers kept master's data-only flags in ``metrics()``,
    which blanked all 15 levels (the cold one, read afresh, did not)."""
    record = _record(case)
    output = record["outputs"][variant]
    model = _loads(record["models"][variant])
    names, se, _ = _summary(model)
    master_names, master_se = _master_summary(output)
    tolerance = _se_tolerance(model)
    for name, value in zip(names, se, strict=True):
        saved = master_se[master_names.index(name)]
        assert np.isfinite(value) and abs(value / saved - 1.0) <= tolerance, name
    metrics = _metrics_se(model, *_rows(model, record["frame"], record["y"]))
    hessian, design = _poisson_hessian(model, record["frame"])
    oracle, bound = _reduced_se(hessian, np.arange(design.shape[1]), design.shape[0])
    for group in model._groups:
        np.testing.assert_allclose(
            metrics[group.name], oracle[1 + group.start : 1 + group.end], rtol=bound, atol=0.0
        )


@pytest.mark.pins_behaviour
def test_a_compact_model_saved_now_keeps_its_decided_record():
    """Pins that only pickles from before the record are frozen: a compact fit of the
    duplicate saved by this version loads ``decided``, with its record (``x2`` held, ``x1``'s
    SE given) and no footnote about saved SEs.  It passed at ``253c6d353``, which had no
    frozen state to fall into; a load move that ignored the tag would freeze it."""
    from superglm import Numeric, Spline, SuperGLM

    record = _record("duplicate")
    features = {"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}
    model = SuperGLM(
        family="poisson", features=features, selection_penalty=0.0, retain_fit_state=False
    )
    _quiet(model.fit_reml, record["frame"], record["y"])
    loaded = _loads(pickle.dumps(model))
    # the tag as ``state_ops.record_tag`` reads it (a name ``253c6d353`` lacks)
    assert loaded.__dict__.get("_record_tag", "decided") == "decided"
    names, se, notes = _summary(loaded)
    assert [n for n, v in zip(names, se, strict=True) if np.isnan(v)] == ["x2"]
    assert FROZEN not in notes
