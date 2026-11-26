"""``cross_validate(..., fit_mode="fit_ncv")``: each outer fold chooses its smoothing by held-out
prediction on its own training rows.

References.  Each fold's fit is compared with the ``fit_ncv`` call a user would make on that
fold's rows: the same model with its levels bound on the whole frame (``bind_levels``, the
pass ``cross_validate`` applies), the fold's training rows, weights and offset, and the outer
``groups`` sliced to those rows.  With the warm start off, the fold's fit is that call, so the
two agree exactly.  The criterion of a ``fit_ncv(cv=splitter, groups=g)`` fit is compared with
the pooled held-out deviance ``cross_validate(..., fit_mode="fit")`` reports on the same splits
at the chosen smoothing, on a model whose knots and levels are the same in every fold.

Tolerances.  ``U`` is the unit roundoff and ``kappa`` the condition of the Jacobi-scaled
penalised Hessian (``tests.test_fit_ncv.hessian_condition``).  A Gaussian criterion and the
fold refits it stands for are exact computations of one quantity and agree within
``64 kappa U`` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002,
section 10.1).  A Poisson criterion takes one Newton step from the full fit for each
neighbourhood where the folds refit to convergence; the step's error is second order in the
change the neighbourhood makes (``POISSON_ONE_STEP``).
"""

import logging
import re
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold

from superglm import (
    Categorical,
    GammaLS,
    Numeric,
    Spline,
    SuperGLM,
    SuperLSS,
    cross_validate,
    s,
)
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.model import fit_method
from superglm.ncv import engine
from tests.test_fit_ncv import (
    Book,
    U,
    book,
    hessian_condition,
    lone_level,
    model_of,
    refusal_case,
)

TOL = engine.DEFAULT_NCV_TOL
OUTER = GroupKFold(3)
INNER = {"groups": {}, "inner cv": {"cv": GroupKFold(5)}}
POISSON_ONE_STEP = 1e-3
"""The Poisson bar, relative: a tenth of the 1% the acceptance table allows a one-step held-out
fit against a refit (its row 3).  The gap is the one-step method's own error, so it does not
move with rounding: measured 1.5e-5 with five shuffled folds and 5.7e-4 with five folds of
whole policies, each removing a fifth of the rows."""


def quiet(call, *args, **kwargs):
    """``call`` with its warnings recorded; the result and the warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        return call(*args, **kwargs), caught


def outer_cv(model, b: Book, **arguments):
    """``cross_validate(fit_mode="fit_ncv")`` on a book with the outer groups, every fold kept."""
    arguments = {
        "cv": OUTER,
        "groups": b.labels,
        "return_estimators": True,
        "return_oof": True,
        **arguments,
    }
    return cross_validate(
        model,
        b.frame,
        b.y,
        sample_weight=b.weights,
        offset=b.offset,
        fit_mode="fit_ncv",
        **arguments,
    )


def direct_fit(model, b: Book, train, **arguments):
    """The ``fit_ncv`` call on a fold's training rows, levels bound on the whole frame."""
    model.bind_levels(b.frame, sample_weight=b.weights)
    model.fit_ncv(
        b.frame.take(train),
        b.y[train],
        sample_weight=b.weights[train],
        offset=b.offset[train],
        groups=b.labels[train],
        **arguments,
    )
    return model


# ── row 10: each fold is the fit_ncv call on its own training rows ────────────


@pytest.mark.parametrize("inner", list(INNER))
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_each_fold_is_fit_ncv_on_its_training_rows_with_its_groups(family, inner):
    """Smoothing, criterion, EDF, held-out predictions and the fold's record, exactly.

    The outer groups reach each fold's ``fit_ncv`` sliced to its training rows: as ``groups``
    held out one at a time, or forwarded to the inner splitter.  Mutation checks: the outer
    groups passed whole, the outer splitter drawn again as the inner hold-out, ``fit_reml``
    called for ``"fit_ncv"``, and the fold's weights or offset dropped each fail here.
    """
    b = book(family)
    result, _ = quiet(outer_cv, model_of(family), b, fit_kwargs=INNER[inner])
    assert result.fit_mode == "fit_ncv" and result.warm_start is False
    folds = result.fold_scores
    assert not any(folds["warm_started"])
    for fold, (train, test) in enumerate(result.fold_indices):
        direct, _ = quiet(direct_fit, model_of(family), b, train, **INNER[inner])
        est, record = result.estimators[fold], folds.iloc[fold]
        assert fit_method.method_of(est) == "fit_ncv"
        got, want = est.ncv_diagnostics(), direct.ncv_diagnostics()
        assert got["lambdas"] == want["lambdas"]
        assert got["criterion"] == want["criterion"]
        assert got["neighbourhoods"] == want["neighbourhoods"]
        assert got["neighbourhoods"]["count"] == (
            len(np.unique(b.labels[train])) if inner == "groups" else 5
        )
        assert record["effective_df"] == direct.result.effective_df
        assert record["n_ncv_iter"] == want["n_iter"]
        assert record["converged"] == (want["converged"] and direct.result.converged)
        np.testing.assert_array_equal(
            result.oof_predictions[test], direct.predict(b.frame.take(test), offset=b.offset[test])
        )


def test_warm_folds_start_from_the_first_converged_fold_and_choose_alike():
    """``warm_start=True`` starts later folds from fold 0's smoothing; they reach what cold reaches.

    Each search stops where its slopes are within ``TOL`` of the criterion, so a warm and a
    cold stop near one minimum of a fold differ in the criterion by the second-order
    remainder of those slopes, inside ``TOL`` of it.  The criterion is what agrees within the
    search's tolerance; the smoothing, EDF and predictions of two first-order stops on a flat
    criterion are not pinned by it, so they are not compared.
    """
    b = book("gaussian")
    cold, _ = quiet(outer_cv, model_of("gaussian"), b)
    warm, _ = quiet(outer_cv, model_of("gaussian"), b, warm_start=True)
    assert warm.warm_start is True
    assert list(warm.fold_scores["warm_started"]) == [False, True, True]
    assert all(cold.fold_scores["converged"]) and all(warm.fold_scores["converged"])
    first = cold.estimators[0].ncv_diagnostics()["lambdas"]
    assert warm.estimators[0].ncv_diagnostics()["lambdas"] == first
    for fold in (1, 2):
        start = warm.estimators[fold].ncv_diagnostics()["start_lambdas"]
        assert start == {name: first[name] for name in start}
    for cold_est, warm_est in zip(cold.estimators, warm.estimators, strict=True):
        c, w = cold_est.ncv_diagnostics(), warm_est.ncv_diagnostics()
        assert abs(w["criterion"] - c["criterion"]) <= TOL * c["criterion"]


# ── row 4: the criterion is cross_validate's pooled held-out deviance ─────────


def fixed_book(family) -> Book:
    """A book on which knots and levels do not depend on the fold: 1200 rows, 150 policies."""
    n = 1200
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(-2, 2, n),
            "z": rng.normal(0.0, 1.0, n),
            "f": rng.choice(list("abc"), n, p=[0.5, 0.3, 0.2]),
        }
    )
    weights = rng.choice([1.0, 2.0, 3.0], n)
    exposed = rng.uniform(0.3, 1.5, n)
    signal = (
        np.sin(6 * frame.x1) + 0.3 * frame.x2**2 + 0.2 * frame.z + 0.4 * (frame.f == "b")
    ).to_numpy()
    if family == "gaussian":
        offset = 0.5 * np.log(exposed)
        y = signal + offset + rng.normal(0, 0.5, n) / np.sqrt(weights)
    else:
        offset = np.log(exposed)
        y = rng.poisson(np.exp(signal - 0.6 + offset)).astype(float)
    return Book(frame, y, weights, offset, rng.integers(0, 150, n))


def fixed_model(family):
    """Explicit interior knots and boundaries, so every fold builds the same basis."""
    return SuperGLM(
        features={
            "x1": Spline(kind="ps", knots=np.linspace(0, 1, 9)[1:-1], boundary=(0.0, 1.0)),
            "x2": Spline(kind="ps", knots=np.linspace(-2, 2, 9)[1:-1], boundary=(-2.0, 2.0)),
            "z": Numeric(),
            "f": Categorical(),
        },
        family=family,
        selection_penalty=0,
        spline_penalty=1.0,
        tol=1e-12,
    )


SPLITTERS = {
    "KFold": (KFold(5, shuffle=True, random_state=3), False),
    "GroupKFold": (GroupKFold(5), True),
}


@pytest.mark.parametrize("splitter", list(SPLITTERS))
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_criterion_per_scored_row_is_cross_validates_pooled_deviance(family, splitter):
    """``criterion_mean`` equals the pooled held-out deviance of ``fit`` at the chosen smoothing.

    Gaussian to rounding (``64 kappa U``); Poisson within the one-step error
    (``POISSON_ONE_STEP``).  ``cv`` and ``groups`` are passed alike to both.
    """
    b = fixed_book(family)
    cv, grouped = SPLITTERS[splitter]
    groups = b.labels if grouped else None
    model, _ = quiet(
        fixed_model(family).fit_ncv,
        b.frame,
        b.y,
        sample_weight=b.weights,
        offset=b.offset,
        cv=cv,
        groups=groups,
    )
    diagnostics = model.ncv_diagnostics()
    oracle = model._clone_without_features(set(), smoothing_resolved=True)
    oracle.lambda2 = {**oracle.lambda2, **diagnostics["lambdas"]}
    pooled = cross_validate(
        oracle,
        b.frame,
        b.y,
        cv=cv,
        groups=groups,
        sample_weight=b.weights,
        offset=b.offset,
        fit_mode="fit",
        error_score="raise",
    )
    assert all(pooled.fold_scores["converged"])
    want = pooled.pooled_scores["deviance"]
    bar = 64 * hessian_condition(model, b) * U if family == "gaussian" else POISSON_ONE_STEP
    assert abs(diagnostics["criterion_mean"] - want) <= bar * want


# ── refusals and disclosures ──────────────────────────────────────────────────


class NoSplit:
    """A splitter whose split must not be drawn: what is refused is refused before any fold."""

    def split(self, X, y=None, groups=None):
        raise AssertionError("cross_validate drew the folds before refusing")


MODEL_REFUSALS = [
    "binned route",
    "family",
    "link",
    "selection penalty",
    "selection penalty chosen by the fit",
    "random effect",
    "factor smooth",
    "search tolerance",
    "step limit",
    "start that is no mapping",
    "start that is no smoothing",
]


@pytest.mark.parametrize("name", MODEL_REFUSALS)
def test_what_fit_ncv_refuses_without_rows_is_refused_before_any_fold_in_its_words(name):
    """The sentence ``fit_ncv`` gives, raised before the splitter is drawn."""
    model, b, arguments, _ = refusal_case(name)
    with pytest.raises((ValueError, TypeError)) as direct:
        model.fit_ncv(b.frame, b.y, sample_weight=b.weights, offset=b.offset, **arguments)
    with pytest.raises(direct.type) as outer:
        cross_validate(
            model,
            b.frame,
            b.y,
            cv=NoSplit(),
            sample_weight=b.weights,
            offset=b.offset,
            fit_mode="fit_ncv",
            fit_kwargs=arguments,
        )
    assert str(outer.value) == str(direct.value)


@pytest.mark.parametrize(
    ("fit_kwargs", "message"),
    [
        (
            {"groups": np.arange(300) % 15},
            r"fit_kwargs cannot set \['groups'\]: cross_validate slices groups to each fold's "
            r"training rows and passes them to fit_ncv; pass them as its own groups argument",
        ),
        (
            {"neighbours": [[i] for i in range(300)]},
            r"fit_kwargs cannot set \['neighbours'\]: its row positions refer to the whole "
            r"frame, not to a fold's training rows; pass groups to cross_validate",
        ),
    ],
)
def test_hold_out_arguments_that_would_not_follow_the_fold_are_refused(fit_kwargs, message):
    b = book("gaussian")
    with pytest.raises(ValueError, match=message):
        cross_validate(
            model_of(), b.frame, b.y, cv=NoSplit(), fit_mode="fit_ncv", fit_kwargs=fit_kwargs
        )


def test_superlss_has_no_fit_ncv_and_is_refused():
    b = book("gaussian")
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x1", kind="cr", k=6)), family.scale())
    with pytest.raises(ValueError, match=r"a SuperLSS has no fit_ncv; use fit_mode='fit_reml'"):
        cross_validate(model, b.frame, b.y, cv=NoSplit(), fit_mode="fit_ncv")


def test_an_unknown_fit_mode_names_the_three():
    b = book("gaussian")
    with pytest.raises(
        ValueError, match=r"fit_mode must be 'fit', 'fit_reml' or 'fit_ncv', got 'fit_gcv'"
    ):
        cross_validate(model_of(), b.frame, b.y, cv=NoSplit(), fit_mode="fit_gcv")


class Splits:
    """A splitter that yields the folds it is given."""

    def __init__(self, *folds):
        self.folds = [(np.asarray(train), np.asarray(test)) for train, test in folds]

    def split(self, X, y=None, groups=None):
        yield from self.folds


def fold_refusals(b: Book):
    """Two folds: fold 0 trains on ten of the fifteen groups, fold 1 on group g00 only."""
    first = np.isin(b.labels, [f"g{k:02d}" for k in range(10)])
    alone = b.labels == "g00"
    return Splits(
        (np.flatnonzero(first), np.flatnonzero(~first)),
        (np.flatnonzero(alone), np.flatnonzero(~alone)),
    )


def test_fold_with_one_group_gives_fit_ncvs_message_naming_the_fold(caplog):
    """A fold-dependent refusal fails its fold only, in ``fit_ncv``'s words, and names the fold."""
    b = book("gaussian")
    cv = fold_refusals(b)
    sentence = r"`groups` holds one label only \('g00'\), so holding it out would leave no rows"
    with caplog.at_level(logging.WARNING, logger="superglm.model_selection"):
        result, _ = quiet(outer_cv, model_of(), b, cv=cv)
    failed = [r.getMessage() for r in caplog.records if "failed" in r.getMessage()]
    assert len(failed) == 1 and failed[0].startswith("Fold 1 failed: ValueError(")
    assert re.search(sentence, failed[0])
    assert np.isfinite(result.fold_scores["deviance"].iloc[0])
    assert np.isnan(result.fold_scores["deviance"].iloc[1])
    with pytest.raises(ValueError, match=sentence) as raised:
        quiet(outer_cv, model_of(), b, cv=cv, error_score="raise")
    assert raised.value.__notes__ == ["cross_validate: raised in fold 1."]


def test_fold_leaving_a_level_in_one_group_gives_fit_ncvs_message():
    """Within a fold, every row of level 'd' in one held-out group is ``fit_ncv``'s refusal."""
    b = book("gaussian")
    b = replace(b, labels=lone_level(b, "d"))
    message = (
        r"Every weighted row of level 'd' of 'f' lies in group 'apart', so with it held out "
        r"nothing in the fit estimates that level"
    )
    with pytest.raises(ValueError, match=message) as raised:
        quiet(
            outer_cv, model_of(), b, cv=KFold(2, shuffle=True, random_state=0), error_score="raise"
        )
    assert raised.value.__notes__ == ["cross_validate: raised in fold 0."]


def test_monotone_term_is_refused_by_each_fold_in_fit_ncvs_words():
    """Whether a term carries fit-time constraints is known once its basis is built on rows."""
    model, b, _, message = refusal_case("monotone term")
    with pytest.raises(ValueError, match=message):
        quiet(outer_cv, model, b, error_score="raise")


def test_fold_whose_search_stops_short_is_disclosed_not_refused():
    """A step limit gives each fold a fit, a warning and ``converged=False``, as for fit_reml."""
    b = book("gaussian")
    result, caught = quiet(outer_cv, model_of(), b, fit_kwargs={"max_ncv_iter": 1})
    folds = result.fold_scores
    assert not any(folds["converged"])
    assert list(folds["n_ncv_iter"]) == [1, 1, 1]
    assert np.all(np.isfinite(folds["deviance"]))
    assert all(est.result.converged for est in result.estimators)
    stops = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(stops) == 3
    assert all("because it reached the max_ncv_iter limit" in text for text in stops)


def test_scorer_cannot_write_the_ncv_step_column():
    b = book("gaussian")

    def steps(model, X, y, *, sample_weight=None, offset=None):
        return {"n_ncv_iter": 0}

    with pytest.raises(ValueError, match=r"Scorer returned reserved column name 'n_ncv_iter'"):
        quiet(outer_cv, model_of(), b, scoring=steps, error_score="raise")
