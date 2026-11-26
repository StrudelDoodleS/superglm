"""Bitwise characterisation of SuperLSS's smoothing search against another tree on the same machine.

Fits a fixed set of cases and records everything ``fit_distributional_efs``
(``superglm.distributional.smoothing.loop``) does, so a refactor of the search
can be checked bit for bit against master.  It is not a CI test: it checks
same-machine, same-thread identity, not identity across machines or BLAS builds.

    list                          the case names
    run --out DIR [--jobs N] [--only REGEX] [--coverage]
                                  fit the cases, one fresh process each
    compare A B                   exit 1 unless every case is bitwise identical
    census DIR                    the joint states of the search's flags, and the calls

Run it on each tree with that tree's interpreter and the same harness file::

    master/.venv/bin/python tree/benchmarks/lss_efs_characterisation.py run --out A
    tree/.venv/bin/python tree/benchmarks/lss_efs_characterisation.py run --out B
    python tree/benchmarks/lss_efs_characterisation.py compare A B

Every numerically meaningful thing the search does is a call to a function it
looks up in its own module's globals, so the ordered list of those calls, with
what went in and what came out, is its behaviour.  At the entry of every call of
``fit_distributional_efs`` the case process wraps, in the module's namespace,
every plain function the module imported from elsewhere in ``superglm`` (found
by inspection of the unpatched module, not listed by hand; the imported
classes are left alone, since the search uses them in ``isinstance``),
``_within_ceiling`` and ``_trial_laplace_objective``, and puts the module's
bindings back on exit.  Wrapping at entry wraps whatever a case or a test put
there, so a replaced callee is traced under its name.  The accelerator's
``record_accepted``, ``propose``, ``reject`` and ``reset`` are traced when the
search calls them.  Each call is recorded in order with its depth, its
arguments as passed (positional and keyword) and its result, or the error it
raised.  A coefficient fit is written as its index in the search's
``coefficient_fits`` when it is in that list, by identity, and walked
otherwise, so a warm start taken from another fit shows as another index.

At the top of every pass, through the wrap of ``compact_coefficient_history``,
the search's state is read from the caller's frame (its locals, or the
attributes of its ``self`` and of ``self.state`` and ``self.windows`` once the
search is an object): the nine flags of the design's section 1.4,
``endgame_stationary``, whether ``pending_cap`` is set, ``newton_budget``, the
four plateau counters, ``fall`` and ``previous_fall``, the face's components,
the lengths of ``history`` and ``coefficient_fits``, ``terminal_fit_index``
and which fits have been compacted.  On exit the whole
``DistributionalEFSResult`` is recorded, and with it the type name of
``converged``.  The ``m/``, ``f/`` and ``b/`` cases also record what a user
reads: coefficients, held-out linear predictors with their standard errors,
predictions, the summary table (EDF, statistics, p-values), the smoothing
parameters and reason, the covariance, ``diagnose()`` (less its wall-clock
profile), the warnings raised, and what the pickled model predicts once loaded
again (the pickle's bytes hold wall-clock timings, so they are not hashed).

The cases:

- ``m/``: the eight families of ``tests/test_distributional_reorg_golden.py``
  on its data (700 rows fitted, 200 held out), exact and binned, under each of
  the four ``outer`` searches with ``practical_reml`` on and off, and under
  multisecant acceleration with plain ``efs``;
- ``t/``: quick-suite tests run as they are, each in its own process, chosen so
  that together with the matrix they reach every branch arm of the search the
  quick suite reaches; a test pins the calls and results of every search it
  runs and its own outcome;
- ``f/``: paths no test reaches, forced by replacing a callee (each case's
  docstring says which, and which lines of the search at f5204839b it reaches);
- ``b/``: the three ``benchmarks/books.py`` SuperLSS cases, where the freMTPL2
  files are present.

Each case runs in its own process with every thread pool pinned to one thread
and ``PYTHONHASHSEED=0``.  Floats are kept as ``float.hex``, arrays as a hash
of dtype, shape and bytes.  Wall-clock fields, ids, and the contents of caches,
memos and the reuse session are left out.  Output per case:
``cases/<id>.hash.json`` (the hashes ``compare`` reads) and
``cases/<id>.leaves.json.gz`` (every recorded value, read only for a case that
differs).  ``--coverage`` adds the branch arms of the search the run reached.

The general parts (the digest, the leaf hashes, the comparison) are imported
from ``irls_direct_characterisation.py``.  Only the standard library is
imported at module level: the thread pins must reach the case processes before
numpy is loaded.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import pickle
import re
import subprocess
import sys
import time
import types
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import irls_direct_characterisation as irls  # noqa: E402

PINNED_ENV = irls.PINNED_ENV
CASE_TIMEOUT_S = 600
TRAIN_ROWS = 700
LOOP_MODULE = "superglm.distributional.smoothing.loop"
OUTERS = ("efs", "efs+finish", "efs+newton", "efs+stationary")
FAMILIES = ("gaussian", "lognormal", "gamma", "nb2", "tweedie", "gengamma", "twopiece", "gpd")
BOOKS = ("lss_nb", "lss_tweedie", "lss_gamma")

# Quick-suite tests run whole (``t/``). Together with the matrix they reach every branch
# arm of the search the quick suite reaches: a greedy cover of the arms over the
# suite's per-test coverage, seeded with the paths the design names, from the
# SuperLSS smoothing test files (cross-validation and book-data tests left out).
SUITE_TESTS = (
    "test_distributional_efs.py::test_inner_and_smoothing_convergence_are_reported_separately",
    "test_distributional_efs.py::test_non_gfs_fallback_refuses_and_resets_multisecant",
    "test_distributional_efs.py::test_refused_trial_breaks_strict_plateau_window[provenance]",
    "test_distributional_efs.py::test_rejected_acceleration_is_not_backtracked_before_raw_trials_all_fail",
    "test_distributional_efs.py::test_warming_and_current_duplicate_refusal_add_no_coefficient_fit",
    "test_distributional_endpoint_laml.py::test_efs_releases_a_face_when_terminal_direction_resolves_finite_and_carries_on",
    "test_distributional_endpoint_laml.py::test_efs_releases_a_joint_face_when_one_recheck_is_finite_and_carries_on",
    "test_distributional_endpoint_laml.py::test_efs_retracts_a_face_when_the_fresh_cap_has_a_better_objective",
    "test_distributional_endpoint_laml.py::test_terminal_failure_rolls_back_the_whole_multi_face",
    "test_distributional_endpoint_roundoff.py::test_the_iteration_budget_does_not_refuse_a_stationary_endgames_face_check",
    "test_distributional_endpoint_trial.py::test_a_bound_past_an_interior_optimum_is_refused_and_tried_once",
    "test_distributional_endpoint_trial.py::test_a_bound_the_objective_refuses_costs_one_fit",
    "test_distributional_endpoint_trial.py::test_a_finish_that_stalls_after_a_rejected_round_leaves_the_round_s_verdict[derivative_unavailable]",
    "test_distributional_endpoint_trial.py::test_a_finish_that_stalls_after_a_rejected_round_leaves_the_round_s_verdict[gradient_unresolved]",
    "test_distributional_endpoint_trial.py::test_a_finish_that_stalls_after_a_rejected_round_leaves_the_round_s_verdict[max_iterations]",
    "test_distributional_endpoint_trial.py::test_a_finish_that_stalls_after_a_rejected_round_leaves_the_round_s_verdict[objective_rejected]",
    "test_distributional_endpoint_trial.py::test_a_plateau_stands_on_an_exact_face_under_the_question_a_held_cap_asks",
    "test_distributional_endpoint_trial.py::test_a_refused_hand_over_leaves_the_stationary_verdict_standing",
    "test_distributional_endpoint_trial.py::test_a_search_holding_a_trials_cap_goes_on_through_a_plateau_still_falling",
    "test_distributional_endpoint_trial.py::test_a_stationary_endgame_hands_nothing_over_beside_another_components_cap",
    "test_distributional_endpoint_trial.py::test_handed_caps_that_were_all_there_was_to_estimate_keep_the_stationary_reason",
    "test_distributional_endpoint_trial.py::test_held_caps_are_not_assessed_beside_another_components_cap[efs+finish]",
    "test_distributional_endpoint_trial.py::test_held_caps_are_not_assessed_beside_another_components_cap[efs]",
    "test_distributional_history_rows.py::test_compact_and_full_history_preserve_fit_and_pickling[efs+newton]",
    "test_distributional_history_rows.py::test_compact_and_full_history_preserve_fit_and_pickling[efs]",
    "test_distributional_history_rows.py::test_plateau_and_exact_face_keep_replay_authority[True]",
    "test_distributional_newton_endgame.py::test_endgame_steps_a_free_term_beside_one_its_cap_holds[efs+newton]",
    "test_distributional_newton_endgame.py::test_flat_smooth_is_certified_not_capped",
    "test_distributional_newton_endgame.py::test_gradient_unresolved_when_certificate_dominates",
    "test_distributional_newton_endgame.py::test_unavailable_gradient_after_cap_release_retains_uncertified_fit",
    "test_distributional_newton_endgame.py::test_unavailable_newton_gradient_resumes_efs_from_accepted_fit[1-3-1]",
    "test_distributional_newton_practical_handoff.py::test_a_refused_cap_lets_the_exact_gradient_decide_first[efs+newton]",
    "test_distributional_newton_practical_handoff.py::test_an_outward_stop_keeps_its_plateau_when_the_finish_stalls",
    "test_distributional_practical_stop.py::test_a_search_that_cannot_step_still_reports_its_refused_cap[0]",
    "test_gamma_ls_joint_infinity_face.py::test_public_gamma_reml_accepts_the_strict_joint_infinity_face",
    "test_gamma_ls_vertical_slice.py::test_gamma_multisecant_cross_family_correctness_receipt",
    "test_negative_binomial_lss_efs.py::test_coefficients_a_face_removes_are_not_unidentified",
    "test_negative_binomial_lss_efs.py::test_size_at_the_poisson_limit_is_disclosed_with_its_row_count",
    "test_superlss_api.py::test_a_spent_iteration_budget_leaves_the_fellner_schall_stop_in_place[practical]",
    "test_superlss_api.py::test_a_spent_iteration_budget_leaves_the_fellner_schall_stop_in_place[strict]",
    "test_superlss_api.py::test_public_fit_reml_exposes_only_efs_and_honours_fixed_policy",
    "test_tweedie_lss_vertical_slice.py::test_nonconverged_efs_trial_is_never_compared_as_laml",
)


# --------------------------------------------------------------------------- the tree


def _tree_root() -> Path:
    """The checkout whose ``superglm`` this interpreter imports, put on ``sys.path``."""
    import importlib.util

    spec = importlib.util.find_spec("superglm")
    root = Path(spec.origin).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(1, str(root))
    return root


# --------------------------------------------------------------------------- case registry


def _registry() -> dict[str, tuple[Callable, dict]]:
    cases: dict[str, tuple[Callable, dict]] = {}
    for family in FAMILIES:
        for route in ("exact", "binned"):
            for outer in OUTERS:
                for stop in ("practical", "strict"):
                    cases[f"m/{family}/{route}/{outer}/{stop}"] = (
                        _matrix_case,
                        {"family": family, "route": route, "outer": outer, "stop": stop},
                    )
            cases[f"m/{family}/{route}/efs/multisecant"] = (
                _matrix_case,
                {"family": family, "route": route, "outer": "efs", "stop": "practical",
                 "acceleration": "multisecant"},
            )  # fmt: skip
    for test in SUITE_TESTS:
        cases[f"t/{test.removeprefix('test_').replace('.py::', '::')}"] = (
            _suite_test,
            {"test": test},
        )
    for name, builder in _FORCED.items():
        cases[f"f/{name}"] = (builder, {})
    if _have_books():
        for book in BOOKS:
            cases[f"b/{book}"] = (_book_case, {"book": book})
    return cases


def _have_books() -> bool:
    """Both freMTPL2 files in one of the places ``examples/fremtpl2_data.py`` looks."""
    places = [os.environ.get("SUPERGLM_DATA_DIR", ""), Path.home() / ".cache" / "superglm",
              HERE.parent / "data"]  # fmt: skip
    files = ("freMTPL2freq.parquet", "freMTPL2sev.parquet")
    return any(place and all((Path(place) / f).exists() for f in files) for place in places)


def _case_order(name: str) -> tuple[int, str]:
    """Slow kinds first, so the pool's tail is short."""
    return (0 if name.startswith(("b/", "t/")) else 1, name)


# --------------------------------------------------------------------------- case builders


def _attempt(function: Callable) -> object:
    """A user-facing output, or the error it raised (recorded, not raised)."""
    try:
        return function()
    except Exception as exc:  # noqa: BLE001 - the refusal is the output
        return {"error": f"{type(exc).__qualname__}: {_scrub(str(exc))}"}


def _frame_digest(frame) -> dict:
    import numpy as np

    return {
        "columns": [str(c) for c in frame.columns],
        "index": [str(i) for i in frame.index],
        "values": {str(c): np.asarray(frame[c]).tolist() for c in frame.columns},
    }


def _model_outputs(model, test, offsets=None) -> dict:
    """What a user reads from a fitted ``SuperLSS``."""
    import numpy as np

    def link():
        values, se = model.predict_link(test, offsets=offsets, se_fit=True)
        return {"link": values.to_numpy(np.float64), "se": se.to_numpy(np.float64)}

    return {
        "coef": _attempt(lambda: dict(model.coef_)),
        "link": _attempt(link),
        "predict": _attempt(lambda: np.asarray(model.predict(test, offsets=offsets), np.float64)),
        "summary": _attempt(lambda: _frame_digest(model.summary())),
        "smoothing": _attempt(lambda: dict(model.smoothing_parameters_)),
        "reason": _attempt(lambda: model.smoothing_convergence_reason_),
        "certified": _attempt(lambda: model.smoothing_certified_),
        "covariance": _attempt(lambda: np.asarray(model.covariance_, np.float64)),
        "diagnose": _attempt(model.diagnose),
        "reloaded": _attempt(lambda: _reloaded(model, test, offsets)),
    }


def _reloaded(model, test, offsets) -> dict:
    """The pickled model, loaded again (its bytes hold wall-clock timings): what it predicts."""
    import numpy as np

    again = pickle.loads(pickle.dumps(model))
    values, se = again.predict_link(test, offsets=offsets, se_fit=True)
    return {"link": values.to_numpy(np.float64), "se": se.to_numpy(np.float64),
            "smoothing": dict(again.smoothing_parameters_)}  # fmt: skip


def _matrix_case(
    family: str, route: str, outer: str, stop: str, acceleration: str = "none"
) -> dict:
    """The golden test's fit for one family, fitted on its first rows from the default start."""
    _tree_root()
    from tests.bound_predictor_fixtures import model_from_templates
    from tests.test_distributional_reorg_golden import _cases

    cases, frame = _cases()
    distribution, predictors, y = cases[family]
    model = model_from_templates(
        family=distribution, predictors=predictors, discrete=route == "binned"
    )
    model.fit_reml(
        frame.iloc[:TRAIN_ROWS],
        y[:TRAIN_ROWS],
        outer=outer,
        practical_reml=stop == "practical",
        acceleration=acceleration,
    )
    return _model_outputs(model, frame.iloc[TRAIN_ROWS:])


def _golden_fit(family: str, **options) -> dict:
    """A matrix fit, exact, with ``fit_reml`` options of the case's own."""
    _tree_root()
    from tests.bound_predictor_fixtures import model_from_templates
    from tests.test_distributional_reorg_golden import _cases

    cases, frame = _cases()
    distribution, predictors, y = cases[family]
    model = model_from_templates(family=distribution, predictors=predictors)
    model.fit_reml(frame.iloc[:TRAIN_ROWS], y[:TRAIN_ROWS], **options)
    return _model_outputs(model, frame.iloc[TRAIN_ROWS:])


class _Outcomes:
    """A pytest plugin that keeps each phase's outcome of the tests it runs."""

    def __init__(self) -> None:
        self.reports: list[list[str]] = []
        self.failures: list[str] = []

    def pytest_runtest_logreport(self, report) -> None:
        self.reports.append([report.when, report.outcome])
        if report.failed:
            self.failures.append(report.longreprtext[-1500:])


def _suite_test(test: str) -> dict:
    """One quick-suite test, run as it is: its searches are traced, its outcome pinned."""
    import pytest

    root = _tree_root()
    outcomes = _Outcomes()
    os.chdir(root)
    code = pytest.main(
        [f"tests/{test}", "-q", "-p", "no:cacheprovider", "--no-header", "-o", "addopts="],
        plugins=[outcomes],
    )
    # the failure text names the checkout, so it is reported, not compared
    _DETAIL.extend(outcomes.failures)
    return {"exit": int(code), "reports": outcomes.reports}


_DETAIL: list[str] = []


def _book_case(book: str) -> dict:
    """A ``benchmarks/books.py`` SuperLSS case, fitted as ``books.py`` fits it, without the warm-up."""
    import books

    spec = books.CASES[book]
    train, test = books.load_book(spec["dataset"])
    model = books._lss_model(spec, offsets=books._lss_inputs(spec, train)[3])
    books._fit(spec, model, train)
    X, _, _, offsets = books._lss_inputs(spec, test)
    return _model_outputs(model, X, offsets)


# Paths no test reaches, each forced by replacing a callee on the search's module. The
# docstrings name the lines of ``fit_distributional_efs`` at f5204839b each case reaches.
_FORCED: dict[str, Callable[[], dict]] = {}


def _forced(name: str) -> Callable:
    def register(builder: Callable[[], dict]) -> Callable[[], dict]:
        _FORCED[name] = builder
        return builder

    return register


def _loop():
    from superglm.distributional.smoothing import loop

    return loop


def _fit_outputs(fitted) -> dict:
    """What ``fit_dense_distributional`` returns, walked whole."""
    return {"fit": fitted}


def _endpoint_trial_fit(outer: str, **options) -> dict:
    """``tests/test_distributional_endpoint_trial.py``'s fit: a strong smooth beside linear ones."""
    import numpy as np
    import pandas as pd

    _tree_root()
    from tests.test_distributional_endpoint_trial import _fit

    model, _ = _fit(outer, **options)
    grid = np.linspace(-0.95, 0.95, 40)
    probe = pd.DataFrame({"x": grid, "z": grid[::-1], "w": np.sin(3.0 * grid)})
    return _model_outputs(model, probe)


def _history_rows_fit(**config):
    """``tests/test_distributional_history_rows.py``'s 96-row Gaussian fit, by config."""
    _tree_root()
    from tests.test_distributional_history_rows import _fit

    return _fit(**config)


def _rows_before_newton(fitted) -> int:
    history = fitted.smoothing.history
    return next((k for k, row in enumerate(history) if row.stage == "newton"), len(history))


def _gamma_joint_face() -> dict:
    """``tests/test_gamma_ls_joint_infinity_face.py``'s fit, on a joint face of two caps."""
    _tree_root()
    from tests.test_gamma_ls_joint_infinity_face import (
        _fit_public_gamma_joint_face,
        _gamma_joint_face_data,
    )

    frame, response, weights = _gamma_joint_face_data()
    model = _fit_public_gamma_joint_face(frame, response, weights)
    return _model_outputs(model, frame.iloc[::37])


def _rounds() -> list[int]:
    """Count the Fellner-Schall rounds: each proposal reads ``_maximum_step`` once."""
    loop = _loop()
    real, count = loop._maximum_step, [0]

    def counted(*args, **kwargs):
        count[0] += 1
        return real(*args, **kwargs)

    loop._maximum_step = counted
    return count


def _endgame_kinds(*kinds: str | None, when: Callable[[dict], bool] = lambda kwargs: True) -> None:
    """Replace the endgame's outcome kind on its successive calls (``None`` leaves one alone)."""
    from dataclasses import replace

    loop = _loop()
    real, script = loop.run_newton_endgame, list(kinds)

    def scripted(*args, **kwargs):
        outcome = real(*args, **kwargs)
        if not script or not when(kwargs):
            return outcome
        kind = script.pop(0)
        if kind is None:
            return outcome
        changes = {"kind": kind}
        if kind in ("derivative_unavailable", "cap_pressure"):
            changes["derivatives"] = None
        if kind == "cap_pressure":
            changes["cap_pressure"] = ()
        return replace(outcome, **changes)

    loop.run_newton_endgame = scripted


def _floor_evidence_after(record: str, *, once: bool) -> None:
    """Fresh evidence after the first call of the history-row builder ``record`` is not
    stationary: its largest step is at least ten times the tolerance, at the next
    evidence (``once``) or at every one after."""
    from dataclasses import replace

    loop = _loop()
    real_record, real_evidence, armed = getattr(loop, record), loop._fresh_raw_evidence, [0]

    def recorded(*args, **kwargs):
        armed[0] = armed[0] or 1
        return real_record(*args, **kwargs)

    def evidence(*args, **kwargs):
        fresh = real_evidence(*args, **kwargs)
        if armed[0] != 1:
            return fresh
        armed[0] = 2 if once else 1
        config = args[3]
        return replace(fresh, maximum=max(fresh.maximum, 10.0 * config.tolerance))

    setattr(loop, record, recorded)
    loop._fresh_raw_evidence = evidence


def _no_hand_off_on_a_round() -> None:
    """The tail of an accepted round never hands off; every stop is consulted as it is."""
    loop = _loop()
    real = loop.should_hand_off

    def hand_off(reason, **kwargs):
        return False if reason is None else real(reason, **kwargs)

    loop.should_hand_off = hand_off


@_forced("resume/multisecant-reset")
def _resume_resets_the_accelerator() -> dict:
    """Lines 719-720: a resumed Fellner-Schall search resets the multisecant window.

    The first endgame of a multisecant ``efs+newton`` fit reports its derivatives
    unavailable, so the search resumes Fellner-Schall rounds.
    """
    _endgame_kinds("derivative_unavailable")
    return _golden_fit(
        "gaussian", outer="efs+newton", acceleration="multisecant", practical_reml=False
    )


@_forced("strict-stop/no-budget")
def _strict_stop_without_budget() -> dict:
    """Lines 733-734 and 1124-1125: a strict stop hands an ``efs+stationary`` search off
    with no round left, and the endgame ends the fit.

    The hand-off is put out of reach of the accepted rounds and the practical plateau
    is off, so the search runs to its strict stop; refitted with the round limit at
    that stop, the hand-off finds the budget spent.
    """
    options = dict(outer="efs+stationary", practical_convergence=False, handoff_step=1e-300,
                   handoff_iterations=10**6, tolerance=1e-6, max_log_step=5.0)  # fmt: skip
    first = _history_rows_fit(max_iterations=60, **options)
    second = _history_rows_fit(max_iterations=_rows_before_newton(first), **options)
    return _fit_outputs(second)


@_forced("stationary-plateau/endgame")
def _stationary_plateau_hands_off() -> dict:
    """Lines 1825-1828 and 1829-1831: an ``efs+stationary`` search reaches its practical
    plateau before its hand-off and runs the endgame from it, which goes on.

    The matrix's Gamma fit, whose plain search ends at a practical plateau; no
    accepted round hands off.
    """
    _no_hand_off_on_a_round()
    return _golden_fit("gamma", outer="efs+stationary")


@_forced("stationary-plateau/no-budget")
def _stationary_plateau_without_budget() -> dict:
    """Lines 1829-1830 and 733-734: the same plateau with its round limit spent: the
    endgame it hands to ends the fit."""
    _tree_root()
    from tests.bound_predictor_fixtures import model_from_templates
    from tests.test_distributional_reorg_golden import _cases

    _no_hand_off_on_a_round()
    cases, frame = _cases()
    distribution, predictors, y = cases["gamma"]
    first = model_from_templates(family=distribution, predictors=predictors)
    first.fit_reml(frame.iloc[:TRAIN_ROWS], y[:TRAIN_ROWS], outer="efs+stationary")
    rows = _rows_before_newton(first._require_fitted())
    return _golden_fit("gamma", outer="efs+stationary", max_reml_iter=rows)


@_forced("bracket/release-goes-on")
def _bracket_release_goes_on() -> dict:
    """Lines 993-995 and 1444-1449: after a bracket release the endgame returns nothing,
    and the passes after it run the endgame for the component beyond the cap.

    The base is ``test_unavailable_gradient_after_cap_release_retains_uncertified_fit``;
    the first two endgames after the release report cap pressure on no component and
    the third an unresolved gradient, which ends the fit.
    """
    import numpy as np
    import pandas as pd

    from superglm.distributional import GaussianLS, Predictor
    from superglm.distributional.model import fit_dense_distributional
    from superglm.distributional.results.iteration import DistributionalEFSConfig
    from superglm.distributional.results.solver import DenseSolverConfig
    from superglm.distributional.weights import WeightContract
    from superglm.features import Spline

    _endgame_kinds(
        "cap_pressure",
        "cap_pressure",
        "gradient_unresolved",
        when=lambda kwargs: bool(kwargs["upper_bounds"]),
    )
    rng = np.random.default_rng(17)
    x = np.linspace(0.0, 1.0, 200)
    y = 2.0 * np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.5, len(x))
    fitted = fit_dense_distributional(
        pd.DataFrame({"x": x}),
        y,
        family=GaussianLS(),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {}),
        ),
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(max_iterations=200, tolerance=1.0e-10),
        efs_config=DistributionalEFSConfig(
            outer="efs+newton",
            maximum_lambda=1.0e-4,
            initial_lambda=1.0e-4,
            maximum_lambda_conditioning=1.0,
            max_iterations=12,
        ),
        retain_rows=True,
    )
    return _fit_outputs(fitted)


@_forced("bracket/joint-attempt")
def _bracket_after_a_joint_refusal() -> dict:
    """Lines 809-810, 810-814 and 814-834: a refusal under cap pressure is a joint attempt
    with no direction, so the bracket beyond the cap has no component to search.

    The base is the bracket release fixture; every joint assessment is scripted as a
    refusal without a direction.
    """
    import numpy as np
    import pandas as pd

    from superglm.distributional import GaussianLS, Predictor
    from superglm.distributional.model import fit_dense_distributional
    from superglm.distributional.results.iteration import DistributionalEFSConfig
    from superglm.distributional.results.solver import DenseSolverConfig
    from superglm.distributional.smoothing.authority import _face_authority_config
    from superglm.distributional.smoothing.faces import _JointFaceDirectionAttempt
    from superglm.distributional.weights import WeightContract
    from superglm.features import Spline

    loop = _loop()

    def refused(*args, **kwargs):
        return None, _JointFaceDirectionAttempt(
            component_names=tuple(kwargs["terminal_evidence"].unresolved_upper_bound),
            direction=None,
            assessment_fits=(),
            coefficient_tolerance=_face_authority_config(kwargs["solver_config"]).tolerance,
            failure_reason="joint_analytic_unavailable",
        )

    loop._try_joint_exact_face = refused
    rng = np.random.default_rng(17)
    x = np.linspace(0.0, 1.0, 200)
    y = 2.0 * np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.5, len(x))
    fitted = fit_dense_distributional(
        pd.DataFrame({"x": x}),
        y,
        family=GaussianLS(),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {}),
        ),
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(max_iterations=200, tolerance=1.0e-10),
        efs_config=DistributionalEFSConfig(
            outer="efs+newton",
            maximum_lambda=1.0e-4,
            initial_lambda=1.0e-4,
            maximum_lambda_conditioning=1.0,
            max_iterations=12,
        ),
        retain_rows=True,
    )
    return _fit_outputs(fitted)


@_forced("face/revalidation-out-of-rounds")
def _face_revalidation_out_of_rounds() -> dict:
    """Lines 1137-1138: the round limit refuses a face revalidation at a strict stop.

    The endpoint trial fixture under plain ``efs`` stops strictly on ``z``'s face;
    refitted with the round limit at that stop, the revalidation has no round.
    """
    _tree_root()
    from tests.test_distributional_endpoint_trial import _fit

    _, plain = _fit("efs", practical_reml=False)
    rows = next(k for k, row in enumerate(plain.history) if row.revalidated_face_components)
    return _endpoint_trial_fit("efs", practical_reml=False, max_reml_iter=rows)


@_forced("face/scalar-kept-goes-on")
def _scalar_face_kept_goes_on() -> dict:
    """Lines 1429-1433: a kept scalar face whose fresh evidence is not yet stationary.

    The evidence after the first revalidation is scripted above the tolerance, so the
    search goes on from the face and stops at the next revalidation.
    """
    _floor_evidence_after("_face_revalidation_iteration", once=True)
    return _endpoint_trial_fit("efs", practical_reml=False)


@_forced("face/joint-kept-goes-on")
def _joint_face_kept_goes_on() -> dict:
    """Lines 1229-1233: the same for a kept joint face (the joint infinity face fixture)."""
    _floor_evidence_after("_joint_face_revalidation_iteration", once=True)
    return _gamma_joint_face()


@_forced("face/joint-without-finite-faces")
def _joint_face_without_finite_faces() -> dict:
    """Lines 1156-1157: building a joint face's finite companions raises
    ``PenaltyFaceError``, so the joint revalidation is unavailable and the face is
    rolled back (the joint infinity face fixture)."""
    from superglm.distributional.smoothing.penalty_face import PenaltyFaceError

    def unavailable(*args, **kwargs):
        raise PenaltyFaceError("finite companion face unavailable (forced)")

    _loop()._optional_penalty_face = unavailable
    return _gamma_joint_face()


@_forced("rejected-round/endgame-ends")
def _rejected_round_endgame_ends() -> dict:
    """Lines 1678-1679: an ``efs+newton`` search hands a rejected round to the endgame,
    which ends the fit.

    Every trial of the first round is scripted as rejected, and that endgame reports
    an unresolved gradient.
    """
    loop, rounds = _loop(), _rounds()
    real = loop._within_ceiling

    def ceiling(*args, **kwargs):
        return False if rounds[0] == 1 else real(*args, **kwargs)

    loop._within_ceiling = ceiling
    _endgame_kinds("gradient_unresolved")
    return _golden_fit("gaussian", outer="efs+newton")


@_forced("outward-plateau/goes-on")
def _outward_plateau_goes_on() -> dict:
    """Lines 1834-1836: an outward plateau beside a trial's held cap is not yet a result.

    The base is ``test_a_search_holding_a_trials_cap_goes_on_through_a_plateau_still_falling``;
    every practical window reads as outward, and the first remaining fall is scripted
    beyond the tolerance.
    """
    import math

    loop = _loop()
    real_fall, asked = loop.remaining_fall, [0]

    def fall(*args, **kwargs):
        asked[0] += 1
        return math.inf if asked[0] == 1 else real_fall(*args, **kwargs)

    loop._practical_outward_window = lambda **kwargs: True
    loop.remaining_fall = fall
    _tree_root()
    from tests.test_distributional_endpoint_trial import _PLATEAU

    return _endpoint_trial_fit("efs", **_PLATEAU)


@_forced("multisecant/box-blocked")
def _accelerated_proposal_box_blocked() -> dict:
    """Lines 1570-1572: the box blocks every accelerated proposal (the matrix's
    multisecant Tweedie fit, binned)."""
    _loop()._accelerated_proposal = lambda *args, **kwargs: None
    return _matrix_case("tweedie", "binned", "efs", "practical", acceleration="multisecant")


@_forced("start/supplied-lambdas")
def _supplied_lambdas() -> dict:
    """The golden test's own start: supplied lambdas per knot interval, under plain ``efs``."""
    _tree_root()
    from tests.bound_predictor_fixtures import model_from_templates
    from tests.test_distributional_reorg_golden import _cases, _per_knot

    cases, frame = _cases()
    distribution, predictors, y = cases["gaussian"]
    starts = {key: 0.1 * value for key, value in _per_knot(frame, predictors).items()}
    model = model_from_templates(family=distribution, predictors=predictors)
    model.fit_reml(frame.iloc[:TRAIN_ROWS], y[:TRAIN_ROWS], outer="efs", initial_lambda=0.1,
                   lambdas=starts)  # fmt: skip
    return _model_outputs(model, frame.iloc[TRAIN_ROWS:])


@_forced("box/absolute-max-lambda")
def _absolute_box() -> dict:
    """``max_lambda`` set, so the box is absolute (the endpoint trial fixture, ``efs+finish``)."""
    return _endpoint_trial_fit("efs+finish", max_lambda=1.0e6)


@_forced("unearned/retracted-face-then-stalled-finish")
def _unearned_survives() -> dict:
    """Design section 8.2: does ``unearned`` from an earlier rejected round reach a later
    stalled finish that has since earned its plateau?

    ``efs+finish`` on the endpoint trial fixture. With ``z`` held at the cap a trial
    reached, every trial of the round after it is scripted as rejected, so the round
    is unearned and goes to the finish, which first has the held cap judged and
    promoted. That finish runs two Newton steps and is reported stationary on the
    face (on this fixture a real finish spends its whole budget); the strict stop's
    revalidation is scripted to find the face's direction finite, so the face is
    released and Fellner-Schall rounds go on. They are accepted up to a practical
    plateau, which hands to a second finish, scripted to stall: the search reports
    ``objective_rejected``, the rejected round's verdict, though accepted rounds have
    since earned the plateau.
    """

    from dataclasses import replace

    import superglm.distributional.smoothing.faces as faces

    loop, rounds = _loop(), _rounds()
    real_ceiling, real_check, checks = loop._within_ceiling, faces._check_face_direction, [0]
    trial_round: list[int] = []
    real_endpoint = loop._endpoint_proposal

    def endpoint(*args, **kwargs):
        trial_round.append(rounds[0])
        return real_endpoint(*args, **kwargs)

    def ceiling(*args, **kwargs):
        rejected = bool(trial_round) and rounds[0] == trial_round[0] + 1
        return False if rejected else real_ceiling(*args, **kwargs)

    def check(*args, **kwargs):
        # the face's direction at the strict stop's revalidation, made finite once
        attempt = real_check(*args, **kwargs)
        checks[0] += 1
        if checks[0] != 2 or attempt.check is None:
            return attempt
        direction = attempt.check.direction
        error = direction.numerical_error
        finite = replace(direction, decision="finite", analytic_derivative=-1.0,
                         profile_score_term=-2.0, curvature_schur_term=0.0,
                         curvature_drift_term=0.0, lower_bound=-1.0 - error,
                         upper_bound=-1.0 + error)  # fmt: skip
        return replace(attempt, check=replace(attempt.check, direction=finite))

    loop._endpoint_proposal = endpoint
    loop._within_ceiling = ceiling
    faces._check_face_direction = check
    _endgame_kinds("stationary", "objective_rejected")
    scripted, finishes = loop.run_newton_endgame, [0]

    def two_steps_first(*args, **kwargs):
        finishes[0] += 1
        if finishes[0] == 1:
            kwargs = {**kwargs, "budget": min(2, kwargs["budget"])}
        return scripted(*args, **kwargs)

    loop.run_newton_endgame = two_steps_first
    _tree_root()
    from tests.test_distributional_endpoint_trial import _PLATEAU

    return _endpoint_trial_fit("efs+finish", **_PLATEAU)


# --------------------------------------------------------------------------- digests

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _scrub(text: str) -> str:
    return _ADDRESS.sub("0x?", text)


class _Fits:
    """The coefficient fits of the search running now, by identity."""

    def __init__(self) -> None:
        self.fits: list | None = None
        self.index: dict[int, int] = {}

    def refresh(self) -> None:
        fits = self.fits
        self.index = {} if fits is None else {id(fit): k for k, fit in enumerate(fits)}


_FITS = _Fits()
_STOP_SUMMARY = irls._stop_summary
# Runtime machinery the search hands its callees: named, not walked.
_NAMED = re.compile(r"(?:Session|Memo|Recorder|Anderson|Layout|Family|Plan)$")


def _stop_summary(obj) -> str | None:
    """A fit in the search's list is its index; runtime machinery is named, not walked."""
    from superglm.distributional.results.solver import DenseSolverResult

    if isinstance(obj, DenseSolverResult):
        index = _FITS.index.get(id(obj))
        return None if index is None else f"<fit {index}>"
    name = type(obj).__qualname__
    if _NAMED.search(name):
        return f"<{name}>"
    return _STOP_SUMMARY(obj)


# ``irls._digest`` reads ``_stop_summary`` and ``_SKIPPED_KEY`` from its module at call
# time. ``diagnose()``'s phase profile is wall-clock too: its seconds and their shares.
irls._stop_summary = _stop_summary
irls._SKIPPED_KEY = re.compile(irls._SKIPPED_KEY.pattern + r"|^(?:seconds|fit_share)$")


def _plain(value: object) -> object:
    """A callable passed as an argument is its name: a closure and a method alike."""
    if isinstance(value, types.FunctionType | types.MethodType | types.BuiltinFunctionType):
        return f"<callable {value.__name__}>"
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_plain(item) for item in value)
    return value


def _digest(obj) -> object:
    _FITS.refresh()
    return irls._digest(_plain(obj))


# --------------------------------------------------------------------------- the search's state

_MISSING = object()
# Where a name of the function's locals lives once the search is an object.
_ALIASES = {
    "current_lambdas": ("state.lambdas",),
    "current_fit": ("state.fit",),
    "current_objective": ("state.objective",),
    "current_face": ("state.face",),
    "terminal_evidence": ("state.evidence",),
    "terminal_fit_index": ("state.terminal_fit_index",),
    "plateau_run": ("windows.plateau_run",),
    "practical_run": ("windows.practical_run",),
    "plateau_qualified": ("windows.plateau_qualified",),
    "previous_accepted_step": ("windows.previous_accepted_step",),
}
_PASS_FIELDS = (
    "cap_held", "caps_final", "refused_fit", "endpoint_tried", "endpoint_caps", "endgame_due",
    "handed_over", "plateau_earned", "unearned", "endgame_stationary", "pending_cap",
    "newton_budget", "plateau_run", "practical_run", "plateau_qualified",
    "previous_accepted_step", "fall", "previous_fall", "current_face", "beyond_cap",
    "estimated_names", "endgame_outcome", "terminal_fit_index", "current_objective",
    "terminal_evidence",
)  # fmt: skip


def _read(frame, name: str) -> object:
    """``name`` from a frame of the search: a local, or an attribute of its ``self``."""
    local = frame.f_locals
    if name in local:
        return local[name]
    owner = local.get("self")
    if owner is None:
        return _MISSING
    for path in (name, *_ALIASES.get(name, ())):
        value = owner
        for part in path.split("."):
            value = getattr(value, part, _MISSING)
            if value is _MISSING:
                break
        if value is not _MISSING:
            return value
    return _MISSING


def _search_fits(frame) -> list | None:
    """The search's ``coefficient_fits`` list, from the nearest frame of the search."""
    loop_file = sys.modules[LOOP_MODULE].__file__
    while frame is not None:
        if frame.f_code.co_filename == loop_file:
            fits = _read(frame, "coefficient_fits")
            if isinstance(fits, list):
                return fits
        frame = frame.f_back
    return None


def _pass_state(frame) -> dict:
    """The search's state at the top of a pass, as the census and the comparison read it."""
    out: dict = {}
    for name in _PASS_FIELDS:
        value = _read(frame, name)
        if value is _MISSING:
            out[name] = "<missing>"
        elif name == "current_face":
            out[name] = None if value is None else list(value.component_names)
        elif name in ("pending_cap", "endgame_outcome"):
            out[name] = None if value is None else value.kind
        elif name == "terminal_evidence":
            out[name] = {
                "pressed": list(value.unresolved_upper_bound),
                "maximum": float(value.maximum),
                "working_infinity": list(value.working_infinity),
                "cap_held": sorted(value.cap_held),
            }
        elif isinstance(value, set | frozenset):
            out[name] = sorted(value)
        elif isinstance(value, dict):
            out[name] = {key: value[key] for key in sorted(value)}
        else:
            out[name] = value
    history = _read(frame, "history")
    out["history"] = len(history) if isinstance(history, list) else "<missing>"
    out["outer"] = getattr(_read(frame, "outer_config"), "outer", "<missing>")
    return out


def _census_state(state: dict) -> str:
    """One pass's joint state in the columns of the design's section 1.4 table."""

    def flag(key, label):
        value = state.get(key)
        return label if value not in (None, False, [], {}, "<missing>") else "-"

    evidence = state.get("terminal_evidence") or {}
    return "|".join(
        [
            str(state.get("outer")),
            flag("cap_held", "held"),
            flag("caps_final", "final"),
            flag("refused_fit", "refused"),
            flag("endpoint_tried", "tried"),
            flag("endpoint_caps", "trialcap"),
            flag("endgame_due", "due"),
            flag("handed_over", "handed"),
            flag("plateau_earned", "earned"),
            flag("unearned", "unearned"),
            flag("endgame_stationary", "stationary"),
            flag("pending_cap", "pending"),
            flag("current_face", "face"),
            "pressed" if evidence.get("pressed") else "-",
            flag("beyond_cap", "beyond"),
        ]
    )


# --------------------------------------------------------------------------- recording


class Recorder:
    """Every call the search makes out of its module, every pass's state, every result."""

    _ACCELERATOR = ("record_accepted", "propose", "reject", "reset")

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.census: Counter = Counter()
        self.depth = 0
        self.searches = 0
        self.targets: list[str] = []

    def install(self) -> dict:
        import importlib

        import superglm.distributional.model  # noqa: F401 - binds the entry point
        from superglm.distributional.smoothing import acceleration

        loop = importlib.import_module(LOOP_MODULE)
        self.loop = loop
        self.targets = sorted(
            name
            for name, value in vars(loop).items()
            if isinstance(value, types.FunctionType)
            and (value.__module__ or "").startswith("superglm")
            and value.__module__ != LOOP_MODULE
        ) + ["_trial_laplace_objective", "_within_ceiling"]
        for method in self._ACCELERATOR:
            original = getattr(acceleration.WindowedTypeIIAnderson, method)
            setattr(
                acceleration.WindowedTypeIIAnderson, method, self._accelerator(method, original)
            )
        entry = loop.fit_distributional_efs
        return {"fit_distributional_efs": irls._rebind(entry, self._entry(entry)),
                "traced_names": len(self.targets)}  # fmt: skip

    def _entry(self, original: Callable) -> Callable:
        def fit_distributional_efs(*args, **kwargs):
            namespace = vars(self.loop)
            saved = {name: namespace[name] for name in self.targets}
            search = self.searches
            self.searches += 1
            fits_before, _FITS.fits = _FITS.fits, None
            self.calls.append({"search": search, "callee": "<enter>", "args": _digest([args[0].__class__.__qualname__, *args[2:], kwargs])})  # fmt: skip
            for name, value in saved.items():
                namespace[name] = self._traced(name, value)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.calls.append({"search": search, "callee": "<exit>", "error": _error(exc)})
                self.census[f"search_error:{type(exc).__qualname__}"] += 1
                raise
            finally:
                for name, value in saved.items():
                    namespace[name] = value
                # the result's fits are walked whole, not named by their index
                _FITS.fits = fits_before
            self.calls.append({
                "search": search, "callee": "<exit>", "result": _digest(result),
                "converged_type": type(result.converged).__qualname__,
            })  # fmt: skip
            self.census["searches"] += 1
            self.census[f"reason:{result.convergence_reason}"] += 1
            self.census["outer_iterations"] += int(result.iterations)
            self.census["coefficient_fits"] += len(result.coefficient_fits)
            return result

        return fit_distributional_efs

    def _traced(self, name: str, original: Callable) -> Callable:
        recorder = self

        def traced(*args, **kwargs):
            caller = sys._getframe(1)
            if _FITS.fits is None:
                _FITS.fits = _search_fits(caller)
            record: dict = {"callee": name, "depth": recorder.depth}
            if name == "compact_coefficient_history":
                if caller.f_code.co_name == "_efs_result":
                    record["pass"] = "result"
                else:
                    state = _pass_state(caller)
                    recorder.census[f"pass:{_census_state(state)}"] += 1
                    record["pass"] = _digest(state)
            record["args"] = _digest([list(args), kwargs])
            recorder.calls.append(record)
            recorder.census[f"call:{name}"] += 1
            recorder.depth += 1
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                record["error"] = _error(exc)
                raise
            finally:
                recorder.depth -= 1
            if name == "compact_coefficient_history" and _FITS.fits is not None:
                compacted = [k for k, fit in enumerate(_FITS.fits) if fit.eta is None]
                record["compacted"] = irls._digest(compacted)
            record["result"] = _digest(result)
            return result

        traced.__name__ = getattr(original, "__name__", name)
        traced.__qualname__ = getattr(original, "__qualname__", name)
        traced.__wrapped__ = original
        return traced

    def _accelerator(self, method: str, original: Callable) -> Callable:
        recorder = self

        def traced(accelerator, *args, **kwargs):
            if sys._getframe(1).f_code.co_filename != recorder.loop.__file__:
                return original(accelerator, *args, **kwargs)
            record = {"callee": f"accelerator.{method}", "depth": recorder.depth,
                      "args": _digest([list(args), kwargs])}  # fmt: skip
            recorder.calls.append(record)
            recorder.census[f"call:accelerator.{method}"] += 1
            result = original(accelerator, *args, **kwargs)
            record["result"] = _digest(result)
            return result

        return traced


def _error(exc: BaseException) -> dict:
    return {"type": type(exc).__qualname__, "message": _scrub(str(exc))}


# --------------------------------------------------------------------------- case process


def _child(name: str, out: Path) -> int:
    for key, value in PINNED_ENV.items():
        if os.environ.get(key) != value:
            raise SystemExit(f"case process needs {key}={value}; run it through `run`")
    import warnings

    recorder = Recorder()
    rebound = recorder.install()
    builder, kwargs = _registry()[name]
    status, error, outputs = "ok", None, {}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            outputs = builder(**kwargs)
        except Exception as exc:
            status, error = "error", _error(exc)
    elapsed = time.perf_counter() - started
    if name.startswith("t/") and outputs.get("exit") != 0:
        status, error = "error", {"type": "pytest", "message": f"exit {outputs.get('exit')}"}
    warned = Counter(f"{w.category.__qualname__}: {_scrub(str(w.message))}" for w in caught)
    tree = {
        "status": status,
        "error": error,
        "calls": recorder.calls,
        "model": _digest(outputs),
        "warnings": sorted(f"{count}x {text}" for text, count in warned.items()),
    }
    leaves: dict = {}
    irls._flatten(tree, "", leaves)
    by_call: dict[int, dict] = {}
    for path, value in leaves.items():
        if path.startswith("calls["):
            by_call.setdefault(int(path[6 : path.index("]")]), {})[path] = value
    call_hashes = [irls._leaves_hash(by_call.get(k, {})) for k in range(len(recorder.calls))]
    model_leaves = {p: v for p, v in leaves.items() if not p.startswith("calls[")}
    header = {
        "case": name,
        "status": status,
        "error": error,
        "case_hash": irls._leaves_hash(leaves),
        "model_hash": irls._leaves_hash(model_leaves),
        "call_hashes": call_hashes,
        "callees": [record["callee"] for record in recorder.calls],
        "leaf_count": len(leaves),
        "census": dict(recorder.census),
        "rebound": rebound,
        "elapsed_s": round(elapsed, 3),
        "detail": _DETAIL,
    }
    stem = out / "cases" / irls._case_id(name)
    with gzip.open(f"{stem}.leaves.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(leaves, handle)
    Path(f"{stem}.hash.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- run, compare


def _run(args) -> int:
    out = Path(args.out).resolve()
    (out / "cases").mkdir(parents=True, exist_ok=True)
    names = sorted(_registry(), key=_case_order)
    if args.only:
        pattern = re.compile(args.only)
        names = [name for name in names if pattern.search(name)]
    env = {**os.environ, **PINNED_ENV}
    command = [sys.executable]
    if args.coverage:
        (out / "coverage").mkdir(exist_ok=True)
        command += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={_coverage_rc(out)}",
                    f"--data-file={out / 'coverage' / '.coverage'}"]  # fmt: skip
    command += [str(Path(__file__).resolve()), "_case"]
    started = time.perf_counter()

    def one(name: str) -> tuple[str, str, float]:
        stem = out / "cases" / irls._case_id(name)
        Path(f"{stem}.hash.json").unlink(missing_ok=True)
        t0 = time.perf_counter()
        try:
            done = subprocess.run(
                [*command, name, "--out", str(out)],
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            status = None if done.returncode == 0 else f"crash (exit {done.returncode})"
            detail = done.stderr[-2000:]
        except subprocess.TimeoutExpired:
            status, detail = f"timeout ({args.timeout} s)", ""
        if status is not None:
            header = {"case": name, "status": status, "error": {"type": status, "message": detail},
                      "case_hash": status, "model_hash": status, "call_hashes": [], "census": {}}  # fmt: skip
            Path(f"{stem}.hash.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
        header = json.loads(Path(f"{stem}.hash.json").read_text(encoding="utf-8"))
        return name, header["status"], time.perf_counter() - t0

    statuses: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, name) for name in names]
        for future in as_completed(futures):
            name, status, seconds = future.result()
            statuses[status] += 1
            if status != "ok" or args.verbose:
                print(f"{status:>8}  {seconds:7.1f} s  {name}", flush=True)
    wall = time.perf_counter() - started
    manifest = {
        "tree": irls._tree_identity(),
        "harness": irls._hash(Path(__file__).read_bytes()),
        "env": {key: env.get(key) for key in (*PINNED_ENV, "NUMBA_CACHE_DIR")},
        "selection": {"only": args.only},
        "cases": {},
    }
    for name in names:
        header = json.loads((out / "cases" / f"{irls._case_id(name)}.hash.json").read_text())
        manifest["cases"][name] = header["case_hash"]
    (out / "manifest-all.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"{len(names)} cases in {wall:.0f} s on {args.jobs} jobs: "
        + ", ".join(f"{v} {k}" for k, v in statuses.items())
    )
    if args.coverage:
        _coverage_report(out)
    return 0


def _coverage_rc(out: Path) -> Path:
    """The run's own coverage settings: branch coverage of the search's module only."""
    rc = out / "coverage" / "coveragerc"
    include = "include =\n    */superglm/distributional/smoothing/loop.py\n"
    rc.write_text(f"[run]\nbranch = True\n{include}[report]\n{include}", encoding="utf-8")
    return rc


def _search_spans(path: Path) -> list[tuple[str, int, int]]:
    """``fit_distributional_efs`` and every class of the module: the search, in either shape."""
    spans = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        wanted = isinstance(node, ast.FunctionDef) and node.name == "fit_distributional_efs"
        if wanted or isinstance(node, ast.ClassDef):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            spans.append((node.name, first, node.end_lineno))
    return spans


def _coverage_report(out: Path) -> None:
    import coverage

    data, rc = out / "coverage" / ".coverage", _coverage_rc(out)
    subprocess.run([sys.executable, "-m", "coverage", "combine", "--append", f"--rcfile={rc}",
                    f"--data-file={data}"], cwd=out / "coverage", capture_output=True, check=False)  # fmt: skip
    cov = coverage.Coverage(data_file=str(data), config_file=str(rc), branch=True)
    cov.load()
    files = [f for f in cov.get_data().measured_files() if f.endswith("smoothing/loop.py")]
    if not files:
        print("coverage: no data")
        return
    path = Path(files[0])
    analysis = cov._analyze(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    executed = set(analysis.arcs_executed)
    branches = set(analysis._branch_lines())
    report: dict = {"file": str(path), "spans": {}}
    for name, first, last in _search_spans(path):
        arms = sorted(
            (a, b) for a, b in analysis.arc_possibilities if first <= a <= last and a in branches
        )
        if not arms:
            continue
        missed = [(a, b) for a, b in arms if (a, b) not in executed]
        report["spans"][name] = {"arms": len(arms), "missed": missed}
        print(f"coverage {name} (lines {first}-{last}): {len(arms) - len(missed)} of "
              f"{len(arms)} branch arms taken")  # fmt: skip
        for a, b in missed:
            target = "exit" if b < 0 else str(b)
            print(f"  never taken {a} -> {target}: {lines[a - 1].strip()[:90]}")
    (out / "coverage" / "arms.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def _compare(args) -> int:
    code = irls._compare(args)
    a_headers, b_headers = irls._headers(Path(args.a)), irls._headers(Path(args.b))
    for name in sorted(set(a_headers) & set(b_headers)):
        a, b = a_headers[name], b_headers[name]
        if a["case_hash"] == b["case_hash"]:
            continue
        pairs = zip(a.get("call_hashes", []), b.get("call_hashes", []), strict=False)
        first = next((k for k, (x, y) in enumerate(pairs) if x != y), None)
        if first is None:
            first = min(len(a.get("call_hashes", [])), len(b.get("call_hashes", [])))
        a_callees, b_callees = a.get("callees", []), b.get("callees", [])

        def callee(names, k):
            return names[k] if k < len(names) else "<none>"

        print(f"{name}: first differing call {first}: A {callee(a_callees, first)}, "
              f"B {callee(b_callees, first)}")  # fmt: skip
    return code


def _census(args) -> int:
    headers = irls._headers(Path(args.directory))
    states: Counter = Counter()
    calls: Counter = Counter()
    for header in headers.values():
        for key, value in header.get("census", {}).items():
            (states if key.startswith("pass:") else calls)[key] += value
    columns = ("outer held final refused tried trialcap due handed earned unearned "
               "stationary pending face pressed beyond").split()  # fmt: skip
    print(f"{sum(states.values())} passes in {len(headers)} cases, {len(states)} joint states")
    print("  passes  " + " ".join(f"{c:>10}" for c in columns))
    for key, count in states.most_common():
        print(f"  {count:6d}  " + " ".join(f"{v:>10}" for v in key[5:].split("|")))
    print("calls and results (total over the cases):")
    for key in sorted(calls):
        print(f"  {key:56s} {calls[key]:9d}")
    errors = [name for name, header in headers.items() if header["status"] != "ok"]
    if errors:
        print(f"{len(errors)} cases did not finish ok (their outcome is pinned too):")
        for name in errors:
            error = headers[name].get("error") or {}
            print(f"  {headers[name]['status']}: {name}: {error.get('type')}: "
                  f"{str(error.get('message'))[:160]}")  # fmt: skip
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--jobs", type=int, default=8)
    run.add_argument("--only")
    run.add_argument("--timeout", type=int, default=CASE_TIMEOUT_S)
    run.add_argument("--coverage", action="store_true")
    run.add_argument("--verbose", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--show", type=int, default=12)
    census = sub.add_parser("census")
    census.add_argument("directory")
    case = sub.add_parser("_case")
    case.add_argument("name")
    case.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        for name in sorted(_registry(), key=_case_order):
            print(name)
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "census":
        return _census(args)
    return _child(args.name, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
