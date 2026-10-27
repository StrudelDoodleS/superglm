"""Never break userspace: what v0.40.0 accepted keeps working (the 0.41 audit's guard).

The fixtures in ``fixtures/saved_v0_40_0`` were written by the v0.40.0 release
(commit 542f5826) with ``scripts/make_saved_v040_fixtures.py``.  A model saved
by that release loads, predicts what it predicted then, and refits.

``fixtures/release_differential_v0_40_0.jsonl`` holds v0.40.0's outcomes on the
release differential's compact subset (``benchmarks/release_differential.py
run --subset`` on 542f5826): every case converged there, so each must converge
here, at an objective and held-out results no worse than the harness's
tolerances.

``fixtures/release_differential_history_v0_39_0.jsonl`` holds v0.39.0's outcomes
on the three history cases rows 2, 3 and 13 of the release-history sweep. It was
produced on the v0.39.0 release (tag commit 30167bfb), installed non-editable
into a venv from its frozen lock (``git archive v0.39.0``, ``uv sync --frozen
--no-dev --no-editable``), by running this branch's harness with that venv's
interpreter: ``python benchmarks/release_differential.py run --only '^hist/'
--out FILE``. The harness needs only the public API, so no shim was used; the
three rows were kept from the output. Rerunning it reproduces the file's
objectives and held-out deviances bit for bit.

``fixtures/saved_v0_17_0``, ``saved_v0_18_0`` and ``saved_v0_25_0`` hold small
models those releases saved (the same script, run with each release): one per
class and era of a term that gained predict-time state after its saver, which
the 0.41 audit found failing at predict until ``__setstate__`` gave it the
state its saver scored with.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from benchmarks import release_differential

from superglm._group_matrix import _group_matrix_centered

from .test_saved_fs_models import assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_40_0"
GLM_RECORDS = sorted(path.stem for path in FIXTURES.glob("glm_*.pkl")) + ["sklearn_regressor"]


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def test_the_fixture_set_covers_every_saved_glm_route() -> None:
    assert len(GLM_RECORDS) == 11


@pytest.mark.parametrize("name", GLM_RECORDS)
def test_a_model_saved_by_v0_40_0_loads_predicts_and_refits(name: str) -> None:
    """Predictions are v0.40.0's within two evaluations' rounding
    (``assert_predicts_as_saved``), and the loaded model refits the same data
    with the fit method that made it."""
    record = _load(name)
    assert record["version"] == "0.40.0"
    loaded, frame, y = record["model"], record["frame"], record["y"]
    model = getattr(loaded, "_model", loaded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model is loaded:
            assert_predicts_as_saved(model, frame, record["prediction"])
        else:  # the sklearn wrapper predicts through its SuperGLM
            np.testing.assert_array_equal(loaded.predict(frame), model.predict(frame))
            assert_predicts_as_saved(model, frame, record["prediction"])
        assert str(model.summary())
        getattr(loaded, record["fit"])(frame, y)
    assert np.all(np.isfinite(np.asarray(loaded.predict(frame), dtype=np.float64)))


def test_a_model_saved_with_fast_prediction_metadata_loads_without_it() -> None:
    """Releases through 0.40.0 pickled the metadata of a fast discrete predictor.

    The binned fixture carries it; the model loads without it and predicts as saved.
    """
    record = _load("glm_poisson_binned")
    model = record["model"]
    assert "_fast_prediction_state" not in model.__dict__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_predicts_as_saved(model, record["frame"], record["prediction"])


LEGACY_RECORDS = [
    (saver, path.stem)
    for saver in ("0.17.0", "0.18.0", "0.25.0")
    for path in sorted((FIXTURES.parent / f"saved_v{saver.replace('.', '_')}").glob("*.pkl"))
]


def test_the_older_savers_cover_every_term_that_gained_predict_state() -> None:
    """A spline-by-category term saved before 0.18 lacks its parent's grouping
    and, before 0.26, its level universe and unseen policy; an
    OrderedCategorical saved before 0.19 lacks specials.  The numeric,
    polynomial and categorical by-category terms share that restore
    (``_restore_parent_policy``)."""
    assert LEGACY_RECORDS == [
        ("0.17.0", "spline_cat"),
        ("0.18.0", "ordered_cat"),
        ("0.25.0", "spline_cat"),
    ]


def _load_legacy(saver: str, name: str) -> dict:
    with open(
        FIXTURES.parent / f"saved_v{saver.replace('.', '_')}" / f"{name}.pkl", "rb"
    ) as handle:
        return pickle.load(handle)


@pytest.mark.parametrize(("saver", "name"), LEGACY_RECORDS)
def test_a_model_saved_by_an_older_release_predicts_as_it_did_then(
    saver: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Predictions are the saver's own within two evaluations' rounding
    (``assert_predicts_as_saved``); without the ``__setstate__`` defaults the
    first predict raises ``AttributeError``.  The loaded model also summarises
    and refits.

    ``summary`` runs twice: as these small designs route it, and through the
    raw-moment route a design of ``_MIN_MIXED_RAW_MOMENT_CELLS`` cells takes,
    which forms each group's Gram (a spline-by-category block saved by 0.17
    has no ``_dense_level``).
    """
    record = _load_legacy(saver, name)
    assert record["version"] == saver
    model, frame = record["model"], record["frame"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_predicts_as_saved(model, frame, record["prediction"])
        assert str(model.summary())
        getattr(model, record["fit"])(frame, record["y"])
        assert np.all(np.isfinite(np.asarray(model.predict(frame), dtype=np.float64)))

        monkeypatch.setattr(_group_matrix_centered, "_MIN_MIXED_RAW_MOMENT_CELLS", 0)
        assert str(_load_legacy(saver, name)["model"].summary())


RELEASE_OUTCOMES = release_differential.load(FIXTURES.parent / "release_differential_v0_40_0.jsonl")


def test_the_pinned_outcomes_are_the_subset_and_all_converged_on_v0_40_0() -> None:
    assert list(RELEASE_OUTCOMES) == list(release_differential.SUBSET)
    assert all(release_differential.sensible(record) for record in RELEASE_OUTCOMES.values())


def _record(**changes) -> dict:
    record = {
        "status": "ok",
        "converged": True,
        "objective": 1000.0,
        "test_nll": 0.5,
        "test_deviance": 800.0,
        "gini": 0.3,
        "max_mu": 4.0,
    }
    return {**record, **changes}


@pytest.mark.parametrize(
    "broken",
    [
        {"objective": float("nan")},
        {"test_nll": float("nan")},
        {"test_deviance": float("inf")},
        {"gini": float("nan")},
        {"max_mu": float("inf")},
        {"max_mu": float("nan")},
    ],
)
def test_the_differential_flags_a_converged_candidate_with_non_finite_results(broken) -> None:
    """Fails on eedfa48e, where non-finite candidate metrics skipped every comparison."""
    name = "glm/poisson/log/small/mains/exact/fit_reml"
    assert release_differential.flags(name, _record(), _record(), "v0.40.0") == []
    assert release_differential.flags(name, _record(), _record(**broken), "v0.40.0")


def test_the_criterion_change_exempts_only_bases_that_predate_it() -> None:
    """The objective of a pair whose criterion changed in v0.41 is compared again from a
    v0.41 base on. Fails on eedfa48e, which exempted the pair for every base."""
    name = "glm/gamma/log/small/mains/binned/fit_reml"
    worse = _record(objective=1000.0 + 1.0)
    assert release_differential.flags(name, _record(), worse, "v0.40.0") == []
    assert release_differential.flags(name, _record(), worse, "v0.41.0")
    assert release_differential.flags(name, _record(), worse, "v0.42.0")
    assert release_differential.flags(name, _record(), worse, "v0.41")  # read as 0.41.0


def test_the_engine_change_exempts_only_bases_that_predate_it() -> None:
    """A constrained P-spline fits on the QP from v0.42, a different model whose REML
    criterion is not SCOP's: its objective is compared again only from a v0.42 base on,
    and its held-out results always."""
    name = "glm/poisson/log/small/scop/exact/fit_reml"
    worse = _record(objective=1000.0 + 1.0)
    assert release_differential.flags(name, _record(), worse, "v0.41.0") == []
    assert (
        release_differential.flags("hist/r01/convex-beside-cr/exact", _record(), worse, "v0.41.0")
        == []
    )
    assert release_differential.flags(name, _record(), worse, "v0.42.0")
    assert release_differential.flags(name, _record(), _record(test_deviance=900.0), "v0.41.0")
    assert release_differential.flags(name.replace("/scop/", "/qp/"), _record(), worse, "v0.41.0")


@pytest.mark.parametrize("ref", ["master", "3681b0c1", "v0.41.0rc1"])
def test_a_base_that_is_not_a_release_tag_is_compared_without_a_criterion_exemption(ref) -> None:
    """``release REF`` ends in a compare against REF. Fails on df503618, where a REF other
    than ``vX.Y.Z`` raised ValueError there after both corpora had run."""
    name = "glm/gamma/log/small/mains/binned/fit_reml"
    assert release_differential.flags(name, _record(), _record(objective=1001.0), ref)


@pytest.mark.parametrize(
    ("ref", "stem"), [("v0.40.0", "v0.40.0"), ("origin/master", "origin_master")]
)
def test_the_release_command_gives_the_baseline_an_absolute_output_path(
    tmp_path, monkeypatch, ref, stem
) -> None:
    """The baseline runs with its own tree as the working directory, so a relative --out
    would point inside that tree. Fails on eedfa48e, which passed the path as given. A
    ref with a slash names a flat file (fails on 951f0e4: ``results/origin/master``)."""
    from types import SimpleNamespace

    launched = []
    books = SimpleNamespace(
        _resolve=lambda ref: ref,
        prepare_tree=lambda ref: tmp_path / "tree",
        _venv_python=lambda tree: "python",
        _clean_env=dict,
    )
    monkeypatch.setitem(sys.modules, "books", books)
    monkeypatch.setattr(
        release_differential.subprocess, "run", lambda command, **kwargs: launched.append(command)
    )
    monkeypatch.setattr(release_differential, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(release_differential, "load", lambda path: {})
    monkeypatch.chdir(tmp_path)

    assert release_differential.main(["release", ref, "--out", "results"]) == 0
    baseline = Path(launched[0][launched[0].index("--out") + 1])
    assert baseline.is_absolute()
    assert baseline == tmp_path / "results" / f"{stem}.jsonl"


@pytest.mark.slow
@pytest.mark.parametrize("name", release_differential.SUBSET)
def test_a_fit_v0_40_0_converged_still_converges_no_worse(name: str) -> None:
    """The fit is not refused, converges, and its objective (where the criterion
    is unchanged), held-out NLL, deviance and Gini are no worse than v0.40.0's
    beyond the harness's tolerances (``release_differential.flags``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = {"status": "ok", **release_differential.run_case(name)}
    assert release_differential.flags(name, RELEASE_OUTCOMES[name], record, "v0.40.0") == []


HISTORY_OUTCOMES = release_differential.load(
    FIXTURES.parent / "release_differential_history_v0_39_0.jsonl"
)


@pytest.mark.parametrize("name", release_differential.HISTORY_SUBSET)
def test_a_fit_an_earlier_release_converged_converges_here(name: str) -> None:
    """Breaks the release-history sweep found, fixed: v0.40.0 refused rows 2 and 13 and
    left row 3 unconverged. v0.39.0 (``release_differential_history_v0_39_0.jsonl``) fitted
    rows 2 and 13 to convergence and row 3 to the same objective, at its iteration cap.
    The fit here converges at that objective and held-out deviance. Both are held on both
    sides, as these fixes restore the same fit rather than a no-worse one: the objective
    to the harness's ``OBJECTIVE_TOLERANCE`` (relative to ``1 + |V|``, above the REML
    stop's own resolution), the deviance to its ``HELDOUT_TOLERANCE``, which is above
    criterion-level moves."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = {"status": "ok", **release_differential.run_case(name)}
    assert release_differential.sensible(record), record
    recorded = HISTORY_OUTCOMES[name]
    scale = 1.0 + abs(recorded["objective"])
    assert abs(record["objective"] - recorded["objective"]) <= (
        release_differential.OBJECTIVE_TOLERANCE * scale
    ), (record["objective"], recorded["objective"])
    assert abs(record["test_deviance"] - recorded["test_deviance"]) <= (
        release_differential.HELDOUT_TOLERANCE * abs(recorded["test_deviance"])
    ), (record["test_deviance"], recorded["test_deviance"])
