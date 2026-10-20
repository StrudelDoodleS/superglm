"""Never break userspace: what v0.40.0 accepted keeps working (the 0.41 audit's guard).

The fixtures in ``fixtures/saved_v0_40_0`` were written by the v0.40.0 release
(commit 542f5826) with ``scripts/make_saved_v040_fixtures.py``.  A model saved
by that release loads, predicts what it predicted then, and refits.

``fixtures/release_differential_v0_40_0.jsonl`` holds v0.40.0's outcomes on the
release differential's compact subset (``benchmarks/release_differential.py
run --subset`` on 542f5826): every case converged there, so each must converge
here, at an objective and held-out results no worse than the harness's
tolerances.

``fixtures/saved_v0_17_0``, ``saved_v0_18_0`` and ``saved_v0_25_0`` hold small
models those releases saved (the same script, run with each release): one per
class and era of a term that gained predict-time state after its saver, which
the 0.41 audit found failing at predict until ``__setstate__`` gave it the
state its saver scored with.
"""

from __future__ import annotations

import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from benchmarks import release_differential

from superglm import SuperLSS
from superglm._group_matrix import _group_matrix_centered

from .test_saved_fs_models import _gamma, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_40_0"
GLM_RECORDS = sorted(path.stem for path in FIXTURES.glob("glm_*.pkl")) + ["sklearn_regressor"]
LSS_RECORDS = sorted(path.stem for path in FIXTURES.glob("lss_*.pkl"))


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def test_the_fixture_set_covers_every_saved_route() -> None:
    assert len(GLM_RECORDS) == 11
    assert LSS_RECORDS == ["lss_gamma", "lss_gaussian", "lss_nb", "lss_tweedie", "lss_two_piece"]


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


def _lss_eta_rounding(model: SuperLSS, frame) -> np.ndarray:
    """Per row and parameter, the bound two float64 evaluations of the linear predictors meet.

    Each predictor is ``b0 + sum_t X_t beta_t`` over its terms, a dot product
    of at most ``p + 1`` terms, so two evaluations differ by at most
    ``2 gamma_{p+1} (|b0| + sum_j |X_ij| |beta_j|)`` (Higham 2002, eq. 3.5),
    the magnitude formed one basis column at a time.
    """
    from superglm._frame import as_eager_frame
    from superglm.distributional.model import _score_feature, _term_indices

    fitted = model._require_fitted()
    rows = as_eager_frame(frame)
    columns = []
    for predictor in fitted.compiled_predictors:
        state = fitted.layout.predictors[predictor.parameter_index]
        local = np.asarray(fitted.result.coefficients[state.coefficient_slice], dtype=np.float64)
        width = int(state.intercept_index is not None)
        slopes = local[width:]
        magnitude = np.full(len(rows), abs(float(local[0])) if width else 0.0)
        assert not predictor.compiled.interaction_order
        for name in predictor.compiled.feature_order:
            indices = np.atleast_1d(
                np.arange(len(slopes))[_term_indices(predictor.compiled.groups, name)]
            )
            spec, values = predictor.compiled.specs[name], rows.column_array(name)
            for position, j in enumerate(indices):
                unit = np.zeros(len(indices))
                unit[position] = 1.0
                contribution = np.asarray(_score_feature(spec, values, unit), dtype=np.float64)
                magnitude += np.abs(contribution).reshape(-1) * abs(slopes[j])
        columns.append(2.0 * _gamma(local.size + 1) * magnitude)
    return np.column_stack(columns)


@pytest.mark.parametrize("name", LSS_RECORDS)
def test_a_superlss_model_saved_by_v0_40_0_loads_and_predicts(name: str) -> None:
    """``from_bytes`` restores v0.40.0's coefficients bit for bit, and the linear
    predictors are v0.40.0's within two evaluations' rounding."""
    record = _load(name)
    assert record["version"] == "0.40.0"
    model = SuperLSS.from_bytes(record["serialized"])
    frame = record["frame"]
    restored = dict(model.coef_)
    assert list(restored) == list(record["coefficients"])
    assert all(
        math.copysign(1.0, restored[key]) == math.copysign(1.0, value) and restored[key] == value
        for key, value in record["coefficients"].items()
    )
    eta = model.predict_link(frame).to_numpy(dtype=np.float64)
    assert eta.shape == record["eta"].shape
    excess = np.abs(eta - record["eta"]) / _lss_eta_rounding(model, frame)
    assert np.all(excess <= 1.0), float(np.max(excess))
    assert np.all(np.isfinite(np.asarray(model.predict(frame), dtype=np.float64)))


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


@pytest.mark.parametrize("ref", ["master", "3681b0c1", "v0.41.0rc1"])
def test_a_base_that_is_not_a_release_tag_is_compared_without_a_criterion_exemption(ref) -> None:
    """``release REF`` ends in a compare against REF. Fails on df503618, where a REF other
    than ``vX.Y.Z`` raised ValueError there after both corpora had run."""
    name = "glm/gamma/log/small/mains/binned/fit_reml"
    assert release_differential.flags(name, _record(), _record(objective=1001.0), ref)


def test_the_release_command_gives_the_baseline_an_absolute_output_path(
    tmp_path, monkeypatch
) -> None:
    """The baseline runs with its own tree as the working directory, so a relative --out
    would point inside that tree. Fails on eedfa48e, which passed the path as given."""
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

    assert release_differential.main(["release", "v0.40.0", "--out", "results"]) == 0
    baseline = Path(launched[0][launched[0].index("--out") + 1])
    assert baseline.is_absolute()
    assert baseline == tmp_path / "results" / "v0.40.0.jsonl"


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


@pytest.mark.parametrize("name", release_differential.HISTORY_SUBSET)
def test_a_fit_an_earlier_release_converged_converges_here(name: str) -> None:
    """Breaks the release-history sweep found, fixed: v0.40.0 refused rows 2 and
    13 and left row 3 unconverged; v0.39.0 converged all three to the same fits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = {"status": "ok", **release_differential.run_case(name)}
    assert release_differential.sensible(record), record
