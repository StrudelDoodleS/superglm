"""Historical row retention must not change smoothing or replay authority."""

import base64
import gc
import hashlib
import json
import math
import pickle
import weakref
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm.distributional import GaussianLS, Predictor
from superglm.distributional import serialization as serialization_module
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
)
from superglm.distributional.serialization import (
    DistributionalSerializationError,
    deserialize_distributional_model,
    serialize_distributional_model,
)
from superglm.distributional.weights import WeightContract
from superglm.features import RandomEffect, Spline
from tests.bound_predictor_fixtures import model_from_templates


def _data():
    rng = np.random.default_rng(153)
    x = np.linspace(-1.0, 1.0, 96)
    return pd.DataFrame({"x": x}), np.sin(3 * x) + rng.normal(size=len(x)) * 0.35


def _fit(**config):
    frame, y = _data()
    options = dict(max_iterations=14, tolerance=1e-10, max_log_step=0.3)
    options.update(config)
    return fit_dense_distributional(
        frame,
        y,
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Spline(n_knots=5)}), Predictor("scale", {})),
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(tolerance=1e-9),
        efs_config=DistributionalEFSConfig(**options),
        retain_rows=False,
    )


def test_default_history_releases_obsolete_row_arrays():
    smoothing = _fit().smoothing
    assert len(smoothing.coefficient_fits) > 8
    assert smoothing.coefficient_fits[0].eta is None
    assert smoothing.coefficient_fits[0].theta is None
    assert smoothing.terminal_fit.eta.shape == (96, 2)


def test_public_full_history_opt_in():
    frame, y = _data()
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Spline(n_knots=5)}), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, max_reml_iter=2, retain_history_rows=True, retain_rows=False)
    assert model._model.smoothing.config.retain_history_rows is True
    assert all(fit.eta is not None for fit in model._model.smoothing.coefficient_fits)


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
def test_compact_and_full_history_preserve_fit_and_serialization(outer):
    compact = _fit(outer=outer, retain_history_rows=False, handoff_iterations=2)
    full = _fit(outer=outer, retain_history_rows=True, handoff_iterations=2)
    left, right = compact.smoothing, full.smoothing
    assert left.history == right.history
    assert left.lambdas == right.lambdas
    assert left.objective == right.objective
    assert left.convergence_reason == right.convergence_reason
    assert left.converged == right.converged
    assert left.terminal_endpoint_directions == right.terminal_endpoint_directions
    assert diagnose_distributional_fit(compact) == diagnose_distributional_fit(full)
    np.testing.assert_array_equal(left.terminal_fit.coefficients, right.terminal_fit.coefficients)
    np.testing.assert_array_equal(
        compact.fit_state.inference.covariance, full.fit_state.inference.covariance
    )
    frame, _ = _data()
    np.testing.assert_array_equal(compact.predict_parameters(frame), full.predict_parameters(frame))
    for model in (compact, full):
        restored = deserialize_distributional_model(serialize_distributional_model(model))
        np.testing.assert_array_equal(
            restored.predict_parameters(frame), model.predict_parameters(frame)
        )
        assert restored.smoothing.history == model.smoothing.history
        assert [fit.eta is None for fit in restored.smoothing.coefficient_fits] == [
            fit.eta is None for fit in model.smoothing.coefficient_fits
        ]


def test_compact_shape_is_anchored_to_terminal_rows():
    smoothing = _fit().smoothing
    fits = list(smoothing.coefficient_fits)
    assert fits[0].eta is None
    fits[0] = replace(fits[0], row_shape=(97, 2))
    with pytest.raises(ValueError, match="row shape"):
        replace(smoothing, coefficient_fits=tuple(fits))
    fits = list(smoothing.coefficient_fits)
    fits[smoothing.terminal_fit_index] = replace(smoothing.terminal_fit, eta=None, theta=None)
    with pytest.raises(ValueError, match="terminal.*rows"):
        replace(smoothing, coefficient_fits=tuple(fits))


@pytest.mark.parametrize(
    "changes",
    [
        {"eta": None},
        {"eta": None, "theta": None, "row_shape": None},
        {"eta": None, "theta": None, "row_shape": (-1, 2)},
        {"eta": None, "theta": None, "row_shape": (96, True)},
        {"eta": None, "theta": None, "row_shape": (96,)},
    ],
)
def test_rowless_results_refuse_partial_or_malformed_provenance(changes):
    fit = _fit(max_iterations=1).smoothing.terminal_fit
    with pytest.raises(ValueError, match="rows|row shape"):
        replace(fit, **changes)


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
def test_history_row_storage_is_bounded_during_optimization(monkeypatch, outer):
    from superglm.distributional.smoothing import loop, newton
    from superglm.distributional.smoothing.history import compact_coefficient_history

    samples = []
    rows = []
    original_post_init = DenseSolverResult.__post_init__

    def track_rows(self):
        original_post_init(self)
        if self.eta is not None:
            rows.append(weakref.ref(self.eta))

    monkeypatch.setattr(DenseSolverResult, "__post_init__", track_rows)

    def measured(fits, history, **kwargs):
        compact_coefficient_history(fits, history, **kwargs)
        samples.append((len(fits), sum(fit.eta is not None for fit in fits)))
        # The terminal fit plus two endpoints per plateau iteration is an
        # upper bound even when source/accepted indices are not contiguous.
        assert samples[-1][1] <= 1 + 2 * kwargs["config"].plateau_iterations
        # Also inspect actual array lifetimes: compact list entries alone do
        # not prove release if an execution cache owns the old full results.
        # Four extra live references allow the initial fit and the current
        # outer/Newton transition and retry states outside the history window.
        live = sum(reference() is not None for reference in rows)
        assert live <= 5 + 2 * kwargs["config"].plateau_iterations

    monkeypatch.setattr(loop, "compact_coefficient_history", measured)
    monkeypatch.setattr(newton, "compact_coefficient_history", measured)
    smoothing = _fit(outer=outer, handoff_iterations=2).smoothing
    assert len(samples) > 4  # A final-only cleanup cannot satisfy this contract.
    assert samples[-1][0] > samples[-1][1]
    if outer == "efs+newton":
        assert smoothing.newton_iterations > 0


def _mutated_artifact(model, mutate, *, legacy=False):
    artifact = json.loads(serialize_distributional_model(model))
    restored = pickle.loads(base64.b64decode(artifact["payload"]["data"]))
    mutate(restored)
    raw = serialization_module._pickle_model(restored)
    artifact["payload"]["data"] = base64.b64encode(raw).decode("ascii")
    artifact["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()
    if legacy:
        artifact["manifest"]["smoothing"]["config"].pop("retain_history_rows")
    return json.dumps(artifact).encode()


def test_legacy_full_history_without_new_fields_round_trips():
    model = _fit(retain_history_rows=True)

    def old_fields(restored):
        vars(restored.smoothing.config).pop("retain_history_rows")
        for fit in restored.smoothing.coefficient_fits:
            vars(fit).pop("row_shape")

    restored = deserialize_distributional_model(_mutated_artifact(model, old_fields, legacy=True))
    assert restored.smoothing.config.retain_history_rows is False
    assert all(fit.row_shape == (96, 2) for fit in restored.smoothing.coefficient_fits)
    assert all(fit.eta is not None for fit in restored.smoothing.coefficient_fits)


def test_rehashed_compact_shape_tampering_is_refused():
    model = _fit()

    def wrong_shape(restored):
        object.__setattr__(restored.smoothing.coefficient_fits[0], "row_shape", (9700, 2))

    with pytest.raises(DistributionalSerializationError, match="row shape"):
        deserialize_distributional_model(_mutated_artifact(model, wrong_shape))


def test_legacy_migration_does_not_repair_mismatched_terminal_identity():
    model = _fit(retain_history_rows=True)

    def corrupt_legacy(restored):
        vars(restored.smoothing.config).pop("retain_history_rows")
        for fit in restored.smoothing.coefficient_fits:
            vars(fit).pop("row_shape")
        object.__setattr__(
            restored.fit_state, "solver_result", restored.smoothing.coefficient_fits[0]
        )

    with pytest.raises(DistributionalSerializationError, match="terminal EFS fit"):
        deserialize_distributional_model(_mutated_artifact(model, corrupt_legacy, legacy=True))


def test_dense_reuse_keeps_live_authority_without_owning_old_fits(monkeypatch):
    from superglm.distributional.solver.solver import _DenseObservedReuseSession

    original = _DenseObservedReuseSession.remember
    sources = []

    def observed(self, result, owner, **kwargs):
        original(self, result, owner, **kwargs)
        if self.remembers(result, owner):
            assert not self.remembers(result, replace(owner, response=object()))
            sources.append((self, id(result), weakref.ref(result), owner))

    monkeypatch.setattr(_DenseObservedReuseSession, "remember", observed)
    model = _fit()
    assert sources
    assert any(reference() is None for _, _, reference, _ in sources)
    for session, _key, reference, owner in sources:
        result = reference()
        if result is not None:
            assert session.remembers(result, owner)
            live_session, live_result, live_owner = session, result, owner
    # Check eviction before another fit can reuse the collected object's id.
    probe = replace(live_result)
    live_session.remember(probe, live_owner)
    assert live_session.remembers(probe, live_owner)
    probe_key, probe_reference = id(probe), weakref.ref(probe)
    del probe
    assert probe_reference() is None
    assert probe_key not in live_session._results
    assert model.smoothing.terminal_fit.eta is not None


def test_initial_rows_are_collected_after_the_plateau_window(monkeypatch):
    from superglm.distributional.smoothing import loop

    original = loop.compact_coefficient_history
    initial_rows = []
    observed_release = []

    def measured(fits, history, **kwargs):
        if not initial_rows:
            initial_rows.extend((weakref.ref(fits[0].eta), weakref.ref(fits[0].theta)))
        original(fits, history, **kwargs)
        if fits[0].eta is None:
            assert all(reference() is None for reference in initial_rows)
            observed_release.append(len(history))

    monkeypatch.setattr(loop, "compact_coefficient_history", measured)
    _fit()
    assert observed_release


@pytest.mark.parametrize("chunk_size", [None, 16])
def test_reuse_session_and_caches_are_collected_without_cyclic_gc(chunk_size):
    from superglm.distributional.solver.solver import _DenseObservedReuseSession

    from .test_distributional_chunk_reuse import _fit as fit_problem
    from .test_distributional_chunk_reuse import _problem

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        session = _DenseObservedReuseSession()
        source = fit_problem(_problem(), session=session, chunk_size=chunk_size)
        session_reference = weakref.ref(session)
        if chunk_size is None:
            assert id(source) in session._results
            cache_arrays = [
                weakref.ref(array) for _, arrays in session._dense.values() for array in arrays
            ]
        else:
            cache_arrays = [weakref.ref(session._chunk_results[id(source)].score_data)]
        assert cache_arrays
        # Keep the source alive, so its callback remains registered, while
        # dropping the caller's sole ownership of the completed fit session.
        del session
        assert session_reference() is None
        assert all(reference() is None for reference in cache_arrays)
        assert source.eta is not None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("practical", [True, False])
def test_plateau_and_exact_face_keep_replay_authority(practical):
    frame = pd.DataFrame({"effect": np.repeat(["a", "b", "c", "d"], 10)})
    y = np.random.default_rng(7).normal(size=len(frame))
    models = []
    for retain_history_rows in (False, True):
        model = model_from_templates(
            family=GaussianLS(scale_floor=1e-4),
            predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
        ).fit_reml(
            frame,
            y,
            lambdas={"location:effect#wiggle": 1e6},
            max_lambda=1e6 * math.exp(1.5),
            max_log_step=0.5,
            max_reml_iter=20,
            reml_tol=1e-8,
            inner_tol=1e-10,
            reml_plateau_tol=1e-6,
            practical_reml=practical,
            retain_history_rows=retain_history_rows,
        )
        models.append(model._model)
    left, right = (model.smoothing for model in models)
    assert left.history == right.history
    assert left.terminal_endpoint_directions == right.terminal_endpoint_directions
    assert left.convergence_reason == right.convergence_reason
    assert left.matched_certified == right.matched_certified
    np.testing.assert_array_equal(models[0].inference.covariance, models[1].inference.covariance)
    if practical:
        assert left.convergence_reason == "practical_plateau"
        fits = list(left.coefficient_fits)
        source_index = left.history[-left.config.plateau_iterations].source_fit_index
        fits[source_index] = replace(fits[source_index], eta=None, theta=None)
        with pytest.raises(ValueError, match="rows.*plateau replay"):
            replace(left, coefficient_fits=tuple(fits))
    else:
        assert left.terminal_fit.coefficient_face is not None
    for model in models:
        restored = deserialize_distributional_model(serialize_distributional_model(model))
        assert restored.smoothing.history == model.smoothing.history
        assert (
            restored.smoothing.terminal_endpoint_directions
            == model.smoothing.terminal_endpoint_directions
        )
        assert restored.smoothing.matched_certified == model.smoothing.matched_certified
