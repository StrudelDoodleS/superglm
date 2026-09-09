"""Automatic global batching belongs to the complete solver context."""

import numpy as np
import pytest

from superglm.distributional.solver import chunks, solver

from .test_distributional_support_predictions import _support_problem


@pytest.mark.parametrize("requested", [None, 1, 7, "auto"])
def test_unadmitted_and_explicit_requests_keep_existing_resolution(monkeypatch, requested):
    family, layout, _, plan, _ = _support_problem(False)
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted or explicit request attempted global batch growth")

    monkeypatch.setattr(chunks, "global_moment_chunk_size", forbidden, raising=False)
    expected = (
        None
        if requested is None
        else chunks.resolve_chunk_size(
            layout.predictors[0].design.n,
            len(layout.predictors),
            requested,
            p_coefficients=layout.n_coefficients,
        )
    )
    assert chunks._resolve_fitting_chunk_size(family, layout, plan, requested) == expected


def test_explicit_request_stays_explicit_even_when_global_is_admitted(monkeypatch):
    family, layout, _, plan, _ = _support_problem(False)
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)

    def forbidden(*args, **kwargs):
        pytest.fail("explicit request attempted global batch growth")

    monkeypatch.setattr(chunks, "global_moment_chunk_size", forbidden, raising=False)
    assert chunks._resolve_fitting_chunk_size(family, layout, plan, 7) == 7


def test_admitted_auto_resolution_reaches_every_phase_and_reuse_certificate(monkeypatch):
    family, layout, response, plan, initial = _support_problem(False)
    monkeypatch.setattr(chunks, "AUTO_CHUNK_MEMORY_BYTES", 128)
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    selections = []

    def select(source, *, byte_budget, minimum_chunk_size):
        assert source is layout
        assert byte_budget == 64 << 20
        assert minimum_chunk_size < 17
        selections.append(17)
        return 17

    monkeypatch.setattr(chunks, "global_moment_chunk_size", select, raising=False)
    penalty = np.eye(layout.n_coefficients)
    kwargs = dict(coefficient_curvature="observed", coefficient_face=None)
    context = solver._validated_context(
        family, layout, response, plan, penalty, chunk_size="auto", **kwargs
    )
    old_context = solver._validated_context(
        family, layout, response, plan, penalty, chunk_size=7, **kwargs
    )
    assert context.chunk_size == 17
    assert solver._chunk_reuse_data_certificate(context) is not None
    assert solver._chunk_reuse_data_certificate(context) != solver._chunk_reuse_data_certificate(
        old_context
    )
    observed = {}
    for name in (
        "assemble_chunked_geometry",
        "evaluate_chunked_log_likelihood",
        "maximum_chunked_predictor_change",
        "materialize_terminal_predictions",
    ):
        original = getattr(chunks, name)

        def record(*args, _name=name, _original=original, **kwargs):
            observed.setdefault(_name, set()).add(kwargs["chunk_size"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(chunks, name, record)
    fitted = solver.fit_dense_fixed_lambda(
        family, layout, response, plan, penalty, initial=initial, chunk_size="auto"
    )
    assert selections == [17, 17]
    assert fitted.resolved_chunk_size == 17
    assert set(observed) == {
        "assemble_chunked_geometry",
        "evaluate_chunked_log_likelihood",
        "maximum_chunked_predictor_change",
        "materialize_terminal_predictions",
    }
    assert all(sizes == {17} for sizes in observed.values())
