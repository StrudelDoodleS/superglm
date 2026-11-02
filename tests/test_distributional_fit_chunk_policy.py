"""A fit resolves one row bound in its solver context, and every phase uses it."""

import numpy as np
import pytest

from superglm.distributional.solver import chunks, solver

from .test_distributional_support_predictions import _support_problem


@pytest.mark.parametrize("requested", [None, 1, 7, "auto"])
def test_requests_resolve_through_the_layout_selector(requested):
    family, layout, _, plan, _ = _support_problem(False)
    expected = None if requested is None else chunks.layout_chunk_size(layout, requested)
    assert chunks._resolve_fitting_chunk_size(family, layout, plan, requested) == expected


def test_automatic_resolution_reaches_every_phase(monkeypatch):
    family, layout, response, plan, initial = _support_problem(False)
    original_selector = chunks.layout_chunk_size
    selections = []

    def select(source, chunk_size):
        if chunk_size != "auto":
            return original_selector(source, chunk_size)
        assert source is layout
        selections.append(17)
        return 17

    monkeypatch.setattr(chunks, "layout_chunk_size", select)
    penalty = np.eye(layout.n_coefficients)
    kwargs = dict(coefficient_curvature="observed", coefficient_face=None)
    context = solver._validated_context(
        family, layout, response, plan, penalty, chunk_size="auto", **kwargs
    )
    assert context.chunk_size == 17
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
