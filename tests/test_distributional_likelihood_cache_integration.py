"""Prepared likelihood rows are reused within a fit, with unchanged algebra."""

import sys
from dataclasses import fields, replace

import numpy as np
import pytest

import superglm._group_matrix._group_matrix_algebra as algebra
from superglm.distributional.solver import solver

from .test_distributional_support_predictions import _support_problem


def test_session_retains_only_one_likelihood_source(monkeypatch):
    class Cache:
        cleared = False

        def clear(self):
            self.cleared = True

    built = []

    def build(family, plan):
        cache = Cache()
        built.append((family, plan, cache))
        return cache

    monkeypatch.setattr(solver, "build_likelihood_cache", build, raising=False)
    session = solver._DenseObservedReuseSession()
    family, first, second = object(), object(), object()
    cache = session.likelihood_cache(family, first)
    assert session.likelihood_cache(family, first) is cache
    assert len(built) == 1
    assert session.likelihood_cache(family, second) is built[1][2]
    assert cache.cleared


def _context(problem, *, session=None, chunk_size=7):
    family, layout, response, plan, _ = problem
    return solver._validated_context(
        family,
        layout,
        response,
        plan,
        np.eye(layout.n_coefficients),
        coefficient_curvature="observed",
        chunk_size=chunk_size,
        coefficient_face=None,
        _reuse_session=session,
    )


def test_contexts_share_preparation_only_within_their_fit_session():
    problem = _support_problem(False)
    session = solver._DenseObservedReuseSession()
    first = _context(problem, session=session)
    second = _context(problem, session=session)
    assert first.likelihood_cache is not None
    assert first.likelihood_cache is second.likelihood_cache
    assert _context(problem).likelihood_cache is not first.likelihood_cache
    assert _context(problem, chunk_size=None).likelihood_cache is None
    assert first.pass_factors is second.pass_factors is session.pass_factors()
    assert _context(problem).pass_factors is not first.pass_factors
    assert _context(problem, session=session, chunk_size=None).pass_factors is None


@pytest.mark.parametrize("derived_root", [False, True])
def test_value_and_geometry_passes_reuse_children_with_identical_outputs(derived_root):
    problem = _support_problem(False)
    if derived_root:
        family, layout, response, plan, coefficients = problem
        plan = family.bind_chunked_likelihood(response, plan.weights, plan.observation)
        problem = (family, layout, response, plan, coefficients)
    context = _context(problem)
    code = type(problem[3]).take.__code__
    takes = 0

    def profile(frame, event, arg):
        nonlocal takes
        if event == "call" and frame.f_code is code:
            takes += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        state = solver._evaluate_state(context, problem[-1], derivative_order=0)
        assert state is not None
        geometry = solver._geometry(context, state, "observed")
        repeated = solver._evaluate_state(context, problem[-1], derivative_order=0)
    finally:
        sys.setprofile(previous)
    assert takes == 4  # Three passes, but each of the four row ranges is prepared once.
    assert context.likelihood_cache.hits == 8
    uncached = replace(context, likelihood_cache=None)
    expected_state = solver._evaluate_state(uncached, problem[-1], derivative_order=0)
    expected_geometry = solver._geometry(uncached, expected_state, "observed")
    assert state.optimizing_log_likelihood == expected_state.optimizing_log_likelihood
    assert repeated.optimizing_log_likelihood == expected_state.optimizing_log_likelihood
    for field in fields(geometry):
        np.testing.assert_array_equal(
            getattr(geometry, field.name), getattr(expected_geometry, field.name)
        )


def test_complete_fit_reuses_one_cache_and_matches_uncached_fit(monkeypatch):
    problem = _support_problem(False)
    family, layout, response, plan, initial = problem
    factory = solver.build_likelihood_cache
    caches = []

    def build(*args, **kwargs):
        result = factory(*args, **kwargs)
        caches.append(result)
        return result

    monkeypatch.setattr(solver, "build_likelihood_cache", build)
    kwargs = dict(initial=initial, chunk_size=7)
    penalty = np.eye(layout.n_coefficients)
    fitted = solver.fit_dense_fixed_lambda(family, layout, response, plan, penalty, **kwargs)
    assert len(caches) == 1
    assert caches[0].entry_count == 4
    assert caches[0].hits > caches[0].misses
    monkeypatch.setattr(solver, "build_likelihood_cache", lambda *args: None)
    expected = solver.fit_dense_fixed_lambda(family, layout, response, plan, penalty, **kwargs)
    np.testing.assert_array_equal(fitted.coefficients, expected.coefficients)
    np.testing.assert_array_equal(fitted.terminal_data_curvature, expected.terminal_data_curvature)
    assert fitted.history == expected.history
    assert "likelihood_cache" not in {field.name for field in fields(fitted)}


def test_a_fit_projects_each_support_once_across_its_assemblies(monkeypatch):
    problem = _support_problem(False)
    family, layout, response, plan, initial = problem
    original = algebra.PassFactors.projection
    owners, formed = set(), []

    def counted(self, B_unique, R_inv):
        owners.add(id(self))
        if (id(B_unique), id(R_inv)) not in self.projections:
            formed.append(1)
        return original(self, B_unique, R_inv)

    monkeypatch.setattr(algebra.PassFactors, "projection", counted)
    fitted = solver.fit_dense_fixed_lambda(
        family, layout, response, plan, np.eye(layout.n_coefficients), initial=initial, chunk_size=7
    )
    assert fitted.iterations > 1
    # Every assembly of the fit reads one PassFactors, so a support is projected once.
    assert len(owners) == 1 and len(formed) > 0
