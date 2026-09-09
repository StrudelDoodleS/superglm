"""Bounded child-plan caching preserves live prepared likelihood contracts."""

from __future__ import annotations

import gc
import importlib
import tracemalloc
import warnings
import weakref

import numpy as np
import pytest

from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights


def _problem(family_type=GaussianLS, semantics="prior", n=80):
    family = family_type()
    values = np.linspace(0.8, 1.2, n) if semantics == "prior" else np.arange(n) % 3 + 1
    weights = resolve_likelihood_weights(
        values, n_observations=n, contract=WeightContract(semantics)
    )
    y = np.linspace(1.0, 2.0, n)
    return family, family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)


def _build(family, plan, **kwargs):
    module = importlib.import_module("superglm.distributional.solver._likelihood_cache")
    return module.build_likelihood_cache(family, plan, **kwargs)


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_repeated_ranges_call_legacy_take_once(family_type, semantics, monkeypatch):
    family, plan = _problem(family_type, semantics)
    cache = _build(family, plan)
    assert cache is not None
    calls = 0
    original = type(plan).take

    def counted(self, indices):
        nonlocal calls
        calls += 1
        return original(self, indices)

    monkeypatch.setattr(type(plan), "take", counted)
    indices = np.arange(3, 23)
    first = cache.take(plan, indices, start=3, stop=23)
    second = cache.take(plan, indices, start=3, stop=23)
    assert second is first
    assert calls == 1
    assert cache.hits == 1 and cache.misses == 1


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_cached_child_is_numerically_the_legacy_prepared_likelihood(family_type, semantics):
    family, plan = _problem(family_type, semantics)
    cache = _build(family, plan)
    indices = np.arange(4, 24)
    expected = plan.take(indices)
    actual = cache.take(plan, indices, start=4, stop=24)
    assert actual.plan_identifier == expected.plan_identifier
    for name in ("values", "geometry_values", "root_take_map", "input_positions"):
        np.testing.assert_array_equal(
            getattr(actual.weights, name), getattr(expected.weights, name)
        )
    y = np.linspace(1.0, 2.0, 80)[indices]
    theta = np.column_stack((np.full(len(indices), 1.4), np.full(len(indices), 0.6)))
    reference = family.evaluate_natural(y, theta, expected)
    evaluated = family.evaluate_natural(y, theta, actual)
    for name in (
        "optimizing_log_likelihood",
        "parameter_independent_carrier",
        "score",
        "hessian_packed",
    ):
        np.testing.assert_array_equal(getattr(evaluated, name), getattr(reference, name))


_ARRAY_PATHS = (
    "weights.values",
    "weights.geometry_values",
    "weights.root_take_map",
    "weights.input_positions",
    "weights.dropped_input_positions",
    "parameter_independent_carrier",
)


def _owner(plan, path):
    names = path.split(".")
    for name in names[:-1]:
        plan = getattr(plan, name)
    return plan, names[-1]


def _change_array(plan, path, mutation):
    owner, name = _owner(plan, path)
    original = getattr(owner, name)
    changed = original.copy()
    if changed.size:
        changed.flat[0] += 1
    if mutation == "replace":
        changed = np.frombuffer(changed.tobytes(), dtype=changed.dtype)
        object.__setattr__(owner, name, changed)
    elif mutation == "setstate":
        original.__setstate__(changed.__reduce__()[2])
    elif mutation == "shape":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            original.shape = (original.size, 1)
    elif mutation == "mutable":
        changed.flags.writeable = False
        object.__setattr__(owner, name, changed)
    else:

        class CustomArray(np.ndarray):
            pass

        object.__setattr__(owner, name, original.view(CustomArray))


@pytest.mark.parametrize("target", ["source", "child"])
@pytest.mark.parametrize("path", (*_ARRAY_PATHS, "exact_response"))
@pytest.mark.parametrize("mutation", ["replace", "setstate", "shape", "mutable", "custom"])
def test_live_source_and_cached_child_arrays_invalidate(target, path, mutation, monkeypatch):
    family, plan = _problem(GammaLS)
    cache = _build(family, plan)
    indices = np.arange(3, 23)
    child = cache.take(plan, indices, start=3, stop=23)
    _change_array(plan if target == "source" else child, path, mutation)

    def fresh_required(*args, **kwargs):
        raise RuntimeError("fresh legacy take required")

    monkeypatch.setattr(type(plan), "take", fresh_required)
    with pytest.raises(RuntimeError, match="fresh legacy take required"):
        cache.take(plan, indices, start=3, stop=23)
    assert cache.entry_count == 0


@pytest.mark.parametrize("target", ["source", "child"])
@pytest.mark.parametrize(
    "path,value",
    [
        ("carrier_digest", "forged"),
        ("weights.digest", "forged"),
        ("weights.provenance.root_digest", "forged"),
        ("weights.provenance.contract.semantics", "frequency"),
        ("weights.provenance.weight_sum", 200.0),
        ("row_law", "forged"),
        ("family_config", ("GaussianLS/v1", 0.2)),
        ("observation.schema_version", 2),
    ],
)
def test_forged_frozen_scalar_metadata_invalidates(target, path, value, monkeypatch):
    family, plan = _problem()
    # Avoid mutating the globally shared complete-observation singleton.
    from dataclasses import replace

    object.__setattr__(plan, "observation", replace(plan.observation))
    cache = _build(family, plan)
    indices = np.arange(3, 23)
    child = cache.take(plan, indices, start=3, stop=23)
    owner, name = _owner(plan if target == "source" else child, path)
    object.__setattr__(owner, name, value)

    def fresh_required(*args, **kwargs):
        raise RuntimeError("fresh legacy take required")

    monkeypatch.setattr(type(plan), "take", fresh_required)
    with pytest.raises(RuntimeError, match="fresh legacy take required"):
        cache.take(plan, indices, start=3, stop=23)


@pytest.mark.parametrize("path", _ARRAY_PATHS)
def test_frozen_owning_array_source_is_ineligible(path):
    family, plan = _problem()
    _change_array(plan, path, "mutable")
    assert _build(family, plan) is None


@pytest.mark.parametrize("which", ["family", "plan", "weights"])
def test_custom_types_are_ineligible(which):
    family, plan = _problem()
    source = {"family": family, "plan": plan, "weights": plan.weights}[which]

    class Custom(type(source)):
        pass

    custom = Custom.__new__(Custom)
    for name, value in vars(source).items():
        object.__setattr__(custom, name, value)
    if which == "family":
        family = custom
    elif which == "plan":
        plan = custom
    else:
        object.__setattr__(plan, "weights", custom)
    assert _build(family, plan) is None


def test_instance_take_override_is_not_hidden_by_a_hit():
    family, plan = _problem()
    cache = _build(family, plan)
    indices = np.arange(3, 23)
    cache.take(plan, indices, start=3, stop=23)
    marker = object()
    object.__setattr__(plan, "take", lambda indices: marker)
    assert cache.take(plan, indices, start=3, stop=23) is marker


@pytest.mark.parametrize(
    "indices,start,stop",
    [
        (np.array([3, 5, 4]), 3, 6),
        (np.arange(4, 7), 3, 6),
        (np.array([3, 3, 4]), 3, 6),
        (np.array([-1, 3, 4]), 3, 6),
        (np.array([3.0, 4.0]), 3, 5),
        (np.array([], dtype=int), 3, 3),
        (np.arange(3, 83), 3, 83),
        (np.arange(3, 6), np.int64(3), 6),
    ],
)
def test_nonranges_and_invalid_indices_preserve_legacy_take(indices, start, stop, monkeypatch):
    family, plan = _problem()
    cache = _build(family, plan)
    original = type(plan).take
    calls = 0

    def counted(self, selected):
        nonlocal calls
        calls += 1
        return original(self, selected)

    monkeypatch.setattr(type(plan), "take", counted)
    try:
        expected = original(plan, indices)
    except Exception as error:
        with pytest.raises(type(error), match=str(error)):
            cache.take(plan, indices, start=start, stop=stop)
    else:
        actual = cache.take(plan, indices, start=start, stop=stop)
        assert actual.plan_identifier == expected.plan_identifier
    assert calls == 1
    assert cache.entry_count == 0


def test_fixed_admission_preserves_hits_after_budget_refusal():
    family, plan = _problem()
    probe = _build(family, plan)
    probe.take(plan, np.arange(20), start=0, stop=20)
    budget = probe.retained_bytes + 128
    cache = _build(family, plan, byte_budget=budget)
    first = cache.take(plan, np.arange(20), start=0, stop=20)
    cache.take(plan, np.arange(20, 40), start=20, stop=40)
    cache.take(plan, np.arange(40, 41), start=40, stop=41)
    assert cache.entry_count == 1
    assert cache.retained_bytes <= budget
    assert cache.take(plan, np.arange(20), start=0, stop=20) is first
    assert cache.max_fresh_child_bytes > 0


def test_entry_count_has_an_independent_cap():
    family, plan = _problem(n=270)
    cache = _build(family, plan)
    for start in range(270):
        cache.take(plan, np.array([start]), start=start, stop=start + 1)
    assert cache.entry_count == 256
    assert cache.retained_bytes <= cache.byte_budget


def test_clear_releases_children_and_does_not_keep_source_plan_alive():
    family, plan = _problem()
    cache = _build(family, plan)
    child = cache.take(plan, np.arange(20), start=0, stop=20)
    child_ref, source_ref = weakref.ref(child), weakref.ref(plan)
    del child, plan
    gc.collect()
    assert source_ref() is None
    assert child_ref() is not None
    cache.clear()
    gc.collect()
    assert child_ref() is None
    assert cache.retained_bytes == 0 and cache.entry_count == 0


@pytest.mark.parametrize("budget", [0, -1, True, 64.0, 1])
def test_unusable_budget_declines_without_new_errors(budget):
    family, plan = _problem()
    assert _build(family, plan, byte_budget=budget) is None


def test_source_swap_and_family_scalar_change_start_fresh_generation():
    from dataclasses import replace

    family, plan = _problem()
    cache = _build(family, plan)
    indices = np.arange(20)
    first = cache.take(plan, indices, start=0, stop=20)
    replacement = replace(plan)
    second = cache.take(replacement, indices, start=0, stop=20)
    assert second is not first
    object.__setattr__(family, "scale_floor", 0.2)
    third = cache.take(replacement, indices, start=0, stop=20)
    assert third is not second


def test_same_identity_base_storage_change_refuses_and_releases_old_bytes():
    family, plan = _problem()
    cache = _build(family, plan)
    indices = np.arange(20)
    first = cache.take(plan, indices, start=0, stop=20)
    # The visible array itself keeps its identity and metadata. Changing its
    # intermediate base still invalidates the complete storage authority.
    original = plan.parameter_independent_carrier
    base = original.base
    assert type(base) is np.ndarray
    base.__setstate__(base.copy().__reduce__()[2])
    second = cache.take(plan, indices, start=0, stop=20)
    assert second is not first
    assert cache.entry_count == 0


def test_mutable_source_falls_back_and_observes_each_live_value():
    family, plan = _problem()
    cache = _build(family, plan)
    indices = np.arange(20)
    cache.take(plan, indices, start=0, stop=20)
    values = plan.weights.values.copy()
    object.__setattr__(plan.weights, "values", values)
    for value in (1.5, 2.5):
        values[0] = value
        child = cache.take(plan, indices, start=0, stop=20)
        assert child.weights.values[0] == value
    assert cache.entry_count == 0


def test_shared_dropped_row_buffers_are_charged_against_retained_budget():
    family = GaussianLS()
    values = np.zeros(100_000)
    values[:80] = 1.0
    weights = resolve_likelihood_weights(
        values, n_observations=len(values), contract=WeightContract("prior")
    )
    plan = family.bind_likelihood(np.ones(80), weights, COMPLETE_OBSERVATION)
    cache = _build(family, plan, byte_budget=64 * 1024)
    assert cache is not None
    child = cache.take(plan, np.arange(20), start=0, stop=20)
    assert child.weights.dropped_input_positions is weights.dropped_input_positions
    assert cache.entry_count == 0
    assert cache.retained_bytes <= cache.byte_budget
    assert cache.max_fresh_child_bytes >= weights.dropped_input_positions.nbytes


def test_root_authority_and_warm_hit_do_not_copy_or_scan_source_rows(monkeypatch):
    family, plan = _problem(n=100_000)
    tracemalloc.start()
    try:
        cache = _build(family, plan)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert cache is not None
    # Authority is constant-size metadata, well below one float64 source copy.
    assert peak < 128 * 1024
    indices = np.arange(64)
    child = cache.take(plan, indices, start=0, stop=64)
    original_equal = np.array_equal

    def bounded_equal(left, right, *args, **kwargs):
        assert np.size(left) <= len(indices)
        assert np.size(right) <= len(indices)
        return original_equal(left, right, *args, **kwargs)

    monkeypatch.setattr(np, "array_equal", bounded_equal)
    tracemalloc.start()
    try:
        assert cache.take(plan, indices, start=0, stop=64) is child
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 128 * 1024
