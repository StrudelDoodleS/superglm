"""Smoke coverage for the structured credibility profiling harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from superglm import FactorSmooth, PSpline, RandomEffect

_HARNESS_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "profile_structured_credibility.py"
)
_SPEC = importlib.util.spec_from_file_location("structured_credibility_benchmark", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HARNESS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HARNESS
_SPEC.loader.exec_module(_HARNESS)


def test_factor_smooth_profile_case_builds_block_model() -> None:
    config = _HARNESS.CaseConfig(
        n=600,
        levels=20,
        family="poisson",
        discrete=True,
        random_effects=1,
        secondary_levels=7,
        small_width=2,
        weights="nonuniform",
        seed=917,
        structured_term="factor_smooth",
        block_size=5,
        global_spline=True,
    )

    prepared = _HARNESS.prepare_case(config)
    model = _HARNESS._new_model(prepared, "structured")

    assert {"curve_x", "curve_group", "branch", "x0", "x1"} <= set(prepared.X)
    assert prepared.config.dominant_width == 100
    assert "fs_k5" in prepared.config.slug
    assert isinstance(model._specs["curve_x"], PSpline)
    assert isinstance(model._specs["branch"], RandomEffect)
    assert isinstance(model._interaction_specs["curve_x:curve_group:fs"], FactorSmooth)


def test_random_effect_profile_case_retains_scalar_contract() -> None:
    config = _HARNESS.CaseConfig(
        n=200,
        levels=20,
        family="gaussian",
        discrete=False,
        random_effects=1,
        secondary_levels=None,
        small_width=2,
        weights="unit",
        seed=918,
    )

    prepared = _HARNESS.prepare_case(config)
    model = _HARNESS._new_model(prepared, "structured")

    assert "policy" in prepared.X
    assert prepared.config.dominant_width == 20
    assert isinstance(model._specs["policy"], RandomEffect)


def test_sum_to_zero_profile_case_builds_constrained_model() -> None:
    config = _HARNESS.CaseConfig(
        n=600,
        levels=20,
        family="poisson",
        discrete=True,
        random_effects=0,
        secondary_levels=None,
        small_width=2,
        weights="nonuniform",
        seed=919,
        structured_term="factor_smooth",
        block_size=6,
        global_spline=True,
        factor_basis="sz",
    )

    prepared = _HARNESS.prepare_case(config)
    model = _HARNESS._new_model(prepared, "structured")
    term = model._interaction_specs["curve_x:curve_group:sz"]

    assert term.basis == "sz"
    assert config.dominant_width == 114
    assert "sz_k6" in config.slug
    assert isinstance(model._specs["curve_x"], PSpline)


def test_sum_to_zero_profile_case_requires_meaningful_configuration() -> None:
    with pytest.raises(ValueError, match="global_spline"):
        _HARNESS.prepare_case(
            _HARNESS.CaseConfig(
                n=200,
                levels=10,
                family="gaussian",
                discrete=False,
                random_effects=0,
                secondary_levels=None,
                small_width=2,
                weights="unit",
                seed=920,
                structured_term="factor_smooth",
                block_size=6,
                factor_basis="sz",
            )
        )

    with pytest.raises(ValueError, match="factor_basis"):
        _HARNESS.prepare_case(
            _HARNESS.CaseConfig(
                n=200,
                levels=10,
                family="gaussian",
                discrete=False,
                random_effects=1,
                secondary_levels=None,
                small_width=2,
                weights="unit",
                seed=921,
                factor_basis="sz",
            )
        )
