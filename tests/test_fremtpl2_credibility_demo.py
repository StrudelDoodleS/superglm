"""Offline smoke coverage for the real French MTPL credibility demo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import Categorical, FactorSmooth, RandomEffect

_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "fremtpl2_credibility.py"
_SPEC = importlib.util.spec_from_file_location("fremtpl2_credibility_demo", _DEMO_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DEMO = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DEMO
_SPEC.loader.exec_module(_DEMO)


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": np.arange(8),
            "ClaimNb": [0, 1, 0, 2, 0, 1, 0, 7],
            "Exposure": [0.5, 1.2, 0.8, 0.0, 0.7, 0.4, 0.9, 1.0],
            "Area": list("ABCDEFAB"),
            "VehPower": [4, 5, 6, 7, 8, 9, 10, 11],
            "VehAge": [1, 2, 3, 4, 30, 6, 7, 8],
            "DrivAge": [18, 21, 35, 42, 55, 99, 63, 72],
            "BonusMalus": [50, 55, 60, 70, 80, 90, 160, 100],
            "VehBrand": [f"B{index}" for index in range(8)],
            "VehGas": ["Diesel", "Regular"] * 4,
            "Density": [10, 20, 40, 80, 160, 320, 640, 1280],
            "Region": [f"R{index}" for index in range(8)],
        }
    )


def test_prepare_frame_applies_documented_insurance_caps() -> None:
    frame = _DEMO.prepare_frame(_tiny_frame())

    assert frame["ClaimNb"].max() == 4
    assert frame["Exposure"].between(1.0e-3, 1.0).all()
    assert frame["VehAge"].max() == 20
    assert frame["DrivAge"].max() == 90
    assert frame["BonusMalus"].max() == 150
    np.testing.assert_allclose(frame["LogDensity"], np.log1p(frame["Density"]))


def test_demo_variants_add_re_and_fs_to_one_common_baseline() -> None:
    base = _DEMO.make_model("baseline")
    fixed = _DEMO.make_model("brand_fixed")
    random_effect = _DEMO.make_model("re")
    factor_smooth = _DEMO.make_model("fs")
    combined = _DEMO.make_model("re_fs")

    assert "VehBrand" not in base._specs
    assert isinstance(fixed._specs["VehBrand"], Categorical)
    assert isinstance(random_effect._specs["VehBrand"], RandomEffect)
    assert isinstance(
        factor_smooth._interaction_specs["DrivAge:Region:fs"],
        FactorSmooth,
    )
    assert isinstance(combined._specs["VehBrand"], RandomEffect)
    assert isinstance(
        combined._interaction_specs["DrivAge:Region:fs"],
        FactorSmooth,
    )
    assert base._direct_solve == "auto"
    assert factor_smooth._direct_solve == "structured"


def test_exposure_normalized_poisson_metrics_are_zero_at_perfect_prediction() -> None:
    claims = np.array([0.0, 1.0, 2.0])
    exposure = np.array([0.5, 1.0, 2.0])
    prediction = claims.copy()

    metrics = _DEMO.poisson_metrics(claims, prediction, exposure)

    assert metrics["mean_poisson_deviance"] == 0.0
    assert metrics["claim_calibration"] == 1.0
