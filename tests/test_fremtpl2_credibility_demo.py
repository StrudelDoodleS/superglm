"""Offline smoke coverage for the real French MTPL credibility demo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from superglm import Categorical, RandomEffect

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


def _tiny_fit_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260727)
    n_levels = 8
    repeats = 24
    codes = np.repeat(np.arange(n_levels), repeats)
    age = np.tile(np.linspace(18.0, 82.0, repeats), n_levels)
    exposure = rng.uniform(0.4, 1.0, size=len(codes))
    regional_slope = np.linspace(-0.015, 0.015, n_levels)
    mean = exposure * np.exp(-2.4 + 0.01 * (age - 45.0) + regional_slope[codes] * (age - 45.0))
    raw = pd.DataFrame(
        {
            "IDpol": np.arange(len(codes)),
            "ClaimNb": rng.poisson(mean),
            "Exposure": exposure,
            "Area": np.array(
                [f"A{value}" for value in rng.integers(0, 4, size=len(codes))],
                dtype=object,
            ),
            "VehPower": rng.integers(4, 9, size=len(codes)),
            "VehAge": rng.integers(1, 16, size=len(codes)),
            "DrivAge": age,
            "BonusMalus": rng.integers(50, 100, size=len(codes)),
            "VehBrand": np.array(
                [f"B{value}" for value in rng.integers(0, 6, size=len(codes))],
                dtype=object,
            ),
            "VehGas": np.where(
                rng.integers(0, 2, size=len(codes)) == 0,
                "Diesel",
                "Regular",
            ),
            "Density": rng.uniform(10.0, 500.0, size=len(codes)),
            "Region": np.array([f"R{code}" for code in codes], dtype=object),
        }
    )
    return _DEMO.prepare_frame(raw)


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

    assert "VehBrand" not in base.features
    assert isinstance(fixed.features["VehBrand"], Categorical)
    assert isinstance(random_effect.features["VehBrand"], RandomEffect)
    assert isinstance(combined.features["VehBrand"], RandomEffect)
    assert "fs" in _DEMO.DEFAULT_VARIANTS
    assert "re_fs" in _DEMO.DEFAULT_VARIANTS
    assert set(factor_smooth.features) == set(base.features)


def test_demo_factor_smooth_variant_has_public_fitted_report() -> None:
    frame = _tiny_fit_frame()
    model = _DEMO.make_model("fs", discrete=True, n_bins=64)
    model.fit_reml(
        frame.loc[:, _DEMO.MODEL_COLUMNS],
        frame["ClaimNb"].to_numpy(),
        offset=np.log(frame["Exposure"].to_numpy()),
        max_reml_iter=1,
        pirls_tol=1.0e-8,
        max_pirls_iter=60,
        runtime_validation="skip",
    )

    report = model.factor_smooth("DrivAge:Region:fs", grid=12)

    assert report.basis == "fs"
    assert set(report.table["level"]) == set(frame["Region"])
    assert len(report.curves) == frame["Region"].nunique() * 12
    assert model.result.direct_backend == "structured"


def test_fit_variants_requires_outer_and_inner_convergence(monkeypatch) -> None:
    frame = _DEMO.prepare_frame(_tiny_frame())

    class PublicModel:
        def __init__(self):
            self.result = SimpleNamespace(
                direct_backend="gram",
                effective_df=2.5,
                converged=False,
            )

        def fit_reml(self, *_args, **_kwargs):
            return self

        def predict(self, X, *, offset=None):
            del offset
            return np.full(len(X), 0.5)

        def diagnostics(self):
            return {"_model": {"converged": True, "n_iter": 3}}

    monkeypatch.setattr(_DEMO, "make_model", lambda *_args, **_kwargs: PublicModel())

    _models, metrics = _DEMO.fit_variants(
        frame,
        frame,
        variants=("baseline",),
        discrete=False,
        n_bins=32,
        max_reml_iter=1,
        reml_tol=1.0e-5,
    )

    assert not bool(metrics.loc[0, "converged"])
    assert metrics.loc[0, "reml_iterations"] == 3


def test_write_outputs_uses_known_variants_and_public_reports(tmp_path) -> None:
    train = _DEMO.prepare_frame(_tiny_frame())
    re_table = pd.DataFrame(
        {
            "level": ["B0", "B1"],
            "effect": [-0.1, 0.2],
            "credibility": [0.3, 0.7],
        }
    )
    fs_table = pd.DataFrame(
        {
            "level": ["R0", "R1"],
            "fit_weight": [3.0, 2.0],
        }
    )
    fs_curves = pd.DataFrame(
        {
            "level": ["R0", "R0", "R1", "R1"],
            "DrivAge": [20.0, 60.0, 20.0, 60.0],
            "effect": [0.1, 0.2, -0.1, -0.2],
        }
    )

    class RandomEffectModel:
        def random_effects(self, name, *, exposure):
            assert name == "VehBrand"
            assert len(exposure) == len(train)
            return SimpleNamespace(table=re_table)

    class FactorSmoothModel:
        def factor_smooth(self, name, *, grid, levels=None):
            assert name == "DrivAge:Region:fs"
            assert grid == 80
            if levels is not None:
                assert levels == ["R0", "R1"]
            return SimpleNamespace(table=fs_table, curves=fs_curves)

    metrics = pd.DataFrame(
        {
            "variant": ["re", "fs"],
            "mean_poisson_deviance": [0.6, 0.58],
        }
    )

    _DEMO.write_outputs(
        tmp_path,
        {"re": RandomEffectModel(), "fs": FactorSmoothModel()},
        metrics,
        train,
        plot_levels=2,
        make_plot=False,
    )

    assert (tmp_path / "vehicle_brand_random_effect.csv").exists()
    assert (tmp_path / "driver_age_region_credibility.csv").exists()
    assert (tmp_path / "driver_age_region_curves.csv").exists()


def test_exposure_normalized_poisson_metrics_are_zero_at_perfect_prediction() -> None:
    claims = np.array([0.0, 1.0, 2.0])
    exposure = np.array([0.5, 1.0, 2.0])
    prediction = claims.copy()

    metrics = _DEMO.poisson_metrics(claims, prediction, exposure)

    assert metrics["mean_poisson_deviance"] == 0.0
    assert metrics["claim_calibration"] == 1.0
