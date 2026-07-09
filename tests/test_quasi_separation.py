"""Tests for quasi-separation detection and numerical stability."""

import logging

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric


def _make_sparse_tweedie_data(n=10_000, n_rare=10, seed=42):
    """Create Tweedie data with one near-separated level (few obs, all y=0)."""
    rng = np.random.default_rng(seed)

    # Build most observations from well-identified levels
    n_main = n - n_rare
    cat_main = rng.choice(["base", "mid", "hi", "lo"], n_main, p=[0.45, 0.30, 0.15, 0.10])
    cat_rare = np.array(["rare"] * n_rare)
    cat = np.concatenate([cat_main, cat_rare])

    exposure = rng.uniform(0.3, 1.0, n)
    eta = 5.0 + 0.3 * (cat == "hi") - 0.2 * (cat == "lo") + 0.1 * (cat == "mid")
    mu = np.exp(eta) * exposure

    from superglm.profiling.tweedie import generate_tweedie_cpg

    y = generate_tweedie_cpg(n, mu=mu, phi=2.0, p=1.5, rng=rng)
    y[cat == "rare"] = 0.0  # force near-separation

    # Shuffle so rare obs aren't all at the end
    idx = rng.permutation(n)
    cat = cat[idx]
    y = y[idx]
    exposure = exposure[idx]

    df = pd.DataFrame({"cat": cat})
    return df, y, exposure


class TestSEFiniteForRareLevel:
    """SE must be large but finite (not 0, not inf) for separated levels."""

    def test_se_finite_for_rare_level(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))
        s = m.summary()

        rare_row = next(r for r in s._coef_rows if "rare" in r.name)
        assert rare_row.se is not None
        assert rare_row.se > 0, "SE should not be zero for separated level"
        assert np.isfinite(rare_row.se), "SE should be finite"
        assert np.isfinite(rare_row.z), "z-score should be finite"
        # SE should be large (indicating undetermined)
        assert rare_row.se > 1.0, "SE should be large for near-separated level"


class TestSeparatedTailDiagnostics:
    """Diagnostics should expose, not hide, separated-tail IRLS geometry."""

    def test_first_direct_diagnostic_separates_working_and_updated_state(self):
        rng = np.random.default_rng(321)
        n = 200
        x = rng.normal(size=n)
        y = rng.gamma(shape=2.0, scale=np.exp(0.4 * x), size=n)
        df = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        model.fit(df, y, max_iter=1, record_diagnostics=True)

        first = model.iteration_diagnostics().iloc[0]
        updated_eta = model._dm.matvec(model.result.beta) + model.result.intercept
        assert first["working_eta_min_unclipped"] == pytest.approx(
            first["working_eta_max_unclipped"]
        )
        assert not bool(first["working_eta_clipped"])
        assert first["eta_min_unclipped"] == pytest.approx(float(updated_eta.min()))
        assert first["eta_max_unclipped"] == pytest.approx(float(updated_eta.max()))
        assert first["intercept"] == pytest.approx(model.result.intercept)
        assert first["eta_min_unclipped"] != pytest.approx(first["working_eta_min_unclipped"])

    def test_first_pirls_diagnostic_separates_working_and_updated_state(self):
        rng = np.random.default_rng(654)
        x = rng.normal(size=200)
        y = rng.gamma(shape=2.0, scale=np.exp(0.3 * x), size=200)
        df = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.01,
            features={"x": Numeric()},
        )

        model.fit(df, y, max_iter=1, record_diagnostics=True)

        first = model.iteration_diagnostics().iloc[0]
        updated_eta = model._dm.matvec(model.result.beta) + model.result.intercept
        assert first["working_eta_min_unclipped"] == pytest.approx(
            first["working_eta_max_unclipped"]
        )
        assert first["eta_min_unclipped"] == pytest.approx(float(updated_eta.min()))
        assert first["eta_max_unclipped"] == pytest.approx(float(updated_eta.max()))
        assert first["intercept"] == pytest.approx(model.result.intercept)

    def test_working_weight_ratio_is_not_artificially_capped(self):
        rng = np.random.default_rng(123)
        n = 1_000
        n_rare = 5
        cat = np.array(["base"] * (n - n_rare) + ["rare"] * n_rare)
        y = rng.gamma(shape=2.0, scale=3.0, size=n)
        y[cat == "rare"] = 0.0

        idx = rng.permutation(n)
        df = pd.DataFrame({"cat": cat[idx]})
        y = y[idx]

        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        with pytest.warns(UserWarning, match="coefficient-based convergence"):
            model.fit(
                df,
                y,
                convergence="coefficients",
                max_iter=100,
                tol=0.0,
                record_diagnostics=True,
            )

        diagnostics = model.iteration_diagnostics()
        assert diagnostics["W_ratio"].max() > 1e12
        assert diagnostics["raw_W_ratio"].max() == pytest.approx(diagnostics["W_ratio"].max())
        assert diagnostics["eta_min"].min() == pytest.approx(-80.0)
        assert diagnostics["eta_min_unclipped"].min() < -80.0
        assert diagnostics["eta_clipped"].any()

    def test_direct_path_warns_on_extreme_working_weight_ratio(self, caplog):
        rng = np.random.default_rng(123)
        n = 1_000
        n_rare = 5
        cat = np.array(["base"] * (n - n_rare) + ["rare"] * n_rare)
        y = rng.gamma(shape=2.0, scale=3.0, size=n)
        y[cat == "rare"] = 0.0

        idx = rng.permutation(n)
        df = pd.DataFrame({"cat": cat[idx]})
        y = y[idx]

        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
            with pytest.warns(UserWarning, match="coefficient-based convergence"):
                model.fit(
                    df,
                    y,
                    convergence="coefficients",
                    max_iter=100,
                    tol=0.0,
                )

        assert any("extreme W ratio" in record.message for record in caplog.records)


class TestZeroWeightDiagnostics:
    """Zero-frequency rows stay visible without creating false W-ratio alerts."""

    @staticmethod
    def _data():
        x = np.linspace(-1.0, 1.0, 50)
        y = 1.0 + 2.0 * x
        weights = np.ones_like(x)
        weights[0] = 0.0
        return pd.DataFrame({"x": x}), y, weights

    @pytest.mark.parametrize(
        ("selection_penalty", "logger_name"),
        [
            (0.0, "superglm.solvers.irls_direct"),
            (0.01, "superglm.solvers.pirls"),
        ],
    )
    def test_zero_weight_row_does_not_inflate_ratio(self, selection_penalty, logger_name, caplog):
        df, y, weights = self._data()
        model = SuperGLM(
            family="gaussian",
            selection_penalty=selection_penalty,
            features={"x": Numeric()},
        )

        with caplog.at_level(logging.WARNING, logger=logger_name):
            model.fit(
                df,
                y,
                sample_weight=weights,
                max_iter=1,
                record_diagnostics=True,
            )

        first = model.iteration_diagnostics().iloc[0]
        assert first["raw_W_min"] == 0.0
        assert first["W_ratio"] == pytest.approx(1.0)
        assert first["raw_W_ratio"] == pytest.approx(1.0)
        assert not any("extreme W ratio" in record.message for record in caplog.records)


class TestQuasiSeparatedMarker:
    """The ? marker must appear on rare levels."""

    def test_marker_on_rare_level(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))
        s = m.summary()

        rare_row = next(r for r in s._coef_rows if "rare" in r.name)
        assert rare_row.quasi_separated is True


class TestPerLevelDiagnostics:
    """Per-level observation count and exposure share must be populated."""

    def test_diagnostics_populated(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))
        s = m.summary()

        for r in s._coef_rows:
            if r.name == "Intercept" or r.is_spline:
                continue
            assert r.level_n_obs is not None, f"{r.name} missing level_n_obs"
            assert r.level_exposure_share is not None, f"{r.name} missing exposure_share"

        rare_row = next(r for r in s._coef_rows if "rare" in r.name)
        assert rare_row.level_n_obs < 200  # ~1% of 10k
        assert rare_row.level_exposure_share < 0.02


class TestWellDeterminedUnaffected:
    """Well-identified levels must have unchanged SEs after regularization."""

    def test_well_determined_se_reasonable(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))
        s = m.summary()

        # Well-identified levels should have moderate SEs
        for r in s._coef_rows:
            if r.name == "Intercept" or "rare" in r.name:
                continue
            if r.se is not None and r.se > 0:
                assert r.se < 1.0, f"{r.name} SE={r.se} seems too large for well-identified level"
                assert not r.quasi_separated, f"{r.name} should not be quasi-separated"


class TestEDFNotRegressed:
    """Total EDF from summary should be close to fit's effective_df."""

    def test_edf_consistency(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))

        fit_edf = m.result.effective_df
        s = m.summary()
        # EDF from summary header should match fit
        assert s._info["effective_df"] == pytest.approx(fit_edf, rel=1e-6)


class TestFootnoteInSummary:
    """Quasi-separated footnote must appear in ASCII output."""

    def test_footnote_present(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))
        s = m.summary()
        text = str(s)

        assert "Quasi-separated" in text
        assert "rare" in text.lower()
        assert "obs" in text


class TestColumnAlignment:
    """Sig and QS columns must not break border alignment."""

    def test_all_data_rows_same_width(self):
        df, y, exposure = _make_sparse_tweedie_data()
        m = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        m.fit(df, y, sample_weight=exposure, offset=np.log(exposure))
        s = m.summary()
        text = str(s)

        # All lines between the box borders should have the same length
        lines = text.split("\n")
        box_lines = [line for line in lines if line.startswith("║")]
        if box_lines:
            widths = {len(line) for line in box_lines}
            assert len(widths) == 1, f"Inconsistent line widths: {widths}"
