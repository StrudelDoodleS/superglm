"""Tests for quasi-separation detection and numerical stability."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.categorical import Categorical


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

    from superglm.tweedie_profile import generate_tweedie_cpg

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
