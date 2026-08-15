"""Tests for the low-credibility advisory, its renderers, and IRLS stability."""

import logging
import re

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

    def test_direct_path_keeps_extreme_working_weight_ratio_out_of_warning_log(self, caplog):
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
        with caplog.at_level(logging.DEBUG, logger="superglm.solvers.irls_direct"):
            with pytest.warns(UserWarning, match="coefficient-based convergence"):
                model.fit(
                    df,
                    y,
                    convergence="coefficients",
                    max_iter=100,
                    tol=0.0,
                )

        ratio_records = [record for record in caplog.records if "extreme W ratio" in record.message]
        assert ratio_records
        assert all(record.levelno == logging.DEBUG for record in ratio_records)

    def test_pirls_keeps_extreme_working_weight_ratio_out_of_warning_log(self, monkeypatch, caplog):
        import superglm.solvers.pirls as pirls

        monkeypatch.setattr(
            pirls,
            "_positive_working_weight_stats",
            lambda _weights: (1e-20, 1.0, 1e20),
        )
        frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.01,
            features={"x": Numeric()},
        )

        with caplog.at_level(logging.DEBUG, logger="superglm.solvers.pirls"):
            model.fit(frame, np.linspace(0.0, 1.0, len(frame)), max_iter=1)

        ratio_records = [record for record in caplog.records if "extreme W ratio" in record.message]
        assert ratio_records
        assert all(record.levelno == logging.DEBUG for record in ratio_records)


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
    """The per-level footnote must appear in ASCII output.

    It names the levels the flag fired on and the experience behind each, which
    is the whole of what the rule measured.
    """

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

        assert "Low-credibility levels" in text
        assert "rare" in text.lower()
        assert "obs" in text


def _thin_but_cleanly_estimated():
    """A level with 15 rows, a large effect, and a tight interval.

    Gaussian with an identity link, so a coefficient CANNOT diverge here: the
    normal has no mass point for the predictor to be driven towards, and the
    MLE is the least-squares solution, which exists and is unique whenever the
    design has full column rank (Wedderburn 1976).  Any surface that reports
    separation for this fit is reporting something the data cannot show.
    """
    rng = np.random.default_rng(0)
    levels = np.array(["A"] * 2000 + ["B"] * 2000 + ["RARE"] * 15)
    mu = np.where(levels == "RARE", 8.0, np.where(levels == "B", 1.0, 0.0))
    y = mu + rng.normal(0, 0.5, size=levels.size)
    X = pd.DataFrame({"cat": pd.Categorical(levels, categories=["A", "B", "RARE"])})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"cat": Categorical(base="first")},
    )
    model.fit(X, y)
    return model


def _export_term(model, name):
    from superglm.export.summary import build_summary_export_payload

    payload = build_summary_export_payload(model)
    return next(term for term in payload.terms if term.term == name), payload


def _editor_row(model, name):
    from superglm.editor.summaries import _compact_summary_row

    rows = model.summary(detail="compact")._coef_rows
    return next(_compact_summary_row(row) for row in rows if str(row.name) == name)


class TestTheFlagIsAdvisoryOnEveryRenderer:
    """The flag says "thin cell"; no renderer may report it as a verdict.

    The rule that sets it reads ``level_n_obs`` and ``level_exposure_share``
    and nothing else -- not the coefficient, not its standard error, not its
    p-value, not the link.  A level can therefore be both strongly significant
    and flagged, and the console has always said so while the export and the
    editor replaced the significance code with the flag (issue #239).
    """

    def test_every_renderer_reports_the_same_significance_for_a_flagged_level(self):
        model = _thin_but_cleanly_estimated()
        rare = next(r for r in model.summary()._coef_rows if r.name == "cat[RARE]")
        assert rare.quasi_separated is True
        assert rare.p < 0.001
        assert np.isfinite(rare.se) and rare.se > 0

        console_line = next(
            line for line in str(model.summary()).split("\n") if "cat[RARE]" in line
        )
        assert "***" in console_line

        html_row = next(
            fragment
            for fragment in re.findall(r"<tr.*?</tr>", model.summary()._repr_html_(), re.S)
            if "cat[RARE]" in fragment
        )
        assert "***" in html_row

        term, _ = _export_term(model, "cat[RARE]")
        assert term.significance == "***"

        editor = _editor_row(model, "cat[RARE]")
        assert editor["sig_code"] == "***"
        assert editor["sig_class"] == "sig-strong"

    def test_every_renderer_still_carries_the_advisory(self):
        """Keeping the stars may not cost the reader the flag itself."""
        model = _thin_but_cleanly_estimated()

        console_line = next(
            line for line in str(model.summary()).split("\n") if "cat[RARE]" in line
        )
        other_line = next(line for line in str(model.summary()).split("\n") if "cat[B]" in line)
        assert "?" in console_line
        assert "?" not in other_line

        html = model.summary()._repr_html_()
        rare_row = next(
            fragment
            for fragment in re.findall(r"<tr.*?</tr>", html, re.S)
            if "cat[RARE]" in fragment
        )
        assert "?" in rare_row

        term, _ = _export_term(model, "cat[RARE]")
        other, _ = _export_term(model, "cat[B]")
        assert term.warning != ""
        assert other.warning == ""

        editor = _editor_row(model, "cat[RARE]")
        assert editor["quasi_separated"] is True
        assert editor["advisory_code"] == "?"
        assert _editor_row(model, "cat[B]")["advisory_code"] == ""


class TestTheLegendDescribesTheRuleThatExists:
    """A reader trusts a legend absolutely, so it may not out-run the rule.

    Separation and an infinite MLE component are logically equivalent (Albert
    and Anderson, *Biometrika* 71(1), 1984), and separation is a joint property
    of the design and the response that a marginal cell count can neither
    establish nor rule out.  The legend used to assert both, plus a log-link
    divergence, of every flagged level -- on a fit that has no log link.
    """

    _FABRICATED = (
        "no finite MLE",
        "diverges",
        "perfectly or nearly predicts",
        "log-link",
        "log link",
        "separation",
        "separated",
    )

    def _surfaces(self, model):
        term, payload = _export_term(model, "cat[RARE]")
        return {
            "ascii": str(model.summary()),
            "html": model.summary()._repr_html_(),
            "export": "\n".join(payload.notes) + "\n" + term.warning,
        }

    def test_no_surface_asserts_a_diagnosis_the_rule_never_made(self):
        model = _thin_but_cleanly_estimated()
        for surface, text in self._surfaces(model).items():
            lowered = text.lower()
            for claim in self._FABRICATED:
                assert claim not in lowered, f"{surface} still asserts {claim!r}"

    def test_every_surface_states_the_rule_and_what_it_licenses(self):
        model = _thin_but_cleanly_estimated()
        for surface, text in self._surfaces(model).items():
            lowered = text.lower()
            assert "20 observations" in lowered, f"{surface} does not state the count trigger"
            assert "0.05%" in lowered, f"{surface} does not state the exposure trigger"
            assert "credibility" in lowered, f"{surface} does not say what the flag means"


class TestTheNoteIsTrueOfTheSecondTriggerToo:
    """The SE branch is not a volume test, so the note may not call it one.

    Its threshold is this model's median parametric standard error times fifty,
    which is not scale-invariant: rescaling ONE predictor moves its standard
    error without moving the median, so the flag fires on units alone. A note
    that told the reader little data stood behind such a row would be as wrong
    as the separation claim it replaced, in the other direction.
    """

    @staticmethod
    def _rescaled_predictor(scale):
        rng = np.random.default_rng(4)
        n = 5000
        a = rng.normal(0, 1, n)
        b = rng.normal(0, 1, n)
        c = rng.normal(0, 1, n)
        y = 1.0 + 0.5 * a + 0.3 * b + 0.2 * c + rng.normal(0, 0.5, n)
        X = pd.DataFrame({"a": a, "b": b, "c": c * scale})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"a": Numeric(), "b": Numeric(), "c": Numeric()},
        )
        model.fit(X, y)
        return model, len(X)

    def test_the_flag_fires_on_units_alone(self):
        plain, _ = self._rescaled_predictor(1.0)
        rescaled, n_rows = self._rescaled_predictor(1e-6)

        before = next(r for r in plain.summary()._coef_rows if r.name == "c")
        after = next(r for r in rescaled.summary()._coef_rows if r.name == "c")

        # Same fit: the rescaling is a change of units, not of information.
        # Compared RELATIVELY -- both p-values here are ~1e-176, so an absolute
        # tolerance would pass for any two numbers below it -- and the z keeps
        # its sign, which a positive rescale preserves.
        assert after.p == pytest.approx(before.p, rel=1e-9)
        assert after.z == pytest.approx(before.z, rel=1e-6)
        # But the standard error moves by the scale factor, and the flag follows.
        assert after.se > before.se * 1e5
        assert before.quasi_separated is False
        assert after.quasi_separated is True
        # With every one of the fit's rows behind it.
        assert n_rows == 5000

    def test_the_note_sends_the_reader_to_the_units_first(self):
        rescaled, _ = self._rescaled_predictor(1e-6)
        text = str(rescaled.summary()).lower()

        assert "units" in text
        # And it does not attribute this row to a shortage of data.
        assert "little data stands behind" not in text

    def test_the_threshold_has_an_absolute_floor_the_note_declares(self):
        """The SE branch is a screen, not a ranking, and the note says so.

        ``max(50 * median_se, 10.0)``: the floor binds whenever the median
        parametric standard error is under 0.2, which is the ordinary case, so
        a coefficient can be hundreds of times the typical width and go
        unflagged.  A reader who took "far above this model's typical one" as
        the whole rule would scan the LC column for the widest-relative
        coefficients and not find this one.
        """
        rescaled, _ = self._rescaled_predictor(1e-3)
        rows = {r.name: r for r in rescaled.summary()._coef_rows}
        c = rows["c"]
        typical = float(
            np.median([r.se for name, r in rows.items() if name != "Intercept" and r.se])
        )

        # Nearly a thousand times the model's typical width...
        assert c.se / typical > 500.0
        # ...and below the absolute floor, so not flagged.
        assert c.se < 10.0
        assert c.quasi_separated is False

        # The note is only printed when something is flagged, so the wording is
        # read off a fit where the flag fires -- and it has to warn that this
        # unflagged 985x coefficient exists.
        flagged, _ = self._rescaled_predictor(1e-6)
        # Whitespace-normalised: the note is hard-wrapped, so a phrase test on
        # the raw text would be a test of where the line breaks fall.
        note = re.sub(r"\s+", " ", str(flagged.summary())).lower()
        assert "above a fixed floor" in note
        assert "screen rather than a ranking" in note

    def test_the_exported_warning_cell_names_the_trigger_that_fired(self):
        """That cell lands in a spreadsheet column without its legend.

        A downstream consumer reads the Warning value on its own, so it has to
        be a true statement about THAT row rather than the union of the two
        triggers. The branches are disjoint -- the SE fallback skips any row
        with per-level counts -- so the row itself says which one fired.
        """
        rescaled, _ = self._rescaled_predictor(1e-6)
        se_flagged, _ = _export_term(rescaled, "c")
        assert se_flagged.warning == "Outsized standard error"
        assert se_flagged.significance != ""

        thin = _thin_but_cleanly_estimated()
        level_flagged, _ = _export_term(thin, "cat[RARE]")
        assert level_flagged.warning == "Low credibility"


class TestColumnAlignment:
    """Sig and LC columns must not break border alignment."""

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
