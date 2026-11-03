"""Convergence disclosures name the stop that happened and point at the caller (#461, #469)."""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import (
    ConvergenceWarning,
    GaussianLS,
    Numeric,
    Spline,
    SuperGLM,
    SuperLSS,
    families,
    s,
)
from superglm.diagnostics.convergence import (
    lss_nonconvergence_message,
    warn_coefficient_nonconvergence,
)
from superglm.editor.refit import fit_refit_model


def _scalar_data():
    rng = np.random.default_rng(469)
    x = np.linspace(-1.0, 1.0, 160)
    return pd.DataFrame({"x": x}), rng.poisson(np.exp(0.3 + 1.4 * x)).astype(float)


def _lss_data():
    rng = np.random.default_rng(469)
    x = np.linspace(-1.0, 1.0, 240)
    return pd.DataFrame({"x": x}), np.sin(3.0 * x) + rng.normal(0.0, 0.3, len(x))


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_diagnostics_and_summary_report_the_only_coefficient_fit_stop(method):
    frame, y = _scalar_data()
    model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=0)
    budget = {"max_iter": 1} if method == "fit" else {"max_pirls_iter": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        getattr(model, method)(frame, y, **budget)

    assert model.result.converged is False
    diagnostic = model.diagnostics()["_model"]
    assert diagnostic["termination_reason"] == "max_iter"
    assert "did not converge" in diagnostic["convergence_note"]
    assert "larger max_iter" in diagnostic["convergence_note"]
    assert "did not converge" in model.summary()._info["convergence_note"]


@pytest.mark.parametrize("reason", ["score_stagnated", "step_rejected", "nonfinite_deviance"])
def test_a_stop_that_is_not_a_budget_does_not_advise_a_larger_budget(reason):
    result = SimpleNamespace(converged=False, termination_reason=reason, n_iter=83)
    with pytest.warns(ConvergenceWarning) as caught:
        warn_coefficient_nonconvergence(result)
    message = str(caught[0].message)
    assert "larger max_iter" not in message and "larger max_pirls_iter" not in message
    assert "weights" in message and "offsets" in message
    assert caught[0].filename == __file__


def test_an_edited_plain_fit_does_not_claim_the_solver_failed():
    from superglm.editor import EditorSession

    frame, y = _scalar_data()
    model = SuperGLM(family="poisson", features={"x": Spline(k=6)}, selection_penalty=0).fit(
        frame, y
    )
    session = EditorSession.from_model(model, terms=["x"])
    session.select_x("x", -0.5, 0.5)
    session.shift("x", 0.2)
    edited = session.to_model()

    assert edited.result.termination_reason == "coefficients_revised"
    assert edited.diagnostics()["_model"]["convergence_note"] is None
    assert "convergence_note" not in edited.summary()._info


@pytest.mark.parametrize("inner_cap", [False, True], ids=["smoothing", "coefficient"])
def test_lss_budget_disclosure_names_the_budget_that_was_exhausted(inner_cap):
    frame, y = _lss_data()
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("x", k=6)), family.scale())
    with pytest.warns(ConvergenceWarning) as caught:
        model.fit_reml(
            frame,
            y,
            max_inner_iter=1 if inner_cap else 100,
            max_reml_iter=1,
            practical_reml=False,
        )
    fitted = model._require_fitted()
    reason = fitted.result.convergence_reason if inner_cap else fitted.smoothing.convergence_reason
    assert reason == "max_iterations"
    message = next(
        str(item.message) for item in caught if "SuperLSS fit did not converge" in str(item.message)
    )
    assert "max_iterations" in message
    assert f"larger {'max_inner_iter' if inner_cap else 'max_reml_iter'}" in message
    assert all("max_iterations" in note for note in model.summary()["note"])


def test_lss_no_ascent_disclosure_does_not_claim_an_iteration_cap():
    frame, y = _lss_data()
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("x", k=6)), family.scale()).fit(
        frame, y, lambdas={"location:x#wiggle": 1.0}, max_inner_iter=1
    )
    message = lss_nonconvergence_message(
        model.result_, None, coefficient_reason="no_ascent_direction"
    )
    assert "no_ascent_direction" in message
    assert "larger max_inner_iter" not in message
    assert "larger max_reml_iter" not in message


@pytest.mark.parametrize("edf", [-3.0e-17, -0.125], ids=["roundoff", "material"])
def test_summary_flags_only_a_material_negative_edf(monkeypatch, edf):
    import superglm.distributional.terms as terms

    frame, y = _lss_data()
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("x", k=6)), family.scale()).fit(
        frame, y, lambdas={"location:x#wiggle": 1.0}
    )
    original = terms._term_test_from_covariance

    def with_edf(prepared, matrix, *, note=""):
        return replace(original(prepared, matrix, note=note), edf=edf)

    monkeypatch.setattr(terms, "_term_test_from_covariance", with_edf)
    row = model.summary().loc[lambda table: table["term"] == "x"].iloc[0]
    assert row["edf"] == edf
    assert ("negative EDF" in row["note"]) == (edf == -0.125)


def test_editor_refit_convergence_warning_points_to_the_external_caller():
    frame, y = _scalar_data()
    source = SuperGLM(family="poisson", features={"x": Spline(k=6)}, selection_penalty=0)
    refit = SuperGLM(family="poisson", features={"x": Spline(k=6)}, selection_penalty=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_refit_model(
            source, refit, method="fit_reml", X=frame, y=y, fit_kwargs={"max_reml_iter": 1}
        )
    convergence = [item for item in caught if isinstance(item.message, ConvergenceWarning)]
    assert convergence
    assert all(item.filename == __file__ for item in convergence)


def test_theta_estimate_published_with_an_unconverged_fit_is_not_converged(monkeypatch):
    """Force only the publication refit to stop; the theta search itself settles."""
    from superglm.model import profile_ops

    real = profile_ops._refit_selected

    def stop_publication(final_model, validated, references, fit_mode, budget, retain):
        return real(final_model, validated, references, fit_mode, 1, retain)

    monkeypatch.setattr(profile_ops, "_refit_selected", stop_publication)
    rng = np.random.default_rng(469)
    frame = pd.DataFrame({"x": rng.uniform(0, 1, 240)})
    mean = np.exp(0.3 + 0.8 * np.sin(6 * frame["x"]))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mean)).astype(float)
    model = SuperGLM(family=families.nb2(theta=1), selection_penalty=0, features={"x": Spline(k=7)})
    with pytest.warns(ConvergenceWarning):
        published = model.estimate_theta(frame, y, fit_mode="reml")
    assert published.converged is False
    assert model._nb_profile_result.converged is False
