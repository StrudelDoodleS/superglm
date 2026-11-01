"""Bounded regressions for convergence and boundary disclosures in the issue audit."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from superglm import (
    Constraint,
    ConvergenceWarning,
    GaussianLS,
    GeneralizedParetoLSS,
    LambdaPolicy,
    Numeric,
    Spline,
    SuperGLM,
    SuperLSS,
    s,
)
from superglm.distributional import Predictor
from superglm.editor.refit import fit_refit_model
from tests.bound_predictor_fixtures import model_from_templates


def _scalar_data():
    rng = np.random.default_rng(469)
    x = np.linspace(-1.0, 1.0, 160)
    return pd.DataFrame({"x": x}), rng.poisson(np.exp(0.3 + 1.4 * x)).astype(float)


def _lss_data():
    rng = np.random.default_rng(469)
    x = np.linspace(-1.0, 1.0, 240)
    return pd.DataFrame({"x": x}), np.sin(3.0 * x) + rng.normal(0.0, 0.3, len(x))


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_scalar_diagnostics_disclose_the_only_coefficient_fit_stop(method):
    """Kills taking the termination reason only from a nonexistent REML result."""
    frame, y = _scalar_data()
    model = SuperGLM(family="poisson", features={"x": Numeric()}, selection_penalty=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        getattr(model, method)(
            frame, y, **({"max_iter": 1} if method == "fit" else {"max_pirls_iter": 1})
        )

    assert model.result.converged is False
    diagnostic = model.diagnostics()["_model"]
    assert diagnostic["termination_reason"] == "max_iter"
    assert "did not converge" in diagnostic["convergence_note"]
    if method == "fit":
        assert "fit_reml" not in diagnostic["convergence_note"]
    assert "did not converge" in model.summary()._info["convergence_note"]


def test_scalar_probability_boundary_advice_does_not_suggest_a_larger_budget():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 4)})
    model = SuperGLM(
        family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
    ).fit(frame, np.array([0.0, 0.0, 1.0, 1.0]))
    assert not model.result.converged
    assert model.result.termination_reason == "mean_space_boundary"
    note = model.diagnostics()["_model"]["convergence_note"]
    assert "boundary" in note
    assert "larger max_iter" not in note
    assert "logit" in note


@pytest.mark.parametrize("inner_cap", [False, True], ids=["smoothing", "coefficient"])
def test_lss_budget_disclosure_names_the_budget_that_was_exhausted(inner_cap):
    """Kills generic restart advice that omits an exhausted user-supplied budget."""
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
    """Terminal no-ascent evidence must not turn into an instruction to raise a budget."""
    from superglm.diagnostics.convergence import lss_nonconvergence_message

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
    unknown = lss_nonconvergence_message(model.result_, None)
    assert "max_iterations" not in unknown
    assert "larger max_inner_iter" not in unknown


@pytest.mark.parametrize("edf", [-3.0e-17, -0.125], ids=["roundoff", "material"])
def test_summary_distinguishes_roundoff_negative_edf_from_material_negative_edf(monkeypatch, edf):
    """Kills classifying a roundoff residual as an uninterpretable degree of freedom."""
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
    assert model.term_test("location", "x").edf == edf
    if edf == -0.125:
        assert row["edf"] == -0.125
        assert "negative EDF" in row["note"]
    else:
        assert row["edf"] == 0.0
        assert "negative EDF" not in row["note"]


def test_editor_refit_convergence_warning_points_to_the_external_caller():
    """Kills a fixed stacklevel that exposes the editor's refit dispatcher as the caller."""
    frame, y = _scalar_data()
    source = SuperGLM(family="poisson", features={"x": Spline(k=6)}, selection_penalty=0)
    refit = SuperGLM(family="poisson", features={"x": Spline(k=6)}, selection_penalty=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_refit_model(
            source,
            refit,
            method="fit_reml",
            X=frame,
            y=y,
            fit_kwargs={"max_reml_iter": 1},
        )
    convergence = [item for item in caught if isinstance(item.message, ConvergenceWarning)]
    assert convergence
    assert all(item.filename == __file__ for item in convergence)


def test_edited_plain_fit_does_not_claim_the_solver_failed():
    """A manual revision invalidates the mode; the editor's existing note explains it."""
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
    info = edited.summary()._info
    assert "convergence_note" not in info
    assert info["editor_inference_stale"] is True


@pytest.mark.parametrize("fixed_lambda", [True, False], ids=["fixed", "estimated"])
def test_feasible_qp_without_kkt_certificate_is_returned_and_disclosed(monkeypatch, fixed_lambda):
    """Forces the terminal event rather than relying on BLAS to miss a KKT tolerance."""
    import superglm.solvers.irls_direct._qp as qp

    original = qp.solve_constrained_qp

    def incomplete(*args, **kwargs):
        result = original(*args, **kwargs)
        result.converged = False
        return result

    monkeypatch.setattr(qp, "solve_constrained_qp", incomplete)
    frame, y = _scalar_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={
            "x": Spline(
                k=6,
                constraint=Constraint.fit.increasing,
                shape_engine="qp",
                lambda_policy=LambdaPolicy.fixed(1.0) if fixed_lambda else None,
            )
        },
    )
    with pytest.warns(ConvergenceWarning, match="KKT certificate"):
        model.fit_reml(frame, y, max_pirls_iter=3)
    assert model.result.converged is False
    assert model.result.termination_reason == "constraint_kkt_incomplete"
    assert np.min(np.diff(model.predict(frame))) >= -1.0e-10
    diagnostic = model.diagnostics()["_model"]
    assert diagnostic["converged"] is False
    assert "KKT certificate" in diagnostic["convergence_note"]


def test_unfinished_interior_gpd_fit_reports_current_wall_movement_conditionally():
    """At eight iterations an interior optimum can still look like two escaping walls."""
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(20.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(len(x)))) / shape
    frame = pd.DataFrame({"x": x})
    models = []
    for cap in (8, 40):
        model = model_from_templates(
            family=GeneralizedParetoLSS(),
            predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(frame, y, max_inner_iter=cap)
        models.append(model)
    capped, settled = models
    assert capped.result_.coefficient_converged is False
    wall = next(message for message in capped.result_.warnings if "tail shape at the" in message)
    assert "not converged" in wall
    assert "the data put the shape" not in wall
    assert "if" in wall.lower()
    assert settled.result_.coefficient_converged is True
    assert not [message for message in settled.result_.warnings if "tail shape at the" in message]


def test_custom_wall_hook_keeps_the_existing_keyword_contract():
    """A built-in's extra disclosure context must not break a custom hook's signature."""
    from superglm.distributional.fit_state import _wall_disclosure

    class LegacyWallFamily(GeneralizedParetoLSS):
        def wall_disclosure(self, pressed, clipped, n_rows, *, unresolved=False):
            return "custom: " + super().wall_disclosure(
                pressed, clipped, n_rows, unresolved=unresolved
            )

    family = LegacyWallFamily()
    y = np.expm1(-1.5 * np.log(np.random.default_rng(8).random(600))) / 1.5
    model = model_from_templates(
        family=family, predictors=(Predictor("scale", {}), Predictor("shape", {}))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(pd.DataFrame({"x": np.zeros(len(y))}), y, max_inner_iter=8)
    fitted = model._require_fitted()
    message = _wall_disclosure(family, fitted.result, fitted.layout)
    assert message.startswith("custom: ")
    assert "600 of 600" in message


def test_unsettled_wall_disclosure_preserves_factual_clipping_counts():
    """Conditional optimum claims do not erase rows already held at a float64 wall."""
    family = GeneralizedParetoLSS()
    message = family.wall_disclosure((0, 0), (3, 4), 19, settled=False, unresolved=True)
    assert "3 of 19 rows" in message and "last value above the lower wall" in message
    assert "4 of 19 rows" in message and "last value below the upper wall" in message
    assert "float64 can represent" in message
    assert "not reliable" in message


@pytest.mark.parametrize("cap", [8, 12])
def test_unfinished_gpd_fit_still_discloses_true_wall_pressure(cap):
    """Kills suppressing every wall warning merely because coefficients are unconverged."""
    rng = np.random.default_rng(8)
    x = rng.uniform(-1.0, 1.0, 3_000)
    y = np.exp(0.4 + 0.7 * np.sin(np.pi * x)) * np.expm1(-1.5 * np.log(rng.random(len(x)))) / 1.5
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    )
    with pytest.warns(ConvergenceWarning) as caught:
        model.fit(pd.DataFrame({"x": x}), y, max_inner_iter=cap)
    assert any("at the upper wall 1" in str(item.message) for item in caught)
    assert any("infinite" in message for message in model.result_.warnings)
