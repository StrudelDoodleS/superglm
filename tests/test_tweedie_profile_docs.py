"""Documentation contracts for Tweedie profile estimation."""

from __future__ import annotations

import json
from pathlib import Path

from superglm import SuperGLM
from superglm.profiling.tweedie import estimate_tweedie_p

_ROOT = Path(__file__).resolve().parents[1]


def _notebook_source() -> str:
    notebook = json.loads((_ROOT / "docs/notebooks/tweedie_profile_estimation.ipynb").read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )


def _notebook_all_source() -> str:
    notebook = json.loads((_ROOT / "docs/notebooks/tweedie_profile_estimation.ipynb").read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_tweedie_notebook_first_code_cell_executes_supported_public_imports():
    path = _ROOT / "docs/notebooks/tweedie_profile_estimation.ipynb"
    notebook = json.loads(path.read_text())
    first_code_cell = next(cell for cell in notebook["cells"] if cell.get("cell_type") == "code")
    source = "".join(first_code_cell.get("source", []))
    namespace: dict[str, object] = {}

    exec(compile(source, str(path), "exec"), namespace)

    from superglm import generate_tweedie_cpg

    assert namespace["generate_tweedie_cpg"] is generate_tweedie_cpg
    assert all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_tweedie_family_guide_documents_current_profile_contract():
    guide = (_ROOT / "docs/guide/families.md").read_text()
    tweedie = guide.split("## Tweedie: estimating the power parameter", maxsplit=1)[1]

    assert "per-exposure response" in tweedie
    assert "p_range=" not in tweedie
    assert "p_bounds=(1.1, 1.9)" in tweedie
    assert '`phi_method="mle"` and `method="auto"` are the defaults' in tweedie
    assert '`phi_method="pearson"` is an explicit fast plug-in' in tweedie
    assert "`result.ci()` is explicit and potentially expensive" in tweedie
    assert "ci_alpha=0.05" in tweedie
    assert "detached returned result" in tweedie
    assert "cached for `model.summary(alpha=0.05)`" in tweedie
    assert "summaries and exports display it after this explicit call" not in tweedie
    assert "φ / wᵢ" in tweedie
    assert "does not enter the linear predictor or automatically scale" in tweedie
    assert "Use an explicit offset" in tweedie
    assert "Zero-weight observations must be removed" in tweedie
    assert "REML selects spline smoothing penalties" in tweedie
    assert "does not jointly estimate *p* and φ" in tweedie
    assert "saddlepoint fallback" in tweedie
    assert "near *p*=1 and *p*=2" in tweedie


def test_binomial_family_guide_block_executes_a_real_fit() -> None:
    import numpy as np
    import pandas as pd

    from superglm import Numeric

    guide = (_ROOT / "docs/guide/families.md").read_text()
    section = guide.split("## Binomial (binary classification)", maxsplit=1)[1]
    block = section.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]
    frame = pd.DataFrame({"age": np.linspace(18.0, 80.0, 80)})
    response = (np.arange(len(frame)) % 3 == 0).astype(np.float64)
    namespace = {
        "df": frame,
        "features": {"age": Numeric()},
        "y": response,
    }

    exec(compile(block, "docs/guide/families.md#binomial", "exec"), namespace)

    probabilities = namespace["probabilities"]
    assert probabilities.shape == response.shape
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_tweedie_notebook_removes_stale_pearson_default_claims():
    source = _notebook_source()
    all_source = _notebook_all_source()

    assert "default Pearson profile" not in source
    assert "Pearson moments by default" not in source
    assert "Exact joint ML is the default" in source
    assert "Pearson plug-in is explicit" in source
    assert "`result.ci()`" in source
    assert "ci_alpha=0.05" in all_source
    assert "detached returned result" in source
    assert "model's independently owned summary cache" in source
    assert "preceding `result.ci()` call populated the cache" not in source
    assert "Zero-weight observations must be removed" in source
    assert "REML selects spline smoothing penalties" in source
    assert "saddlepoint fallback" in source
    assert "near `p=1` and `p=2`" in source


def test_selection_penalty_docs_make_calibration_explicit():
    readme = (_ROOT / "README.md").read_text()
    fitting = (_ROOT / "docs/guide/fitting.md").read_text()

    for document in (readme, fitting):
        assert 'SuperGLM(selection_penalty="auto")' in document
        assert "`None` and `0.0` disable sparse selection" in document
        assert "REML accepts only `None` or `0.0`" in document


def test_tweedie_profile_api_docstrings_use_prior_weight_convention():
    for function in (SuperGLM.estimate_p, estimate_tweedie_p):
        docstring = function.__doc__ or ""
        assert "EDM prior weights" in docstring
        assert "Var(Y_i | x_i) = phi * mu_i**p / w_i" in docstring
        assert "not replication or frequency weights" in docstring
        assert "Remove zero-weight rows consistently" in docstring
        assert "Frequency weights. Must be frequency weights" not in docstring


def test_public_fit_docstrings_explain_family_specific_tweedie_prior_weights():
    for function in (SuperGLM.fit, SuperGLM.fit_reml):
        docstring = " ".join((function.__doc__ or "").split())
        assert "likelihood interpretation is family-specific" in docstring
        assert "EDM prior weights" in docstring
        assert "Var(Y_i | x_i) = phi * mu_i**p / w_i" in docstring
        assert "not replication counts" in docstring
        assert "do not enter the linear predictor or automatically scale" in docstring
        assert "case/frequency weights" in docstring
        assert "once feature geometry is fixed" in docstring
        assert "likelihood-equivalent to row replication" in docstring
        assert "sum(w) - edf" in docstring


def test_plotting_docstrings_use_family_specific_weight_language() -> None:
    for function in (SuperGLM.plot, SuperGLM.plot_diagnostics):
        docstring = " ".join((function.__doc__ or "").split())
        assert "case/frequency weights for non-Tweedie families" in docstring
        assert "EDM prior weights for Tweedie" in docstring
    plot_docstring = " ".join((SuperGLM.plot_diagnostics.__doc__ or "").split())
    assert "sample-weighted observed vs predicted" in plot_docstring
    assert "exposure-weighted observed vs predicted" not in plot_docstring

    main_effects = (_ROOT / "src/superglm/plotting/main_effects.py").read_text()
    assert "Weights for the display-density overlay" in main_effects
    assert "Exposure / frequency weights." not in main_effects


def test_family_guide_makes_weight_contracts_mutually_exclusive():
    guide = (_ROOT / "docs/guide/families.md").read_text()
    weights = " ".join(
        guide.split("## Weight semantics", maxsplit=1)[1]
        .split("## Negative binomial", maxsplit=1)[0]
        .split()
    )

    assert "case/frequency weights" in weights
    assert "repeating the corresponding rows" in weights
    assert "`sum(sample_weight) - edf`" in weights
    assert "EDM prior weights" in weights
    assert "not replication counts" in weights
    assert "These contracts are not interchangeable" in weights
    assert "`n - edf`" in weights
    assert "Gamma fit does not implement that contract" in weights
    assert "retain the individual claim-severity rows" in weights
    assert "conditional on an identical constructed design" in weights
    assert "`quantile_rows` and `quantile_tempered` knot strategies use frequency mass" in weights
    assert "ignore zero-weight rows" in weights
    assert "Tweedie prior weights intentionally leave spline geometry" in weights
    assert "tensor-interaction marginal centering" in weights
    assert "categorical/ordered/factor feature geometry" in weights
    assert "fixed or preconstructed feature geometry" in weights
    assert "derive spline knots from the physical row distribution" not in weights
    assert "are not removed before feature construction" not in weights


def test_screening_guides_use_the_fitted_family_weight_contract() -> None:
    guide = (_ROOT / "docs/guide/screening.md").read_text()
    evaluation = (_ROOT / "docs/guide/screening-evaluation.md").read_text()
    normalized_guide = " ".join(guide.split())

    assert "`sum(sample_weight) - edf`" in normalized_guide
    assert "Tweedie EDM prior weights use the observation count minus `edf`" in normalized_guide
    assert "positive-weight-row count" not in guide
    assert "frequency-weight user can supply their own `phi`" not in guide
    assert "house exposure contract" not in guide
    assert "house exposure contract" not in evaluation
    assert "Poisson case/frequency-weight contract" in evaluation


def test_validation_and_diagnostic_guides_follow_family_weight_contracts() -> None:
    workflows = " ".join((_ROOT / "docs/guide/workflows.md").read_text().split())
    results = " ".join((_ROOT / "docs/guide/results.md").read_text().split())
    validation = " ".join((_ROOT / "docs/guide/validation.md").read_text().split())

    assert "`sum(sample_weight)`" in workflows
    assert "`sum(sample_weight) - edf`" in results
    for document in (workflows, results):
        assert "Tweedie EDM prior weights" in document
        assert "observation count" in document or "`n - edf`" in document
    assert "row's response distribution is unchanged by its weight" in results
    assert r"\(\phi / w_i\)" in results
    assert "diagnose raw claim counts with `log(exposure)` as an offset" in results
    assert "Poisson rate response" in validation
    assert "exposure is a case/frequency weight" in validation
    assert "Tweedie uses strictly positive EDM prior weights" in validation
    assert "portfolio aggregation weight" in validation
    assert "sample-weighted calibration" in validation
    assert "exposure-weighted calibration" not in validation
    assert (
        "a rate plus frequency weight cannot reconstruct the corresponding count CDF" in validation
    )


def test_monotone_guide_scopes_fit_curvature_to_cubic_or_lower_degree() -> None:
    guide = (_ROOT / "docs/guide/monotone.md").read_text()
    normalized = " ".join(guide.split())

    assert "convex/concave only when `degree <= 3`" in normalized
    assert "`PSpline` and `BSplineSmooth` with `degree > 3`" in normalized
    assert "still support fit-time increasing/decreasing constraints" in normalized
    assert "not `Constraint.fit.convex` or `Constraint.fit.concave`" in normalized
    assert "`CubicRegressionSpline` is always degree three" in normalized
    assert "This restriction does not apply to the separate `Constraint.postfit.*`" in normalized


def test_quickstart_scopes_frequency_weight_guidance_to_poisson():
    quickstart = (_ROOT / "docs/getting-started/quickstart.md").read_text()
    weights = " ".join(quickstart.split("## Weights And Offsets", maxsplit=1)[1].split())

    assert "For the Poisson frequency examples on this page" in weights
    assert "Weight semantics are family-specific" in weights
    assert "Var(Y_i | x_i) = phi * mu_i**p / w_i" in weights
    assert "not replication counts" in weights
    assert "does not enter the linear predictor or automatically multiply the mean" in weights
    assert "`sample_weight=` is interpreted as exposure / frequency weight" not in weights
