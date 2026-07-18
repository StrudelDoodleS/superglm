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
    assert '`phi_method="mle"` and `method="brent"` are the defaults' in tweedie
    assert '`phi_method="pearson"` is an explicit fast plug-in' in tweedie
    assert "`result.ci()` is explicit and potentially expensive" in tweedie
    assert "φ / wᵢ" in tweedie
    assert "does not enter the linear predictor or automatically scale" in tweedie
    assert "Use an explicit offset" in tweedie
    assert "Zero-weight observations must be removed" in tweedie
    assert "REML selects spline smoothing penalties" in tweedie
    assert "does not jointly estimate *p* and φ" in tweedie
    assert "certified compound-Poisson/Gamma series" in tweedie
    assert "never select a saddlepoint approximation" in tweedie
    assert "near *p*=1 and *p*=2" in tweedie


def test_tweedie_notebook_removes_stale_pearson_default_claims():
    source = _notebook_source()

    assert "default Pearson profile" not in source
    assert "Pearson moments by default" not in source
    assert "Nested MLE is the default" in source
    assert "Pearson plug-in is explicit" in source
    assert "`result.ci()`" in source
    assert "Zero-weight observations must be removed" in source
    assert "REML selects spline smoothing penalties" in source
    assert "certified compound-Poisson/Gamma series" in source
    assert "never selects a saddlepoint approximation" in source
    assert "near `p=1` and `p=2`" in source


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
        assert "Non-Tweedie families retain their existing weighting behavior" in docstring


def test_quickstart_scopes_frequency_weight_guidance_to_poisson():
    quickstart = (_ROOT / "docs/getting-started/quickstart.md").read_text()
    weights = " ".join(quickstart.split("## Weights And Offsets", maxsplit=1)[1].split())

    assert "For the Poisson frequency examples on this page" in weights
    assert "Weight semantics are family-specific" in weights
    assert "Var(Y_i | x_i) = phi * mu_i**p / w_i" in weights
    assert "not replication counts" in weights
    assert "does not enter the linear predictor or automatically multiply the mean" in weights
    assert "`sample_weight=` is interpreted as exposure / frequency weight" not in weights
