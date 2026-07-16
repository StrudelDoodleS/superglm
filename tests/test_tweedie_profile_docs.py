"""Documentation contracts for Tweedie profile estimation."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _notebook_source() -> str:
    notebook = json.loads((_ROOT / "docs/notebooks/tweedie_profile_estimation.ipynb").read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )


def test_tweedie_family_guide_documents_current_profile_contract():
    guide = (_ROOT / "docs/guide/families.md").read_text()
    tweedie = guide.split("## Tweedie: estimating the power parameter", maxsplit=1)[1]

    assert "p_range=" not in tweedie
    assert "p_bounds=(1.1, 1.9)" in tweedie
    assert '`phi_method="mle"` and `method="brent"` are the defaults' in tweedie
    assert '`phi_method="pearson"` is an explicit fast plug-in' in tweedie
    assert "`result.ci()` is explicit and potentially expensive" in tweedie
    assert "φ / wᵢ" in tweedie
    assert "Zero-weight observations are removed" in tweedie
    assert "REML selects spline smoothing penalties" in tweedie
    assert "does not jointly estimate *p* and φ" in tweedie
    assert "saddlepoint fallback" in tweedie
    assert "near *p*=1 and *p*=2" in tweedie


def test_tweedie_notebook_removes_stale_pearson_default_claims():
    source = _notebook_source()

    assert "default Pearson profile" not in source
    assert "Pearson moments by default" not in source
    assert "Nested MLE is the default" in source
    assert "Pearson plug-in is explicit" in source
    assert "`result.ci()`" in source
    assert "Zero-weight observations are removed" in source
    assert "REML selects spline smoothing penalties" in source
    assert "saddlepoint fallback" in source
    assert "near `p=1` and `p=2`" in source
