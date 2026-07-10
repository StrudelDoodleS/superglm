import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.debug_weights import compare_irls_weights
from superglm.features.numeric import Numeric
from superglm.solvers.pirls import _positive_working_weight_stats


def test_positive_working_weight_stats_are_scale_invariant_for_subnormals():
    tiny = np.nextafter(0.0, 1.0)

    w_min, w_max, w_ratio = _positive_working_weight_stats(np.array([0.0, tiny, 4.0 * tiny]))

    assert w_min == tiny
    assert w_max == 4.0 * tiny
    assert w_ratio == 4.0


def test_positive_working_weight_stats_for_all_zero_weights():
    w_min, w_max, w_ratio = _positive_working_weight_stats(np.zeros(3))

    assert np.isnan(w_min)
    assert np.isnan(w_max)
    assert np.isinf(w_ratio)


def test_compare_irls_weights_ignores_zero_frequency_rows():
    pytest.importorskip("statsmodels")
    x = np.linspace(-1.0, 1.0, 30)
    X = pd.DataFrame({"x": x})
    y = 2.0 + 0.5 * x + 0.01 * np.sin(3.0 * x)
    weights = np.ones_like(x)
    weights[0] = 0.0
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y, sample_weight=weights, record_diagnostics=True)

    report = compare_irls_weights(model, X, y, sample_weight=weights, max_iter=1)

    assert set(report["source"]) == {"statsmodels", "superglm"}
    assert np.all(report["W_min"].to_numpy() == 0.0)
    np.testing.assert_allclose(report["W_ratio"].to_numpy(), 1.0)
    superglm_row = report.loc[report["source"] == "superglm"].iloc[0]
    assert superglm_row["step_halvings"] == 0
    assert not bool(superglm_row["step_rejected"])
    assert not bool(model.iteration_diagnostics().iloc[0]["step_rejected"])
