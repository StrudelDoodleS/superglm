import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.debug_weights import compare_irls_weights
from superglm.features.numeric import Numeric


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
