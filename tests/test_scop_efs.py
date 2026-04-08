"""Tests for SCOP state returned from fit_irls_direct.

Verifies that the return_scop_state=True parameter correctly exposes
converged SCOP Newton state (beta_eff, H_scop_penalized, etc.) needed
by the EFS outer loop for automatic smoothing selection.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.families import Gaussian
from superglm.features.spline import PSpline
from superglm.model.base import model_build_design_matrix
from superglm.solvers.irls_direct import fit_irls_direct


@pytest.fixture
def scop_model_inputs():
    """Build a minimal SCOP model ready for fit_irls_direct."""
    rng = np.random.default_rng(42)
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    y = 2 * x + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family=Gaussian(),
        selection_penalty=0,
        discrete=True,
        features={"x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")},
    )
    # Do NOT call auto_detect — features= dict already populates _specs.
    # auto_detect would overwrite the PSpline spec with Numeric().
    y_out, sample_weight, offset = model_build_design_matrix(model, df, y, np.ones(n), None)
    return model, y_out, sample_weight, offset


class TestReturnSCOPState:
    """Tests for return_scop_state parameter of fit_irls_direct."""

    @pytest.mark.slow
    def test_return_scop_state_with_xtwx_returns_4_tuple(self, scop_model_inputs):
        """return_scop_state=True with return_xtwx=True returns a 4-tuple."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_xtwx=True,
            return_scop_state=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 4, f"Expected 4-tuple, got {len(out)}-tuple"

        result, XtWX_S_inv, XtWX, scop_states = out
        assert scop_states is not None
        assert isinstance(scop_states, dict)

    @pytest.mark.slow
    def test_return_scop_state_without_xtwx_returns_3_tuple(self, scop_model_inputs):
        """return_scop_state=True without return_xtwx returns a 3-tuple."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_xtwx=False,
            return_scop_state=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 3, f"Expected 3-tuple, got {len(out)}-tuple"

        result, XtWX_S_inv, scop_states = out
        assert scop_states is not None
        assert isinstance(scop_states, dict)

    @pytest.mark.slow
    def test_scop_states_has_one_entry_per_scop_group(self, scop_model_inputs):
        """scop_states dict should have one entry per SCOP group."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_scop_state=True,
        )
        _, _, scop_states = out

        # Count SCOP groups in the model
        n_scop = sum(
            1 for g in model._groups if getattr(g, "scop_reparameterization", None) is not None
        )
        assert len(scop_states) == n_scop
        assert len(scop_states) >= 1, "Expected at least one SCOP group"

    @pytest.mark.slow
    def test_scop_state_has_required_keys(self, scop_model_inputs):
        """Each SCOP state entry must have all required keys."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_scop_state=True,
        )
        _, _, scop_states = out

        required_keys = {
            "beta_eff",
            "H_scop_penalized",
            "S_scop",
            "B_scop",
            "reparam",
            "bin_idx",
            "group_sl",
            "group_name",
        }

        for gi, state in scop_states.items():
            missing = required_keys - set(state.keys())
            assert not missing, f"Group {gi} missing keys: {missing}"

    @pytest.mark.slow
    def test_H_penalized_positive_definite(self, scop_model_inputs):
        """H_scop_penalized should be positive definite (all eigenvalues > -1e-8)."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_scop_state=True,
        )
        _, _, scop_states = out

        for gi, state in scop_states.items():
            H = state["H_scop_penalized"]
            assert H is not None, f"H_scop_penalized is None for group {gi}"
            eigvals = np.linalg.eigvalsh(H)
            assert np.all(eigvals > -1e-8), (
                f"Group {gi}: H not PD, min eigenvalue = {eigvals.min():.2e}"
            )

    @pytest.mark.slow
    def test_default_return_scop_state_false_returns_unchanged(self, scop_model_inputs):
        """Default return_scop_state=False returns standard 2-tuple (no SCOP state)."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
        )
        assert isinstance(out, tuple)
        assert len(out) == 2, f"Expected 2-tuple, got {len(out)}-tuple"

    @pytest.mark.slow
    def test_default_return_scop_state_false_with_xtwx_returns_3_tuple(self, scop_model_inputs):
        """Default return_scop_state=False with return_xtwx=True returns standard 3-tuple."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_xtwx=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 3, f"Expected 3-tuple, got {len(out)}-tuple"

    @pytest.mark.slow
    def test_beta_eff_shape_matches_group(self, scop_model_inputs):
        """beta_eff shape should match the SCOP basis dimension."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_scop_state=True,
        )
        _, _, scop_states = out

        for gi, state in scop_states.items():
            beta = state["beta_eff"]
            S = state["S_scop"]
            B = state["B_scop"]
            assert beta.ndim == 1
            assert S.shape[0] == S.shape[1] == len(beta)
            assert B.shape[1] == len(beta)

    @pytest.mark.slow
    def test_H_penalized_shape_matches_beta(self, scop_model_inputs):
        """H_scop_penalized shape should be (q_eff, q_eff) matching beta_eff."""
        model, y, sample_weight, offset = scop_model_inputs
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_scop_state=True,
        )
        _, _, scop_states = out

        for gi, state in scop_states.items():
            q = len(state["beta_eff"])
            H = state["H_scop_penalized"]
            assert H.shape == (q, q), f"Expected ({q},{q}), got {H.shape}"
