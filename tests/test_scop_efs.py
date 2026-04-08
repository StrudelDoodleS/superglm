"""Tests for SCOP EFS infrastructure.

Part 1: Tests for SCOP state returned from fit_irls_direct.
Part 2: Tests for build_scop_penalty_components.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.families import Gaussian
from superglm.features.spline import PSpline
from superglm.model.base import model_build_design_matrix
from superglm.reml.scop_efs import (
    assemble_joint_hessian,
    build_scop_penalty_components,
    compute_scop_aware_penalty_quad,
    scop_efs_lambda_update,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import PenaltyComponent


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


# ---------------------------------------------------------------------------
# Part 2: Tests for build_scop_penalty_components
# ---------------------------------------------------------------------------


def _first_diff_penalty(q):
    """Build first-difference penalty D'D for q parameters."""
    D = np.diff(np.eye(q), axis=0)
    return D.T @ D


class TestBuildSCOPPenaltyComponents:
    """Tests for build_scop_penalty_components (pure unit tests, no model fitting)."""

    def test_one_group_one_component(self):
        """One SCOP group produces exactly one PenaltyComponent."""
        q = 8
        S = _first_diff_penalty(q)
        scop_states = {
            0: {
                "S_scop": S,
                "group_sl": slice(1, 1 + q),
                "group_name": "x",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        assert len(pcs) == 1
        assert isinstance(pcs[0], PenaltyComponent)

    def test_omega_ssp_equals_S_scop(self):
        """omega_ssp should be S_scop directly, not an SSP transform."""
        q = 10
        S = _first_diff_penalty(q)
        scop_states = {
            0: {
                "S_scop": S,
                "group_sl": slice(1, 1 + q),
                "group_name": "x",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        np.testing.assert_array_equal(pcs[0].omega_ssp, S)
        np.testing.assert_array_equal(pcs[0].omega_raw, S)

    def test_rank_equals_q_minus_1(self):
        """Rank of D'D on q params is q-1 (one null space dimension)."""
        for q in [5, 8, 12, 20]:
            S = _first_diff_penalty(q)
            scop_states = {
                0: {
                    "S_scop": S,
                    "group_sl": slice(0, q),
                    "group_name": f"var_q{q}",
                    "beta_eff": np.zeros(q),
                }
            }
            pcs = build_scop_penalty_components(scop_states)
            assert pcs[0].rank == q - 1, f"q={q}: expected rank {q - 1}, got {pcs[0].rank}"

    def test_log_det_omega_plus_finite(self):
        """log_det_omega_plus should be finite for a valid first-diff penalty."""
        q = 10
        S = _first_diff_penalty(q)
        scop_states = {
            0: {
                "S_scop": S,
                "group_sl": slice(0, q),
                "group_name": "x",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        assert np.isfinite(pcs[0].log_det_omega_plus)

    def test_name_and_group_name_match(self):
        """pc.name and pc.group_name should match the group name from input."""
        q = 6
        S = _first_diff_penalty(q)
        scop_states = {
            3: {
                "S_scop": S,
                "group_sl": slice(5, 5 + q),
                "group_name": "driver_age",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        assert pcs[0].name == "driver_age"
        assert pcs[0].group_name == "driver_age"

    def test_group_sl_matches_input(self):
        """pc.group_sl should match the slice from scop_states."""
        q = 7
        sl = slice(10, 10 + q)
        S = _first_diff_penalty(q)
        scop_states = {
            2: {
                "S_scop": S,
                "group_sl": sl,
                "group_name": "age",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        assert pcs[0].group_sl == sl

    def test_group_index_preserved(self):
        """pc.group_index should match the key from scop_states."""
        q = 5
        S = _first_diff_penalty(q)
        scop_states = {
            7: {
                "S_scop": S,
                "group_sl": slice(0, q),
                "group_name": "feat",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        assert pcs[0].group_index == 7

    def test_multiple_groups(self):
        """Multiple SCOP groups produce one PenaltyComponent each."""
        states = {}
        for i, (q, name) in enumerate([(6, "age"), (9, "income"), (4, "tenure")]):
            S = _first_diff_penalty(q)
            states[i] = {
                "S_scop": S,
                "group_sl": slice(i * 20, i * 20 + q),
                "group_name": name,
                "beta_eff": np.zeros(q),
            }
        pcs = build_scop_penalty_components(states)
        assert len(pcs) == 3
        assert [pc.name for pc in pcs] == ["age", "income", "tenure"]
        # Check ranks
        assert pcs[0].rank == 5  # q=6 -> rank=5
        assert pcs[1].rank == 8  # q=9 -> rank=8
        assert pcs[2].rank == 3  # q=4 -> rank=3

    def test_eigvals_omega_length_matches_rank(self):
        """eigvals_omega should have exactly rank positive eigenvalues."""
        q = 10
        S = _first_diff_penalty(q)
        scop_states = {
            0: {
                "S_scop": S,
                "group_sl": slice(0, q),
                "group_name": "x",
                "beta_eff": np.zeros(q),
            }
        }
        pcs = build_scop_penalty_components(scop_states)
        assert len(pcs[0].eigvals_omega) == int(pcs[0].rank)
        assert np.all(pcs[0].eigvals_omega > 0)


class TestAssembleJointHessian:
    """Tests for assemble_joint_hessian."""

    def test_no_scop_returns_original(self):
        """Empty scop_states returns the original matrix and empty mapping."""
        rng = np.random.default_rng(42)
        p = 10
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)

        H_joint, mapping = assemble_joint_hessian(XtWX_plus_S, {})
        np.testing.assert_array_equal(H_joint, XtWX_plus_S)
        assert mapping == {}

    def test_scop_block_replaced(self):
        """SCOP block in H_joint should equal H_scop_penalized, not the original."""
        p = 12
        q_scop = 5
        scop_sl = slice(7, 12)  # last 5 cols

        rng = np.random.default_rng(99)
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)
        original_scop_block = XtWX_plus_S[scop_sl, scop_sl].copy()

        # Build a distinct H_scop
        B = rng.standard_normal((q_scop, q_scop))
        H_scop = B.T @ B + 3.0 * np.eye(q_scop)

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
                "beta_eff": np.zeros(q_scop),  # identity Jacobian
            }
        }

        H_joint, mapping = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # SCOP block should be H_scop, not the original
        np.testing.assert_array_equal(H_joint[scop_sl, scop_sl], H_scop)
        assert not np.allclose(H_joint[scop_sl, scop_sl], original_scop_block)

    def test_linear_block_unchanged(self):
        """Non-SCOP (linear) diagonal block must be unchanged after assembly."""
        p = 12
        q_scop = 5
        scop_sl = slice(7, 12)
        linear_sl = slice(0, 7)

        rng = np.random.default_rng(77)
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)

        B = rng.standard_normal((q_scop, q_scop))
        H_scop = B.T @ B + np.eye(q_scop)
        beta_eff = rng.standard_normal(q_scop) * 0.5

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
                "beta_eff": beta_eff,
            }
        }

        H_joint, _ = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # Linear diagonal block unchanged
        np.testing.assert_array_equal(
            H_joint[linear_sl, linear_sl], XtWX_plus_S[linear_sl, linear_sl]
        )

    def test_cross_blocks_scaled_by_jacobian(self):
        """Cross-blocks between linear and SCOP must be scaled by exp(beta_eff)."""
        p = 10
        q_scop = 4
        scop_sl = slice(6, 10)
        linear_sl = slice(0, 6)

        rng = np.random.default_rng(77)
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)

        B = rng.standard_normal((q_scop, q_scop))
        H_scop = B.T @ B + np.eye(q_scop)
        beta_eff = np.array([0.5, -0.3, 0.1, 0.8])
        j_diag = np.exp(beta_eff)

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
                "beta_eff": beta_eff,
            }
        }

        H_joint, _ = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # Cross-block [linear, scop] should be original * j_diag (column-wise)
        expected_cross = XtWX_plus_S[linear_sl, scop_sl] * j_diag[np.newaxis, :]
        np.testing.assert_allclose(H_joint[linear_sl, scop_sl], expected_cross, rtol=1e-12)

        # Cross-block [scop, linear] should be original * j_diag (row-wise)
        expected_cross_t = XtWX_plus_S[scop_sl, linear_sl] * j_diag[:, np.newaxis]
        np.testing.assert_allclose(H_joint[scop_sl, linear_sl], expected_cross_t, rtol=1e-12)

        # Verify cross-blocks are NOT unchanged (they were transformed)
        assert not np.allclose(H_joint[linear_sl, scop_sl], XtWX_plus_S[linear_sl, scop_sl])

    def test_mapping_correct(self):
        """Mapping dict has correct group_name -> slice entries."""
        p = 15
        sl_a = slice(5, 10)
        sl_b = slice(10, 15)

        XtWX_plus_S = np.eye(p)

        scop_states = {
            0: {
                "group_sl": sl_a,
                "H_scop_penalized": 2.0 * np.eye(5),
                "group_name": "spline_a",
                "beta_eff": np.zeros(5),
            },
            1: {
                "group_sl": sl_b,
                "H_scop_penalized": 3.0 * np.eye(5),
                "group_name": "spline_b",
                "beta_eff": np.zeros(5),
            },
        }

        _, mapping = assemble_joint_hessian(XtWX_plus_S, scop_states)

        assert "spline_a" in mapping
        assert "spline_b" in mapping
        assert mapping["spline_a"] == sl_a
        assert mapping["spline_b"] == sl_b

    def test_block_diagonal_logdet_additive(self):
        """For true block-diagonal (zero off-diag), log|H| = sum of log|block|."""
        p_lin = 4
        q_scop = 6
        p = p_lin + q_scop
        scop_sl = slice(p_lin, p)

        rng = np.random.default_rng(123)

        # Build block-diagonal XtWX_plus_S (zeros in off-diagonal blocks)
        A_lin = rng.standard_normal((p_lin, p_lin))
        linear_block = A_lin.T @ A_lin + np.eye(p_lin)

        XtWX_plus_S = np.zeros((p, p))
        XtWX_plus_S[:p_lin, :p_lin] = linear_block
        # Put placeholder in SCOP block (will be replaced)
        XtWX_plus_S[scop_sl, scop_sl] = np.eye(q_scop)

        # Build H_scop
        B = rng.standard_normal((q_scop, q_scop))
        S_scop = _first_diff_penalty(q_scop)
        H_scop = B.T @ B + S_scop + 0.5 * np.eye(q_scop)

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
                "beta_eff": np.zeros(q_scop),  # j_diag=1, off-diag zero → block-additive
            }
        }

        H_joint, _ = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # log|H_joint| should = log|linear_block| + log|H_scop|
        _, logdet_joint = np.linalg.slogdet(H_joint)
        _, logdet_linear = np.linalg.slogdet(linear_block)
        _, logdet_scop = np.linalg.slogdet(H_scop)

        np.testing.assert_allclose(logdet_joint, logdet_linear + logdet_scop, rtol=1e-10)

    def test_inverse_valid(self):
        """H_joint @ inv(H_joint) should approximate identity."""
        p_lin = 5
        q_scop = 7
        p = p_lin + q_scop
        scop_sl = slice(p_lin, p)

        rng = np.random.default_rng(456)

        # Build positive-definite XtWX_plus_S
        A = rng.standard_normal((2 * p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)

        # Build H_scop
        C = rng.standard_normal((q_scop, q_scop))
        H_scop = C.T @ C + 2.0 * np.eye(q_scop)

        beta_eff = rng.standard_normal(q_scop) * 0.3
        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
                "beta_eff": beta_eff,
            }
        }

        H_joint, _ = assemble_joint_hessian(XtWX_plus_S, scop_states)
        H_joint_inv = np.linalg.inv(H_joint)
        product = H_joint @ H_joint_inv

        np.testing.assert_allclose(product, np.eye(p), atol=1e-10)

    def test_original_matrix_not_mutated(self):
        """assemble_joint_hessian must not modify the input matrix."""
        p = 8
        q_scop = 3
        scop_sl = slice(5, 8)

        rng = np.random.default_rng(789)
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)
        original_copy = XtWX_plus_S.copy()

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": 5.0 * np.eye(q_scop),
                "group_name": "mono_z",
                "beta_eff": np.zeros(q_scop),
            }
        }

        assemble_joint_hessian(XtWX_plus_S, scop_states)
        np.testing.assert_array_equal(XtWX_plus_S, original_copy)

    def test_two_scop_cross_blocks_scaled_by_both_jacobians(self):
        """SCOP_i-SCOP_j cross-blocks get diag(j_i) @ H_ij @ diag(j_j)."""
        p_linear = 4
        q_a, q_b = 3, 5
        p = p_linear + q_a + q_b
        sl_lin = slice(0, p_linear)
        sl_a = slice(p_linear, p_linear + q_a)
        sl_b = slice(p_linear + q_a, p)

        rng = np.random.default_rng(123)
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)

        H_scop_a = rng.standard_normal((q_a, q_a))
        H_scop_a = H_scop_a.T @ H_scop_a + 2 * np.eye(q_a)
        H_scop_b = rng.standard_normal((q_b, q_b))
        H_scop_b = H_scop_b.T @ H_scop_b + 2 * np.eye(q_b)

        beta_eff_a = np.array([0.5, -0.3, 0.2])
        beta_eff_b = np.array([0.1, -0.4, 0.6, -0.1, 0.3])
        j_a = np.exp(beta_eff_a)
        j_b = np.exp(beta_eff_b)

        scop_states = {
            0: {
                "group_sl": sl_a,
                "H_scop_penalized": H_scop_a,
                "group_name": "age",
                "beta_eff": beta_eff_a,
            },
            1: {
                "group_sl": sl_b,
                "H_scop_penalized": H_scop_b,
                "group_name": "power",
                "beta_eff": beta_eff_b,
            },
        }

        H_joint, mapping = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # Each SCOP diagonal block replaced by its Newton Hessian
        np.testing.assert_array_equal(H_joint[sl_a, sl_a], H_scop_a)
        np.testing.assert_array_equal(H_joint[sl_b, sl_b], H_scop_b)

        # SCOP_a-SCOP_b cross-block: H_ab(beta_eff) = diag(j_a) @ H_ab(gamma) @ diag(j_b)
        H_ab_gamma = XtWX_plus_S[sl_a, sl_b]
        expected_ab = np.diag(j_a) @ H_ab_gamma @ np.diag(j_b)
        np.testing.assert_allclose(H_joint[sl_a, sl_b], expected_ab, rtol=1e-12)

        # Symmetric: H_ba(beta_eff) = diag(j_b) @ H_ba(gamma) @ diag(j_a)
        H_ba_gamma = XtWX_plus_S[sl_b, sl_a]
        expected_ba = np.diag(j_b) @ H_ba_gamma @ np.diag(j_a)
        np.testing.assert_allclose(H_joint[sl_b, sl_a], expected_ba, rtol=1e-12)

        # Linear-SCOP cross-blocks still scaled by single Jacobian
        expected_lin_a = XtWX_plus_S[sl_lin, sl_a] * j_a[np.newaxis, :]
        np.testing.assert_allclose(H_joint[sl_lin, sl_a], expected_lin_a, rtol=1e-12)
        expected_lin_b = XtWX_plus_S[sl_lin, sl_b] * j_b[np.newaxis, :]
        np.testing.assert_allclose(H_joint[sl_lin, sl_b], expected_lin_b, rtol=1e-12)

        # Linear diagonal block unchanged
        np.testing.assert_array_equal(H_joint[sl_lin, sl_lin], XtWX_plus_S[sl_lin, sl_lin])

        # Overall symmetry preserved
        np.testing.assert_allclose(H_joint, H_joint.T, atol=1e-12)

        # Mapping has both groups
        assert "age" in mapping and "power" in mapping


# ---------------------------------------------------------------------------
# Part 3: Tests for compute_scop_aware_penalty_quad
# ---------------------------------------------------------------------------


class TestSCOPPenaltyQuad:
    """Tests for compute_scop_aware_penalty_quad (pure unit tests, no model fitting)."""

    def test_scop_only_model(self):
        """Pure SCOP model: penalty_quad uses beta_eff, not gamma_eff.

        For a SCOP-only model, the full penalty matrix S = lam * S_scop.
        The naive quad is gamma_eff @ S @ gamma_eff (wrong).
        The correct quad is lam * beta_eff @ S_scop @ beta_eff.
        These should differ because gamma = exp(beta) != beta.
        """
        q = 8
        S_scop = _first_diff_penalty(q)
        lam = 2.5

        rng = np.random.default_rng(42)
        beta_eff = rng.standard_normal(q)
        gamma_eff = np.exp(beta_eff)

        # Full penalty matrix is just lam * S_scop for a single SCOP group
        S_full = lam * S_scop

        scop_states = {
            0: {
                "S_scop": S_scop,
                "beta_eff": beta_eff,
                "group_sl": slice(0, q),
                "group_name": "x",
            }
        }
        lambdas = {"x": lam}

        # result_beta contains gamma_eff for SCOP groups
        result_beta = gamma_eff.copy()

        pq = compute_scop_aware_penalty_quad(result_beta, S_full, scop_states, lambdas)

        # Should equal lam * beta_eff @ S_scop @ beta_eff
        expected = lam * float(beta_eff @ S_scop @ beta_eff)
        np.testing.assert_allclose(pq, expected, rtol=1e-12)

        # Should differ from the naive gamma-space quad
        naive_pq = float(gamma_eff @ S_full @ gamma_eff)
        assert not np.isclose(pq, naive_pq, rtol=1e-6), (
            "SCOP penalty quad should differ from naive gamma-space quad"
        )

    def test_mixed_ssp_and_scop(self):
        """Mixed model: SSP part uses gamma (correct), SCOP part uses beta_eff.

        Build a block-diagonal penalty matrix with an SSP block and a SCOP block.
        Verify that the SSP contribution is gamma @ S_ssp @ gamma and the
        SCOP contribution is lam_scop * beta_eff @ S_scop @ beta_eff.
        """
        q_ssp = 5
        q_scop = 6
        p = q_ssp + q_scop
        lam_ssp = 1.5
        lam_scop = 3.0

        rng = np.random.default_rng(99)

        # SSP block (linear group): coefficients are used as-is
        S_ssp = _first_diff_penalty(q_ssp)
        beta_ssp = rng.standard_normal(q_ssp)

        # SCOP block
        S_scop = _first_diff_penalty(q_scop)
        beta_eff = rng.standard_normal(q_scop)
        gamma_eff = np.exp(beta_eff)

        # Full penalty matrix (block-diagonal)
        S_full = np.zeros((p, p))
        S_full[:q_ssp, :q_ssp] = lam_ssp * S_ssp
        S_full[q_ssp:, q_ssp:] = lam_scop * S_scop

        # result_beta: SSP coefficients as-is, SCOP as gamma_eff
        result_beta = np.concatenate([beta_ssp, gamma_eff])

        scop_states = {
            1: {
                "S_scop": S_scop,
                "beta_eff": beta_eff,
                "group_sl": slice(q_ssp, p),
                "group_name": "mono_x",
            }
        }
        lambdas = {"mono_x": lam_scop}

        pq = compute_scop_aware_penalty_quad(result_beta, S_full, scop_states, lambdas)

        # Expected: SSP contribution + SCOP contribution in beta_eff space
        ssp_contrib = lam_ssp * float(beta_ssp @ S_ssp @ beta_ssp)
        scop_contrib = lam_scop * float(beta_eff @ S_scop @ beta_eff)
        expected = ssp_contrib + scop_contrib

        np.testing.assert_allclose(pq, expected, rtol=1e-12)

    def test_no_scop_terms_fallback(self):
        """No SCOP terms: falls back to standard result.beta @ S @ result.beta."""
        p = 10
        rng = np.random.default_rng(77)
        S = _first_diff_penalty(p)
        beta = rng.standard_normal(p)

        pq = compute_scop_aware_penalty_quad(beta, S, {}, {})

        expected = float(beta @ S @ beta)
        np.testing.assert_allclose(pq, expected, rtol=1e-14)

    def test_zero_lambda_scop_contributes_zero(self):
        """When lambda=0 for SCOP term, its contribution is zero."""
        q = 7
        S_scop = _first_diff_penalty(q)
        lam = 0.0

        rng = np.random.default_rng(123)
        beta_eff = rng.standard_normal(q)
        gamma_eff = np.exp(beta_eff)

        # With lambda=0, the S_full SCOP block is all zeros
        S_full = np.zeros((q, q))

        scop_states = {
            0: {
                "S_scop": S_scop,
                "beta_eff": beta_eff,
                "group_sl": slice(0, q),
                "group_name": "x",
            }
        }
        lambdas = {"x": lam}

        pq = compute_scop_aware_penalty_quad(gamma_eff, S_full, scop_states, lambdas)

        # With lambda=0, both subtracting and adding contribute zero
        np.testing.assert_allclose(pq, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Part 4: Tests for scop_efs_lambda_update
# ---------------------------------------------------------------------------


class TestSCOPEFSLambdaUpdate:
    """Tests for scop_efs_lambda_update (pure unit tests, no model fitting)."""

    def test_ssp_component_uses_gamma_space(self):
        """SSP PenaltyComponent with known beta, H_inv, omega produces finite positive lambda."""
        rng = np.random.default_rng(42)
        p = 5
        beta = rng.standard_normal(p)
        # Make a PD H_inv
        A = rng.standard_normal((p, p))
        H_joint_inv = np.linalg.inv(A.T @ A + np.eye(p))

        pc = PenaltyComponent(
            name="smooth",
            group_name="smooth",
            group_index=0,
            group_sl=slice(0, 5),
            omega_raw=np.eye(5) * 0.5,
            omega_ssp=np.eye(5) * 0.5,
            rank=4.0,
            log_det_omega_plus=0.0,
        )

        lam_old = 1.0
        inv_phi = 1.0
        scop_states = {}  # no SCOP groups

        lam_new = scop_efs_lambda_update(pc, beta, H_joint_inv, inv_phi, lam_old, scop_states)
        assert np.isfinite(lam_new)
        assert lam_new > 0

    def test_scop_component_uses_beta_eff(self):
        """SCOP component lambda uses beta_eff, NOT gamma_eff from result.beta."""
        rng = np.random.default_rng(99)
        q_eff = 5
        p = 8  # total param dimension

        S_scop = _first_diff_penalty(q_eff)

        # Two different coefficient vectors for the SCOP group
        beta_eff = rng.standard_normal(q_eff) * 2.0  # solver space
        gamma_eff = rng.standard_normal(q_eff) * 0.5  # gamma space (different)

        # Full beta vector with gamma_eff in the SCOP slice
        beta_full = np.zeros(p)
        beta_full[:3] = rng.standard_normal(3)
        beta_full[3:8] = gamma_eff

        # PD H_joint_inv
        A = rng.standard_normal((p, p))
        H_joint_inv = np.linalg.inv(A.T @ A + 5.0 * np.eye(p))

        pc = PenaltyComponent(
            name="age",
            group_name="age",
            group_index=1,
            group_sl=slice(3, 8),
            omega_raw=S_scop,
            omega_ssp=S_scop,
            rank=float(q_eff - 1),
            log_det_omega_plus=0.0,
        )
        scop_states = {
            1: {
                "beta_eff": beta_eff,
                "S_scop": S_scop,
                "group_sl": slice(3, 8),
                "group_name": "age",
            }
        }

        lam_old = 1.0
        inv_phi = 1.0

        # Compute with SCOP state (should use beta_eff)
        lam_scop = scop_efs_lambda_update(pc, beta_full, H_joint_inv, inv_phi, lam_old, scop_states)

        # Compute without SCOP state (would use gamma_eff from beta_full)
        lam_ssp = scop_efs_lambda_update(pc, beta_full, H_joint_inv, inv_phi, lam_old, {})

        # They should differ because beta_eff != gamma_eff
        assert lam_scop != lam_ssp, f"SCOP and SSP lambdas should differ: {lam_scop} vs {lam_ssp}"

        # Verify SCOP version manually: quad should use beta_eff
        quad_expected = float(beta_eff @ S_scop @ beta_eff)
        trace_expected = float(np.trace(H_joint_inv[3:8, 3:8] @ S_scop))
        denom_expected = inv_phi * quad_expected + trace_expected
        lam_raw_expected = float(q_eff - 1) / denom_expected
        log_step_expected = np.clip(
            np.log(max(lam_raw_expected, 1e-10)) - np.log(max(lam_old, 1e-10)),
            -5.0,
            5.0,
        )
        lam_expected = lam_old * np.exp(log_step_expected)
        np.testing.assert_allclose(lam_scop, lam_expected, rtol=1e-12)

    def test_uphill_guard_clips_log_step(self):
        """Extreme case: log-step must be clipped to [-5, 5]."""
        p = 5
        # Very small beta_eff and H_inv -> large lam_raw -> large positive log-step
        beta_eff = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        S_scop = _first_diff_penalty(p)

        H_joint_inv = 1e-10 * np.eye(p)

        pc = PenaltyComponent(
            name="x",
            group_name="x",
            group_index=0,
            group_sl=slice(0, 5),
            omega_raw=S_scop,
            omega_ssp=S_scop,
            rank=float(p - 1),
            log_det_omega_plus=0.0,
        )
        scop_states = {
            0: {
                "beta_eff": beta_eff,
                "S_scop": S_scop,
                "group_sl": slice(0, 5),
                "group_name": "x",
            }
        }

        lam_old = 1.0
        inv_phi = 1.0

        lam_new = scop_efs_lambda_update(
            pc, np.zeros(p), H_joint_inv, inv_phi, lam_old, scop_states
        )

        # log-step should be clipped, so lam_new = lam_old * exp(5)
        max_ratio = np.exp(5.0)
        min_ratio = np.exp(-5.0)
        ratio = lam_new / lam_old
        assert ratio <= max_ratio + 1e-10, f"Ratio {ratio} exceeds exp(5)"
        assert ratio >= min_ratio - 1e-10, f"Ratio {ratio} below exp(-5)"

    def test_near_zero_beta_returns_old_lambda(self):
        """If beta_g norm < 1e-12, returns lam_old unchanged."""
        p = 5
        beta = np.zeros(p)  # all zeros
        H_joint_inv = np.eye(p)

        pc = PenaltyComponent(
            name="smooth",
            group_name="smooth",
            group_index=0,
            group_sl=slice(0, 5),
            omega_raw=np.eye(5),
            omega_ssp=np.eye(5),
            rank=4.0,
            log_det_omega_plus=0.0,
        )

        lam_old = 42.0
        lam_new = scop_efs_lambda_update(pc, beta, H_joint_inv, 1.0, lam_old, {})
        assert lam_new == lam_old

        # Also test near-zero for SCOP
        scop_states = {
            0: {
                "beta_eff": np.full(5, 1e-15),
                "S_scop": np.eye(5),
                "group_sl": slice(0, 5),
                "group_name": "smooth",
            }
        }
        lam_new_scop = scop_efs_lambda_update(pc, beta, H_joint_inv, 1.0, lam_old, scop_states)
        assert lam_new_scop == lam_old

    def test_returns_positive(self):
        """Lambda is always positive for valid inputs."""
        rng = np.random.default_rng(123)
        p = 8
        q = 5

        for trial in range(20):
            beta = rng.standard_normal(p)
            A = rng.standard_normal((2 * p, p))
            H_joint_inv = np.linalg.inv(A.T @ A + np.eye(p))
            S = _first_diff_penalty(q)

            pc = PenaltyComponent(
                name="feat",
                group_name="feat",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=S,
                omega_ssp=S,
                rank=float(q - 1),
                log_det_omega_plus=0.0,
            )
            lam_old = rng.uniform(0.01, 100.0)
            inv_phi = rng.uniform(0.5, 2.0)

            lam_new = scop_efs_lambda_update(pc, beta, H_joint_inv, inv_phi, lam_old, {})
            assert lam_new > 0, f"Trial {trial}: lambda={lam_new} is not positive"


# ---------------------------------------------------------------------------
# Part 6: Tests for SCOP-aware REML objective
# ---------------------------------------------------------------------------


class TestSCOPAwareObjective:
    """Tests for reml_laml_objective with scop_states parameter."""

    @pytest.mark.slow
    def test_objective_accepts_scop_state(self, scop_model_inputs):
        """reml_laml_objective with scop_states returns a finite float."""
        from superglm.reml.objective import reml_laml_objective

        model, y, sample_weight, offset = scop_model_inputs
        offset_arr = offset if offset is not None else np.zeros_like(y)
        lambdas = {"x": 1.0}

        # Get PIRLS result + XtWX + scop_states
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
            return_scop_state=True,
        )
        result, _, XtWX, scop_states = out

        val = reml_laml_objective(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            result=result,
            lambdas=lambdas,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            XtWX=XtWX,
            scop_states=scop_states,
        )
        assert isinstance(val, float)
        assert np.isfinite(val), f"Objective returned non-finite value: {val}"

    @pytest.mark.slow
    def test_objective_without_scop_state_unchanged(self, scop_model_inputs):
        """Without scop_states (None), result matches the standard objective path."""
        from superglm.reml.objective import reml_laml_objective

        model, y, sample_weight, offset = scop_model_inputs
        offset_arr = offset if offset is not None else np.zeros_like(y)
        lambdas = {"x": 1.0}

        # Get PIRLS result + XtWX (no scop_states needed for baseline)
        out = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
        )
        result, _, XtWX = out

        # Call without scop_states (default None)
        val_none = reml_laml_objective(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            result=result,
            lambdas=lambdas,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            XtWX=XtWX,
        )

        # Call with explicit scop_states=None
        val_explicit_none = reml_laml_objective(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            result=result,
            lambdas=lambdas,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            XtWX=XtWX,
            scop_states=None,
        )

        assert isinstance(val_none, float)
        assert np.isfinite(val_none)
        assert val_none == val_explicit_none, (
            f"Default and explicit None should be identical: {val_none} vs {val_explicit_none}"
        )


# ---------------------------------------------------------------------------
# Part 7: Tests for optimize_scop_efs_reml (full SCOP EFS outer loop)
# ---------------------------------------------------------------------------

from superglm.reml.result import REMLResult  # noqa: E402
from superglm.reml.scop_efs import optimize_scop_efs_reml  # noqa: E402


class TestSCOPEFSOuterLoop:
    """Tests for the full SCOP-aware EFS outer loop."""

    @pytest.fixture
    def scop_reml_model(self):
        """Build SCOP model inputs for REML outer loop tests."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
        )
        y_out, sample_weight, offset = model_build_design_matrix(model, df, y, np.ones(n), None)
        offset_arr = np.zeros(n) if offset is None else np.array(offset)
        return model, y_out, np.array(sample_weight), offset_arr, df

    @pytest.mark.slow
    def test_converges(self, scop_reml_model):
        """optimize_scop_efs_reml should return REMLResult and converge."""
        model, y, sample_weight, offset, _ = scop_reml_model
        lambdas = {"x": 1.0}
        estimated_names = {"x"}

        result = optimize_scop_efs_reml(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            lambdas=lambdas,
            estimated_names=estimated_names,
            max_reml_iter=20,
            reml_tol=1e-6,
            verbose=False,
        )

        assert isinstance(result, REMLResult)
        # Should converge or at least finish within max_reml_iter
        assert result.converged or result.n_reml_iter < 20

    @pytest.mark.slow
    def test_lambda_responds_to_noise(self, scop_reml_model):
        """Higher noise should produce higher lambda (more smoothing)."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))

        results = {}
        for noise_label, sigma in [("low", 0.1), ("high", 1.0)]:
            y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, sigma, n)
            df = pd.DataFrame({"x": x})

            model = SuperGLM(
                family=Gaussian(),
                selection_penalty=0,
                discrete=True,
                features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
            )
            y_out, sw, off = model_build_design_matrix(model, df, y, np.ones(n), None)

            res = optimize_scop_efs_reml(
                dm=model._dm,
                distribution=model._distribution,
                link=model._link,
                groups=model._groups,
                y=y_out,
                sample_weight=np.array(sw),
                offset_arr=np.array(off) if off is not None else np.zeros(n),
                lambdas={"x": 1.0},
                estimated_names={"x"},
                max_reml_iter=20,
                reml_tol=1e-6,
            )
            results[noise_label] = res

        lam_lo = results["low"].lambdas["x"]
        lam_hi = results["high"].lambdas["x"]
        assert lam_hi > lam_lo, (
            f"Expected lambda_high > lambda_low, got {lam_hi:.4g} vs {lam_lo:.4g}"
        )

    @pytest.mark.slow
    def test_predictions_are_monotone(self, scop_reml_model):
        """After EFS convergence, predictions should be monotonically increasing."""
        model, y, sample_weight, offset, df = scop_reml_model
        lambdas = {"x": 1.0}

        result = optimize_scop_efs_reml(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            lambdas=lambdas,
            estimated_names={"x"},
            max_reml_iter=20,
            reml_tol=1e-6,
        )

        # Compute fitted values using final coefficients
        beta = result.pirls_result.beta
        intercept = result.pirls_result.intercept
        eta = model._dm.matvec(beta) + intercept
        if offset is not None:
            eta = eta + offset

        mu = model._link.inverse(eta)

        # x is sorted, so fitted values should be monotone increasing
        x = df["x"].values
        sort_idx = np.argsort(x)
        mu_sorted = mu[sort_idx]
        diffs = np.diff(mu_sorted)
        assert np.all(diffs >= -1e-6), f"Predictions not monotone: min diff = {diffs.min():.2e}"

    @pytest.mark.slow
    def test_returns_reml_result_with_history(self, scop_reml_model):
        """Result should have lambda_history with multiple entries and correct keys."""
        model, y, sample_weight, offset, _ = scop_reml_model
        lambdas = {"x": 1.0}

        result = optimize_scop_efs_reml(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            lambdas=lambdas,
            estimated_names={"x"},
            max_reml_iter=20,
            reml_tol=1e-6,
        )

        assert isinstance(result, REMLResult)
        assert len(result.lambda_history) > 1, (
            f"Expected multiple history entries, got {len(result.lambda_history)}"
        )
        assert isinstance(result.lambdas, dict)
        assert "x" in result.lambdas
        assert result.lambdas["x"] > 0

        # Each history entry should be a dict with "x" key
        for entry in result.lambda_history:
            assert isinstance(entry, dict)
            assert "x" in entry


# ── fit_reml integration tests ──────────────────────────────────────────────────

from superglm.features.spline import BSplineSmooth  # noqa: E402
from superglm.types import LambdaPolicy  # noqa: E402


class TestSCOPFitRemlIntegration:
    """Integration tests: fit_reml routes to SCOP EFS for auto-lambda monotone."""

    @pytest.mark.slow
    def test_fit_reml_scop_auto_lambda(self):
        """fit_reml with SCOP monotone PSpline, no lambda_policy, discrete=True.

        Should converge, estimate lambda, and produce monotone predictions.
        """
        rng = np.random.default_rng(42)
        n = 400
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None
        assert any(v > 0 for v in model._reml_lambdas.values())

        # Predictions should be monotone increasing
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-8), f"Predictions not monotone: min diff = {diffs.min():.2e}"

    @pytest.mark.slow
    def test_fit_reml_mixed_scop_and_ssp(self):
        """Mixed: SCOP monotone x1 + unconstrained PSpline x2, discrete=True.

        Both terms should get lambdas, and x1 predictions should be monotone.
        """
        rng = np.random.default_rng(42)
        n = 400
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None

        # Both terms should have lambdas estimated
        assert len(model._reml_lambdas) >= 2

        # x1 partial effect should be monotone: hold x2 at median
        x1_grid = np.linspace(0, 1, 200)
        pred_df = pd.DataFrame({"x1": x1_grid, "x2": np.median(x2)})
        pred = model.predict(pred_df)
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-6), (
            f"x1 partial effect not monotone: min diff = {diffs.min():.2e}"
        )

    @pytest.mark.slow
    def test_fixed_lambda_policy_still_works(self):
        """Phase 4 path: SCOP with fixed lambda_policy still uses single-fit path."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        # Lambda should be exactly 1.0 (fixed)
        assert model._reml_lambdas is not None
        for v in model._reml_lambdas.values():
            assert v == 1.0

    @pytest.mark.slow
    def test_mixed_fixed_and_estimated_lambda(self):
        """Mixed model: fixed-lambda SSP + auto-lambda SCOP through EFS path."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        fixed_val = 5.0
        model = SuperGLM(
            family=Gaussian(),
            discrete=True,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(
                    n_knots=8,
                    lambda_policy=LambdaPolicy(mode="fixed", value=fixed_val),
                ),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        # x2 lambda must stay exactly at fixed value (SSP uses "x2:wiggle" key)
        x2_key = next(k for k in model._reml_lambdas if k.startswith("x2"))
        assert model._reml_lambdas[x2_key] == pytest.approx(fixed_val)
        # x1 lambda was estimated
        assert "x1" in model._reml_lambdas
        assert model._reml_lambdas["x1"] > 0

    def test_qp_monotone_still_raises(self):
        """BSplineSmooth with monotone still raises NotImplementedError for auto lambda."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        with pytest.raises(NotImplementedError, match="QP monotone"):
            model.fit_reml(df[["x"]], df["y"])


class TestSCOPEFSRegression:
    """Regression and edge-case tests for SCOP EFS auto-lambda.

    Ensures Phase 5a changes do not break unconstrained REML, fixed-lambda SCOP,
    EFS-only models, and that SCOP auto-lambda works across families, directions,
    and summary output.
    """

    @pytest.mark.slow
    def test_unconstrained_reml_unchanged(self):
        """fit_reml with no monotone terms works identically to pre-Phase-5a."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(family=Gaussian(), features={"x": PSpline(n_knots=10)})
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        # Unconstrained REML should produce a valid lambda
        assert model._reml_lambdas is not None
        assert all(v > 0 for v in model._reml_lambdas.values())

    @pytest.mark.slow
    def test_fixed_scop_lambda_unchanged(self):
        """Phase 4 fixed-lambda path still works exactly after Phase 5a changes."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None
        for v in model._reml_lambdas.values():
            assert v == pytest.approx(1.0)

    @pytest.mark.slow
    def test_efs_only_model_unchanged(self):
        """EFS-only model (selection_penalty > 0, no monotone) unaffected by Phase 5a."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0.01,
            features={"x1": PSpline(n_knots=8), "x2": PSpline(n_knots=8)},
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_discrete_scop_auto_lambda(self):
        """discrete=True + SCOP + auto lambda works and produces monotone predictions."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None

        # Check monotone predictions
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-6), f"Predictions not monotone: min diff = {diffs.min():.2e}"

    @pytest.mark.slow
    def test_poisson_scop_auto_lambda(self):
        """Poisson family (known scale) with SCOP auto lambda converges."""
        from superglm.families import Poisson

        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 5, n))
        log_mu = 0.3 * x - 0.5
        y = rng.poisson(np.exp(log_mu))
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Poisson(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result is not None
        assert model._reml_lambdas is not None
        assert all(v > 0 for v in model._reml_lambdas.values())

    @pytest.mark.slow
    def test_summary_after_scop_auto_lambda(self):
        """summary() works after SCOP auto-lambda fit_reml."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        summary = model.summary()
        text = str(summary)
        assert "x" in text

    @pytest.mark.slow
    def test_decreasing_scop_auto_lambda(self):
        """Decreasing monotone also works with auto lambda."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        # Decreasing relationship: y = -2x + noise
        y = -2 * x + 3 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None

        # Check decreasing predictions
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        diffs = np.diff(pred)
        assert np.all(diffs <= 1e-6), f"Predictions not decreasing: max diff = {diffs.max():.2e}"

    @pytest.mark.slow
    def test_reml_penalties_stored_with_scop_components(self):
        """model._reml_penalties includes SCOP PenaltyComponents after auto-lambda fit."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        # model._reml_penalties must include the SCOP penalty component
        assert model._reml_penalties is not None
        assert len(model._reml_penalties) > 0
        scop_pc_names = [pc.name for pc in model._reml_penalties]
        assert "x" in scop_pc_names, f"SCOP component 'x' not in stored penalties: {scop_pc_names}"

    @pytest.mark.slow
    def test_stored_state_reproduces_objective(self):
        """Stored model state reproduces the SCOP-aware REML objective without rerunning solver."""
        from superglm.reml.objective import reml_laml_objective

        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        # Verify scop_states is persisted on the REMLResult
        assert model._reml_result.scop_states is not None
        assert len(model._reml_result.scop_states) > 0

        # Reconstruct XtWX from stored model state (no rerunning solver)
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.group_matrix import _block_xtwx
        from superglm.links import stabilize_eta

        result = model._result
        eta = model._dm.matvec(result.beta) + result.intercept
        eta = stabilize_eta(eta + np.zeros(n), model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
        V = model._distribution.variance(mu)
        dmu = model._link.deriv_inverse(eta)
        W = np.ones(n) * dmu**2 / np.maximum(V, _VARIANCE_FLOOR)

        XtWX = _block_xtwx(
            model._dm.group_matrices,
            model._groups,
            W,
            tabmat_split=model._dm.tabmat_split,
        )

        # Recompute objective from stored state only — no fit_irls_direct call
        obj_recomputed = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            result,
            model._reml_lambdas,
            np.ones(n),
            np.zeros(n),
            XtWX=XtWX,
            reml_penalties=model._reml_penalties,
            scop_states=model._reml_result.scop_states,
        )

        # Must match the objective stored during optimization
        obj_stored = model._reml_result.objective
        assert np.isfinite(obj_recomputed)
        assert np.isfinite(obj_stored)
        assert obj_recomputed == pytest.approx(obj_stored, rel=1e-8), (
            f"Recomputed {obj_recomputed:.6f} != stored {obj_stored:.6f}"
        )

    @pytest.mark.slow
    def test_model_wrapper_objective_matches_stored(self):
        """model._reml_laml_objective wrapper reproduces stored objective for SCOP fits."""
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.group_matrix import _block_xtwx
        from superglm.links import stabilize_eta

        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        # Reconstruct XtWX to call the wrapper
        result = model._result
        sw = np.ones(n)
        offset_arr = np.zeros(n)
        eta = model._dm.matvec(result.beta) + result.intercept + offset_arr
        eta = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
        V = model._distribution.variance(mu)
        dmu = model._link.deriv_inverse(eta)
        W = sw * dmu**2 / np.maximum(V, _VARIANCE_FLOOR)
        XtWX = _block_xtwx(
            model._dm.group_matrices,
            model._groups,
            W,
            tabmat_split=model._dm.tabmat_split,
        )

        # Call through the model wrapper (the path that was broken)
        obj_wrapper = model._reml_laml_objective(
            y,
            result,
            model._reml_lambdas,
            sw,
            offset_arr,
            XtWX=XtWX,
        )

        obj_stored = model._reml_result.objective
        assert np.isfinite(obj_wrapper)
        assert obj_wrapper == pytest.approx(obj_stored, rel=1e-8), (
            f"Wrapper {obj_wrapper:.6f} != stored {obj_stored:.6f}"
        )


class TestSCOPNewtonLineSearchSafety:
    """Newton step-halving rejects non-finite trial states cleanly."""

    def _make_scop_inputs(self, q_eff=7, n=100, seed=42):
        """Build synthetic SCOP Newton inputs."""
        from superglm.solvers.scop import build_scop_solver_reparam

        rng = np.random.default_rng(seed)
        reparam = build_scop_solver_reparam(q_eff + 1, direction="increasing")
        B_scop = rng.standard_normal((n, q_eff))
        W = np.abs(rng.standard_normal(n)) + 0.1
        beta_scop = rng.standard_normal(q_eff) * 0.3
        gamma = reparam.forward(beta_scop)
        z = B_scop @ gamma + rng.standard_normal(n) * 0.1
        S_scop = reparam.penalty_matrix()
        return B_scop, W, z, beta_scop, reparam, S_scop

    def test_overflow_starting_point_noop_no_warning(self):
        """Starting from huge beta_eff (overflow in exp) → no-op, no warning."""
        import warnings

        from superglm.solvers.scop_newton import scop_newton_step

        B_scop, W, z, beta_scop, reparam, S_scop = self._make_scop_inputs()
        beta_huge = np.full_like(beta_scop, 600.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = scop_newton_step(
                B_scop,
                W,
                z,
                beta_huge,
                reparam,
                S_scop,
                lambda2=1.0,
                max_halving=10,
            )

        # Step rejected entirely — beta unchanged
        np.testing.assert_array_equal(result.beta_new, beta_huge)
        assert result.step_norm == 0.0
        assert result.objective_after == result.objective_before

    def test_overflow_trial_halved_to_safety(self):
        """Moderate beta_eff where full step overflows but halving recovers."""
        import warnings

        from superglm.solvers.scop_newton import scop_newton_step

        B_scop, W, z, beta_scop, reparam, S_scop = self._make_scop_inputs()
        # Moderate starting point — finite obj_before, but Newton step may overshoot
        beta_mod = np.full_like(beta_scop, 3.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = scop_newton_step(
                B_scop,
                W,
                z,
                beta_mod,
                reparam,
                S_scop,
                lambda2=1.0,
                max_halving=20,
            )

        assert np.isfinite(result.objective_after)
        assert result.objective_after <= result.objective_before + 1e-14

    def test_exhausted_halvings_rejects_step(self):
        """When all halvings fail, step is rejected: beta unchanged, step_norm=0."""
        from superglm.solvers.scop_newton import scop_newton_step

        B_scop, W, z, beta_scop, reparam, S_scop = self._make_scop_inputs()
        # Very large beta_eff + tiny max_halving → all trials overflow
        beta_huge = np.full_like(beta_scop, 700.0)

        result = scop_newton_step(
            B_scop,
            W,
            z,
            beta_huge,
            reparam,
            S_scop,
            lambda2=1.0,
            max_halving=2,
        )

        np.testing.assert_array_equal(result.beta_new, beta_huge)
        assert result.step_norm == 0.0
        assert result.objective_after == result.objective_before

    @pytest.mark.slow
    def test_mixed_model_no_overflow_warning(self):
        """Mixed SCOP + unconstrained model produces no RuntimeWarning."""
        import warnings

        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            discrete=True,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8),
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.fit_reml(df[["x1", "x2"]], y)


# ---------------------------------------------------------------------------
# Part 10: Multi-SCOP integration tests
# ---------------------------------------------------------------------------


class TestMultiSCOPIntegration:
    """Integration tests for models with multiple SCOP monotone terms.

    Multi-SCOP models need generous max_iter because the EFS outer loop calls
    multiple PIRLS fits and the SCOP Newton reparameterization slows
    inner-loop convergence compared to ordinary splines.
    """

    @pytest.mark.slow
    def test_two_scop_terms_auto_lambda(self):
        """Two SCOP terms (x1 increasing, x2 decreasing), discrete=True, auto lambda.

        Both lambdas should be estimated; predictions should respect monotonicity.
        """
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=200,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None
        assert len(model._reml_lambdas) >= 2

        # x1 partial effect: hold x2 at median, predictions should be increasing
        x1_grid = np.linspace(0, 1, 200)
        pred_df = pd.DataFrame({"x1": x1_grid, "x2": np.median(x2)})
        pred = model.predict(pred_df)
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-6), (
            f"x1 predictions not increasing: min diff = {diffs.min():.2e}"
        )

        # x2 partial effect: hold x1 at median, predictions should be decreasing
        x2_grid = np.linspace(0, 1, 200)
        pred_df = pd.DataFrame({"x1": np.median(x1), "x2": x2_grid})
        pred = model.predict(pred_df)
        diffs = np.diff(pred)
        assert np.all(diffs <= 1e-6), f"x2 predictions not decreasing: max diff = {diffs.max():.2e}"

    @pytest.mark.slow
    def test_three_scop_terms(self):
        """Three SCOP terms, all increasing, discrete=True, auto lambda."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        x3 = np.sort(rng.uniform(0, 1, n))
        y = x1 + 0.5 * x2 + 0.3 * x3 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=500,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x3": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x1", "x2", "x3"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None
        assert len(model._reml_lambdas) >= 3

    @pytest.mark.slow
    def test_mixed_scop_and_ordinary_ssp(self):
        """Two SCOP monotone + one ordinary PSpline, discrete=True.

        All terms should get lambdas estimated.
        """
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        x3 = rng.uniform(0, 1, n)
        y = 2 * x1 - 1.5 * x2 + 0.5 * x3 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=500,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
                "x3": PSpline(n_knots=8),
            },
        )
        model.fit_reml(df[["x1", "x2", "x3"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None
        # All three terms must have lambdas
        assert len(model._reml_lambdas) >= 3

    @pytest.mark.slow
    def test_mixed_fixed_and_estimated_multi_scop(self):
        """One SCOP estimated, one SCOP fixed at 5.0.

        Fixed lambda must stay exactly 5.0.
        """
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        fixed_val = 5.0
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=200,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(
                    n_knots=8,
                    monotone="decreasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=fixed_val),
                ),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        # x2 lambda must stay exactly at fixed value
        x2_key = next(k for k in model._reml_lambdas if k.startswith("x2"))
        assert model._reml_lambdas[x2_key] == pytest.approx(fixed_val)
        # x1 lambda was estimated
        x1_key = next(k for k in model._reml_lambdas if k.startswith("x1"))
        assert model._reml_lambdas[x1_key] > 0

    @pytest.mark.slow
    def test_discrete_two_scop(self):
        """discrete=True with 2 SCOP terms. Assert model fitted."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=200,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_stored_objective_reproduction_multi_scop(self):
        """Reconstruct REML objective from stored model state (no solver rerun).

        Must match model._reml_result.objective to rel=1e-8.
        """
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.group_matrix import _block_xtwx
        from superglm.links import stabilize_eta
        from superglm.reml.objective import reml_laml_objective

        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=200,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        result = model._result
        sw = np.ones(n)
        offset_arr = np.zeros(n)
        eta = model._dm.matvec(result.beta) + result.intercept + offset_arr
        eta = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
        V = model._distribution.variance(mu)
        dmu = model._link.deriv_inverse(eta)
        W = sw * dmu**2 / np.maximum(V, _VARIANCE_FLOOR)
        XtWX = _block_xtwx(
            model._dm.group_matrices,
            model._groups,
            W,
            tabmat_split=model._dm.tabmat_split,
        )

        obj_recomputed = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            result,
            model._reml_lambdas,
            sw,
            offset_arr,
            XtWX=XtWX,
            reml_penalties=model._reml_penalties,
            scop_states=model._reml_result.scop_states,
        )
        assert obj_recomputed == pytest.approx(model._reml_result.objective, rel=1e-8)

    @pytest.mark.slow
    def test_lambda_responds_to_noise_multi_scop(self):
        """Two SCOP terms: low noise (sigma=0.1) vs high noise (sigma=1.0).

        Higher noise should produce larger lambdas for both terms.
        """
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))

        lambdas_by_noise = {}
        for sigma in [0.1, 1.0]:
            y = 2 * x1 - 1.5 * x2 + rng.normal(0, sigma, n)
            df = pd.DataFrame({"x1": x1, "x2": x2})

            model = SuperGLM(
                family=Gaussian(),
                selection_penalty=0,
                discrete=True,
                max_iter=200,
                features={
                    "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                    "x2": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
                },
            )
            model.fit_reml(df[["x1", "x2"]], y)
            lambdas_by_noise[sigma] = model._reml_lambdas.copy()

        lam_lo = lambdas_by_noise[0.1]
        lam_hi = lambdas_by_noise[1.0]

        for key in lam_lo:
            assert lam_hi[key] > lam_lo[key], (
                f"Lambda for {key} did not increase with noise: "
                f"lo={lam_lo[key]:.4f}, hi={lam_hi[key]:.4f}"
            )

    @pytest.mark.slow
    def test_plain_fit_with_two_scop(self):
        """fit() (not fit_reml) with 2 SCOP terms, discrete=True, fixed lambda.

        Uses a loose tolerance (1e-3) because the SCOP Newton reparameterization
        causes limit-cycle oscillations in the deviance convergence criterion
        at ~2e-4 relative change. The solution quality is fine — deviance is
        stable to 4 significant figures.
        """
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=1.0,
            discrete=True,
            max_iter=200,
            tol=1e-3,
            features={
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit"),
            },
        )
        model.fit(df[["x1", "x2"]], y)

        assert model._result.converged

        # x1 predictions should be increasing
        x1_grid = np.linspace(0, 1, 200)
        pred_df = pd.DataFrame({"x1": x1_grid, "x2": np.median(x2)})
        pred = model.predict(pred_df)
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-6), (
            f"x1 predictions not increasing: min diff = {diffs.min():.2e}"
        )

        # x2 predictions should be decreasing
        x2_grid = np.linspace(0, 1, 200)
        pred_df = pd.DataFrame({"x1": np.median(x1), "x2": x2_grid})
        pred = model.predict(pred_df)
        diffs = np.diff(pred)
        assert np.all(diffs <= 1e-6), f"x2 predictions not decreasing: max diff = {diffs.max():.2e}"

    @pytest.mark.slow
    def test_single_scop_still_works(self):
        """Single SCOP term regression — no breakage from multi-SCOP changes."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged
        assert model._reml_lambdas is not None

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-6)

    @pytest.mark.slow
    def test_no_scop_model_unchanged(self):
        """No SCOP terms — completely unaffected."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            features={"x": PSpline(n_knots=10)},
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged

    @pytest.mark.slow
    def test_qp_monotone_still_raises(self):
        """QP monotone auto-lambda still raises NotImplementedError."""
        from superglm.features.spline import BSplineSmooth

        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        with pytest.raises(NotImplementedError, match="QP monotone"):
            model.fit_reml(df[["x"]], y)

    @pytest.mark.slow
    def test_diagnostics_populated(self):
        """Convergence diagnostics are populated after fit_reml."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)

        reml_result = model._reml_result
        assert reml_result.inner_iter_history is not None
        assert len(reml_result.inner_iter_history) > 0
        assert all(isinstance(v, int) for v in reml_result.inner_iter_history)

        assert reml_result.objective_history is not None
        assert len(reml_result.objective_history) > 0
        assert all(np.isfinite(v) for v in reml_result.objective_history)


# ---------------------------------------------------------------------------
# Joint SCOP Newton step tests
# ---------------------------------------------------------------------------


class TestJointSCOPNewton:
    """Tests for scop_joint_newton_step."""

    def _build_single_group_inputs(self, rng=None, q_eff=7, n=100, lam=1.0):
        """Build single-group SCOP problem inputs for testing."""
        from superglm.solvers.scop import build_scop_solver_reparam

        if rng is None:
            rng = np.random.default_rng(42)

        reparam = build_scop_solver_reparam(q_eff + 1, direction="increasing")
        B_scop = rng.standard_normal((n, q_eff))
        W = np.abs(rng.standard_normal(n)) + 0.1
        beta_scop = rng.standard_normal(q_eff) * 0.3
        gamma = reparam.forward(beta_scop)
        z = B_scop @ gamma + rng.standard_normal(n) * 0.1
        S_scop = reparam.penalty_matrix()

        return B_scop, W, z, beta_scop, reparam, S_scop

    def _build_two_group_inputs(self, rng=None, q1=7, q2=5, n=200, discretized=False):
        """Build two-group SCOP problem inputs for testing."""
        from superglm.solvers.scop import build_scop_solver_reparam

        if rng is None:
            rng = np.random.default_rng(99)

        reparam1 = build_scop_solver_reparam(q1 + 1, direction="increasing")
        reparam2 = build_scop_solver_reparam(q2 + 1, direction="increasing")

        if discretized:
            n_bins1 = 50
            n_bins2 = 40
            B1 = rng.standard_normal((n_bins1, q1))
            B2 = rng.standard_normal((n_bins2, q2))
            bi1 = rng.integers(0, n_bins1, size=n)
            bi2 = rng.integers(0, n_bins2, size=n)
        else:
            B1 = rng.standard_normal((n, q1))
            B2 = rng.standard_normal((n, q2))
            bi1 = None
            bi2 = None

        W = np.abs(rng.standard_normal(n)) + 0.1
        beta1 = rng.standard_normal(q1) * 0.3
        beta2 = rng.standard_normal(q2) * 0.3

        gamma1 = reparam1.forward(beta1)
        gamma2 = reparam2.forward(beta2)

        eta1 = B1 @ gamma1
        eta2 = B2 @ gamma2
        if bi1 is not None:
            eta1 = eta1[bi1]
        if bi2 is not None:
            eta2 = eta2[bi2]

        z = eta1 + eta2 + rng.standard_normal(n) * 0.1

        S1 = reparam1.penalty_matrix()
        S2 = reparam2.penalty_matrix()

        scop_states = {
            0: {
                "B_scop": B1,
                "S_scop": S1,
                "beta_scop": beta1,
                "reparam": reparam1,
                "bin_idx": bi1,
                "group_sl": slice(0, q1),
                "group_name": "x1",
            },
            1: {
                "B_scop": B2,
                "S_scop": S2,
                "beta_scop": beta2,
                "reparam": reparam2,
                "bin_idx": bi2,
                "group_sl": slice(q1, q1 + q2),
                "group_name": "x2",
            },
        }

        return scop_states, W, z

    def _make_mock_groups(self, scop_states):
        """Create minimal mock GroupSlice objects for testing."""
        from dataclasses import dataclass

        @dataclass
        class MockGroup:
            name: str
            sl: slice

        groups = []
        for gi in sorted(scop_states.keys()):
            st = scop_states[gi]
            groups.append(MockGroup(name=st["group_name"], sl=st["group_sl"]))
        return groups

    def test_single_group_matches_existing(self):
        """Joint step with one group should match sequential scop_newton_step."""
        from superglm.solvers.scop_newton import scop_joint_newton_step, scop_newton_step

        B_scop, W, z, beta_scop, reparam, S_scop = self._build_single_group_inputs()
        q_eff = len(beta_scop)

        # Single-group result via existing sequential step
        result_single = scop_newton_step(B_scop, W, z, beta_scop, reparam, S_scop, lambda2=1.0)

        # Joint result (one group)
        scop_states = {
            0: {
                "B_scop": B_scop,
                "S_scop": S_scop,
                "beta_scop": beta_scop.copy(),
                "reparam": reparam,
                "bin_idx": None,
                "group_sl": slice(0, q_eff),
                "group_name": "x",
            }
        }
        groups = self._make_mock_groups(scop_states)
        joint_results = scop_joint_newton_step(scop_states, W, z, {"x": 1.0}, groups)

        np.testing.assert_allclose(joint_results[0].beta_new, result_single.beta_new, rtol=1e-8)
        np.testing.assert_allclose(
            joint_results[0].objective_after, result_single.objective_after, rtol=1e-8
        )

    def test_single_group_discretized_matches(self):
        """Joint step with one discretized group matches sequential."""
        from superglm.solvers.scop_newton import scop_joint_newton_step, scop_newton_step

        rng = np.random.default_rng(77)
        q_eff = 6
        n = 200
        n_bins = 40

        from superglm.solvers.scop import build_scop_solver_reparam

        reparam = build_scop_solver_reparam(q_eff + 1, direction="increasing")
        B_scop = rng.standard_normal((n_bins, q_eff))
        W = np.abs(rng.standard_normal(n)) + 0.1
        beta_scop = rng.standard_normal(q_eff) * 0.3
        bin_idx = rng.integers(0, n_bins, size=n)
        gamma = reparam.forward(beta_scop)
        z = (B_scop @ gamma)[bin_idx] + rng.standard_normal(n) * 0.1
        S_scop = reparam.penalty_matrix()

        result_single = scop_newton_step(
            B_scop, W, z, beta_scop, reparam, S_scop, lambda2=1.0, bin_idx=bin_idx
        )

        scop_states = {
            0: {
                "B_scop": B_scop,
                "S_scop": S_scop,
                "beta_scop": beta_scop.copy(),
                "reparam": reparam,
                "bin_idx": bin_idx,
                "group_sl": slice(0, q_eff),
                "group_name": "x",
            }
        }
        groups = self._make_mock_groups(scop_states)
        joint_results = scop_joint_newton_step(scop_states, W, z, {"x": 1.0}, groups)

        np.testing.assert_allclose(joint_results[0].beta_new, result_single.beta_new, rtol=1e-8)
        np.testing.assert_allclose(
            joint_results[0].objective_after, result_single.objective_after, rtol=1e-8
        )

    def test_joint_gradient_finite_differences(self):
        """Joint gradient should match centered finite differences."""
        from superglm.solvers.scop_newton import _safe_joint_objective

        scop_states, W, z = self._build_two_group_inputs()
        scop_items = sorted(scop_states.items())

        # Build slices and lambdas
        lambdas_list = [1.0, 0.5]
        q_effs = [len(st["beta_scop"]) for _, st in scop_items]
        slices = []
        off = 0
        for q in q_effs:
            slices.append(slice(off, off + q))
            off += q
        q_total = off

        beta_joint = np.concatenate([st["beta_scop"] for _, st in scop_items])

        # Compute gradient analytically (same as in scop_joint_newton_step)
        # Re-derive: forward map, shared residual, per-group grad
        j_diags = []
        etas = []
        for gi, st in scop_items:
            gamma_i = st["reparam"].forward(st["beta_scop"])
            j_diags.append(gamma_i)
            eta_i = st["B_scop"] @ gamma_i
            if st["bin_idx"] is not None:
                eta_i = eta_i[st["bin_idx"]]
            etas.append(eta_i)

        total_eta = sum(etas)
        residual = z - total_eta

        grad = np.zeros(q_total)
        for idx, (gi, st) in enumerate(scop_items):
            sl_i = slices[idx]
            B_i = st["B_scop"]
            bi_i = st["bin_idx"]
            j_i = j_diags[idx]
            lam_i = lambdas_list[idx]
            beta_i = beta_joint[sl_i]

            if bi_i is not None:
                n_bins = B_i.shape[0]
                Wr_agg = np.bincount(bi_i, weights=W * residual, minlength=n_bins)
                r_eff_i = B_i.T @ Wr_agg
            else:
                r_eff_i = B_i.T @ (W * residual)

            grad_data_i = -(j_i * r_eff_i)
            grad[sl_i] = grad_data_i + lam_i * (st["S_scop"] @ beta_i)

        # Finite difference gradient
        eps = 1e-5
        grad_fd = np.zeros(q_total)
        for k in range(q_total):
            bp = beta_joint.copy()
            bm = beta_joint.copy()
            bp[k] += eps
            bm[k] -= eps
            fp = _safe_joint_objective(scop_items, W, z, bp, slices, lambdas_list)
            fm = _safe_joint_objective(scop_items, W, z, bm, slices, lambdas_list)
            grad_fd[k] = (fp - fm) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, atol=1e-4)

    def test_cross_gram_disc_disc(self):
        """Cross-gram for two discretized groups matches naive matmul."""
        from superglm.solvers.scop_newton import _compute_cross_gram

        rng = np.random.default_rng(10)
        n = 300
        nb1, nb2 = 50, 40
        q1, q2 = 7, 5

        B1 = rng.standard_normal((nb1, q1))
        B2 = rng.standard_normal((nb2, q2))
        bi1 = rng.integers(0, nb1, size=n)
        bi2 = rng.integers(0, nb2, size=n)
        W = np.abs(rng.standard_normal(n)) + 0.1

        st_i = {"B_scop": B1, "bin_idx": bi1}
        st_j = {"B_scop": B2, "bin_idx": bi2}

        result = _compute_cross_gram(st_i, st_j, W)

        # Naive: scatter to observation level
        B1_full = B1[bi1]
        B2_full = B2[bi2]
        expected = B1_full.T @ (B2_full * W[:, None])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cross_gram_dense_dense(self):
        """Cross-gram for two dense groups matches naive matmul."""
        from superglm.solvers.scop_newton import _compute_cross_gram

        rng = np.random.default_rng(11)
        n = 200
        q1, q2 = 7, 5

        B1 = rng.standard_normal((n, q1))
        B2 = rng.standard_normal((n, q2))
        W = np.abs(rng.standard_normal(n)) + 0.1

        st_i = {"B_scop": B1, "bin_idx": None}
        st_j = {"B_scop": B2, "bin_idx": None}

        result = _compute_cross_gram(st_i, st_j, W)
        expected = B1.T @ (B2 * W[:, None])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cross_gram_disc_dense(self):
        """Cross-gram for one disc + one dense group matches naive matmul."""
        from superglm.solvers.scop_newton import _compute_cross_gram

        rng = np.random.default_rng(12)
        n = 200
        nb1 = 50
        q1, q2 = 7, 5

        B1 = rng.standard_normal((nb1, q1))
        B2 = rng.standard_normal((n, q2))
        bi1 = rng.integers(0, nb1, size=n)
        W = np.abs(rng.standard_normal(n)) + 0.1

        st_i = {"B_scop": B1, "bin_idx": bi1}
        st_j = {"B_scop": B2, "bin_idx": None}

        result = _compute_cross_gram(st_i, st_j, W)

        # Naive
        B1_full = B1[bi1]
        expected = B1_full.T @ (B2 * W[:, None])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

        # Also test the reverse (dense, disc)
        st_i2 = {"B_scop": B2, "bin_idx": None}
        st_j2 = {"B_scop": B1, "bin_idx": bi1}
        result2 = _compute_cross_gram(st_i2, st_j2, W)
        expected2 = B2.T @ (B1_full * W[:, None])

        np.testing.assert_allclose(result2, expected2, rtol=1e-10)

    def test_joint_step_reduces_objective(self):
        """Joint Newton step should reduce objective for all groups."""
        from superglm.solvers.scop_newton import scop_joint_newton_step

        scop_states, W, z = self._build_two_group_inputs()
        groups = self._make_mock_groups(scop_states)

        joint_results = scop_joint_newton_step(scop_states, W, z, {"x1": 1.0, "x2": 0.5}, groups)

        # Check that at least one group has obj_after <= obj_before
        # (joint step shares the objective, so all should agree)
        for gi, result in joint_results.items():
            assert result.objective_after <= result.objective_before + 1e-14
            assert np.all(np.isfinite(result.beta_new))

    def test_joint_step_reduces_objective_discretized(self):
        """Joint step reduces objective for discretized two-group problem."""
        from superglm.solvers.scop_newton import scop_joint_newton_step

        scop_states, W, z = self._build_two_group_inputs(discretized=True)
        groups = self._make_mock_groups(scop_states)

        joint_results = scop_joint_newton_step(scop_states, W, z, {"x1": 1.0, "x2": 0.5}, groups)

        for gi, result in joint_results.items():
            assert result.objective_after <= result.objective_before + 1e-14
            assert np.all(np.isfinite(result.beta_new))

    def test_h_penalized_is_diagonal_block(self):
        """H_penalized for each group is the diagonal block of the joint H."""
        from superglm.solvers.scop_newton import scop_joint_newton_step

        scop_states, W, z = self._build_two_group_inputs()
        groups = self._make_mock_groups(scop_states)

        joint_results = scop_joint_newton_step(scop_states, W, z, {"x1": 1.0, "x2": 0.5}, groups)

        for gi, result in joint_results.items():
            q_i = len(scop_states[gi]["beta_scop"])
            assert result.H_penalized.shape == (q_i, q_i)
            # H_penalized should be finite
            assert np.all(np.isfinite(result.H_penalized))

    def test_scalar_lambda(self):
        """Joint step works with scalar lambda (not dict)."""
        from superglm.solvers.scop_newton import scop_joint_newton_step

        scop_states, W, z = self._build_two_group_inputs()
        groups = self._make_mock_groups(scop_states)

        # Scalar lambda
        joint_results = scop_joint_newton_step(scop_states, W, z, 1.0, groups)

        for gi, result in joint_results.items():
            assert result.objective_after <= result.objective_before + 1e-14
            assert np.all(np.isfinite(result.beta_new))
