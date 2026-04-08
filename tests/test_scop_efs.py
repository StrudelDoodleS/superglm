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
