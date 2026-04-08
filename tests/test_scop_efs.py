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
from superglm.reml.scop_efs import assemble_joint_hessian, build_scop_penalty_components
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
            }
        }

        H_joint, mapping = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # SCOP block should be H_scop, not the original
        np.testing.assert_array_equal(H_joint[scop_sl, scop_sl], H_scop)
        assert not np.allclose(H_joint[scop_sl, scop_sl], original_scop_block)

    def test_linear_block_unchanged(self):
        """Non-SCOP (linear) block must be unchanged after assembly."""
        p = 12
        q_scop = 5
        scop_sl = slice(7, 12)
        linear_sl = slice(0, 7)

        rng = np.random.default_rng(77)
        A = rng.standard_normal((p, p))
        XtWX_plus_S = A.T @ A + np.eye(p)

        B = rng.standard_normal((q_scop, q_scop))
        H_scop = B.T @ B + np.eye(q_scop)

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
            }
        }

        H_joint, _ = assemble_joint_hessian(XtWX_plus_S, scop_states)

        # Linear block unchanged
        np.testing.assert_array_equal(
            H_joint[linear_sl, linear_sl], XtWX_plus_S[linear_sl, linear_sl]
        )
        # Off-diagonal blocks also unchanged
        np.testing.assert_array_equal(H_joint[linear_sl, scop_sl], XtWX_plus_S[linear_sl, scop_sl])
        np.testing.assert_array_equal(H_joint[scop_sl, linear_sl], XtWX_plus_S[scop_sl, linear_sl])

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
            },
            1: {
                "group_sl": sl_b,
                "H_scop_penalized": 3.0 * np.eye(5),
                "group_name": "spline_b",
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

        scop_states = {
            0: {
                "group_sl": scop_sl,
                "H_scop_penalized": H_scop,
                "group_name": "mono_x",
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
            }
        }

        assemble_joint_hessian(XtWX_plus_S, scop_states)
        np.testing.assert_array_equal(XtWX_plus_S, original_copy)
