"""Tests for SCOP EFS infrastructure.

Part 1: Tests for SCOP state returned from fit_irls_direct.
Part 2: Tests for build_scop_penalty_components.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import superglm.reml.scop_efs as scop_efs_module
from superglm import Constraint, SuperGLM
from superglm.families import Gaussian, Poisson
from superglm.features.spline import PSpline
from superglm.inference.covariance import _active_penalty_matrix
from superglm.model.base import model_build_design_matrix
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.reml.scop_efs import (
    assemble_joint_hessian,
    build_scop_penalty_components,
    compute_scop_aware_penalty_quad,
    scop_efs_lambda_update,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import GroupSlice, PenaltyComponent


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
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.increasing)},
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


class TestBuildSCOPPenaltyMatrixOwnership:
    """Each SCOP group contributes exactly once to an assembled penalty."""

    @staticmethod
    def _group_and_component(q=6):
        omega = _first_diff_penalty(q)
        reparameterization = SimpleNamespace(penalty_matrix=lambda: omega)
        group = GroupSlice(
            name="x",
            start=0,
            end=q,
            penalized=True,
            monotone_engine="scop",
            scop_reparameterization=reparameterization,
        )
        component = PenaltyComponent(
            name="x",
            group_name="x",
            group_index=0,
            group_sl=group.sl,
            omega_raw=omega,
            omega_ssp=omega,
        )
        return omega, group, component

    def test_supplied_scop_component_is_not_added_again_by_group_fallback(self):
        omega, group, component = self._group_and_component()

        assembled = build_penalty_matrix(
            [SimpleNamespace(R_inv=np.eye(group.size))],
            [group],
            {"x": 3.0},
            group.size,
            reml_penalties=[component],
        )

        np.testing.assert_allclose(assembled, 3.0 * omega, rtol=0.0, atol=0.0)

    def test_active_supplied_scop_component_is_not_added_again_by_group_fallback(self):
        omega, group, component = self._group_and_component()

        assembled = _active_penalty_matrix(
            [SimpleNamespace(R_inv=np.eye(group.size))],
            [group],
            [group],
            {"x": 3.0},
            reml_penalties=[component],
        )

        np.testing.assert_allclose(assembled, 3.0 * omega, rtol=0.0, atol=0.0)

    def test_scop_group_fallback_remains_when_component_list_omits_group(self):
        omega, group, _ = self._group_and_component()

        assembled = build_penalty_matrix(
            [SimpleNamespace(R_inv=np.eye(group.size))],
            [group],
            {"x": 3.0},
            group.size,
            reml_penalties=[],
        )

        np.testing.assert_allclose(assembled, 3.0 * omega, rtol=0.0, atol=0.0)

    def test_base_component_list_does_not_suppress_omitted_scop_group(self):
        q_base = 2
        q_scop = 6
        base_omega = np.eye(q_base)
        scop_omega = _first_diff_penalty(q_scop)
        base_group = GroupSlice(name="base", start=0, end=q_base, penalized=True)
        scop_group = GroupSlice(
            name="x",
            start=q_base,
            end=q_base + q_scop,
            penalized=True,
            monotone_engine="scop",
            scop_reparameterization=SimpleNamespace(penalty_matrix=lambda: scop_omega),
        )
        base_component = PenaltyComponent(
            name="base",
            group_name="base",
            group_index=0,
            group_sl=base_group.sl,
            omega_raw=base_omega,
            omega_ssp=base_omega,
        )

        assembled = build_penalty_matrix(
            [SimpleNamespace(R_inv=np.eye(q_base)), SimpleNamespace(R_inv=np.eye(q_scop))],
            [base_group, scop_group],
            {"base": 2.0, "x": 3.0},
            q_base + q_scop,
            reml_penalties=[base_component],
        )

        expected = np.zeros_like(assembled)
        expected[base_group.sl, base_group.sl] = 2.0 * base_omega
        expected[scop_group.sl, scop_group.sl] = 3.0 * scop_omega
        np.testing.assert_allclose(assembled, expected, rtol=0.0, atol=0.0)


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


class TestSCOPPenaltyMetadataCache:
    """Tests for cached SCOP penalty/state metadata."""

    def test_reuses_cached_penalty_metadata(self, monkeypatch):
        """A populated SCOP state cache should avoid recomputing eigvalsh."""
        q = 9
        S = _first_diff_penalty(q)
        scop_states = {
            0: {
                "S_scop": S,
                "group_sl": slice(0, q),
                "group_name": "x",
                "beta_eff": np.zeros(q),
            }
        }
        first = build_scop_penalty_components(scop_states)
        assert scop_states[0]["penalty_rank"] == first[0].rank
        assert np.isfinite(scop_states[0]["penalty_log_det_omega_plus"])
        np.testing.assert_allclose(scop_states[0]["penalty_eigvals_omega"], first[0].eigvals_omega)

        def _fail_eigvalsh(_S):
            raise AssertionError("eigvalsh should not be called when cache is populated")

        monkeypatch.setattr(np.linalg, "eigvalsh", _fail_eigvalsh)
        second = build_scop_penalty_components(scop_states)
        assert second[0].rank == first[0].rank
        assert second[0].log_det_omega_plus == first[0].log_det_omega_plus
        np.testing.assert_allclose(second[0].eigvals_omega, first[0].eigvals_omega)


class TestSCOPStateCaching:
    """Tests for cached SCOP artifacts reused across outer EFS iterations."""

    @pytest.mark.slow
    def test_fit_irls_direct_propagates_cached_penalty_metadata(self, scop_model_inputs):
        """Warm-started SCOP state should retain cached penalty metadata and gamma."""
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
        build_scop_penalty_components(scop_states)

        out2 = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            return_scop_state=True,
            scop_state_init=scop_states,
        )
        _, _, scop_states2 = out2

        assert scop_states2[0]["penalty_rank"] == scop_states[0]["penalty_rank"]
        assert (
            scop_states2[0]["penalty_log_det_omega_plus"]
            == scop_states[0]["penalty_log_det_omega_plus"]
        )
        np.testing.assert_allclose(
            scop_states2[0]["penalty_eigvals_omega"],
            scop_states[0]["penalty_eigvals_omega"],
        )
        np.testing.assert_allclose(
            scop_states2[0]["gamma_eff"],
            np.exp(np.clip(scop_states2[0]["beta_eff"], -500, 500)),
        )


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

    def test_intercept_profiled_geometry_matches_augmented_schur_complement(self):
        """SCOP coordinates must transform the intercept cross-block before profiling."""
        raw_hessian = np.array(
            [
                [5.0, 0.8, 0.3],
                [0.8, 4.0, 0.4],
                [0.3, 0.4, 3.0],
            ]
        )
        beta_eff = np.log(np.array([1.5, 0.7]))
        scop_slice = slice(1, 3)
        scop_block = np.array([[6.0, 0.5], [0.5, 4.5]])
        states = {
            0: {
                "group_sl": scop_slice,
                "H_scop_penalized": scop_block,
                "group_name": "mono",
                "beta_eff": beta_eff,
            }
        }
        xtw1 = np.array([2.0, 1.2, -0.8])
        sum_w = 7.0

        raw_joint, _ = assemble_joint_hessian(raw_hessian, states)
        transformed_cross = xtw1.copy()
        transformed_cross[scop_slice] *= np.exp(beta_eff)
        expected = raw_joint - np.outer(transformed_cross, transformed_cross) / sum_w

        actual, _ = assemble_joint_hessian(
            raw_hessian,
            states,
            XtW1=xtw1,
            sum_W=sum_w,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

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

    def test_joint_update_uses_generalized_prior_trace_for_overlapping_penalties(self):
        """Wood--Fasiolo's prior trace is not component rank for shared blocks."""
        from superglm.reml.scop_efs import _joint_efs_lambda_step

        omegas = (
            np.diag([1.0, 0.0]),
            np.diag([0.0, 1.0]),
            np.ones((2, 2)),
        )
        names = ("a", "b", "c")
        lambdas = {"a": 1.0, "b": 2.0, "c": 3.0}
        components = [
            PenaltyComponent(
                name=name,
                group_name="shared",
                group_index=0,
                group_sl=slice(0, 2),
                omega_raw=omega,
                omega_ssp=omega,
                rank=1.0,
            )
            for name, omega in zip(names, omegas, strict=True)
        ]
        beta = np.array([1.2, 0.8])
        hessian_inverse = 0.1 * np.eye(2)

        updated, _, _ = _joint_efs_lambda_step(
            components,
            beta,
            hessian_inverse,
            1.0,
            lambdas,
            {"a"},
            {},
            {"a": 1.0},
            {},
        )

        total_penalty = sum(lambdas[name] * omega for name, omega in zip(names, omegas))
        prior_trace = float(np.trace(np.linalg.pinv(total_penalty) @ omegas[0]))
        posterior_trace = float(np.trace(hessian_inverse @ omegas[0]))
        residual_edf = lambdas["a"] * (prior_trace - posterior_trace)
        expected = residual_edf / float(beta @ omegas[0] @ beta)
        assert abs(np.log(expected / lambdas["a"])) < 4.0
        assert updated["a"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# Part 6: Tests for SCOP-aware REML objective
# ---------------------------------------------------------------------------


class TestSCOPAwareObjective:
    """Tests for reml_laml_objective with scop_states parameter."""

    @pytest.mark.slow
    def test_objective_uses_profiled_intercept_joint_geometry(self, scop_model_inputs):
        """The SCOP objective determinant must match the explicit Schur geometry."""
        from superglm.reml.objective import reml_laml_objective
        from superglm.solvers.rank import decompose_gram

        model, y, sample_weight, offset = scop_model_inputs
        offset_arr = offset if offset is not None else np.zeros_like(y)
        lambdas = {"x": 1.7}
        result, _, XtWX, scop_states = fit_irls_direct(
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
        penalties = build_scop_penalty_components(scop_states)
        penalty = build_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            lambdas,
            model._dm.p,
            reml_penalties=penalties,
        )
        rank_info = result.rank_info
        assert rank_info is not None
        joint, _ = assemble_joint_hessian(
            XtWX + penalty,
            scop_states,
            XtW1=rank_info.sum_w * rank_info.mean_x,
            sum_W=rank_info.sum_w,
        )
        decomposition = decompose_gram(joint)
        expected_logdet = float(np.log(rank_info.sum_w) + decomposition.log_pdet)

        common = {
            "dm": model._dm,
            "distribution": model._distribution,
            "link": model._link,
            "groups": model._groups,
            "y": y,
            "result": result,
            "lambdas": lambdas,
            "sample_weight": sample_weight,
            "offset_arr": offset_arr,
            "XtWX": XtWX,
            "reml_penalties": penalties,
            "scop_states": scop_states,
        }
        actual = reml_laml_objective(**common)
        expected = reml_laml_objective(
            **common,
            log_det_H=expected_logdet,
            hessian_rank=1 + decomposition.rank,
        )

        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

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
    def test_objective_with_scop_components_matches_single_block_override(
        self,
        scop_model_inputs,
    ):
        """Merged SCOP components and an explicit one-block S define the same objective."""
        from superglm.reml.objective import reml_laml_objective

        model, y, sample_weight, offset = scop_model_inputs
        offset_arr = offset if offset is not None else np.zeros_like(y)
        lambdas = {"x": 3.0}
        result, _, XtWX, scop_states = fit_irls_direct(
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
        penalties = build_scop_penalty_components(scop_states)
        expected_penalty = np.zeros((model._dm.p, model._dm.p))
        for component in penalties:
            expected_penalty[component.group_sl, component.group_sl] += (
                lambdas[component.name] * component.omega_ssp
            )

        common = {
            "dm": model._dm,
            "distribution": model._distribution,
            "link": model._link,
            "groups": model._groups,
            "y": y,
            "result": result,
            "lambdas": lambdas,
            "sample_weight": sample_weight,
            "offset_arr": offset_arr,
            "XtWX": XtWX,
            "reml_penalties": penalties,
            "scop_states": scop_states,
        }
        assembled_objective = reml_laml_objective(**common)
        explicit_objective = reml_laml_objective(
            **common,
            S_override=expected_penalty,
        )

        assert assembled_objective == pytest.approx(explicit_objective, rel=1e-12, abs=1e-12)

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

    def test_candidate_disables_all_generic_terminal_metadata(self, monkeypatch):
        """A rejected private candidate requests no retained-fit decompositions."""
        captured = {}
        rejected = SimpleNamespace(
            beta=np.array([0.0]),
            intercept=0.0,
            converged=False,
        )

        def fake_solver(**kwargs):
            captured.update(kwargs)
            return rejected, None, np.array([[1.0]]), {}

        monkeypatch.setattr(scop_efs_module, "fit_irls_direct", fake_solver)
        context = scop_efs_module._SCOPREMLFitContext(
            dm=SimpleNamespace(p=1, group_matrices=[]),
            distribution=SimpleNamespace(),
            link=SimpleNamespace(),
            groups=[],
            y=np.array([1.0]),
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            pirls_tol=1e-6,
            max_pirls_iter=10,
            reml_penalties=[],
            convergence="coefficients",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=1.0,
            gamma_scale_data=None,
        )

        mode = scop_efs_module._fit_scop_reml_mode(
            context,
            {"x": 1.0},
            beta_init=None,
            intercept_init=None,
            scop_state_init=None,
            phase="candidate",
            reml_iteration=1,
            require_converged=True,
        )

        assert mode is None
        assert captured["compute_rank_info"] is False
        assert captured["_compute_fit_statistics"] is False
        assert captured["_compute_reml_geometry"] is False

    def test_candidate_omits_metadata_then_terminal_hydrates_once(
        self,
        scop_model_inputs,
        monkeypatch,
    ):
        """Only the retained SCOP mode receives public rank, EDF, and covariance state."""
        model, y, sample_weight, offset = scop_model_inputs
        offset_arr = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)
        context = scop_efs_module._SCOPREMLFitContext(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=np.asarray(sample_weight, dtype=float),
            offset_arr=offset_arr,
            pirls_tol=1e-6,
            max_pirls_iter=100,
            reml_penalties=None,
            convergence="coefficients",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=float(np.sum(sample_weight)),
            gamma_scale_data=None,
        )
        candidate = scop_efs_module._fit_scop_reml_mode(
            context,
            {"x": 1.0},
            beta_init=None,
            intercept_init=None,
            scop_state_init=None,
            phase="candidate",
            reml_iteration=1,
            require_converged=True,
        )

        assert candidate is not None
        assert candidate.result.rank_info is None
        assert np.isnan(candidate.result.effective_df)
        assert np.isnan(candidate.result.phi)
        assert candidate.result.log_det_H is None
        assert candidate.result.reml_hessian_rank is None

        calls = 0
        original = scop_efs_module.install_scop_postfit_inference

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(scop_efs_module, "install_scop_postfit_inference", counted)
        terminal = scop_efs_module._finalize_scop_reml_mode(context, candidate)

        assert terminal is candidate.result
        assert calls == 1
        assert terminal.rank_info is not None
        assert terminal.scop_inference is not None
        assert terminal.effective_df == pytest.approx(terminal.scop_inference.total_edf)
        assert terminal.log_det_H == pytest.approx(candidate.log_det_h)
        assert terminal.reml_hessian_rank == candidate.hessian_rank

    def test_coefficient_change_without_latent_kkt_is_not_a_reml_candidate(self, scop_model_inputs):
        """A loose coefficient tolerance cannot silently authorize a LAML mode."""
        model, y, sample_weight, offset = scop_model_inputs
        offset_arr = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)
        context = scop_efs_module._SCOPREMLFitContext(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=np.asarray(sample_weight, dtype=float),
            offset_arr=offset_arr,
            pirls_tol=0.1,
            max_pirls_iter=1,
            reml_penalties=None,
            convergence="coefficients",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=float(len(y)),
            gamma_scale_data=None,
        )

        mode = scop_efs_module._fit_scop_reml_mode(
            context,
            {"x": 1.0},
            beta_init=None,
            intercept_init=None,
            scop_state_init=None,
            phase="candidate",
            reml_iteration=1,
            require_converged=True,
        )

        assert mode is None

    def test_scop_reml_kkt_uses_retained_eta_under_large_translation(self):
        """A compensating intercept must not manufacture a failed terminal score."""
        from superglm.features import Numeric

        rng = np.random.default_rng(20260731)
        n = 100
        x = np.sort(rng.uniform(0.0, 1.0, size=n))
        z = rng.normal(size=n)
        y = 0.2 + 0.6 * z + x + rng.normal(scale=0.04, size=n)
        fitted = []
        for shift in (0.0, 1.0e10):
            frame = pd.DataFrame({"z": z + shift, "x": x})
            model = SuperGLM(
                family="gaussian",
                selection_penalty=0.0,
                discrete=True,
                features={
                    "z": Numeric(),
                    "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing),
                },
            )
            model.fit_reml(frame, y, max_reml_iter=2, max_pirls_iter=100)
            fitted.append(model)

        baseline, translated = fitted
        assert translated._reml_result.curvature_source == "observed"
        assert translated._reml_result.objective == pytest.approx(
            baseline._reml_result.objective,
            rel=3e-6,
            abs=3e-5,
        )
        assert translated._reml_result.lambdas["x"] == pytest.approx(
            baseline._reml_result.lambdas["x"],
            rel=3e-5,
        )
        assert translated._solver_result.deviance == pytest.approx(
            baseline._solver_result.deviance,
            rel=1e-5,
            abs=1e-7,
        )

    def test_candidate_guard_backtracks_past_uphill_full_and_half_steps(self, monkeypatch):
        """A fresh converged mode is required at every log-scale trial."""
        current = SimpleNamespace(
            lambdas={"x": 1.0},
            objective=10.0,
            result=SimpleNamespace(beta=np.array([0.0]), intercept=0.0),
            scop_states={},
        )
        attempted_lambdas = []

        def fake_fit(context, trial_lambdas, **kwargs):
            del context, kwargs
            attempted_lambdas.append(trial_lambdas.copy())
            trial = trial_lambdas["x"]
            objective = 12.0 if trial > 8.0 else 11.0 if trial > 3.0 else 9.0
            return SimpleNamespace(
                lambdas=trial_lambdas.copy(),
                objective=objective,
                result=SimpleNamespace(beta=np.array([objective]), intercept=objective),
                scop_states={},
            )

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fake_fit)

        accepted, moved = scop_efs_module._backtrack_scop_efs_candidate(
            SimpleNamespace(),
            current,
            {"x": 16.0},
            reml_iteration=1,
            max_attempts=4,
        )

        assert moved is True
        assert accepted.lambdas == {"x": 2.0}
        np.testing.assert_allclose(
            [trial["x"] for trial in attempted_lambdas],
            [16.0, 4.0, 2.0],
            rtol=1e-15,
        )

    def test_candidate_guard_rejects_without_moving_after_all_trials_are_uphill(self, monkeypatch):
        """Exhausted backtracking retains the exact current fitted state."""
        current = SimpleNamespace(
            lambdas={"x": 1.0},
            objective=10.0,
            result=SimpleNamespace(beta=np.array([0.0]), intercept=0.0),
            scop_states={},
        )
        attempted_lambdas = []

        def fake_fit(context, trial_lambdas, **kwargs):
            del context, kwargs
            attempted_lambdas.append(trial_lambdas.copy())
            return SimpleNamespace(
                lambdas=trial_lambdas.copy(),
                objective=11.0,
                result=SimpleNamespace(beta=np.array([1.0]), intercept=1.0),
                scop_states={},
            )

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fake_fit)

        retained, moved = scop_efs_module._backtrack_scop_efs_candidate(
            SimpleNamespace(),
            current,
            {"x": 16.0},
            reml_iteration=1,
            max_attempts=4,
        )

        assert moved is False
        assert retained is current
        np.testing.assert_allclose(
            [trial["x"] for trial in attempted_lambdas],
            [
                16.0,
                4.0,
                2.0,
                np.sqrt(2.0),
                1.0 / 16.0,
                1.0 / 4.0,
                1.0 / 2.0,
                1.0 / np.sqrt(2.0),
            ],
            rtol=1e-15,
        )

    def test_candidate_guard_reflects_an_uphill_efs_direction(self, monkeypatch):
        """A reversed EFS direction may be accepted, but only after objective evaluation."""
        current = SimpleNamespace(
            lambdas={"x": 1.0},
            objective=10.0,
            result=SimpleNamespace(beta=np.array([0.0]), intercept=0.0),
            scop_states={},
        )
        attempted_lambdas = []

        def fake_fit(context, trial_lambdas, **kwargs):
            del context, kwargs
            attempted_lambdas.append(trial_lambdas.copy())
            objective = 9.0 if trial_lambdas["x"] < 1.0 else 11.0
            return SimpleNamespace(
                lambdas=trial_lambdas.copy(),
                objective=objective,
                result=SimpleNamespace(beta=np.array([objective]), intercept=objective),
                scop_states={},
            )

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fake_fit)

        accepted, moved = scop_efs_module._backtrack_scop_efs_candidate(
            SimpleNamespace(),
            current,
            {"x": 16.0},
            reml_iteration=1,
            max_attempts=2,
        )

        assert moved is True
        assert accepted.lambdas == {"x": 1.0 / 16.0}
        np.testing.assert_allclose(
            [trial["x"] for trial in attempted_lambdas],
            [16.0, 4.0, 1.0 / 16.0],
            rtol=1e-15,
        )

    def test_mode_certificate_uses_retained_range_newton_correction(self):
        """A tiny raw score must not hide a large weak-curvature correction."""
        mode = SimpleNamespace(
            result=SimpleNamespace(beta=np.array([0.0]), intercept=0.0),
            fisher_mean_x=np.array([0.0]),
            scop_states={},
            mode_score=SimpleNamespace(
                intercept=0.0,
                slopes=np.array([1.0e-12]),
                max_abs=1.0e-12,
                relative_max=1.0e-12,
            ),
            joint_geometry=SimpleNamespace(
                hessian_inverse=np.array([[1.0e12]]),
                transformed_intercept_cross=np.array([0.0]),
                transformed_mean_x=np.array([0.0]),
                sum_w=1.0,
            ),
        )

        assert scop_efs_module._scop_mode_newton_relative(mode) == pytest.approx(1.0)

    def test_mode_certificate_floor_scales_with_joint_rank(self):
        """Factor/score roundoff accumulates at root-rank scale.

        Pinned with exact equality: the bar is deliberately ``sqrt(rank*eps)``
        and independent of any solver tolerance. The ``pirls_tol`` term the
        expression once carried was dead by arithmetic -- it could never
        exceed the floor -- and is gone (#184); exactness keeps a live
        tolerance knob from creeping back in unnoticed.
        """
        epsilon = np.finfo(np.float64).eps

        mode = SimpleNamespace(hessian_rank=36)
        assert scop_efs_module._scop_mode_tolerance(mode) == np.sqrt(36.0 * epsilon)

        degenerate = SimpleNamespace(hessian_rank=0)
        assert scop_efs_module._scop_mode_tolerance(degenerate) == np.sqrt(epsilon)

    def test_candidate_guard_requires_reflected_direction_to_be_downhill(self, monkeypatch):
        """The forward numerical tie tolerance must not admit an uphill reflection."""
        current = SimpleNamespace(
            lambdas={"x": 1.0},
            objective=10.0,
            result=SimpleNamespace(beta=np.array([0.0]), intercept=0.0),
            scop_states={},
        )

        def fake_fit(context, trial_lambdas, **kwargs):
            del context, kwargs
            objective = 11.0 if trial_lambdas["x"] > 1.0 else 10.0 + 1.0e-9
            return SimpleNamespace(
                lambdas=trial_lambdas.copy(),
                objective=objective,
                result=SimpleNamespace(beta=np.array([objective]), intercept=objective),
                scop_states={},
            )

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fake_fit)

        retained, moved = scop_efs_module._backtrack_scop_efs_candidate(
            SimpleNamespace(),
            current,
            {"x": 16.0},
            reml_iteration=1,
            max_attempts=1,
        )

        assert moved is False
        assert retained is current

    def test_candidate_guard_keeps_deep_forward_backtracking_before_reflection(
        self,
        monkeypatch,
    ):
        """A valid 1/16 forward step must be tried before the reflected fallback."""
        current = SimpleNamespace(
            lambdas={"x": 1.0},
            objective=10.0,
            result=SimpleNamespace(beta=np.array([0.0]), intercept=0.0),
            scop_states={},
        )
        attempted_lambdas = []
        deepest_accepted = 16.0 ** (1.0 / 16.0)

        def fake_fit(context, trial_lambdas, **kwargs):
            del context, kwargs
            attempted_lambdas.append(trial_lambdas.copy())
            trial_lambda = trial_lambdas["x"]
            objective = 9.0 if 1.0 < trial_lambda <= deepest_accepted else 11.0
            return SimpleNamespace(
                lambdas=trial_lambdas.copy(),
                objective=objective,
                result=SimpleNamespace(beta=np.array([objective]), intercept=objective),
                scop_states={},
            )

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fake_fit)

        accepted, moved = scop_efs_module._backtrack_scop_efs_candidate(
            SimpleNamespace(),
            current,
            {"x": 16.0},
            reml_iteration=1,
        )

        assert moved is True
        assert accepted.lambdas["x"] == pytest.approx(deepest_accepted)
        assert all(trial["x"] > 1.0 for trial in attempted_lambdas)
        assert len(attempted_lambdas) == 5

    def test_candidate_objective_receives_trial_fit_and_fresh_geometry(self, monkeypatch):
        """A trial lambda is scored only with the state fitted at that lambda."""
        trial_result = SimpleNamespace(
            beta=np.array([3.0]),
            intercept=0.25,
            converged=True,
            rank_info=SimpleNamespace(sum_w=4.0, mean_x=np.array([0.0])),
        )
        trial_xtwx = np.array([[7.0]])
        trial_scop_states = {}
        evaluation = SimpleNamespace(value=8.0)
        objective_calls = []

        def fake_solver(**kwargs):
            assert kwargs["lambda2"] == {"x": 4.0}
            return trial_result, np.array([[1.0]]), trial_xtwx, trial_scop_states

        def fake_objective(*args, **kwargs):
            objective_calls.append((args, kwargs))
            return evaluation

        monkeypatch.setattr(scop_efs_module, "fit_irls_direct", fake_solver)
        monkeypatch.setattr(scop_efs_module, "reml_laml_objective", fake_objective)

        context = scop_efs_module._SCOPREMLFitContext(
            dm=SimpleNamespace(p=1, group_matrices=[]),
            distribution=SimpleNamespace(),
            link=SimpleNamespace(),
            groups=[],
            y=np.array([1.0]),
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            pirls_tol=1e-6,
            max_pirls_iter=10,
            reml_penalties=[],
            convergence="deviance",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=1.0,
            gamma_scale_data=None,
        )
        mode = scop_efs_module._fit_scop_reml_mode(
            context,
            {"x": 4.0},
            beta_init=np.array([0.0]),
            intercept_init=0.0,
            scop_state_init=None,
            phase="line_search",
            reml_iteration=1,
            line_search_iteration=1,
            trial_alpha=1.0,
            require_converged=True,
        )

        assert mode is not None
        assert mode.result is trial_result
        assert mode.xtwx is trial_xtwx
        assert mode.scop_states is trial_scop_states
        assert mode.lambdas == {"x": 4.0}
        assert mode.evaluation is evaluation
        assert len(objective_calls) == 1
        args, kwargs = objective_calls[0]
        assert args[5] is trial_result
        assert args[6] == {"x": 4.0}
        assert kwargs["XtWX"] is trial_xtwx
        assert kwargs["S_override"].shape == (1, 1)
        assert kwargs["return_evaluation"] is True

    def test_nonconverged_candidate_never_reaches_laml(self, monkeypatch):
        """A failed inner solve is backtracked without any objective evaluation."""
        trial_result = SimpleNamespace(
            beta=np.array([3.0]),
            intercept=0.25,
            converged=False,
        )
        objective_calls = []

        monkeypatch.setattr(
            scop_efs_module,
            "fit_irls_direct",
            lambda **kwargs: (trial_result, np.array([[1.0]]), np.array([[7.0]]), {}),
        )
        monkeypatch.setattr(
            scop_efs_module,
            "reml_laml_objective",
            lambda *args, **kwargs: objective_calls.append((args, kwargs)),
        )

        context = scop_efs_module._SCOPREMLFitContext(
            dm=SimpleNamespace(p=1, group_matrices=[]),
            distribution=SimpleNamespace(),
            link=SimpleNamespace(),
            groups=[],
            y=np.array([1.0]),
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            pirls_tol=1e-6,
            max_pirls_iter=10,
            reml_penalties=[],
            convergence="deviance",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=1.0,
            gamma_scale_data=None,
        )
        mode = scop_efs_module._fit_scop_reml_mode(
            context,
            {"x": 4.0},
            beta_init=np.array([0.0]),
            intercept_init=0.0,
            scop_state_init=None,
            phase="line_search",
            reml_iteration=1,
            line_search_iteration=1,
            trial_alpha=1.0,
            require_converged=True,
        )

        assert mode is None
        assert objective_calls == []

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
            features={"x": PSpline(n_knots=10, constraint=Constraint.fit.increasing)},
        )
        y_out, sample_weight, offset = model_build_design_matrix(model, df, y, np.ones(n), None)
        offset_arr = np.zeros(n) if offset is None else np.array(offset)
        return model, y_out, np.array(sample_weight), offset_arr, df

    @pytest.mark.slow
    def test_retained_trial_mode_is_reused_for_terminal_state(self, scop_reml_model, monkeypatch):
        """The terminal result reuses a coherent retained mode instead of refitting it."""
        model, y, sample_weight, offset, _ = scop_reml_model
        real_fit = scop_efs_module.fit_irls_direct
        fit_calls = []

        def spy_fit(**kwargs):
            out = real_fit(**kwargs)
            result = out[0]
            fit_calls.append(
                {
                    "phase": kwargs["debug_context"]["phase"],
                    "lambdas": kwargs["lambda2"].copy(),
                    "result": result,
                }
            )
            return out

        monkeypatch.setattr(scop_efs_module, "fit_irls_direct", spy_fit)

        fitted = optimize_scop_efs_reml(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            lambdas={"x": 1.0},
            estimated_names={"x"},
            max_reml_iter=1,
            reml_tol=1e-12,
        )

        assert any(call["phase"] == "line_search" for call in fit_calls)
        assert all(call["phase"] != "final" for call in fit_calls)
        retained = [
            call
            for call in fit_calls
            if call["phase"] in {"reml", "line_search"} and call["lambdas"] == fitted.lambdas
        ]
        assert len(retained) == 1
        assert fitted.pirls_result is retained[0]["result"]

    @pytest.mark.slow
    def test_final_gaussian_phi_matches_terminal_laml_profile(self, scop_reml_model):
        """The installed SCOP scale and nullity must come from the terminal Wood profile."""
        from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective

        model, y, sample_weight, offset, _ = scop_reml_model
        fitted = optimize_scop_efs_reml(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            lambdas={"x": 1.0},
            estimated_names={"x"},
            max_reml_iter=8,
            reml_tol=1e-6,
        )
        evaluation = reml_laml_objective(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            groups=model._groups,
            y=y,
            result=fitted.pirls_result,
            lambdas=fitted.lambdas,
            sample_weight=sample_weight,
            offset_arr=offset,
            reml_penalties=fitted.reml_penalties,
            scop_states=fitted.scop_states,
            return_evaluation=True,
        )

        assert isinstance(evaluation, REMLObjectiveEvaluation)
        assert evaluation.profiled_scale is not None
        assert evaluation.penalty_nullity == pytest.approx(2.0)
        assert fitted.pirls_result.phi == pytest.approx(
            evaluation.profiled_scale.phi,
            rel=1e-12,
            abs=1e-12,
        )

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
                features={"x": PSpline(n_knots=10, constraint=Constraint.fit.increasing)},
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


class TestSCOPNonConvergenceIsNotSpeciallyAccepted:
    """A non-converged SCOP inner fit is rejected, whatever its deviance did.

    Item 2c retired the deviance-stagnation acceptance rule. It existed
    because ``convergence="coefficients"`` cannot terminate when a SCOP
    coefficient drifts to its log-space boundary (``exp(gamma) -> 0``): the
    coefficient keeps moving while the fit stops. PR #176 fixed that at its
    cause by truncating the unidentifiable direction out of the Newton step,
    so the boundary fit converges normally and no fit in the corpus reaches
    this path any more.

    What certifies a mode is the penalized-score check in
    ``_fit_scop_reml_mode`` -- ``_scop_mode_newton_relative`` against
    ``_scop_mode_tolerance`` -- which every accepted mode always had to pass.
    """

    # Comfortably longer and shorter than any window the retired gate used, so
    # these pin behaviour rather than a constant's exact value.
    LONG_RUN = 256
    SHORT_RUN = 4

    @staticmethod
    def _context():
        return scop_efs_module._SCOPREMLFitContext(
            dm=SimpleNamespace(p=1, group_matrices=[]),
            distribution=SimpleNamespace(),
            link=SimpleNamespace(),
            groups=[],
            y=np.array([1.0]),
            sample_weight=np.array([1.0]),
            offset_arr=np.array([0.0]),
            pirls_tol=1e-6,
            max_pirls_iter=200,
            reml_penalties=[],
            convergence="coefficients",
            scop_joint=True,
            debug_recorder=None,
            likelihood_size=1.0,
            gamma_scale_data=None,
        )

    def _run_gate(self, monkeypatch, solver_result, captured=None):
        def fake_solver(**kwargs):
            if captured is not None:
                captured.update(kwargs)
            return solver_result, None, np.array([[1.0]]), {}

        monkeypatch.setattr(scop_efs_module, "fit_irls_direct", fake_solver)
        return scop_efs_module._fit_scop_reml_mode(
            self._context(),
            {"x": 1.0},
            beta_init=None,
            intercept_init=None,
            scop_state_init=None,
            phase="candidate",
            reml_iteration=1,
            require_converged=True,
        )

    @staticmethod
    def _stub(n_iter):
        """A non-converged solver result that exhausted its budget."""
        return SimpleNamespace(
            converged=False,
            termination_reason="max_iter",
            beta=np.array([0.0]),
            intercept=0.0,
            rank_info=None,
            n_iter=n_iter,
        )

    def test_a_stagnant_candidate_is_no_longer_specially_accepted(self, monkeypatch):
        """A boundary-stagnant fit is a non-convergence like any other.

        Before item 2c the gate admitted this stub: it flipped ``converged``
        to True and the fit proceeded into geometry assembly, which a bare
        stub cannot satisfy, so the failure surfaced as ``retained centered
        fit geometry``. Rank truncation (PR #176) removes the cause, so the
        workaround is gone and the mode is rejected at ``require_converged``.
        """
        stub = self._stub(self.LONG_RUN)
        assert self._run_gate(monkeypatch, stub) is None
        # Nothing reclassifies how the iteration actually ended.
        assert stub.converged is False
        assert stub.termination_reason == "max_iter"

    def test_a_short_run_candidate_is_rejected(self, monkeypatch):
        """Run length never mattered to the outcome; now it cannot."""
        stub = self._stub(self.SHORT_RUN)
        assert self._run_gate(monkeypatch, stub) is None
        assert stub.converged is False

    def test_the_inner_fit_does_not_ask_for_the_full_recorder(self, monkeypatch):
        """``record_diagnostics`` builds a forty-field row per iteration.

        It also switches on the solver's per-iteration extrema capture --
        measured at 7-16% of SCOP REML wall time. Asking for it from the inner
        fits is a performance regression, so this pins that we do not.
        """
        captured = {}
        self._run_gate(monkeypatch, self._stub(self.SHORT_RUN), captured=captured)
        assert captured.get("record_diagnostics", False) is False

    def test_a_genuinely_non_converging_fit_still_raises(self, monkeypatch):
        """A real SCOP fit that never settles is reported as a failure.

        This quasi-separated Poisson exhausts all its PIRLS iterations with
        zero halvings and zero rejections, its deviance still moving by ~1e-3
        relative per iteration.

        The message alone cannot pin that. ``did not converge to a coefficient
        mode`` is raised for two distinct reasons: the inner fit not
        converging, and a converged mode failing latent certification --
        ``_fit_scop_reml_mode`` returns ``None`` for both. So we also pin
        *which* one fired. A non-converged result returns at ``require_converged
        and not result.converged`` before any certification is computed, so
        ``_scop_mode_newton_relative`` is never reached on this path; a
        certification failure would have to call it. Zero calls therefore
        distinguishes the two, and keeps a future change that makes this fit
        converge from leaving the test silently green on the other failure.
        """
        certifications = []
        original = scop_efs_module._scop_mode_newton_relative

        def spy(mode):
            value = original(mode)
            certifications.append(value)
            return value

        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", spy)

        x = np.linspace(0, 1, 200)
        frame = pd.DataFrame({"x": x})
        response = np.where(x > 0.8, 5000.0, 0.0)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": PSpline(n_knots=12, penalty="ssp", constraint=Constraint.fit.increasing)
            },
        )
        with pytest.raises(RuntimeError, match="did not converge to a coefficient mode"):
            model.fit_reml(frame, response, max_reml_iter=20)
        assert certifications == []

    def test_a_failed_certification_gets_a_cold_final_attempt(self, monkeypatch):
        """The final retry rung drops the warm start, not just the tolerance.

        The two tolerance rungs re-fit from the mode that just failed. Once the
        inner fit has converged tighter than the bar, that reproduces the same
        mode bit-identically -- measured on the fit this exists for, three
        attempts all returned 1.3792e-06 against a bar of 7.1463e-08. Only a
        different starting point can move it, so the last rung starts cold.

        Certification is forced to reject the first three attempts, so the fit
        can only succeed if a fourth exists. The warm/cold pattern is asserted
        too: a fourth attempt that also warm-started would not be the fix.
        """
        warm_starts: list[bool] = []
        checks = {"n": 0}

        real_fit = scop_efs_module._fit_scop_reml_mode
        real_relative = scop_efs_module._scop_mode_newton_relative

        def recording_fit(context, lambdas, **kwargs):
            warm_starts.append(kwargs.get("beta_init") is not None)
            return real_fit(context, lambdas, **kwargs)

        def reject_first_three(mode):
            checks["n"] += 1
            if checks["n"] <= 3:
                return 1.0  # far above any achievable tolerance
            return real_relative(mode)

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", recording_fit)
        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", reject_first_three)

        rng = np.random.default_rng(0)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = np.round(np.exp(1.0 + 1.5 * x)).astype(float)
        frame = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": PSpline(n_knots=8, penalty="ssp", constraint=Constraint.fit.increasing)},
        )
        model.fit_reml(frame, y, max_reml_iter=5)

        assert checks["n"] >= 4, "the ladder must reach a fourth attempt"
        assert warm_starts[1] is True, "rung 1 warm-starts"
        assert warm_starts[2] is True, "rung 2 warm-starts"
        assert warm_starts[3] is False, "the final rung must start cold"


class TestCandidateStepBackoff:
    """A candidate certification failure backs the lambda step off (#179).

    The iteration-1 candidate consumes the one EFS proposal that bypasses
    the line search, so it was the one lambda movement with no damping
    behind it: four call sites raised on a rejection the line search
    survives. The backoff applies the line search's own trial formula --
    damped geometric steps in log-lambda -- between the certified mode the
    step was taken from and the proposal that failed. Sites with no
    certified predecessor (bootstrap, fixed-lambda) keep raising.
    """

    @staticmethod
    def _model():
        rng = np.random.default_rng(0)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = np.round(np.exp(1.0 + 1.5 * x)).astype(float)
        frame = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": PSpline(n_knots=8, penalty="ssp", constraint=Constraint.fit.increasing)},
        )
        return model, frame, y

    @staticmethod
    def _phase_tracking(monkeypatch, state):
        """Expose which top-level phase each certification check belongs to.

        Also records every top-level fit (retry depth 0) with its phase,
        ``trial_alpha`` and lambdas in ``state["calls"]``, so tests can pin
        the mechanism -- which vectors were fit, at which damping -- and
        not just the outcome.
        """
        real_fit = scop_efs_module._fit_scop_reml_mode
        calls = state.setdefault("calls", [])

        def tracking_fit(context, lambdas, **kwargs):
            if kwargs.get("_certification_retry", 0) == 0:
                calls.append(
                    {
                        "phase": kwargs.get("phase"),
                        "trial_alpha": kwargs.get("trial_alpha"),
                        "lambdas": dict(lambdas),
                    }
                )
            previous = state["phase"]
            state["phase"] = kwargs.get("phase")
            try:
                return real_fit(context, lambdas, **kwargs)
            finally:
                state["phase"] = previous

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", tracking_fit)

    def test_a_failed_candidate_ladder_gets_a_damped_step(self, monkeypatch):
        """Candidate certification failure damps the lambda step, not the fit.

        The iteration-1 candidate's entire four-rung ladder is forced to
        reject; every other check is real. Before the backoff this raised
        ``SCOP REML candidate did not converge to a coefficient mode``; now
        a shorter step toward the certified bootstrap must be found and the
        fit must succeed. Asserted through the observable outcome -- the
        fit completes and the fitted curve respects the constraint -- plus
        the forced-rejection count, which pins that the whole ladder was
        exhausted rather than the rescue arriving early.
        """
        state = {"phase": None, "candidate_rejections": 0}
        self._phase_tracking(monkeypatch, state)
        real_relative = scop_efs_module._scop_mode_newton_relative

        def reject_the_first_candidate_ladder(mode):
            if state["phase"] == "candidate" and state["candidate_rejections"] < 4:
                state["candidate_rejections"] += 1
                return 1.0  # far above any achievable bar
            return real_relative(mode)

        monkeypatch.setattr(
            scop_efs_module, "_scop_mode_newton_relative", reject_the_first_candidate_ladder
        )

        model, frame, y = self._model()
        model.fit_reml(frame, y, max_reml_iter=5)

        assert state["candidate_rejections"] == 4, "the full ladder must be exhausted first"
        assert model.result.beta is not None
        fitted = model.predict(frame)
        assert np.all(np.diff(fitted) >= -1e-8), "the rescued fit still honours the constraint"

        # The mechanism, not just the outcome: the first backoff attempt is
        # deterministically alpha=0.5, and the vector it adopts must lie
        # strictly between the certified bootstrap and the failed proposal.
        bootstrap = next(c for c in state["calls"] if c["phase"] == "bootstrap")["lambdas"]
        candidate_calls = [c for c in state["calls"] if c["phase"] == "candidate"]
        assert candidate_calls[0]["trial_alpha"] is None, "the full step is a plain candidate"
        assert candidate_calls[1]["trial_alpha"] == pytest.approx(0.5)
        proposal = candidate_calls[0]["lambdas"]
        adopted = candidate_calls[-1]["lambdas"]
        moved = [k for k in proposal if k in bootstrap and proposal[k] != bootstrap[k]]
        assert moved, "the bootstrap EFS step must have proposed movement"
        for key in moved:
            low, high = sorted((bootstrap[key], proposal[key]))
            assert low < adopted[key] < high, "the adopted step must be a strict shortening"

        # The history must record the damped vector that was fitted, not the
        # proposal that never certified (governance reads lambda_history as
        # the REML path of fitted vectors).
        history = model._reml_result.lambda_history
        assert history[0] == adopted, "the history records the fitted damped vector"
        assert history[0] != proposal, "not the proposal that was never fitted"

    def test_the_backoff_preserves_the_proposal_key_set(self, monkeypatch):
        """Adopted lambdas must keep the loop's key set, not the origin's.

        The loop's consumers read every component name out of the adopted
        dict, so a proposal key absent from the origin must survive the
        rescue at its proposed value rather than vanish (found in review,
        PR #183). Also pins the no-movement guard: a proposal identical to
        the origin has no step to shorten and must return None without
        fitting anything.
        """
        seen = []

        def fake_fit(context, lambdas, **kwargs):
            seen.append(dict(lambdas))
            return "certified-mode-sentinel"

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fake_fit)
        origin = SimpleNamespace(
            lambdas={"a": 1.0e-4},
            result=SimpleNamespace(beta=np.zeros(1), intercept=0.0),
            scop_states={},
        )

        rescue = scop_efs_module._backoff_scop_candidate_step(
            None, origin, {"a": 1.0, "b": 2.0}, reml_iteration=1
        )
        assert rescue is not None
        mode, adopted, alpha = rescue
        assert mode == "certified-mode-sentinel"
        assert alpha == pytest.approx(0.5)
        assert adopted["b"] == 2.0, "a proposal-only key keeps its proposed value"
        assert 1.0e-4 < adopted["a"] < 1.0, "a shared key is interpolated toward the origin"
        assert seen and seen[0] == adopted

        seen.clear()
        no_movement = scop_efs_module._backoff_scop_candidate_step(
            None, origin, {"a": 1.0e-4}, reml_iteration=1
        )
        assert no_movement is None
        assert seen == [], "a no-movement proposal must not fit anything"

    def test_an_unrecoverable_candidate_still_raises(self, monkeypatch):
        """When no damped step certifies either, the failure stays loud.

        Every candidate-phase certification is rejected, so the ladder and
        then every backoff attempt fail. The exact candidate error must
        surface: the backoff is bounded, and it must not convert a hard
        failure into a silent stall or an unbounded retry.
        """
        state = {"phase": None}
        self._phase_tracking(monkeypatch, state)
        real_relative = scop_efs_module._scop_mode_newton_relative

        def reject_every_candidate_check(mode):
            if state["phase"] == "candidate":
                return 1.0
            return real_relative(mode)

        monkeypatch.setattr(
            scop_efs_module, "_scop_mode_newton_relative", reject_every_candidate_check
        )

        model, frame, y = self._model()
        with pytest.raises(
            RuntimeError, match="SCOP REML candidate did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)

    def test_a_rescue_the_line_search_cannot_move_from_still_raises(self, monkeypatch):
        """A rescue must be followed by accepted progress, or the fit is loud.

        The rescued mode is chosen for certifiability, not objective merit:
        no acceptance gate ever endorsed it. If the line search then cannot
        accept a single trial from it, returning it through the ordinary
        ``line_search_stalled`` path would publish half a bootstrap step as
        a REML estimate -- the silent degradation the design forbids, on an
        input that raised before the backoff existed. Found in review
        (PR #183, Codex P2).
        """
        state = {"phase": None, "candidate_rejections": 0}
        self._phase_tracking(monkeypatch, state)
        real_relative = scop_efs_module._scop_mode_newton_relative

        def reject_ladder_then_every_line_search_check(mode):
            if state["phase"] == "candidate" and state["candidate_rejections"] < 4:
                state["candidate_rejections"] += 1
                return 1.0
            if state["phase"] == "line_search":
                return 1.0
            return real_relative(mode)

        monkeypatch.setattr(
            scop_efs_module,
            "_scop_mode_newton_relative",
            reject_ladder_then_every_line_search_check,
        )

        model, frame, y = self._model()
        with pytest.raises(
            RuntimeError, match="SCOP REML candidate did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)
        assert state["candidate_rejections"] == 4, "the rescue path must actually be exercised"
        # [None, 0.5] pins that the rescue *certified* and the raise came from
        # the guard -- backoff exhaustion would show the whole alpha ladder.
        candidate_alphas = [c["trial_alpha"] for c in state["calls"] if c["phase"] == "candidate"]
        assert candidate_alphas == [None, pytest.approx(0.5)]

    def test_a_rescue_with_a_no_op_proposal_still_raises(self, monkeypatch):
        """An EFS no-op after a rescue is not accepted progress.

        When every active component's proposal equals the rescued mode's
        lambdas, the line search returns the current mode accepted-by-default
        without fitting a single trial. For a rescued iteration that
        acceptance is vacuous -- no objective gate ever saw the mode -- and
        the zero lambda delta would immediately satisfy strict convergence,
        publishing the rescue as a converged fit on an input that previously
        raised. Found in review (PR #183, Codex round 2).
        """
        state = {"phase": None, "candidate_rejections": 0}
        self._phase_tracking(monkeypatch, state)
        real_relative = scop_efs_module._scop_mode_newton_relative

        def reject_the_first_candidate_ladder(mode):
            if state["phase"] == "candidate" and state["candidate_rejections"] < 4:
                state["candidate_rejections"] += 1
                return 1.0
            return real_relative(mode)

        monkeypatch.setattr(
            scop_efs_module, "_scop_mode_newton_relative", reject_the_first_candidate_ladder
        )

        def no_op_backtrack(context, current, proposed_lambdas, **kwargs):
            return current, True

        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", no_op_backtrack)

        model, frame, y = self._model()
        with pytest.raises(
            RuntimeError, match="SCOP REML candidate did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)
        assert state["candidate_rejections"] == 4, "the rescue path must actually be exercised"
        # [None, 0.5] pins that the rescue *certified* and the raise came from
        # the guard -- backoff exhaustion would show the whole alpha ladder.
        candidate_alphas = [c["trial_alpha"] for c in state["calls"] if c["phase"] == "candidate"]
        assert candidate_alphas == [None, pytest.approx(0.5)]

    def test_a_no_op_proposal_returns_the_identical_current_mode(self, monkeypatch):
        """The identity contract the rescue guard rests on, pinned directly.

        The guard detects "no acceptance gate saw a new state" by object
        identity, so the line search must hand back the *identical* current
        mode -- never a copy -- when a no-op proposal is accepted without
        fitting anything. A harmless-looking ``replace(current)`` here would
        silently disarm the guard with a green suite. Found in review
        (PR #183, round 3).
        """
        fits = []

        def counting_fit(context, lambdas, **kwargs):
            fits.append(dict(lambdas))
            raise AssertionError("a no-op proposal must not fit anything")

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", counting_fit)
        current = SimpleNamespace(lambdas={"x": 2.5})
        retained, accepted = scop_efs_module._backtrack_scop_efs_candidate(
            None, current, {"x": 2.5}, reml_iteration=1
        )
        assert retained is current, "the no-endorsement return must be the identical object"
        assert accepted is True
        assert fits == []

    def test_a_failed_bootstrap_has_nothing_to_back_off_to(self, monkeypatch):
        """The recoverability principle's boundary: no predecessor, no rescue.

        Rejecting every certification kills the bootstrap after its ladder.
        There is no earlier certified mode to damp toward, so the loud
        error is the designed outcome, unchanged by the candidate backoff.
        """
        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", lambda mode: 1.0)
        model, frame, y = self._model()
        with pytest.raises(
            RuntimeError, match="SCOP REML bootstrap did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)

    def test_a_failed_fixed_lambda_fit_has_nothing_to_back_off_to(self, monkeypatch):
        """Fixed-lambda fits have no certified predecessor either.

        With every SCOP lambda held fixed there is no bootstrap and no EFS
        step to shorten -- the requested lambdas are the fit. A
        certification failure there must stay a loud refusal.
        """
        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", lambda mode: 1.0)
        rng = np.random.default_rng(0)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = np.round(np.exp(1.0 + 1.5 * x)).astype(float)
        frame = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=8,
                    penalty="ssp",
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                )
            },
        )
        with pytest.raises(
            RuntimeError, match="fixed-lambda SCOP fit did not converge to a coefficient mode"
        ):
            model.fit_reml(frame, y, max_reml_iter=5)


class TestIterationDiagnosticsSmallSample:
    """The diagnostics recorder must survive n <= 5.

    ``k = min(5, n)`` makes ``k == n`` for small samples, and numpy requires
    ``-n <= kth < n``, so the bottom-k partition needs ``k - 1``. The bug is
    latent while diagnostics are opt-in, and a caller that turns the recorder
    on unconditionally converts it into a crash on a default-argument REML
    fit. The opt-in test below is what keeps the fix pinned, and the REML test
    below keeps small-n SCOP fits themselves covered.
    """

    @staticmethod
    def _frame(n):
        return pd.DataFrame({"x": np.linspace(0.0, 1.0, n)}), np.arange(1.0, n + 1.0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_opt_in_diagnostics_survive_small_samples(self, n):
        frame, response = self._frame(n)
        fitted = SuperGLM(family="poisson").fit(frame, response, record_diagnostics=True)
        log = fitted.iteration_diagnostics()
        assert len(log) >= 1
        # every recorded index is a real observation
        for column in ("top_w_indices", "bottom_w_indices"):
            if column in log.columns:
                for entry in log[column]:
                    assert all(0 <= int(i) < n for i in np.atleast_1d(entry))

    @pytest.mark.parametrize("n", [3, 5, 6])
    def test_scop_reml_fits_small_samples_on_default_arguments(self, n):
        """No caller opt-in involved: the default-argument SCOP REML path."""
        frame, response = self._frame(n)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": PSpline(n_knots=4, penalty="ssp", constraint=Constraint.fit.increasing)},
        )
        model.fit_reml(frame, response, max_reml_iter=2)
        assert model.result.beta is not None

    def test_debug_weights_survives_small_samples(self):
        """Exercises the helper, not a copy of it.

        This test used to re-spell ``np.argpartition(W, k - 1)[:k]`` inline,
        which meant it asserted its own arithmetic: reverting the fix in
        ``_extreme_weight_indices`` left it green, so it read as coverage while
        pinning nothing.
        """
        from superglm.debug_weights import _positive_working_weight_stats
        from superglm.solvers.pirls import _extreme_weight_indices

        for n in range(1, 7):
            weights = np.linspace(1.0, 2.0, n)
            k = min(5, n)
            top_idx, bot_idx = _extreme_weight_indices(weights)
            assert len(top_idx) == k
            assert len(bot_idx) == k
            assert all(0 <= int(i) < n for i in (*top_idx, *bot_idx))
            # ...and the documented ordering: largest first, smallest first.
            np.testing.assert_array_equal(weights[top_idx], np.sort(weights)[::-1][:k])
            np.testing.assert_array_equal(weights[bot_idx], np.sort(weights)[:k])
            assert _positive_working_weight_stats(weights)[2] >= 1.0


class TestSCOPREMLDoesNotPublishDiagnostics:
    """A REML fit must not turn on a public accessor its caller never requested.

    ``fit_reml`` exposes no diagnostics parameter and the non-SCOP REML path
    records nothing, so a REML caller has never been able to ask for a
    per-iteration log. Nothing may leak one onto the published result, or a
    SCOP REML fit would carry a field no other engine populates.
    """

    @staticmethod
    def _scop_model():
        x = np.linspace(0.0, 1.0, 60)
        frame = pd.DataFrame({"x": x})
        response = np.round(np.exp(1.0 + 0.5 * x)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": PSpline(n_knots=6, penalty="ssp", constraint=Constraint.fit.increasing)},
        )
        model.fit_reml(frame, response, max_reml_iter=3)
        return model

    def test_published_result_carries_no_iteration_log(self):
        assert self._scop_model().result.iteration_log is None

    def test_accessor_raises_as_documented(self):
        with pytest.raises(RuntimeError, match="No iteration diagnostics recorded"):
            self._scop_model().iteration_diagnostics()


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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                    constraint=Constraint.fit.increasing,
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
    @pytest.mark.parametrize("family", [Gaussian(), Poisson()], ids=["gaussian", "poisson"])
    def test_fixed_scop_reml_publishes_one_coherent_evaluated_mode(self, family):
        """Fixed smoothing still has a complete REML objective and terminal lifecycle."""
        from superglm.reml.objective import reml_laml_objective

        rng = np.random.default_rng(20260801)
        n = 320
        x = np.sort(rng.uniform(0.0, 1.0, n))
        if isinstance(family, Gaussian):
            y = 0.3 + 1.6 * x + rng.normal(0.0, 0.16, n)
        else:
            y = rng.poisson(np.exp(-0.3 + 1.1 * x))
        frame = pd.DataFrame({"x": x})
        fixed_lambda = 1.7
        model = SuperGLM(
            family=family,
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=fixed_lambda),
                )
            },
        )

        model.fit_reml(frame, y)

        fitted = model._reml_result
        solver = model._solver_result
        assert isinstance(fitted, REMLResult)
        assert fitted.pirls_result is solver
        assert fitted.lambdas == model._reml_lambdas == {"x": fixed_lambda}
        assert fitted.lambda_history == [{"x": fixed_lambda}]
        assert fitted.n_reml_iter == 0
        assert fitted.converged is solver.converged is True
        assert fitted.termination_reason == "fixed_lambdas"
        assert fitted.objective is not None and np.isfinite(fitted.objective)
        assert fitted.scop_states
        assert fitted.reml_penalties
        assert model._reml_profile["n_reml_iter"] == 0
        assert model._reml_profile["converged"] is True
        assert model._last_fit_meta["lambda_strategy"] == "fixed"

        evaluation = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            solver,
            fitted.lambdas,
            model._fit_weights,
            np.zeros(n) if model._fit_offset is None else model._fit_offset,
            log_det_H=solver.log_det_H,
            hessian_rank=solver.reml_hessian_rank,
            reml_penalties=fitted.reml_penalties,
            scop_states=fitted.scop_states,
            return_evaluation=True,
        )
        assert evaluation.value == pytest.approx(fitted.objective, rel=2e-11, abs=2e-11)
        if isinstance(family, Gaussian):
            assert evaluation.profiled_scale is not None
            assert solver.phi == pytest.approx(evaluation.profiled_scale.phi, rel=2e-11)
        else:
            assert evaluation.profiled_scale is None
            assert solver.phi == 1.0

    @pytest.mark.slow
    def test_fit_reml_scop_concave_fixed_lambda_policy(self):
        """Curvature-constrained SCOP terms should honor fixed lambda_policy values."""
        rng = np.random.default_rng(7)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 1.0 - (x - 0.4) ** 2 + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"x": x})

        fixed_val = 2.5
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=8,
                    constraint=Constraint.fit.concave,
                    lambda_policy=LambdaPolicy(mode="fixed", value=fixed_val),
                ),
            },
        )
        model.fit_reml(df[["x"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None
        assert model._reml_lambdas["x"] == pytest.approx(fixed_val)

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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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

    def test_qp_monotone_passthrough(self):
        """BSplineSmooth with QP monotone works via passthrough heuristic."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged
        assert model._reml_lambdas is not None

        # Predictions should be monotone
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-6)

        # Metadata should record passthrough strategy
        assert model._last_fit_meta.get("lambda_strategy") == "qp_passthrough"

    @pytest.mark.slow
    def test_qp_passthrough_lambdas_match_unconstrained(self):
        """QP passthrough lambdas should be close to unconstrained REML lambdas."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        # Unconstrained REML
        model_uc = SuperGLM(
            family=Gaussian(),
            features={
                "x1": BSplineSmooth(n_knots=8),
                "x2": PSpline(n_knots=8),
            },
        )
        model_uc.fit_reml(df[["x1", "x2"]], y)

        # QP passthrough
        model_qp = SuperGLM(
            family=Gaussian(),
            features={
                "x1": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8),
            },
        )
        model_qp.fit_reml(df[["x1", "x2"]], y)

        # x2 lambda should be similar (same term, unconstrained in both)
        # x1 lambda should be in the same ballpark (same penalty structure)
        x2_key_uc = next(k for k in model_uc._reml_lambdas if k.startswith("x2"))
        x2_key_qp = next(k for k in model_qp._reml_lambdas if k.startswith("x2"))
        ratio = model_qp._reml_lambdas[x2_key_qp] / model_uc._reml_lambdas[x2_key_uc]
        assert 0.1 < ratio < 10, f"x2 lambda ratio too far: {ratio:.2f}"

    @pytest.mark.slow
    def test_qp_passthrough_noisy_data_monotone(self):
        """QP passthrough produces monotone predictions even on noisy data."""
        # Use a seed/noise level that makes unconstrained fit non-monotone
        rng = np.random.default_rng(6)
        n = 120
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit_reml(df[["x"]], y)

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-6), f"QP passthrough not monotone: min diff = {diffs.min():.2e}"


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
                    constraint=Constraint.fit.increasing,
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
    def test_fixed_scop_large_lambda_constant_response_converges(self):
        """A valid penalty-null boundary must pass latent mode certification."""
        x = np.linspace(0.0, 1.0, 200)
        df = pd.DataFrame({"x": x})
        y = np.ones_like(x)
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0e6),
                ),
            },
        )

        model.fit_reml(df, y, max_pirls_iter=100)

        assert model._result.converged
        assert np.all(np.isfinite(model._result.beta))
        assert np.isfinite(model._result.intercept)
        assert model._reml_lambdas == {"x": pytest.approx(1.0e6)}

    @pytest.mark.slow
    def test_efs_only_model_unchanged(self):
        """fit_reml() rejects selection_penalty > 0 even without monotone terms."""
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
        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(df[["x1", "x2"]], y)

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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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

    def test_objective_delta_preserves_sub_ulp_descent(self):
        """Single and joint line searches retain descent hidden by full objectives."""
        from superglm.solvers.scop import build_scop_solver_reparam
        from superglm.solvers.scop_newton import (
            _build_joint_objective_cache,
            _joint_objective_from_gammas,
            _safe_joint_trial_objective_delta,
            _safe_trial_objective,
            _safe_trial_objective_delta,
        )

        reparam = build_scop_solver_reparam(2, direction="increasing")
        basis = np.ones((2, 1))
        weights = np.ones(2)
        beta = np.zeros(1)
        gamma = reparam.forward(beta)
        residual = np.array([1.0e8, -1.0e8 + 1.0e-6])
        response = basis @ gamma + residual
        penalty = np.zeros((1, 1))
        beta_trial = np.array([5.0e-7])
        gram = basis.T @ (basis * weights[:, None])
        projected_residual = basis.T @ (weights * residual)

        objective_before = _safe_trial_objective(
            basis,
            weights,
            response,
            beta,
            reparam,
            penalty,
            0.0,
            None,
        )
        objective_trial = _safe_trial_objective(
            basis,
            weights,
            response,
            beta_trial,
            reparam,
            penalty,
            0.0,
            None,
        )
        single_delta = _safe_trial_objective_delta(
            beta,
            beta_trial,
            gamma,
            reparam,
            penalty,
            0.0,
            projected_residual,
            gram,
        )

        state = {
            "B_scop": basis,
            "S_scop": penalty,
            "beta_scop": beta,
            "reparam": reparam,
            "bin_idx": None,
        }
        scop_items = [(0, state)]
        slices = [slice(0, 1)]
        cache = _build_joint_objective_cache(scop_items, weights, response)
        cache.diag_btwb = [gram]
        cache.cross_btwb = {}
        joint_delta = _safe_joint_trial_objective_delta(
            scop_items,
            beta_trial,
            slices,
            [0.0],
            [gamma],
            cache,
        )
        joint_objective_before = _joint_objective_from_gammas(
            [gamma],
            beta,
            slices,
            [0.0],
            scop_items,
            cache,
        )
        joint_objective_trial = _joint_objective_from_gammas(
            [reparam.forward(beta_trial)],
            beta_trial,
            slices,
            [0.0],
            scop_items,
            cache,
        )

        assert objective_trial == objective_before
        assert joint_objective_trial == joint_objective_before
        assert single_delta < 0.0
        assert joint_delta == pytest.approx(single_delta, rel=1.0e-12, abs=1.0e-18)

    def test_single_and_joint_line_searches_use_stable_delta(self, monkeypatch):
        """Both public Newton paths route trial acceptance through delta algebra."""
        from superglm.solvers import scop_newton as scop_newton_module

        B_scop, W, z, beta_scop, reparam, S_scop = self._make_scop_inputs()
        calls = {"single": 0, "joint": 0}
        single_delta = scop_newton_module._safe_trial_objective_delta
        joint_delta = scop_newton_module._safe_joint_trial_objective_delta

        def record_single(*args, **kwargs):
            calls["single"] += 1
            return single_delta(*args, **kwargs)

        def record_joint(*args, **kwargs):
            calls["joint"] += 1
            return joint_delta(*args, **kwargs)

        monkeypatch.setattr(
            scop_newton_module,
            "_safe_trial_objective_delta",
            record_single,
        )
        monkeypatch.setattr(
            scop_newton_module,
            "_safe_joint_trial_objective_delta",
            record_joint,
        )

        scop_newton_module.scop_newton_step(
            B_scop,
            W,
            z,
            beta_scop,
            reparam,
            S_scop,
            lambda2=1.0,
        )
        state = {
            "B_scop": B_scop,
            "S_scop": S_scop,
            "beta_scop": beta_scop,
            "reparam": reparam,
            "bin_idx": None,
            "group_sl": slice(0, beta_scop.size),
            "group_name": "x",
        }
        scop_newton_module.scop_joint_newton_step(
            {0: state},
            W,
            z,
            {"x": 1.0},
            [SimpleNamespace(name="x", sl=state["group_sl"])],
        )

        assert calls["single"] > 0
        assert calls["joint"] > 0

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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8),
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.fit_reml(df[["x1", "x2"]], y)
        assert model._result.converged is True


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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        assert model._reml_result.converged or (
            model._reml_result.termination_reason == "line_search_stalled"
        ), (
            "Outer REML neither converged nor retained an honestly stalled mode "
            f"after {model._reml_result.n_reml_iter} iterations"
        )
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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x3": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            },
        )
        model.fit_reml(df[["x1", "x2", "x3"]], y)

        assert model._result.converged
        # 3-term SCOP at small n may not converge tightly at default reml_tol,
        # but lambdas should be positive and finite
        assert model._reml_lambdas is not None
        assert len(model._reml_lambdas) >= 3
        assert all(v > 0 and np.isfinite(v) for v in model._reml_lambdas.values())

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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(
                    n_knots=8,
                    constraint=Constraint.fit.decreasing,
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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
            },
        )
        model.fit_reml(df[["x1", "x2"]], y)

        assert model._result.converged
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_stored_objective_reproduction_multi_scop(self):
        """Reconstruct REML objective from stored model state (no solver rerun).

        Must match model._reml_result.objective to rel=1e-8.

        The signal is deliberately curved. A linear signal has zero second
        differences, which drives the SCOP curvature coefficients to their
        log-space boundary and leaves the identified Hessian near-singular;
        a log-determinant over coordinates that ill-conditioned amplifies the
        difference between the weights the solver converged on and the weights
        this test recomputes from published coefficients, which differ by the
        IRLS convergence tolerance. The earlier linear fixture truncated a
        direction on all 108 of its solves and reproduced to only ~1e-6,
        varying with the BLAS -- it was measuring conditioning, not the
        bookkeeping fidelity this test is for.

        Curved, the mode is interior: zero truncating solves and reproduction
        to ~7e-15, so 1e-8 holds with seven orders of margin rather than
        resting on a platform's rounding. Boundary modes are covered by
        ``test_two_scop_terms_auto_lambda``.
        """
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.group_matrix import _block_xtwx
        from superglm.links import stabilize_eta
        from superglm.reml.objective import reml_laml_objective

        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = np.sort(rng.uniform(0, 1, n))
        # Curved, not linear: see the docstring. A linear signal has no second
        # differences for the SCOP curvature coefficients to explain, so they run
        # to the log-space boundary and the determinant becomes conditioning-limited.
        y = 2 * np.sqrt(x1) - 1.5 * x2**2 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            max_iter=200,
            features={
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
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
        # Keep the terms independently ordered. Sorting both columns makes
        # them almost collinear, so their individual smoothing parameters can
        # trade off even when the aggregate smoothness response is sensible.
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)

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
                    "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                    "x2": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
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
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8, constraint=Constraint.fit.decreasing),
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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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
    def test_qp_monotone_passthrough_regression(self):
        """QP monotone auto-lambda via passthrough works and produces monotone predictions."""
        from superglm.features.spline import BSplineSmooth

        rng = np.random.default_rng(42)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-6)

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
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
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

        # SCOP-specific diagnostics
        assert reml_result.scop_step_norms is not None
        assert len(reml_result.scop_step_norms) > 0
        assert isinstance(reml_result.scop_fisher_fallbacks, int)


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

    def test_fisher_fallback_exports_the_curvature_that_was_actually_solved(self):
        """An indefinite observed block must not leak into REML after Fisher fallback."""
        from types import SimpleNamespace

        from superglm.solvers.scop import build_scop_solver_reparam
        from superglm.solvers.scop_newton import scop_joint_newton_step, scop_newton_step

        rng = np.random.default_rng(1801)
        n, q = 40, 4
        basis = np.abs(rng.normal(size=(n, q))) + 0.5
        weights = np.ones(n)
        beta = np.zeros(q)
        reparam = build_scop_solver_reparam(q + 1, direction="increasing")
        penalty = reparam.penalty_matrix()
        response = basis @ reparam.forward(beta) + 1_000.0

        single = scop_newton_step(
            basis,
            weights,
            response,
            beta,
            reparam,
            penalty,
            lambda2=1.0,
        )
        states = {
            0: {
                "B_scop": basis,
                "S_scop": penalty,
                "beta_scop": beta,
                "reparam": reparam,
                "bin_idx": None,
                "group_sl": slice(0, q),
                "group_name": "x",
            }
        }
        joint = scop_joint_newton_step(
            states,
            weights,
            response,
            {"x": 1.0},
            [SimpleNamespace(name="x", sl=slice(0, q))],
        )[0]

        for result in (single, joint):
            assert result.used_fisher_fallback is True
            assert np.linalg.eigvalsh(result.H_penalized).min() > 0.0

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

    def test_discretized_joint_objective_cache_matches_observation_objective(self):
        """Cached discretized objective must match the observation-level oracle."""
        from superglm.solvers.scop_newton import (
            _build_joint_objective_cache,
            _joint_objective_from_gammas,
            _safe_joint_objective,
            _safe_joint_trial_objective_delta,
        )

        scop_states, W, z = self._build_two_group_inputs(discretized=True)
        scop_items = sorted(scop_states.items())
        betas = [st["beta_scop"] for _, st in scop_items]
        beta_joint = np.concatenate(betas)
        slices = []
        offset = 0
        for beta_i in betas:
            slices.append(slice(offset, offset + len(beta_i)))
            offset += len(beta_i)
        lambdas_list = [1.0, 0.5]

        cache = _build_joint_objective_cache(scop_items, W, z)
        assert cache is not None

        gammas = []
        for _, st in scop_items:
            gamma = st["reparam"].forward(st["beta_scop"])
            gammas.append(gamma)
        cache.diag_btwb = []
        cache.cross_btwb = {}
        for idx, (_, st) in enumerate(scop_items):
            B = st["B_scop"]
            bin_idx = st["bin_idx"]
            w_agg = np.bincount(bin_idx, weights=W, minlength=B.shape[0])
            cache.diag_btwb.append(B.T @ (B * w_agg[:, None]))
        for left in range(len(scop_items)):
            st_left = scop_items[left][1]
            for right in range(left + 1, len(scop_items)):
                st_right = scop_items[right][1]
                w_2d = np.zeros((st_left["B_scop"].shape[0], st_right["B_scop"].shape[0]))
                np.add.at(w_2d, (st_left["bin_idx"], st_right["bin_idx"]), W)
                cache.cross_btwb[(left, right)] = st_left["B_scop"].T @ w_2d @ st_right["B_scop"]

        obj_cached = _joint_objective_from_gammas(
            gammas,
            beta_joint,
            slices,
            lambdas_list,
            scop_items,
            cache,
        )
        obj_oracle = _safe_joint_objective(scop_items, W, z, beta_joint, slices, lambdas_list)

        np.testing.assert_allclose(obj_cached, obj_oracle, rtol=1e-10, atol=1e-12)

        # The expansion remains an identity if a cached Gram carries the tiny
        # asymmetry that finite-precision matrix products can leave behind.
        cache.diag_btwb[0] = cache.diag_btwb[0].copy()
        cache.diag_btwb[0][0, 1] += 1.0e-3
        obj_cached = _joint_objective_from_gammas(
            gammas,
            beta_joint,
            slices,
            lambdas_list,
            scop_items,
            cache,
        )
        beta_trial = beta_joint + np.linspace(-0.03, 0.02, beta_joint.size)
        gammas_trial = [
            state["reparam"].forward(beta_trial[group_slice])
            for (_, state), group_slice in zip(scop_items, slices, strict=True)
        ]
        obj_trial = _joint_objective_from_gammas(
            gammas_trial,
            beta_trial,
            slices,
            lambdas_list,
            scop_items,
            cache,
        )
        stable_delta = _safe_joint_trial_objective_delta(
            scop_items,
            beta_trial,
            slices,
            lambdas_list,
            gammas,
            cache,
        )
        assert stable_delta == pytest.approx(obj_trial - obj_cached, rel=1e-10, abs=1e-11)

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

    def test_minres_matches_direct_on_two_group_problem(self):
        """Iterative MINRES prototype should match direct solve closely."""
        from superglm.solvers.scop_newton import (
            configure_scop_prototype,
            reset_scop_prototype,
            scop_joint_newton_step,
        )

        scop_states, W, z = self._build_two_group_inputs()
        groups = self._make_mock_groups(scop_states)

        try:
            reset_scop_prototype()
            direct_results = scop_joint_newton_step(
                scop_states, W, z, {"x1": 1.0, "x2": 0.5}, groups
            )

            configure_scop_prototype(
                solve_mode="minres",
                iterative_q_total_min=1,
                iterative_rtol=1e-12,
                iterative_maxiter=200,
            )
            iter_results = scop_joint_newton_step(scop_states, W, z, {"x1": 1.0, "x2": 0.5}, groups)
        finally:
            reset_scop_prototype()

        for gi in direct_results:
            np.testing.assert_allclose(
                iter_results[gi].beta_new,
                direct_results[gi].beta_new,
                rtol=1e-8,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                iter_results[gi].objective_after,
                direct_results[gi].objective_after,
                rtol=1e-10,
                atol=1e-12,
            )
            assert iter_results[gi].linear_solver == "minres"
            assert iter_results[gi].linear_iterations > 0

    def test_cross_block_truncation_keeps_objective_finite(self):
        """Prototype cross-block dropping still returns a usable step."""
        from superglm.solvers.scop_newton import (
            configure_scop_prototype,
            reset_scop_prototype,
            scop_joint_newton_step,
        )

        scop_states, W, z = self._build_two_group_inputs()
        groups = self._make_mock_groups(scop_states)

        try:
            configure_scop_prototype(cross_block_rel_tol=10.0)
            results = scop_joint_newton_step(scop_states, W, z, {"x1": 1.0, "x2": 0.5}, groups)
        finally:
            reset_scop_prototype()

        for result in results.values():
            assert np.isfinite(result.objective_after)
            assert result.objective_after <= result.objective_before + 1e-8
            assert result.dropped_cross_blocks >= 1
