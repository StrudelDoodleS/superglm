"""Tests for SCOP EFS infrastructure.

Part 1: Tests for SCOP state returned from fit_irls_direct.
Part 2: Tests for build_scop_penalty_components.
"""

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import superglm.reml.scop_efs as scop_efs_module
from superglm import Constraint, Numeric, SuperGLM
from superglm.factor_smooth_geometry import sum_to_zero_contrast
from superglm.families import Gaussian, Poisson
from superglm.features.spline import PSpline
from superglm.inference.covariance import _active_penalty_matrix
from superglm.model.base import model_build_design_matrix
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    penalty_component_dense_matrix,
)
from superglm.reml.scop_efs import (
    assemble_joint_hessian,
    build_scop_penalty_components,
    compute_scop_aware_penalty_quad,
    scop_efs_lambda_update,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import GroupSlice, PenaltyComponent


def _curvature_solver_reparam(q_raw: int, kind: str):
    from superglm.solvers.scop import build_scop_solver_reparam

    degree = 3
    n_interior = q_raw - degree - 1
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.linspace(0.0, 1.0, n_interior + 2)[1:-1] ** 1.7,
            np.ones(degree + 1),
        )
    )
    return build_scop_solver_reparam(
        q_raw,
        kind=kind,
        knots=knots,
        degree=degree,
        domain=(0.0, 1.0),
    )


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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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

    def test_curvature_cross_blocks_use_mixed_identity_exp_jacobian(self):
        """EFS must not scale the retained affine slope as though it were exp-mapped."""
        rng = np.random.default_rng(216)
        reparam = _curvature_solver_reparam(7, "convex")
        p_linear = 3
        scop_slice = slice(p_linear, p_linear + reparam.q)
        linear_slice = slice(0, p_linear)
        raw_factor = rng.standard_normal((20, scop_slice.stop))
        raw_hessian = raw_factor.T @ raw_factor + np.eye(scop_slice.stop)
        beta_eff = np.linspace(-0.6, 0.7, reparam.q)
        retained_block = raw_hessian[scop_slice, scop_slice] + 2.0 * np.eye(reparam.q)
        states = {
            0: {
                "group_sl": scop_slice,
                "H_scop_penalized": retained_block,
                "group_name": "convex_x",
                "beta_eff": beta_eff,
                "reparam": reparam,
            }
        }

        actual, _ = assemble_joint_hessian(raw_hessian, states)
        jacobian = reparam.jacobian_diagonal(beta_eff)
        expected_cross = raw_hessian[linear_slice, scop_slice] * jacobian[None, :]

        np.testing.assert_allclose(
            actual[linear_slice, scop_slice],
            expected_cross,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_array_equal(
            actual[linear_slice, scop_slice.start],
            raw_hessian[linear_slice, scop_slice.start],
        )
        assert not np.allclose(
            actual[linear_slice, slice(scop_slice.start + 1, scop_slice.stop)],
            raw_hessian[linear_slice, slice(scop_slice.start + 1, scop_slice.stop)],
        )

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


class TestSCOPPenaltyQuadCompactComponents:
    """A matched component must be contracted through its own ``penalty_kind``.

    ``compute_scop_aware_penalty_quad`` reads ``omega_ssp``/``omega_raw`` off
    each matched ``PenaltyComponent``.  That array is the *compact* penalty:
    only ``penalty_kind="dense"`` stores a full group-width block.  ``identity``
    stores nothing, ``repeated`` one diagonal block, and ``sum_to_zero`` a
    raw-level block that means nothing without the ``[I; -1]`` contrast.

    No public API builds such a component for a SCOP group today --
    ``monotone_engine="scop"`` is stamped only by the spline builder, and a
    spline group always yields a dense penalty -- so these are unit-level pins
    on the contract, constructed directly, not end-to-end fits.  Their job is to
    stop the trap being reopened, in particular the two-level ``sum_to_zero``
    case, which is shape-compatible at the group width and therefore silently
    off by a factor of two rather than raising.
    """

    @staticmethod
    def _quad(component, beta_eff, lam, *, group_width):
        """Run one matched component through the function under test.

        The SCOP group spans the whole coefficient vector, so the mapped-space
        term the function subtracts exactly cancels the one it starts from and
        the return value is the matched component's contribution alone.
        """
        rng = np.random.default_rng(2024)
        gamma_eff = np.exp(rng.standard_normal(group_width))
        dense = np.asarray(
            penalty_component_dense_matrix(component),
            dtype=np.float64,
        )
        states = {
            0: {
                "group_name": "mono",
                "group_sl": slice(0, group_width),
                "beta_eff": beta_eff,
                "S_scop": dense,
            }
        }
        return compute_scop_aware_penalty_quad(
            gamma_eff,
            lam * dense,
            states,
            {component.name: lam},
            reml_penalties=[component],
        )

    def test_two_level_sum_to_zero_is_not_silently_halved(self):
        """The local block is group-wide at two levels: wrong strength, no error."""
        block_width = 3
        n_levels = 2
        lam = 1.75
        rng = np.random.default_rng(5150)
        local = _first_diff_penalty(block_width)
        beta_eff = rng.standard_normal((n_levels - 1) * block_width)

        component = PenaltyComponent(
            name="mono:wiggle",
            group_name="mono",
            group_index=0,
            group_sl=slice(0, (n_levels - 1) * block_width),
            omega_raw=local,
            omega_ssp=local,
            rank=float((n_levels - 1) * np.linalg.matrix_rank(local)),
            penalty_kind="sum_to_zero",
            repeat_count=n_levels,
            block_width=block_width,
        )

        actual = self._quad(
            component,
            beta_eff,
            lam,
            group_width=(n_levels - 1) * block_width,
        )

        # Truth from the parameterisation itself: the free blocks are lifted to
        # raw level blocks by the contrast, and each raw block is penalized.
        raw = sum_to_zero_contrast(n_levels) @ beta_eff.reshape(n_levels - 1, block_width)
        expected = lam * float(sum(row @ local @ row for row in raw))
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)

        # The discarded reading contracts the local block against the free
        # coefficients directly.  It is shape-compatible here, which is exactly
        # why it never announced itself.
        naive = lam * float(beta_eff @ local @ beta_eff)
        assert actual == pytest.approx(2.0 * naive, rel=1e-13)
        assert not np.isclose(actual, naive, rtol=1e-6)

    def test_three_level_sum_to_zero_uses_the_contrast(self):
        """Above two levels the naive reading is not even shape-compatible."""
        block_width = 2
        n_levels = 3
        lam = 0.6
        rng = np.random.default_rng(4242)
        local = _first_diff_penalty(block_width)
        beta_eff = rng.standard_normal((n_levels - 1) * block_width)

        component = PenaltyComponent(
            name="mono:wiggle",
            group_name="mono",
            group_index=0,
            group_sl=slice(0, (n_levels - 1) * block_width),
            omega_raw=local,
            omega_ssp=local,
            rank=float((n_levels - 1) * np.linalg.matrix_rank(local)),
            penalty_kind="sum_to_zero",
            repeat_count=n_levels,
            block_width=block_width,
        )

        actual = self._quad(
            component,
            beta_eff,
            lam,
            group_width=(n_levels - 1) * block_width,
        )

        raw = sum_to_zero_contrast(n_levels) @ beta_eff.reshape(n_levels - 1, block_width)
        expected = lam * float(sum(row @ local @ row for row in raw))
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)

    def test_repeated_component_penalizes_every_level_block(self):
        """A repeated penalty is ``I_repeat kron local``, not ``local``."""
        block_width = 3
        repeat_count = 4
        lam = 2.25
        rng = np.random.default_rng(31337)
        local = _first_diff_penalty(block_width)
        beta_eff = rng.standard_normal(repeat_count * block_width)

        component = PenaltyComponent(
            name="mono:wiggle",
            group_name="mono",
            group_index=0,
            group_sl=slice(0, repeat_count * block_width),
            omega_raw=local,
            omega_ssp=local,
            rank=float(repeat_count * np.linalg.matrix_rank(local)),
            penalty_kind="repeated",
            repeat_count=repeat_count,
            block_width=block_width,
        )

        actual = self._quad(
            component,
            beta_eff,
            lam,
            group_width=repeat_count * block_width,
        )

        blocks = beta_eff.reshape(repeat_count, block_width)
        expected = lam * float(sum(row @ local @ row for row in blocks))
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)

    def test_identity_component_carries_no_matrix(self):
        """An implicit identity penalty stores no array at all."""
        width = 5
        lam = 3.5
        rng = np.random.default_rng(8128)
        beta_eff = rng.standard_normal(width)

        component = PenaltyComponent(
            name="mono:re",
            group_name="mono",
            group_index=0,
            group_sl=slice(0, width),
            omega_raw=None,
            omega_ssp=None,
            rank=float(width),
            penalty_kind="identity",
        )

        actual = self._quad(component, beta_eff, lam, group_width=width)

        expected = lam * float(beta_eff @ beta_eff)
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)

    def test_dense_component_reading_is_unchanged(self):
        """The kind that already worked must keep its exact value."""
        width = 4
        lam = 1.25
        rng = np.random.default_rng(1848)
        omega = _first_diff_penalty(width)
        beta_eff = rng.standard_normal(width)

        component = PenaltyComponent(
            name="mono:wiggle",
            group_name="mono",
            group_index=0,
            group_sl=slice(0, width),
            omega_raw=omega,
            omega_ssp=omega,
            rank=float(np.linalg.matrix_rank(omega)),
        )

        actual = self._quad(component, beta_eff, lam, group_width=width)

        expected = lam * float(beta_eff @ omega @ beta_eff)
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)

    def test_dense_component_without_a_solver_block_is_refused_by_name(self):
        """``omega_ssp=None`` is a missing conversion, not a licence to guess.

        ``omega_raw`` is the penalty on the group's RAW coordinates; the
        solver-space block is ``R_inv.T @ omega_raw @ R_inv``.  Reading
        ``omega_raw`` in place of ``omega_ssp`` is that conversion with
        ``R_inv`` assumed to be the identity -- shape-compatible at the group
        width whatever the reparametrisation, so a wrong penalty strength
        would be returned rather than raised.  ``compute_scop_aware_penalty_quad``
        matches components without their group matrices, so it cannot perform
        the conversion, and refusing names the component that needs one.
        """
        width = 4
        lam = 0.8
        rng = np.random.default_rng(1914)
        omega = _first_diff_penalty(width)
        beta_eff = rng.standard_normal(width)

        component = PenaltyComponent(
            name="mono:wiggle",
            group_name="mono",
            group_index=0,
            group_sl=slice(0, width),
            omega_raw=omega,
            omega_ssp=None,
            rank=float(np.linalg.matrix_rank(omega)),
        )

        rng_state = np.random.default_rng(2024)
        gamma_eff = np.exp(rng_state.standard_normal(width))
        states = {
            0: {
                "group_name": "mono",
                "group_sl": slice(0, width),
                "beta_eff": beta_eff,
                "S_scop": omega,
            }
        }
        with pytest.raises(ValueError, match="mono:wiggle"):
            compute_scop_aware_penalty_quad(
                gamma_eff,
                lam * omega,
                states,
                {component.name: lam},
                reml_penalties=[component],
            )

        # The same component with its solver-space block filled is accepted,
        # so the refusal is about the missing conversion and nothing else.
        resolved = replace(component, omega_ssp=omega)
        assert compute_scop_aware_penalty_quad(
            gamma_eff,
            lam * omega,
            states,
            {resolved.name: lam},
            reml_penalties=[resolved],
        ) == pytest.approx(lam * float(beta_eff @ omega @ beta_eff), rel=1e-13, abs=1e-13)


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
            weight_semantics="frequency",
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
        actual = reml_laml_objective(**common, weight_semantics="frequency")
        expected = reml_laml_objective(
            **common,
            log_det_H=expected_logdet,
            hessian_rank=1 + decomposition.rank,
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
        assembled_objective = reml_laml_objective(**common, weight_semantics="frequency")
        explicit_objective = reml_laml_objective(
            **common,
            S_override=expected_penalty,
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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

    def test_an_unscoreable_scop_mode_is_infeasible_not_fatal(self, monkeypatch):
        """A SCOP score refusal must reach a power search as an infeasible point.

        ``scop_penalized_mode_score`` refuses a non-finite row score, and that
        quotient -- a residual times a derivative over a floored variance -- is
        not bounded by anything an accepted geometry certifies. It used to be a
        bare ``ValueError``, which sails past every
        ``except ObservedModeNotCertifiedError`` handler because that family is
        RuntimeError-derived, so it killed the fit rather than costing it one
        point. Both call sites now retype it.

        A first start whose mode cannot be scored is retried at the
        Hessian-scaled start, like one with no certified mode; here every
        score is refused, so the retry's refusal is raised, an infeasible
        point to a power search.  Mutation check: a19d2fe4 raised at the
        first start without a retry.
        """
        import superglm.reml.scop_efs as scop_efs_module
        from superglm.reml.observed_geometry import (
            ObservedGeometryInfeasibleError,
            ObservedModeNotConvergedError,
        )

        def refuse(**kwargs):
            raise ObservedGeometryInfeasibleError("SCOP penalized mode score is not finite")

        monkeypatch.setattr(scop_efs_module, "scop_penalized_mode_score", refuse)
        starts = []
        real_attempt = scop_efs_module._bootstrap_attempt

        def attempt(context, lambdas, last_fit):
            starts.append(dict(lambdas))
            return real_attempt(context, lambdas, last_fit)

        monkeypatch.setattr(scop_efs_module, "_bootstrap_attempt", attempt)

        rng = np.random.default_rng(20260818)
        n = 100
        x = np.sort(rng.uniform(0.0, 1.0, size=n))
        z = rng.normal(size=n)
        y = 0.2 + 0.6 * z + x + rng.normal(scale=0.04, size=n)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            discrete=True,
            features={
                "z": Numeric(),
                "x": PSpline(n_knots=6, constraint=Constraint.fit.increasing),
            },
        )

        with pytest.raises(ObservedModeNotConvergedError) as excinfo:
            model.fit_reml(pd.DataFrame({"z": z, "x": x}), y, max_reml_iter=2, max_pirls_iter=100)

        assert isinstance(excinfo.value.__cause__, ObservedGeometryInfeasibleError)
        assert len(starts) == 2 and starts[1] != starts[0]

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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
                weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
    def _stub(n_iter, termination_reason="max_iter"):
        """A non-converged solver result that exhausted its budget."""
        return SimpleNamespace(
            converged=False,
            termination_reason=termination_reason,
            beta=np.array([0.0]),
            intercept=0.0,
            rank_info=None,
            n_iter=n_iter,
        )

    def test_a_budget_exhausted_rejection_names_the_round_off_floor(self, monkeypatch, caplog):
        """The refusal stands, and says which kind of refusal it is.

        Under observed curvature the inner tolerance is a FIXED 1e-10 ceiling,
        not scaled to the problem, so a step-length test cannot fire when the
        iteration's round-off floor sits above it -- the ordinary REML path was
        measured missing that same ceiling by 9x to 646x with the mode in fact
        reached. No shape-constrained fit reaching it has been produced (922
        inner fits at this tolerance, none exhausted), so the gate above is
        deliberately unchanged. But a floor-limited refusal and a fit that
        genuinely will not settle are indistinguishable from the message alone,
        which is what this pins: the distinction is stated, not left to be
        rediscovered.
        """
        with caplog.at_level(logging.WARNING, logger="superglm.reml.scop_efs"):
            assert self._run_gate(monkeypatch, self._stub(self.LONG_RUN)) is None
        assert "round-off floor" in caplog.text
        assert "step-length test" in caplog.text

    def test_only_budget_exhaustion_gets_that_note(self, monkeypatch, caplog):
        """A rejected step is a different failure and must not borrow the wording."""
        with caplog.at_level(logging.WARNING, logger="superglm.reml.scop_efs"):
            assert self._run_gate(monkeypatch, self._stub(self.SHORT_RUN, "step_rejected")) is None
        assert "round-off floor" not in caplog.text

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

    def test_a_genuinely_non_converging_fit_is_published_unconverged(self, monkeypatch):
        """A real SCOP fit that never settles is reported as not converged.

        This quasi-separated Poisson exhausts all its PIRLS iterations with
        zero halvings and zero rejections, its deviance still moving by ~1e-3
        relative per iteration, at the cold bootstrap and again at its
        Hessian-scaled retry. Neither is accepted as a mode; the retry's fit
        is published with ``converged=False`` (owner decision 3, 2026-09-30:
        disclose, never refuse; 37f73863 raised here).

        The outcome alone cannot pin which failure fired: an inner fit that
        does not converge, or a converged mode that fails latent
        certification. A non-converged result returns at ``require_converged
        and not result.converged`` before any certification is computed, so
        both bootstrap attempts leave ``_scop_mode_newton_relative`` uncalled,
        and the publication evaluates the retry's last fit without certifying
        it. Zero calls therefore distinguishes the two, and keeps a future
        change that makes this fit converge from leaving the test silently
        green on the other failure. (2a4e28c7 fitted the retry's start a
        third time to publish it, and certified that refit once.)
        """
        from superglm import ConvergenceWarning

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
        with pytest.warns(ConvergenceWarning, match="max_pirls_iter"):
            model.fit_reml(frame, response, max_reml_iter=20)
        diagnostics = model.reml_diagnostics()
        assert not diagnostics["converged"]
        assert diagnostics["termination_reason"] == "bootstrap_uncertified"
        assert model._reml_result.terminal_refit_termination == "max_iter"
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


def _scop_frequency_fixture(diesel_share: float, seed: int = 3):
    """A small SCOP frequency fit with one binary factor.

    At ``diesel_share=0.48`` the non-base level ("Diesel", rate 1.57x) carries
    about 62% of the IRLS weight, so its 0/1 column fails the raw-centring
    check ``|weighted mean| <= centred RMS``; at 0.25 it carries about 37%
    and passes it.
    """
    from superglm import Categorical, Spline

    rng = np.random.default_rng(seed)
    n = 4000
    bm = rng.integers(50, 151, n).astype(float)
    age = rng.uniform(18, 90, n)
    gas = np.where(rng.uniform(size=n) < diesel_share, "Diesel", "Regular")
    exposure = rng.uniform(0.2, 1.0, n)
    eta = (
        -1.6
        + 0.012 * (bm - 50)
        + 0.4 * np.exp(-(age - 18) / 8)
        + np.where(gas == "Diesel", 0.45, 0.0)
    )
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    frame = pd.DataFrame({"BonusMalus": bm, "DrivAge": age, "VehGas": gas})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "BonusMalus": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
            "DrivAge": Spline(kind="ps", k=8),
            "VehGas": Categorical(base="most_exposed"),
        },
    )
    return model, frame, y, np.log(exposure)


def _record_certification_retries(monkeypatch):
    """Record (rung, warm-started, inner iterations) for every SCOP mode fit.

    Each mode fit makes its own inner fit before any retry it recurses into,
    so the first inner fit after entry is its own.
    """
    records: list[tuple[int, bool, int]] = []
    inner_iterations: list[int] = []
    real_fit = scop_efs_module._fit_scop_reml_mode
    real_irls = scop_efs_module.fit_irls_direct

    def counting_irls(*args, **kwargs):
        out = real_irls(*args, **kwargs)
        inner_iterations.append(int(out[0].n_iter))
        return out

    def recording_fit(context, lambdas, **kwargs):
        own = len(inner_iterations)
        mode = real_fit(context, lambdas, **kwargs)
        records.append(
            (
                int(kwargs.get("_certification_retry", 0)),
                kwargs.get("beta_init") is not None,
                inner_iterations[own],
            )
        )
        return mode

    monkeypatch.setattr(scop_efs_module, "fit_irls_direct", counting_irls)
    monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", recording_fit)
    return records


class TestCertificationRetryStart:
    """A certification retry starts where the certificate says the mode is.

    The SCOP inner solve is block-coordinate (an ordinary-block solve, then a
    SCOP Newton step), so it converges linearly: a mode that met its
    coefficient-step test but failed the sqrt(rank * eps) certificate took
    about ten more inner iterations, each a full Gram build, to pass it at
    1e-10 (157 inner iterations for 15 retries on the 678k-row freMTPL2 fit).
    The certificate has already computed the joint Newton step over every
    coefficient, so rung 1 starts from it. A loose ``pirls_tol`` makes every
    candidate fail the certificate, so retries happen on any platform.
    """

    def test_a_retry_starts_from_the_certificate_newton_step(self, monkeypatch):
        """Rung-1 retries converge in about one inner iteration.

        Mutation check: from the plain warm start (master at 155832e8) the same
        nine retries took 5-8 inner iterations each, 59 in all; started from
        the Newton step they take 1-3, 14 in all.
        """
        model, frame, y, offset = _scop_frequency_fixture(diesel_share=0.25)
        records = _record_certification_retries(monkeypatch)
        model.fit_reml(frame, y, offset=offset, max_reml_iter=100, pirls_tol=1e-3)

        retries = [record for record in records if record[0] == 1]
        assert len(retries) >= 3, "a loose inner tolerance must exercise the retry ladder"
        assert all(warm for _, warm, _ in retries)
        assert sum(inner for _, _, inner in retries) <= 2 * len(retries)
        assert model.reml_diagnostics()["converged"]

    def test_a_retry_stays_warm_when_raw_centring_is_ill_scaled(self, monkeypatch):
        """An ill-scaled raw centring no longer sends the retry back to a cold start.

        Mutation check: with the old ``_raw_centering_well_scaled`` gate every
        retry of this fixture started cold (14 inner iterations for the one
        natural retry at the default tolerance); on the cleaned 678k-row book
        20 cold retries took 808 inner iterations against 20 warm ones.
        """
        model, frame, y, offset = _scop_frequency_fixture(diesel_share=0.48)
        records = _record_certification_retries(monkeypatch)
        model.fit_reml(frame, y, offset=offset, max_reml_iter=100, pirls_tol=1e-3)

        retries = [record for record in records if record[0] in (1, 2)]
        assert retries, "a loose inner tolerance must exercise the retry ladder"
        assert all(warm for _, warm, _ in retries), retries
        rung_one = [inner for rung, _, inner in retries if rung == 1]
        assert sum(rung_one) <= 2 * len(rung_one)
        assert model.reml_diagnostics()["converged"]

    def test_a_newton_step_past_the_exp_clip_keeps_the_plain_warm_start(self):
        """A polished start the forward map cannot represent is refused.

        The forward map clips its exponent at 500, so a latent step beyond it
        lands on a different point whose linear predictor sits at the link's
        overflow guard. On the freMTPL2 Tweedie book the cold bootstrap's
        certificate asked for a latent step of 8.0e4; started there, the retry
        could not take a single inner step and reported separation. Mutation
        check: 37f73863 returned that start. The clip binds at -500 as well,
        where the clipped point is not the Newton point either; 2a4e28c7
        guarded only the upper end and returned the start at -501.
        """
        from superglm.solvers.scop import build_scop_solver_reparam

        reparam = build_scop_solver_reparam(6, kind="increasing")
        state = {"group_sl": slice(1, 6), "reparam": reparam, "beta_eff": np.full(5, -11.0)}
        mode = SimpleNamespace(result=SimpleNamespace(intercept=1.0), scop_states={0: state})
        latent = np.concatenate(([0.2], state["beta_eff"]))

        def correction(step: float):
            slope = np.zeros(6)
            slope[1] = step
            return scop_efs_module._SCOPModeNewtonCorrection(
                latent_beta=latent, slope=slope, intercept=0.0, relative=1.0
            )

        start = scop_efs_module._newton_polished_warm_start(mode, correction(510.0))
        assert start is not None and start[2][0]["beta_eff"][0] == 499.0
        assert scop_efs_module._newton_polished_warm_start(mode, correction(512.0)) is None
        assert scop_efs_module._newton_polished_warm_start(mode, correction(8.0e4)) is None
        start = scop_efs_module._newton_polished_warm_start(mode, correction(-488.0))
        assert start is not None and start[2][0]["beta_eff"][0] == -499.0
        assert scop_efs_module._newton_polished_warm_start(mode, correction(-490.0)) is None
        assert scop_efs_module._newton_polished_warm_start(mode, correction(-8.0e4)) is None


def _tweedie_pure_premium_fixture(n: int, seed: int):
    """A small Tweedie (p=1.5) pure-premium book shaped like freMTPL2.

    About 95% of rows have no claim; BonusMalus holds 60% of the rows at 50
    with a risk effect flat to 70, under an increasing constraint; and the
    DrivAge x VehAge tensor has sparse cells with no claims. At the cold
    bootstrap's 1e-4 every smooth is almost unpenalized and the inner fit
    never reaches a certified mode.
    """
    from superglm import Categorical, Spline, Tweedie

    rng = np.random.default_rng(seed)
    age = np.clip(18 + rng.gamma(4.0, 6.0, n), 18, 95)
    veh = np.clip(rng.exponential(6.0, n), 0, 30).round()
    bm = np.where(
        rng.uniform(size=n) < 0.6, 50.0, np.clip(50 + rng.exponential(18.0, n), 50, 150).round()
    )
    dens = np.clip(rng.normal(6.0, 2.0, n), 0.0, 10.5)
    shares = np.array([20, 15, 12, 10, 10, 9, 8, 7, 5, 4]) / 100
    region = rng.choice(list("ABCDEFGHIJ"), size=n, p=shares)
    exposure = rng.uniform(0.05, 1.0, n)
    region_effect = dict(zip("ABCDEFGHIJ", rng.normal(0, 0.15, 10), strict=True))
    eta = (
        np.log(0.1)
        + np.where(bm <= 70, 0.0, 0.02 * (bm - 70))
        + 0.6 * np.exp(-(age - 18) / 6)
        + 0.2 * ((age - 50) / 30) ** 2
        - 0.03 * veh
        + 0.05 * (dens - 6)
        + np.array([region_effect[r] for r in region])
    )
    counts = rng.poisson(exposure * np.exp(eta))
    amount = np.array([rng.gamma(0.8, 1500.0 / 0.8, c).sum() if c else 0.0 for c in counts])
    frame = pd.DataFrame(
        {"DrivAge": age, "VehAge": veh, "BonusMalus": bm, "LogDensity": dens, "Region": region}
    )
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        discrete=True,
        features={
            "DrivAge": Spline(kind="ps", k=12),
            "VehAge": Spline(kind="ps", k=10),
            "BonusMalus": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
            "LogDensity": Spline(kind="ps", k=8),
            "Region": Categorical(base="most_exposed"),
        },
        interactions=[("DrivAge", "VehAge")],
    )
    return model, frame, np.minimum(amount, 50000.0) / exposure, exposure


def _three_kind_fixture(family="tweedie", n: int = 240, seed: int = 11):
    """An ordinary spline, a SCOP block and a two-margin tensor, with prior weights."""
    from superglm import Spline, Tweedie

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.uniform(0.0, 1.0, n) for name in ("s", "m", "a", "b")})
    mean = np.exp(0.5 + np.sin(3.0 * frame["s"]) + frame["m"] + 0.5 * frame["a"] * frame["b"])
    y = rng.poisson(mean).astype(float)
    if family == "tweedie":
        family = Tweedie(p=1.5)
        y = y * rng.gamma(2.0, 0.5, n)
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        discrete=True,
        features={
            "s": Spline(kind="ps", k=8),
            "m": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing),
            "a": Spline(kind="ps", k=6),
            "b": Spline(kind="ps", k=6),
        },
        interactions=[("a", "b")],
    )
    return model, frame, y, rng.uniform(0.2, 2.0, n)


class _BootstrapCapturedError(Exception):
    """Stops a fit at its first bootstrap start, carrying the fit context."""

    def __init__(self, context, lambdas):
        super().__init__("bootstrap captured")
        self.context = context
        self.lambdas = lambdas


def _bootstrap_context(monkeypatch, model, frame, y, **fit_kwargs):
    """The SCOP REML fit context and first bootstrap start, before any inner fit."""

    def capture(context, lambdas, **kwargs):
        raise _BootstrapCapturedError(context, dict(lambdas))

    with monkeypatch.context() as patch:
        patch.setattr(scop_efs_module, "_fit_scop_reml_mode", capture)
        with pytest.raises(_BootstrapCapturedError) as info:
            model.fit_reml(frame, y, **fit_kwargs)
    return info.value.context, info.value.lambdas


def _scaled_start_oracle(context, lambdas):
    """``{name: (sum A_ii / sum S_j,ii, kappa, m)}`` in exact arithmetic, and the row count.

    ``A`` is the two-pass centred Fisher Gram at the initial mean (no
    offset), from the dense design; ``S_j`` the dense penalty with only
    ``name`` at one; ``kappa = sum g_ii / sum A_ii``, ``g = diag(X' W X)``;
    ``m`` the support's size.
    """
    from fractions import Fraction

    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.solvers.working_rows import coefficient_initial_intercept, fisher_working_weights

    assert not np.any(context.offset_arr)
    weights = np.asarray(context.sample_weight, dtype=np.float64)
    intercept = coefficient_initial_intercept(
        distribution=context.distribution, link=context.link, y=context.y, sample_weight=weights
    )
    eta = stabilize_eta(intercept + context.offset_arr, context.link)
    mu = clip_mu(context.link.inverse(eta), context.distribution)
    fisher = fisher_working_weights(
        distribution=context.distribution, link=context.link, mu=mu, eta=eta, sample_weight=weights
    )
    design = context.dm.toarray()
    w = [Fraction(float(value)) for value in fisher]
    sum_w = sum(w)
    moments: dict[int, tuple[Fraction, Fraction]] = {}

    def column(j: int) -> tuple[Fraction, Fraction]:
        if j not in moments:
            x = [Fraction(float(value)) for value in design[:, j]]
            mean = sum(wi * xi for wi, xi in zip(w, x, strict=True)) / sum_w
            centred = sum(wi * (xi - mean) ** 2 for wi, xi in zip(w, x, strict=True))
            raw = sum(wi * xi * xi for wi, xi in zip(w, x, strict=True))
            moments[j] = (centred, raw)
        return moments[j]

    unit = dict.fromkeys(lambdas, 0.0)
    oracle = {}
    for name in lambdas:
        diag = np.diagonal(
            build_penalty_matrix(
                list(context.dm.group_matrices),
                context.groups,
                {**unit, name: 1.0},
                context.dm.p,
                reml_penalties=context.reml_penalties,
            )
        )
        support = np.flatnonzero(diag > 0.0)
        centred = sum(column(int(j))[0] for j in support)
        raw = sum(column(int(j))[1] for j in support)
        penalty = sum(Fraction(float(diag[j])) for j in support)
        oracle[name] = (float(centred / penalty), float(raw / centred), len(support))
    return oracle, design.shape[0]


class TestColdBootstrapStart:
    """A bootstrap with no certified mode at the cold seeds restarts where it can.

    Wood, Pya and Saefken (2016, section 3.1) start the smoothing-parameter
    search where every smooth's effective degrees of freedom lie away from
    their extremes. The cold seed of 1e-4 puts them at their maximum, and an
    exp-reparameterised SCOP block can then stall short of its mode: the
    freMTPL2 Tweedie book stopped on the inner step test at a point whose
    certificate asked for a latent step of 8.0e4. The retry starts at the
    Hessian-scaled values of ``_hessian_scaled_bootstrap_lambdas``.
    """

    def test_a_failed_cold_bootstrap_restarts_at_hessian_scaled_lambdas(self, monkeypatch):
        """The fixture's cold bootstrap fails; the retry certifies and the search converges.

        Mutation check: 37f73863 raised ``ObservedModeNotConvergedError`` ("SCOP
        REML bootstrap did not converge to a coefficient mode") on this fit.

        The discarded cold start's warnings do not reach the caller: its
        SeparationWarning called the returned coefficients unusable, while
        the published fit converged with its linear predictor far from any
        overflow guard. Under an "error" filter for SeparationWarning the
        cold start no longer aborts the fit before its retry. Mutation check:
        2a4e28c7 raised the cold start's SeparationWarning here.
        """
        import warnings

        from superglm import SeparationWarning

        model, frame, y, exposure = _tweedie_pure_premium_fixture(n=10_000, seed=3)
        boots: list[tuple[dict[str, float], bool]] = []
        real = scop_efs_module._fit_scop_reml_mode

        def recording(context, lambdas, **kwargs):
            mode = real(context, lambdas, **kwargs)
            if kwargs.get("phase") == "bootstrap" and kwargs.get("_certification_retry", 0) == 0:
                boots.append((dict(lambdas), mode is not None))
            return mode

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", recording)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("error", SeparationWarning)
            model.fit_reml(frame, y, sample_weight=exposure, max_reml_iter=200)

        assert not [w for w in caught if issubclass(w.category, SeparationWarning)]
        assert [certified for _, certified in boots] == [False, True]
        cold, scaled = boots[0][0], boots[1][0]
        assert set(cold.values()) == {1e-4}
        assert all(scaled[name] > cold[name] for name in cold)
        diagnostics = model.reml_diagnostics()
        assert diagnostics["converged"]
        assert diagnostics["termination_reason"] == "lambda_tolerance"

    def test_the_scaled_start_is_sum_a_over_sum_s(self, monkeypatch):
        """Each start is ``sum A_ii / sum S_j,ii`` over ``S_j``'s support, clipped to [1e-6, 1e10].

        ``A`` is the intercept-centred Fisher Gram at the initial mean (a SCOP
        block's at Jacobian one: its design columns, at unit increments), on
        an ordinary spline, a SCOP block and a two-margin tensor. The oracle
        forms ``A_ii = sum W (x - xbar)**2`` and the ratio in exact rational
        arithmetic from the same float64 design, weights and penalty
        diagonal, so its one rounding is the final one.

        The production value forms ``A_ii`` from raw moments, ``g - sum_w
        m**2`` with ``g = sum W x**2``, by an unknown summation order (BLAS or
        binned kernels), and its error is relative to ``g``, not to ``A_ii``
        (Higham 2002, section 3.1): ``g``, ``sum W x`` and ``sum_w`` round
        within ``gamma_(2n+4)``, ``gamma_(2n+2)`` (of ``sum W |x|``) and
        ``gamma_n``; with ``(sum W |x|)**2 <= sum_w g`` the four further
        roundings of the mean, its square, the product and the difference
        leave ``|A_ii^ - A_ii| <= gamma_(10n+18) g_ii``.  The sums over the
        ``m`` supported coefficients and the divide add ``gamma_(3m+4)``, so
        the relative error is at most ``kappa gamma_(10n+18) + gamma_(3m+4)``,
        ``kappa = sum g_ii / sum A_ii``, doubled for the second-order cross
        terms.  Weights scaled by ``2**-60`` and ``2**60`` scale every ratio
        exactly, past each clip. Mutation check: a ratio of ``g`` instead of
        ``A`` (no centring: ``s`` off by 7e-4 relative against a bound of
        5e-13), or over every penalized coefficient instead of ``S_j``'s
        support (``s`` 2.7 times the oracle), fails the bound.
        """
        model, frame, y, weights = _three_kind_fixture()
        context, lambdas = _bootstrap_context(monkeypatch, model, frame, y, sample_weight=weights)
        names = set(lambdas)
        assert len(names) >= 5
        scaled = scop_efs_module._hessian_scaled_bootstrap_lambdas(context, lambdas, names)
        oracle, n = _scaled_start_oracle(context, lambdas)
        assert set(oracle) == names

        u = 2.0**-53

        def gamma(k: int) -> float:
            return k * u / (1.0 - k * u)

        for name, (ratio, kappa, m) in oracle.items():
            tol = 2.0 * (kappa * gamma(10 * n + 18) + gamma(3 * m + 4))
            assert abs(scaled[name] - ratio) <= tol * ratio, (name, scaled[name], ratio, tol)
            assert ratio * 2.0**-60 < 1e-6 and ratio * 2.0**60 > 1e10
        for factor, bound in ((2.0**-60, 1e-6), (2.0**60, 1e10)):
            rescaled = replace(context, sample_weight=context.sample_weight * factor)
            assert scop_efs_module._hessian_scaled_bootstrap_lambdas(
                rescaled, lambdas, names
            ) == dict.fromkeys(names, bound)

    def test_an_offset_shift_leaves_the_scaled_start_unchanged(self, monkeypatch):
        """A constant added to every offset is absorbed by the intercept, so the start ignores it.

        The Fisher weights are read at the intercept-only fit given the
        offset, relative to its largest value; with offsets on a 2**-20 grid
        a shift of 3 is exact, and the start is the same float64 value.
        Mutation check: 2a4e28c7 read them at the offset-free intercept plus
        the offset, so the shift scaled every Poisson start by e**3 and every
        Tweedie (p=1.5) start by e**1.5.
        """
        from superglm import Tweedie

        for family in ("poisson", Tweedie(p=1.5)):
            model, frame, y, weights = _three_kind_fixture(family=family)
            exposure = np.random.default_rng(5).uniform(0.05, 1.0, len(y))
            offset = np.round(np.log(exposure) * 2.0**20) / 2.0**20
            context, lambdas = _bootstrap_context(
                monkeypatch, model, frame, y, sample_weight=weights, offset=offset
            )
            assert np.array_equal(context.offset_arr, offset)
            names = set(lambdas)
            base = scop_efs_module._hessian_scaled_bootstrap_lambdas(context, lambdas, names)
            shifted = replace(context, offset_arr=context.offset_arr + 3.0)
            assert np.array_equal(shifted.offset_arr - 3.0, offset)
            moved = scop_efs_module._hessian_scaled_bootstrap_lambdas(shifted, lambdas, names)
            assert moved == base, family

    def test_the_scaled_start_keeps_no_dense_penalty_alive(self, monkeypatch):
        """Each component's p x p penalty is released once its diagonal is read.

        ``np.diag`` returns a view, which kept every component's dense
        penalty alive until the start was formed: ``8 m p**2`` bytes for
        ``m`` components. Mutation check: 2a4e28c7 held all earlier ones.
        """
        import weakref

        model, frame, y, weights = _three_kind_fixture()
        context, lambdas = _bootstrap_context(monkeypatch, model, frame, y, sample_weight=weights)
        made: list[weakref.ref] = []
        alive: list[int] = []
        real = scop_efs_module.build_penalty_matrix

        def tracking(*args, **kwargs):
            alive.append(sum(ref() is not None for ref in made))
            matrix = real(*args, **kwargs)
            made.append(weakref.ref(matrix))
            return matrix

        monkeypatch.setattr(scop_efs_module, "build_penalty_matrix", tracking)
        scop_efs_module._hessian_scaled_bootstrap_lambdas(context, lambdas, set(lambdas))
        assert len(made) == len(lambdas) >= 2
        assert alive == [0] * len(made)

    def test_a_bootstrap_with_no_observed_geometry_is_still_published(self, monkeypatch):
        """A last iterate with signed observed rows is published, not refused.

        Gamma with an identity link has observed-information rows
        ``w (2 y - mu) / mu**3``, negative wherever ``y < mu / 2``, and the
        SCOP moment kernels refuse signed rows. With ``max_pirls_iter=1``
        both bootstrap starts stop on their budget, and the retry's last
        iterate (finite coefficients) has such rows. It is published not
        converged, evaluated on the geometry the inner fit retained (the route
        a Fisher-curvature family always takes). Mutation check: 2a4e28c7
        raised ValueError ("signed observed-information rows are not
        supported") with no ConvergenceWarning and no fit.
        """
        from superglm import ConvergenceWarning
        from superglm.distributions import Gamma
        from superglm.links import IdentityLink

        x = np.linspace(0.0, 1.0, 100)
        frame = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=Gamma(),
            link=IdentityLink(),
            selection_penalty=0.0,
            discrete=True,
            features={"x": PSpline(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        retained: list[object] = []
        real = scop_efs_module.build_cached_scop_joint_geometry

        def recording(**kwargs):
            retained.append(real(**kwargs))
            return retained[-1]

        monkeypatch.setattr(scop_efs_module, "build_cached_scop_joint_geometry", recording)
        with pytest.warns(ConvergenceWarning, match="max_pirls_iter"):
            model.fit_reml(frame, np.exp(8.0 * x), max_pirls_iter=1)
        reml = model._reml_result
        assert not reml.converged
        assert reml.termination_reason == "bootstrap_uncertified"
        assert reml.terminal_refit_termination == "max_iter"
        assert len(retained) == 1
        assert reml.curvature_source == retained[0].curvature_source
        assert len(reml.lambda_history) == 2
        assert np.isfinite(model.result.effective_df)
        assert np.all(np.isfinite(model.predict(frame)))


class _ProbeWarning(UserWarning):
    """A warning only these tests raise."""


def _held_probe_a():
    from superglm import _held_warnings

    _held_warnings.warn("thread A", _ProbeWarning)


def _held_probe_b():
    from superglm import _held_warnings

    _held_warnings.warn("thread B", _ProbeWarning)


def _small_scop_model():
    rng = np.random.default_rng(0)
    n = 200
    x = np.sort(rng.uniform(0, 1, n))
    y = np.round(np.exp(1.0 + 1.5 * x)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, penalty="ssp", constraint=Constraint.fit.increasing)},
    )
    return model, pd.DataFrame({"x": x}), y


class TestBootstrapWarningHold:
    """A bootstrap start's warnings are held in its own context and replayed only for the kept start.

    ``warnings.catch_warnings`` swaps the process-wide filters and
    ``showwarning``: two fits overlapping on threads (the editor's jobs) could
    leave the process recording into a list nobody reads.  The hold is a
    context variable (``superglm._held_warnings``).
    """

    def test_overlapping_bootstraps_leave_the_warnings_module_alone(self, monkeypatch):
        """Two fits on two threads, their bootstraps interleaved A in, B in, A out, B out.

        Each thread's kept start replays its own held probe, and nothing
        else's; the filters and ``showwarning`` are as they were, and a
        later warning on the main thread still reaches ``showwarning``.
        Mutation check: a19d2fe4's ``catch_warnings`` left B's exit restoring
        the state A installed (an "always" filter and a recording
        ``showwarning``).
        """
        import threading
        import warnings

        probes = {"A": _held_probe_a, "B": _held_probe_b}
        a_in, a_out = threading.Event(), threading.Event()
        both_in = threading.Barrier(2, timeout=120)
        started: set[str] = set()
        real_fit = scop_efs_module._fit_scop_reml_mode
        real_attempt = scop_efs_module._bootstrap_attempt

        def fit(context, lambdas, **kwargs):
            name = threading.current_thread().name
            if kwargs.get("phase") == "bootstrap" and name not in started:
                started.add(name)
                probes[name]()
                if name == "A":
                    a_in.set()
                both_in.wait()
                if name == "B":
                    assert a_out.wait(timeout=120)
            return real_fit(context, lambdas, **kwargs)

        def attempt(*args, **kwargs):
            # A enters its attempt first, then B; A leaves first, then B
            if threading.current_thread().name == "B":
                assert a_in.wait(timeout=120)
            out = real_attempt(*args, **kwargs)
            if threading.current_thread().name == "A":
                a_out.set()
            return out

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fit)
        monkeypatch.setattr(scop_efs_module, "_bootstrap_attempt", attempt)
        seen: list[tuple[str, type, str, int]] = []

        def show(message, category, filename, lineno, file=None, line=None):
            seen.append((threading.current_thread().name, category, filename, lineno))

        errors: list[BaseException] = []

        def run():
            try:
                model, frame, y = _small_scop_model()
                model.fit_reml(frame, y, max_reml_iter=5)
            except BaseException as exc:  # reported below
                errors.append(exc)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            filters, shown = list(warnings.filters), warnings.showwarning
            threads = [threading.Thread(target=run, name=name) for name in "AB"]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=600)
            assert not errors, errors
            assert warnings.filters == filters
            assert warnings.showwarning is shown
            warnings.warn("after", _ProbeWarning)
        assert ("MainThread", _ProbeWarning) in {(name, category) for name, category, *_ in seen}
        for name, probe in probes.items():
            own = [
                (filename, lineno)
                for thread, category, filename, lineno in seen
                if thread == name and category is _ProbeWarning and filename == __file__
            ]
            assert own == [(__file__, probe.__code__.co_firstlineno + 3)], (name, own)

    def test_the_kept_start_warnings_reach_the_caller(self, monkeypatch):
        """A warning the bootstrap raises through superglm reaches the caller from its own line.

        On a fit whose first start certifies it arrives once, attributed to
        the line that raised it, and an "error" filter raises it.  With the
        first start forced to fail, only the retry's arrive.  Mutation check:
        a no-op ``replay`` drops every one.
        """
        import warnings

        from superglm import _held_warnings

        real_fit = scop_efs_module._fit_scop_reml_mode
        emitted: list[float] = []

        def fit(context, lambdas, **kwargs):
            if kwargs.get("phase") == "bootstrap":
                _held_warnings.warn(f"probe {lambdas['x']!r}", _ProbeWarning)
                emitted.append(lambdas["x"])
            return real_fit(context, lambdas, **kwargs)

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fit)
        line = fit.__code__.co_firstlineno + 2

        def probes():
            emitted.clear()
            model, frame, y = _small_scop_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit_reml(frame, y, max_reml_iter=5)
            return [
                (str(w.message), w.filename, w.lineno)
                for w in caught
                if w.category is _ProbeWarning
            ]

        kept = probes()
        assert emitted and set(emitted) == {1e-4}
        assert kept == [(f"probe {1e-4!r}", __file__, line)] * len(emitted)
        model, frame, y = _small_scop_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", _ProbeWarning)
            with pytest.raises(_ProbeWarning):
                model.fit_reml(frame, y, max_reml_iter=5)

        real_relative = scop_efs_module._scop_mode_newton_relative
        monkeypatch.setattr(
            scop_efs_module,
            "_scop_mode_newton_relative",
            lambda mode: 1.0 if mode.lambdas == {"x": 1e-4} else real_relative(mode),
        )
        retried = probes()
        assert 1e-4 in emitted
        assert retried == [(f"probe {x!r}", __file__, line) for x in emitted if x != 1e-4]
        assert retried

    def test_numpy_floating_point_warnings_pass_through_as_numpy_raises_them(self, monkeypatch):
        """NumPy's own warnings are not held: its text, and a caller's ``np.seterr`` "log".

        Routed through ``np.errstate(call=...)``, the overflow came back as
        "overflow encountered", without the ufunc a caller's message filter
        matches, and under ``np.seterr(all="log")`` NumPy called ``write`` on
        that handler, raising ``AttributeError`` out of the fit.  Mutation
        check: 71c697b0 failed both.
        """
        import warnings

        real_fit = scop_efs_module._fit_scop_reml_mode

        def fit(context, lambdas, **kwargs):
            if kwargs.get("phase") == "bootstrap":
                np.exp(np.array([1000.0]))
            return real_fit(context, lambdas, **kwargs)

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fit)
        model, frame, y = _small_scop_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit_reml(frame, y, max_reml_iter=5)
        messages = {str(w.message) for w in caught if w.category is RuntimeWarning}
        assert messages == {"overflow encountered in exp"}

        class Log:
            def __init__(self):
                self.lines: list[str] = []

            def write(self, message):
                self.lines.append(message)

        log = Log()
        model, frame, y = _small_scop_model()
        with np.errstate(all="log", call=log):
            model.fit_reml(frame, y, max_reml_iter=5)
        assert "Warning: overflow encountered in exp\n" in log.lines

    @pytest.mark.parametrize("held", [False, True], ids=["warnings_warn", "held_warn"])
    def test_a_warning_from_code_with_no_module_is_not_dropped(self, monkeypatch, held):
        """A warning raised from a notebook cell, whose filename names no module, reaches the caller.

        A custom family defined in a notebook warns from code whose globals
        have no ``__name__``.  Replayed with ``module=None``, Python 3.13 drops
        it; the module is read from the frame, as ``warnings.warn`` reads it
        (``"<string>"`` without a ``__name__``).  Mutation check: a19d2fe4
        raised nothing under "error" for the ``warnings.warn`` case.
        """
        import warnings

        namespace = {"warnings": warnings, "Probe": _ProbeWarning}
        if held:
            from superglm import _held_warnings

            namespace["held_warnings"] = _held_warnings
        source = (
            "def emit(held):\n"
            "    if held:\n"
            "        held_warnings.warn('cell', Probe)\n"
            "    else:\n"
            "        warnings.warn('cell', Probe)\n"
        )
        exec(compile(source, "<notebook-cell-7>", "exec"), namespace)
        assert "__name__" not in namespace
        real_fit = scop_efs_module._fit_scop_reml_mode

        def fit(context, lambdas, **kwargs):
            if kwargs.get("phase") == "bootstrap":
                namespace["emit"](held)
            return real_fit(context, lambdas, **kwargs)

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", fit)
        model, frame, y = _small_scop_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", _ProbeWarning)
            with pytest.raises(_ProbeWarning, match="cell"):
                model.fit_reml(frame, y, max_reml_iter=5)
        model, frame, y = _small_scop_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit_reml(frame, y, max_reml_iter=5)
        located = {(w.filename, w.lineno) for w in caught if w.category is _ProbeWarning}
        assert located == {("<notebook-cell-7>", 3 if held else 5)}


class TestBootstrapRetryCoverage:
    """The scaled retry also covers a first start whose mode cannot be scored, and warm starts."""

    def test_a_first_start_whose_mode_cannot_be_scored_is_retried(self, monkeypatch):
        """The first start's mode score refuses (a non-finite score); the scaled start certifies.

        That error is "no certified mode" at that start, the case the scaled
        start exists for, so the retry runs and the search converges from it.
        Mutation check: a19d2fe4 raised ``ObservedModeNotConvergedError``.
        """
        from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

        real_score = scop_efs_module.scop_penalized_mode_score
        calls = []

        def refuse_once(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ObservedGeometryInfeasibleError("SCOP penalized mode score is not finite")
            return real_score(**kwargs)

        starts = []
        real_attempt = scop_efs_module._bootstrap_attempt

        def attempt(context, lambdas, last_fit):
            starts.append(dict(lambdas))
            return real_attempt(context, lambdas, last_fit)

        monkeypatch.setattr(scop_efs_module, "scop_penalized_mode_score", refuse_once)
        monkeypatch.setattr(scop_efs_module, "_bootstrap_attempt", attempt)
        model, frame, y = _small_scop_model()
        model.fit_reml(frame, y, max_reml_iter=20)
        assert len(starts) == 2 and starts[0] == {"x": 1e-4} and starts[1] != starts[0]
        assert model.reml_diagnostics()["converged"]

    def test_a_fold_whose_warm_start_was_retried_away_reads_cold(self, monkeypatch):
        """``warm_started`` and the profile's warm components describe where the search started.

        Folds two and three are warm-started from fold one; their warm
        bootstrap is refused, and the search starts from the Hessian-scaled
        values instead.  Mutation check: a19d2fe4 read ``warm_started`` True.
        """
        from sklearn.model_selection import KFold

        from superglm import cross_validate

        first_starts: list[dict[str, float]] = []
        warm_calls: list[bool] = []
        real_optimize = scop_efs_module.optimize_scop_efs_reml
        real_attempt = scop_efs_module._bootstrap_attempt
        real_relative = scop_efs_module._scop_mode_newton_relative

        depth = [0]

        def optimize(*args, **kwargs):
            # a guard restart calls this again from inside: one record per fit
            if depth[0] == 0:
                warm_calls.append(kwargs.get("warm_lambdas") is not None)
                first_starts.append({})
            depth[0] += 1
            try:
                return real_optimize(*args, **kwargs)
            finally:
                depth[0] -= 1

        def attempt(context, lambdas, last_fit):
            if not first_starts[-1]:
                first_starts[-1].update(lambdas)
            return real_attempt(context, lambdas, last_fit)

        def relative(mode):
            if warm_calls[-1] and mode.lambdas == first_starts[-1]:
                return 1.0
            return real_relative(mode)

        monkeypatch.setattr(scop_efs_module, "optimize_scop_efs_reml", optimize)
        monkeypatch.setattr(scop_efs_module, "_bootstrap_attempt", attempt)
        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", relative)
        # a step the first fold resolves (a flat term is not passed on warm)
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.0, 1.0, 600))
        y = rng.poisson(np.exp(0.5 + 1.5 / (1.0 + np.exp(-12.0 * (x - 0.5))))).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": PSpline(n_knots=8, penalty="ssp", constraint=Constraint.fit.increasing)},
        )
        result = cross_validate(
            model,
            pd.DataFrame({"x": x}),
            y,
            cv=KFold(3, shuffle=True, random_state=0),
            fit_mode="fit_reml",
            return_estimators=True,
        )
        assert warm_calls == [False, True, True]
        assert result.fold_scores["warm_started"].tolist() == [False, False, False]
        for estimator in result.estimators[1:]:
            profile = estimator.reml_diagnostics()["profile"]
            assert profile["reml_warm_start_components"] == []
            assert estimator._reml_result.warm_start_components == []

    def test_a_zero_weight_row_does_not_move_the_scaled_start(self):
        """Only positive-weight rows set the Hessian-scaled start.

        A zero-weight row's offset of -800 overflowed its rescaled response
        and switched the whole intercept-only fit to its fallback, moving the
        published smoothing parameter of an unconverged fit and its
        predictions.  Mutation check: a19d2fe4 published 7.50 at offset -1
        and 5.13 at -800 here.
        """
        import warnings

        from superglm import Spline

        rng = np.random.default_rng(2)
        x = np.linspace(0.0, 1.0, 80)
        y = rng.poisson(np.exp(0.2 + 0.8 * x)).astype(float)
        frame = pd.DataFrame({"x": np.append(x, 0.0)})
        response = np.append(y, 2.0)
        weight = np.append(np.ones(80), 0.0)
        offsets = np.where(np.arange(80) % 2 == 0, -1.0, 0.0)
        fitted = {}
        for inactive in (-1.0, -800.0):
            model = SuperGLM(
                family="poisson",
                selection_penalty=0.0,
                discrete=True,
                features={"x": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing)},
            )
            offset = np.append(offsets, inactive)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit_reml(
                    frame, response, sample_weight=weight, offset=offset, max_pirls_iter=1
                )
            fitted[inactive] = (
                model.reml_diagnostics()["lambdas"],
                model.predict(frame.iloc[:80], offset=offsets),
            )
        assert fitted[-1.0][0] == fitted[-800.0][0]
        np.testing.assert_array_equal(fitted[-1.0][1], fitted[-800.0][1])


class TestSCOPAitkenTail:
    """The EFS linear tail is extrapolated to its limit, not walked step by step."""

    def test_the_aitken_jump_shortens_a_slow_tail_to_the_same_endpoint(self, monkeypatch):
        """27 EFS iterations become 16, at the same strict endpoint.

        Both runs stop on an accepted log-lambda step below ``reml_tol``. With a
        contraction ratio r <= 0.9 each endpoint lies within
        ``reml_tol * r / (1 - r)`` of the fixed point, so they agree to
        ``reml_tol * (1 + 2 * 0.9 / 0.1)``. Mutation check: master has no jump
        and takes the 27 iterations of the ``_aitken=False`` run.
        """
        import functools

        model, frame, y, offset = _scop_frequency_fixture(diesel_share=0.25, seed=10)
        jumps: list[int] = []
        real = scop_efs_module.optimize_scop_efs_reml
        real_step = scop_efs_module._aitken_step
        # The jump belongs to the EFS step, the Newton step's fallback.
        monkeypatch.setattr(
            scop_efs_module, "optimize_scop_efs_reml", functools.partial(real, _outer_step="efs")
        )

        def counting_step(state, *args, **kwargs):
            before = state.get("jumps", 0)
            out = real_step(state, *args, **kwargs)
            jumps.append(state.get("jumps", 0) - before)
            return out

        monkeypatch.setattr(scop_efs_module, "_aitken_step", counting_step)
        model.fit_reml(frame, y, offset=offset, max_reml_iter=100)
        fast = model.reml_diagnostics()

        monkeypatch.setattr(
            scop_efs_module,
            "optimize_scop_efs_reml",
            functools.partial(real, _aitken=False, _outer_step="efs"),
        )
        plain_model, _, _, _ = _scop_frequency_fixture(diesel_share=0.25, seed=10)
        plain_model.fit_reml(frame, y, offset=offset, max_reml_iter=100)
        plain = plain_model.reml_diagnostics()

        assert sum(jumps) >= 1
        assert fast["termination_reason"] == plain["termination_reason"] == "lambda_tolerance"
        assert plain["n_reml_iter"] >= 25
        assert fast["n_reml_iter"] <= 18
        bound = fast["profile"]["reml_tol_resolved"] * (1.0 + 2.0 * 0.9 / 0.1)
        for name, value in plain["lambdas"].items():
            assert abs(np.log(fast["lambdas"][name] / value)) <= bound, name


class TestSCOPWarmStart:
    """``lambda2_init`` as a mapping starts the SCOP Fellner-Schall search there."""

    def test_a_converged_start_is_kept_and_refits_in_one_iteration(self):
        """Seeded with its own converged lambdas, the refit stops at once.

        The bootstrap is fitted at the warm lambdas and reused as the first
        iterate, so the first EFS step is taken from the converged mode; it is
        below ``reml_tol`` and the search stops. The engine stops on an
        accepted log-lambda step below ``reml_tol``; with the cold fit's last
        steps contracting at ratio ``r``, its endpoint lies within
        ``reml_tol * r / (1 - r)`` of the fixed point, and the warm fit's within
        one more step, which bounds the agreement asserted here.
        Mutation check: on master (155832e8) the start was discarded and the
        refit repeated all 11 iterations.
        """
        model, frame, y, offset = _scop_frequency_fixture(diesel_share=0.25)
        model.fit_reml(frame, y, offset=offset, max_reml_iter=100)
        cold = model.reml_diagnostics()
        assert cold["termination_reason"] == "lambda_tolerance"
        assert cold["n_reml_iter"] > 3

        history = [{k: np.log(v) for k, v in lambdas.items()} for lambdas in cold["lambda_history"]]
        steps = [
            max(abs(after[k] - before[k]) for k in after)
            for before, after in zip(history[:-1], history[1:], strict=True)
        ]
        ratio = min(steps[-1] / steps[-2], 0.9) if steps[-2] > 0 else 0.9
        reml_tol = model.reml_diagnostics()["profile"]["reml_tol_resolved"]
        bound = 2.0 * reml_tol * max(1.0, ratio / (1.0 - ratio)) + reml_tol

        warm_model, _, _, _ = _scop_frequency_fixture(diesel_share=0.25)
        warm_model.fit_reml(
            frame, y, offset=offset, max_reml_iter=100, lambda2_init=cold["lambdas"]
        )
        warm = warm_model.reml_diagnostics()
        assert warm["n_reml_iter"] <= 2
        assert warm["converged"]
        assert warm["profile"]["reml_warm_start_components"] == sorted(cold["lambdas"])
        for name, value in cold["lambdas"].items():
            assert abs(np.log(warm["lambdas"][name]) - np.log(value)) <= bound, name


def _scop_newton_fixture(
    family: str = "poisson", seed: int = 7, n: int = 6000, diesel_share: float | None = None
):
    """A small SCOP fit whose monotone term is curved, so no smoothing parameter
    ends in a suppression hold and the fixed point is a point, not a region.

    ``diesel_share`` puts that share of rows on Diesel and makes Regular the
    base level, so the Diesel indicator carries that share of the weight."""
    from superglm import Categorical, Spline

    rng = np.random.default_rng(seed)
    bm = rng.uniform(50.0, 150.0, n)
    age = rng.uniform(18.0, 90.0, n)
    share = 0.3 if diesel_share is None else diesel_share
    gas = np.where(rng.uniform(size=n) < share, "Diesel", "Regular")
    eta = (
        -1.4
        + 0.9 * (1.0 - np.exp(-(bm - 50.0) / 25.0))
        + 0.4 * np.exp(-(age - 18.0) / 8.0)
        + np.where(gas == "Diesel", 0.3, 0.0)
    )
    frame = pd.DataFrame({"BonusMalus": bm, "DrivAge": age, "VehGas": gas})
    if family == "poisson":
        exposure = rng.uniform(0.3, 1.0, n)
        y = rng.poisson(exposure * np.exp(eta)).astype(float)
        offset = np.log(exposure)
    elif family == "gaussian":
        y = 2.0 * eta + rng.normal(0.0, 0.5, n)
        offset = None
    else:
        y = rng.gamma(3.0, np.exp(eta + 5.0) / 3.0)
        offset = None
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        discrete=True,
        features={
            "BonusMalus": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
            "DrivAge": Spline(kind="ps", k=8),
            "VehGas": Categorical(base="most_exposed" if diesel_share is None else "Regular"),
        },
    )
    return model, frame, y, offset


def _flat_monotone_fixture():
    """Poisson, 6000 rows, a monotone BonusMalus term with no signal at all.

    The Newton step overshoots the EFS fixed point here into a region lower on
    the exact LAML, from which the step back is uphill: the guard fires."""
    from superglm import Categorical, Spline

    n = 6000
    rng = np.random.default_rng(7)
    bm = rng.uniform(50.0, 150.0, n)
    age = rng.uniform(18.0, 90.0, n)
    rng.uniform(0.0, 20.0, n)
    gas = np.where(rng.uniform(size=n) < 0.3, "Diesel", "Regular")
    rng.choice(list("ABCDEFGHIJ"), n)
    eta = -1.4 + 0.4 * np.exp(-(age - 18.0) / 8.0) + np.where(gas == "Diesel", 0.3, 0.0)
    exposure = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    frame = pd.DataFrame({"BonusMalus": bm, "DrivAge": age, "VehGas": gas})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "BonusMalus": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
            "DrivAge": Spline(kind="ps", k=8),
            "VehGas": Categorical(base="most_exposed"),
        },
    )
    return model, frame, y, np.log(exposure)


def _half_flat_tensor_fixture(n: int = 3000, seed: int = 3):
    """A monotone term beside a DrivAge x VehAge tensor the data do not support.

    The fit runs the VehAge margin to working infinity (residual EDF under
    0.05, held against increase) and leaves the DrivAge margin finite."""
    from superglm import Categorical, Spline

    rng = np.random.default_rng(seed)
    bm = rng.uniform(50.0, 150.0, n)
    age = rng.uniform(18.0, 90.0, n)
    vage = rng.uniform(0.0, 20.0, n)
    gas = np.where(rng.uniform(size=n) < 0.3, "Diesel", "Regular")
    eta = (
        -1.4
        + 0.9 * (1 - np.exp(-(bm - 50) / 25))
        + 0.4 * np.exp(-(age - 18) / 8)
        + 0.2 * np.sin(vage / 4)
    )
    exposure = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    frame = pd.DataFrame({"BonusMalus": bm, "DrivAge": age, "VehAge": vage, "VehGas": gas})

    def make():
        return SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "BonusMalus": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
                "DrivAge": Spline(kind="ps", k=8),
                "VehAge": Spline(kind="ps", k=8),
                "VehGas": Categorical(base="most_exposed"),
            },
            interactions=[("DrivAge", "VehAge")],
        )

    return make, frame, y, np.log(exposure)


def _count_irls_fits(monkeypatch) -> list[int]:
    """Count the SCOP engine's inner IRLS fits (one entry per call)."""
    calls: list[int] = []
    real = scop_efs_module.fit_irls_direct

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scop_efs_module, "fit_irls_direct", counting)
    return calls


def _strict_efs_lambdas(monkeypatch, family: str) -> dict[str, float]:
    """The EFS fixed point to 1e-9: no Aitken jump, no plateau exit."""
    import functools

    real = scop_efs_module.optimize_scop_efs_reml
    with monkeypatch.context() as patch:
        patch.setattr(
            scop_efs_module,
            "optimize_scop_efs_reml",
            functools.partial(real, _outer_step="efs", _aitken=False),
        )
        patch.setattr(scop_efs_module, "_scop_plateau_steps_stalled", lambda *a, **k: False)
        model, frame, y, offset = _scop_newton_fixture(family)
        model.fit_reml(frame, y, offset=offset, max_reml_iter=1000, reml_tol=1e-9)
    assert model._reml_result.termination_reason == "lambda_tolerance"
    return dict(model._reml_result.lambdas)


def _assert_stopped_on_a_full_newton_step(result) -> None:
    """The premise of the 2 * reml_tol endpoint bound between two runs.

    A run that stops because its full Newton step is under reml_tol is within
    reml_tol of the local model's fixed point; that iteration adopts no move,
    so its lambda history ends on a repeat. A strict stop on an accepted step
    that the line search shortened, or on an EFS step, bounds nothing of the
    kind and needs its own bound.
    """
    assert result.termination_reason == "lambda_tolerance"
    assert result.scop_outer_steps[-1] == "newton"
    assert result.lambda_history[-1] == result.lambda_history[-2]


class TestSCOPSuppressionHold:
    """The EFS step's decrease hold is read in effective degrees of freedom."""

    @staticmethod
    def _step(scale: float, lam: float, size: float = 1.0) -> float:
        from superglm.reml.scop_efs import _joint_efs_lambda_step

        omega = scale * _first_diff_penalty(5)
        component = PenaltyComponent(
            name="m",
            group_name="m",
            group_index=0,
            group_sl=slice(0, 5),
            omega_raw=omega,
            omega_ssp=omega,
            rank=4.0,
        )
        # tr(H^-1 S) = 0.00125 * 8 * scale; beta' S beta = scale * size^2.
        beta = np.array([0.0, 0.0, 0.0, 0.0, size])
        updated, _, _ = _joint_efs_lambda_step(
            [component], beta, 0.00125 * np.eye(5), 1.0, {"m": lam}, {"m"}, {}, {"m": 1.0}, {}
        )
        return float(np.log(updated["m"] / lam))

    def test_a_large_lambda_can_still_come_down(self):
        """lambda = 200 with tr(H^-1 S) = 0.01: the penalty suppresses 2 EDF and the
        score asks for a decrease (log step -4.6, capped at -4). The old bar read
        tr(H^-1 S) < 0.05 alone and held it at 200. Mutation check: the old bar
        returns a step of 0.
        """
        assert self._step(scale=1.0, lam=200.0) == pytest.approx(-4.0, abs=1e-12)

    def test_the_hold_is_invariant_to_the_penalty_scale(self):
        """S -> 10 S with lambda -> lambda / 10 is the same model and the same step.
        Under the old bar the first was held (tr 0.01) and the second was not
        (tr 0.1)."""
        assert self._step(scale=10.0, lam=20.0) == pytest.approx(self._step(scale=1.0, lam=200.0))

    def test_an_isolated_penalty_s_decrease_is_never_held(self):
        """lambda = 2 with tr(H^-1 S) = 0.01 suppresses only 0.02 EDF, but the score
        asks for a decrease with a slope of g = lambda beta' S beta + 0.02 - 4 = 14
        (2 dV/d rho): the criterion is far from flat. An isolated penalty's slope
        tends to -rank(S) as lambda -> 0, so V has no flat end there and the step
        is the plain EFS one, log(3.98 / 18). Mutation check: the decrease bar
        lambda tr(H^-1 S) < 0.05 held it (step 0)."""
        assert self._step(scale=1.0, lam=2.0, size=3.0) == pytest.approx(np.log(3.98 / 18.0))

    @staticmethod
    def _covered_step(size: float) -> float:
        """The log step of a first-difference penalty under an identity penalty
        on the same block, at lambda 1e-4 against 1: the identity covers the
        difference penalty's range, so d log|S|+ / d rho is 8e-4."""
        from superglm.reml.scop_efs import _joint_efs_lambda_step

        components = [
            PenaltyComponent(
                name=name,
                group_name="m",
                group_index=0,
                group_sl=slice(0, 5),
                omega_raw=omega,
                omega_ssp=omega,
                rank=rank,
            )
            for name, omega, rank in (
                ("m:diff", _first_diff_penalty(5), 4.0),
                ("m:ridge", np.eye(5), 5.0),
            )
        ]
        beta = np.array([0.0, 0.0, 0.0, 0.0, size])
        lambdas = {"m:diff": 1e-4, "m:ridge": 1.0}
        updated, _, _ = _joint_efs_lambda_step(
            components,
            beta,
            0.00125 * np.eye(5),
            1.0,
            lambdas,
            {"m:diff"},
            {},
            {"m:diff": 1.0},
            {},
        )
        return float(np.log(updated["m:diff"] / lambdas["m:diff"]))

    def test_a_covered_penalty_is_held_only_while_its_end_is_flat(self):
        """Where another penalty covers range(S_j), d log|S|+ / d rho_j (8e-4 here)
        falls to zero as lambda_j does, and so does the slope: that end is flat.
        With beta' S beta = 9 the slope asking for the decrease is about 1e-4,
        under 0.05, and the decrease is held. With beta' S beta = 1e4 it is 1.0,
        and the decrease (log(8e-4 / 1), capped at -4) goes ahead. Mutation
        check: holding on d log|S|+ / d rho_j < 0.05 alone holds the second.
        A held step leaves lambda as exp(log(lambda)), two roundings: within a
        few eps of no step."""
        assert abs(self._covered_step(3.0)) <= 4.0 * np.finfo(float).eps
        assert self._covered_step(100.0) == pytest.approx(-4.0, abs=1e-12)

    def test_a_component_at_working_infinity_is_reported_flat(self, monkeypatch):
        """A tensor margin the data do not support runs to working infinity: its
        residual EDF falls under 0.05, so its increase is held (not its decrease:
        the other margin does not cover its range). ``flat_components`` is the
        union of both holds, so it reports the margin, and the warm start
        ``cross_validate`` takes from this fit leaves it cold, with its sibling
        margin (a half-warm tensor is refused at the bootstrap). Mutation check:
        built from the decrease hold alone, ``flat_components`` is empty here."""
        from superglm.model.reml_setup import live_reml_lambdas
        from superglm.reml.scop_efs import _scop_newton_system

        captured = {}
        real_final = scop_efs_module._finalize_scop_reml_mode

        def capture(context, mode):
            captured["mode"] = mode
            return real_final(context, mode)

        monkeypatch.setattr(scop_efs_module, "_finalize_scop_reml_mode", capture)
        make, frame, y, offset = _half_flat_tensor_fixture()
        model = make()
        model.fit_reml(frame, y, offset=offset)
        margin = "DrivAge:VehAge:margin_VehAge"
        assert model._reml_result.flat_components == [margin]
        mode = captured["mode"]
        system = _scop_newton_system(
            mode.penalty_components,
            mode.result.beta,
            mode.hessian_inverse,
            1.0,
            mode.lambdas,
            {pc.name for pc in mode.penalty_components},
            mode.scop_states,
        )
        index = system.names.index(margin)
        assert system.increase_held[index] and not system.decrease_held[index]
        start = live_reml_lambdas(model)
        assert set(start) == {"DrivAge", "VehAge", "BonusMalus"}

    def test_flat_components_describe_the_published_mode(self, monkeypatch):
        """The holds are read at the mode the run publishes, not at the iterate
        its last step left. Capped at eight iterations, the last accepted Newton
        step takes the VehAge margin from lambda 1.2e3 (residual EDF 0.053, not
        held) to 3.1e3 (0.024, held against increase), so the published mode is
        flat there and a warm start taken from it leaves the tensor cold.
        Mutation check: on af53c8d4 the holds came from the iterate before that
        step, ``flat_components`` was empty and the warm start kept the margin."""
        from superglm import ConvergenceWarning
        from superglm.model.reml_setup import live_reml_lambdas
        from superglm.reml.scop_efs import _scop_newton_system

        captured = {}
        real_final = scop_efs_module._finalize_scop_reml_mode

        def capture(context, mode):
            captured["mode"] = mode
            return real_final(context, mode)

        monkeypatch.setattr(scop_efs_module, "_finalize_scop_reml_mode", capture)
        make, frame, y, offset = _half_flat_tensor_fixture()
        model = make()
        with pytest.warns(ConvergenceWarning, match="max_reml_iter"):
            model.fit_reml(frame, y, offset=offset, max_reml_iter=8)
        mode = captured["mode"]
        system = _scop_newton_system(
            mode.penalty_components,
            mode.result.beta,
            mode.hessian_inverse,
            1.0,
            mode.lambdas,
            {pc.name for pc in mode.penalty_components},
            mode.scop_states,
        )
        held = sorted(
            name
            for name, up, down in zip(
                system.names, system.increase_held, system.decrease_held, strict=True
            )
            if up or down
        )
        assert held == ["DrivAge:VehAge:margin_VehAge"]
        assert model._reml_result.flat_components == held
        assert set(live_reml_lambdas(model)) == {"DrivAge", "VehAge", "BonusMalus"}

    def test_a_strongly_identified_term_started_above_its_optimum_comes_back(self):
        """A smooth so well identified that its penalty suppresses about 0.01 EDF at
        its optimum is far from a flat end: V rises on both sides of it. Started
        at 2 and 20 times its optimum, it returns to the cold fit's lambda within
        the Newton stop's bound of 2 * reml_tol (``test_a_warm_start_above_the_
        optimum_comes_back_down``) and is never reported flat. Mutation check:
        under the decrease bar lambda tr(H^-1 S_j) < 0.05 both starts stopped
        'converged' inside the band above the optimum, at 2.0 and 2.9 times it,
        with the term reported flat."""
        from superglm import Spline

        rng = np.random.default_rng(1)
        n = 2000
        x = rng.uniform(0, 1, n)
        z = rng.uniform(0, 1, n)
        eta = 0.6 * (1 - np.exp(-3 * x)) + 0.8 * np.sin(2 * np.pi * z) + 0.5 * np.sin(5 * np.pi * z)
        frame = pd.DataFrame({"x": x, "z": z})
        y = eta + rng.normal(0, 0.025, n)

        def fit(start=None):
            model = SuperGLM(
                family="gaussian",
                selection_penalty=0.0,
                discrete=True,
                features={
                    "x": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
                    "z": Spline(kind="ps", k=12),
                },
            )
            return model.fit_reml(frame, y, lambda2_init=start)

        cold = fit()
        lambdas = dict(cold._reml_result.lambdas)
        reml_tol = cold.reml_diagnostics()["profile"]["reml_tol_resolved"]
        assert cold._reml_result.flat_components == []
        for factor in (2.0, 20.0):
            start = dict(lambdas)
            start["z"] *= factor
            warm = fit(start)._reml_result
            assert warm.converged
            assert warm.flat_components == []
            for name, value in lambdas.items():
                assert abs(np.log(warm.lambdas[name] / value)) <= 2.0 * reml_tol, (factor, name)

    def test_a_warm_start_above_the_optimum_comes_back_down(self):
        """Started at e^5 and e^7 times its optimum, the monotone term's lambda
        returns to the cold fit's optimum. Mutation check: under the old bar the
        start sits in the region where every decrease is held, and the fit stopped
        there, 'converged', at lambda 403 and 2980 against an optimum of 2.72.
        The endpoints agree to the Newton step's stopping bound, 2 * reml_tol
        (both runs stop on a full step under reml_tol)."""
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        cold = dict(model._reml_result.lambdas)
        reml_tol = model.reml_diagnostics()["profile"]["reml_tol_resolved"]
        for jump in (5.0, 7.0):
            start = dict(cold)
            start["BonusMalus"] = cold["BonusMalus"] * np.exp(jump)
            warm, _, _, _ = _scop_newton_fixture()
            warm.fit_reml(frame, y, offset=offset, lambda2_init=start)
            assert warm._reml_result.converged
            for name, value in cold.items():
                got = warm._reml_result.lambdas[name]
                assert abs(np.log(got / value)) <= 2.0 * reml_tol, (jump, name)


class TestSCOPNewtonOuterStep:
    """Newton on log lambda reaches the Fellner-Schall fixed point in fewer steps."""

    def test_newton_reaches_the_strict_efs_fixed_point(self, monkeypatch):
        """Every step is a Newton step, the fit stops on a full step under reml_tol,
        and it lands on the EFS fixed point. Bound: Newton's last step bounds its
        distance to the fixed point by reml_tol (its convergence is quadratic,
        so the true distance is far smaller), and the strict EFS reference stops
        within 1e-9 r / (1 - r) <= 1e-8 of it for a contraction ratio r <= 0.9;
        2 * reml_tol covers both. Measured: 1.3e-7, 8 iterations against 12 for
        the EFS default. Mutation check: no Newton step exists on the unfixed
        tree (``scop_outer_steps`` is absent)."""
        import functools

        reference = _strict_efs_lambdas(monkeypatch, "poisson")
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        reml_tol = model.reml_diagnostics()["profile"]["reml_tol_resolved"]
        assert result.converged
        assert result.termination_reason == "lambda_tolerance"
        assert result.scop_newton_fallback is None
        assert set(result.scop_outer_steps) == {"newton"}
        for name, value in reference.items():
            assert abs(np.log(result.lambdas[name] / value)) <= 2.0 * reml_tol, name

        real = scop_efs_module.optimize_scop_efs_reml
        monkeypatch.setattr(
            scop_efs_module, "optimize_scop_efs_reml", functools.partial(real, _outer_step="efs")
        )
        efs_model, _, _, _ = _scop_newton_fixture()
        efs_model.fit_reml(frame, y, offset=offset)
        assert efs_model._reml_result.scop_newton_fallback == "requested"
        assert set(efs_model._reml_result.scop_outer_steps) == {"efs"}
        assert result.n_reml_iter < efs_model._reml_result.n_reml_iter

    def test_an_estimated_scale_fit_takes_newton_steps_to_the_same_point(self, monkeypatch):
        """Gamma profiles its scale out of the LAML, and the Jacobian carries the
        profiled-scale term (d(1/phi)/dD_p). The fit runs on Newton steps
        throughout and lands on the strict EFS fixed point, within the bound of
        ``test_newton_reaches_the_strict_efs_fixed_point`` (measured 2.3e-7)."""
        reference = _strict_efs_lambdas(monkeypatch, "gamma")
        model, frame, y, offset = _scop_newton_fixture("gamma")
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        reml_tol = model.reml_diagnostics()["profile"]["reml_tol_resolved"]
        assert result.converged
        assert result.scop_newton_fallback is None
        assert set(result.scop_outer_steps) == {"newton"}
        for name, value in reference.items():
            assert abs(np.log(result.lambdas[name] / value)) <= 2.0 * reml_tol, name

    def test_the_jacobian_matches_finite_differences(self, monkeypatch):
        """At a fitted Gaussian mode the working weights are constant, so the only
        terms the Jacobian could miss are the ones it forms: with the SCOP
        reparameterisation terms and the profiled-scale term it must match a
        central difference of the working gradient over refits.

        Bound 1e-3 on entries of 2 to 4: the central difference carries
        h^2 / 6 |g'''| ~ 1e-6 of truncation at h = 1e-3 and about 1e-5 from the
        certified modes' rounding divided by h; measured 6.2e-5 across three
        seeds. Mutation check: without the reparameterisation terms the largest
        error is 0.04 to 0.075, and without the scale term 0.038 to 0.040.
        """
        from superglm import Gaussian, Spline
        from superglm.reml.scop_efs import (
            _fit_scop_reml_mode,
            _reml_evaluation_phi,
            _scop_newton_system,
            _scop_reparam_jacobian_correction,
        )

        rng = np.random.default_rng(3)
        n = 400
        x = rng.uniform(0, 1, n)
        z = rng.uniform(0, 1, n)
        y = 1.5 * (1 - np.exp(-3 * x)) + 0.5 * np.sin(2 * np.pi * z) + rng.normal(0, 0.3, n)
        frame = pd.DataFrame({"x": x, "z": z})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
                "z": Spline(kind="ps", k=8),
            },
        )
        captured = {}
        real_final = scop_efs_module._finalize_scop_reml_mode

        def capture(context, mode):
            captured["context"], captured["mode"] = context, mode
            return real_final(context, mode)

        monkeypatch.setattr(scop_efs_module, "_finalize_scop_reml_mode", capture)
        model.fit_reml(frame, y)
        context, mode = captured["context"], captured["mode"]

        def system(at, *, scale_term=True):
            phi = _reml_evaluation_phi(
                at.evaluation, scale_known=False, fallback_likelihood_size=context.likelihood_size
            )
            derivative = at.evaluation.profiled_scale.d_inverse_phi_d_penalized_deviance
            return _scop_newton_system(
                at.penalty_components,
                at.result.beta,
                at.hessian_inverse,
                phi,
                at.lambdas,
                {pc.name for pc in at.penalty_components},
                at.scop_states,
                inverse_phi_derivative=float(derivative) if scale_term else 0.0,
            )

        base = system(mode)
        correction = _scop_reparam_jacobian_correction(mode, base)
        assert correction is not None
        h = 1e-3
        finite = np.zeros_like(base.hessian)
        for k, name in enumerate(base.names):
            gradients = []
            for sign in (1.0, -1.0):
                lambdas = dict(mode.lambdas)
                lambdas[name] *= np.exp(sign * h)
                refit = _fit_scop_reml_mode(
                    context,
                    lambdas,
                    beta_init=mode.result.beta,
                    intercept_init=float(mode.result.intercept),
                    scop_state_init=mode.scop_states,
                    phase="fixed",
                    reml_iteration=0,
                    require_converged=True,
                )
                gradients.append(system(refit).gradient)
            finite[:, k] = (gradients[0] - gradients[1]) / (2.0 * h)

        assert np.max(np.abs(base.hessian + correction - finite)) <= 1e-3
        assert np.max(np.abs(base.hessian - finite)) > 1e-2
        unscaled = system(mode, scale_term=False)
        assert np.max(np.abs(unscaled.hessian + correction - finite)) > 1e-2

    def test_the_jacobian_matches_finite_differences_on_tensor_margins(self, monkeypatch):
        """A tensor product's two margins share one penalty block, so the term
        -d2 log|S|+ / d rho_j d rho_k is non-zero on and between them; for the
        single-penalty smooths of ``test_the_jacobian_matches_finite_differences``
        it vanishes identically. At a fitted Gaussian mode (constant working
        weights) the Jacobian, reparameterisation and scale terms included,
        matches a central difference of the working gradient within that test's
        bound, 1e-3, on every entry. Mutation check: the fixture sees the term --
        with it removed the margins' block misses the finite difference by more
        than 1e-2 (by 5.0 on a Poisson book-shaped fixture, where the fit then
        converged elsewhere)."""
        from superglm import Gaussian, Spline
        from superglm.reml.penalty_algebra import compute_logdet_s_derivatives
        from superglm.reml.scop_efs import (
            _fit_scop_reml_mode,
            _reml_evaluation_phi,
            _scop_newton_system,
            _scop_reparam_jacobian_correction,
        )

        rng = np.random.default_rng(3)
        n = 1500
        x = rng.uniform(0, 1, n)
        a = rng.uniform(0, 1, n)
        v = rng.uniform(0, 1, n)
        y = (
            1.5 * (1 - np.exp(-3 * x))
            + 0.5 * np.sin(2 * np.pi * a)
            + 0.4 * np.cos(3 * v)
            + 0.8 * (a - 0.5) * (v - 0.5)
            + rng.normal(0, 0.3, n)
        )
        frame = pd.DataFrame({"x": x, "a": a, "v": v})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0.0,
            discrete=True,
            features={
                "x": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
                "a": Spline(kind="ps", k=8),
                "v": Spline(kind="ps", k=8),
            },
            interactions=[("a", "v")],
        )
        captured = {}
        real_final = scop_efs_module._finalize_scop_reml_mode

        def capture(context, mode):
            captured["context"], captured["mode"] = context, mode
            return real_final(context, mode)

        monkeypatch.setattr(scop_efs_module, "_finalize_scop_reml_mode", capture)
        model.fit_reml(frame, y)
        context, mode = captured["context"], captured["mode"]

        def system(at):
            phi = _reml_evaluation_phi(
                at.evaluation, scale_known=False, fallback_likelihood_size=context.likelihood_size
            )
            return _scop_newton_system(
                at.penalty_components,
                at.result.beta,
                at.hessian_inverse,
                phi,
                at.lambdas,
                {pc.name for pc in at.penalty_components},
                at.scop_states,
                inverse_phi_derivative=float(
                    at.evaluation.profiled_scale.d_inverse_phi_d_penalized_deviance
                ),
            )

        base = system(mode)
        correction = _scop_reparam_jacobian_correction(mode, base)
        assert correction is not None
        h = 1e-3
        finite = np.zeros_like(base.hessian)
        for k, name in enumerate(base.names):
            gradients = []
            for sign in (1.0, -1.0):
                lambdas = dict(mode.lambdas)
                lambdas[name] *= np.exp(sign * h)
                refit = _fit_scop_reml_mode(
                    context,
                    lambdas,
                    beta_init=mode.result.beta,
                    intercept_init=float(mode.result.intercept),
                    scop_state_init=mode.scop_states,
                    phase="fixed",
                    reml_iteration=0,
                    require_converged=True,
                )
                gradients.append(system(refit).gradient)
            finite[:, k] = (gradients[0] - gradients[1]) / (2.0 * h)

        jacobian = base.hessian + correction
        assert np.max(np.abs(jacobian - finite)) <= 1e-3
        margins = [i for i, name in enumerate(base.names) if name.startswith("a:v:margin_")]
        assert len(margins) == 2
        _, logdet_hessian = compute_logdet_s_derivatives(mode.lambdas, mode.penalty_components)
        logdet = np.array(
            [
                [logdet_hessian.get((name_j, name_k), 0.0) for name_k in base.names]
                for name_j in base.names
            ]
        )
        block = np.ix_(margins, margins)
        assert np.max(np.abs((jacobian + logdet)[block] - finite[block])) > 1e-2

    def test_the_correction_contracts_the_explicit_trace(self, monkeypatch):
        """``_scop_reparam_jacobian_correction`` contracts each trace in O(p^2 q);
        here it is checked against the explicit ``tr(H^-1 dH_k H^-1 S_j)`` with
        dH_k formed as a p x p matrix, at a Poisson mode, where the weights vary
        and the intercept-profiling term ``(dc c' + c dc') / sum(W)`` is not zero
        (with Gaussian weights the SCOP block's c vanishes, so the
        finite-difference test cannot see that term). Both forms sum the same
        O(p) products, so they agree to rounding: bound 1e-9 of the largest
        entry, measured 2e-14. Mutation check: the explicit form without the
        intercept term differs by 3% of the largest entry."""
        from superglm.reml.penalty_algebra import penalty_component_dense_matrix
        from superglm.reml.scop_efs import (
            _scop_newton_system,
            _scop_reparam_jacobian_correction,
        )

        captured = {}
        real_final = scop_efs_module._finalize_scop_reml_mode

        def capture(context, mode):
            captured["mode"] = mode
            return real_final(context, mode)

        monkeypatch.setattr(scop_efs_module, "_finalize_scop_reml_mode", capture)
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        mode = captured["mode"]
        system = _scop_newton_system(
            mode.penalty_components,
            mode.result.beta,
            mode.hessian_inverse,
            1.0,
            mode.lambdas,
            {pc.name for pc in mode.penalty_components},
            mode.scop_states,
        )
        fast = _scop_reparam_jacobian_correction(mode, system)

        hessian_inverse = mode.hessian_inverse
        width = hessian_inverse.shape[0]
        latent = np.asarray(mode.result.beta, dtype=np.float64).copy()
        positivity = np.zeros(width)
        for state in mode.scop_states.values():
            latent[state["group_sl"]] = state["beta_eff"]
            reparam = state["reparam"]
            positivity[state["group_sl"]] = reparam.second_derivative_diagonal(
                state["beta_eff"]
            ) / reparam.jacobian_diagonal(state["beta_eff"])
        cross = mode.joint_geometry.transformed_intercept_cross
        sum_w = mode.joint_geometry.sum_w
        gradient = mode.penalty @ latent
        curvature = (
            mode.joint_geometry.centered_hessian
            + np.outer(cross, cross) / sum_w
            - mode.penalty
            + np.diag(positivity * gradient)
        )
        components = {pc.name: pc for pc in mode.penalty_components}
        local = {name: penalty_component_dense_matrix(pc) for name, pc in components.items()}

        def explicit(with_intercept_term: bool) -> np.ndarray:
            out = np.zeros_like(fast)
            for k, name_k in enumerate(system.names):
                pc = components[name_k]
                penalty_beta = np.zeros(width)
                penalty_beta[pc.group_sl] = mode.lambdas[name_k] * (
                    local[name_k] @ latent[pc.group_sl]
                )
                v = -(hessian_inverse @ penalty_beta)
                v0 = -float(cross @ v) / sum_w
                s = positivity * v
                derivative = s[:, None] * curvature + curvature * s[None, :]
                derivative += np.diag(positivity * (curvature @ v + cross * v0 - v * gradient))
                if with_intercept_term:
                    dc = s * cross
                    derivative -= (np.outer(dc, cross) + np.outer(cross, dc)) / sum_w
                sandwiched = hessian_inverse @ derivative @ hessian_inverse
                for j, name_j in enumerate(system.names):
                    block = components[name_j].group_sl
                    out[j, k] = -mode.lambdas[name_j] * float(
                        np.sum(sandwiched[block, block] * local[name_j].T)
                    )
            return out

        scale = float(np.max(np.abs(fast)))
        assert np.max(np.abs(fast - explicit(True))) <= 1e-9 * scale
        assert np.max(np.abs(fast - explicit(False))) > 1e-3 * scale

    def test_a_failed_newton_line_search_restarts_the_search_with_efs(self, monkeypatch):
        """The guard: a Newton step none of whose three forward trials the LAML
        accepts restarts the search from the bootstrap with EFS steps. Forced here
        by reversing the third step, an ascent direction of 4 in log lambda (far
        outside the plateau). The fallback is recorded with its iteration, the
        guard spent one bounded forward search (three trials, no reflection), and
        the run returns exactly what the EFS search returns, its iterations
        after the three Newton ones. Mutation check: continuing EFS from the
        guard's iterate (the earlier hand-off) gives other lambdas."""
        import functools

        real_step = scop_efs_module._scop_newton_step
        calls = []

        def reversed_third(system, jacobian=None):
            step = real_step(system, jacobian)
            calls.append(step)
            if len(calls) == 3:
                return -4.0 * np.sign(step) * (np.abs(step) > 0)
            return step

        searches = []
        real_search = scop_efs_module._backtrack_scop_efs_candidate

        def record_search(context, current, proposed, **kwargs):
            out = real_search(context, current, proposed, **kwargs)
            searches.append((kwargs.get("max_attempts"), kwargs.get("reflect", True), out[1]))
            return out

        monkeypatch.setattr(scop_efs_module, "_scop_newton_step", reversed_third)
        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", record_search)
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.scop_newton_fallback == "line_search"
        assert result.scop_newton_fallback_iter == 3
        assert result.scop_outer_steps[:3] == ["newton", "newton", "newton"]
        assert set(result.scop_outer_steps[3:]) == {"efs"}
        assert (3, False, False) in searches
        assert result.converged
        monkeypatch.undo()

        real = scop_efs_module.optimize_scop_efs_reml
        monkeypatch.setattr(
            scop_efs_module, "optimize_scop_efs_reml", functools.partial(real, _outer_step="efs")
        )
        efs_model, _, _, _ = _scop_newton_fixture()
        efs_model.fit_reml(frame, y, offset=offset)
        efs = efs_model._reml_result
        assert result.lambdas == efs.lambdas
        assert result.n_reml_iter == 3 + efs.n_reml_iter
        assert result.termination_reason == efs.termination_reason

    def test_an_overshooting_newton_fit_restarts_and_converges(self, monkeypatch):
        """A monotone term with no signal: Newton overshoots the EFS fixed point
        (log lambda 0.25 past it on DrivAge) into a region lower on the exact LAML,
        and the step back is uphill there, so the guard fires at iteration 4.
        The fit then converges, raises no ConvergenceWarning, and returns exactly
        the EFS search's lambdas. Its inner fits are bounded by the EFS search's
        plus one bootstrap refit and four fits (a candidate and three trials) per
        Newton iteration before the restart: measured 22 against 14 + 1 + 16.
        Mutation check: continuing EFS from the guard's iterate stalled
        unconverged after 28 iterations and 194 inner fits, with a warning and
        BonusMalus's lambda 3.05 log units from the EFS answer."""
        import functools
        import warnings

        from superglm import ConvergenceWarning

        newton_fits = _count_irls_fits(monkeypatch)
        model, frame, y, offset = _flat_monotone_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        monkeypatch.undo()

        efs_fits = _count_irls_fits(monkeypatch)
        real = scop_efs_module.optimize_scop_efs_reml
        monkeypatch.setattr(
            scop_efs_module, "optimize_scop_efs_reml", functools.partial(real, _outer_step="efs")
        )
        efs_model, _, _, _ = _flat_monotone_fixture()
        efs_model.fit_reml(frame, y, offset=offset)
        efs = efs_model._reml_result

        assert result.converged and efs.converged
        assert result.scop_newton_fallback == "line_search"
        assert result.lambdas == efs.lambdas
        assert len(newton_fits) <= len(efs_fits) + 1 + 4 * result.scop_newton_fallback_iter

    def test_a_rejected_step_inside_the_plateau_stops_there(self, monkeypatch):
        """When the rejected Newton step is under the plateau's 0.01 step cap and
        the decrease it predicts, |g' delta| / 4, is under the line search's own
        acceptance tolerance, the mode is published as an objective plateau,
        converged, without restarting. Forced here by rejecting every Newton
        search whose step is under 0.01. The full step bounds the distance to the
        fixed point, so the lambdas agree with an unforced fit's within 0.01.
        Mutation check: without the plateau test the guard restarts the search
        with EFS (``scop_newton_fallback == "line_search"``)."""
        real_search = scop_efs_module._backtrack_scop_efs_candidate

        def reject_small(context, current, proposed, **kwargs):
            moved = max(
                abs(np.log(proposed[name] / current.lambdas[name]))
                for name in proposed
                if name in current.lambdas
            )
            if kwargs.get("reflect", True) is False and 0.0 < moved < 0.01:
                return current, False
            return real_search(context, current, proposed, **kwargs)

        reference, frame, y, offset = _scop_newton_fixture()
        reference.fit_reml(frame, y, offset=offset)
        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", reject_small)
        model, _, _, _ = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.converged
        assert result.termination_reason == "objective_plateau"
        assert result.scop_newton_fallback is None
        assert set(result.scop_outer_steps) == {"newton"}
        for name, value in reference._reml_result.lambdas.items():
            assert abs(np.log(result.lambdas[name] / value)) < 0.01, name

    def test_a_fisher_fallback_iterate_takes_an_efs_step(self, monkeypatch):
        """At an iterate where a SCOP block's inner solve fell back to Fisher
        curvature, the reparameterisation terms of the Newton Jacobian cannot be
        formed, so that iteration takes an EFS step, recorded as "efs_fisher",
        and Newton resumes at the next. Forced by marking the third iterate's
        blocks as Fisher fallbacks. The run still converges to the unforced
        fit's fixed point within the Newton stop's bound, 2 * reml_tol. Mutation
        check: the earlier loop took a Newton step there with the fixed-curvature
        Jacobian and recorded nothing."""
        reference, frame, y, offset = _scop_newton_fixture()
        reference.fit_reml(frame, y, offset=offset)
        reml_tol = reference.reml_diagnostics()["profile"]["reml_tol_resolved"]

        real_search = scop_efs_module._backtrack_scop_efs_candidate
        searches = []

        def mark_second(context, current, proposed, **kwargs):
            mode, accepted = real_search(context, current, proposed, **kwargs)
            searches.append(mode)
            if len(searches) == 2 and accepted:
                states = {
                    index: {**state, "last_fisher_fallback": True}
                    for index, state in mode.scop_states.items()
                }
                mode = replace(mode, scop_states=states)
            return mode, accepted

        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", mark_second)
        model, _, _, _ = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.scop_outer_steps[:3] == ["newton", "newton", "efs_fisher"]
        assert "newton" in result.scop_outer_steps[3:]
        assert result.scop_newton_fallback is None
        assert result.scop_fisher_fallbacks >= 1
        assert result.converged
        _assert_stopped_on_a_full_newton_step(reference._reml_result)
        _assert_stopped_on_a_full_newton_step(result)
        for name, value in reference._reml_result.lambdas.items():
            assert abs(np.log(result.lambdas[name] / value)) <= 2.0 * reml_tol, name

    def test_an_uncorrectable_iterate_takes_an_efs_step(self, monkeypatch):
        """On an observed iterate the reparameterisation terms can still fail to
        form (here a map that is not the exp map; the same holds when they are
        not finite). That iteration takes an EFS step, recorded as
        "efs_uncorrected", and Newton resumes at the next: the fixed-curvature
        Jacobian alone is not the step's Jacobian. Forced at the third iterate's
        correction. The run converges to the unforced fit's fixed point within
        the Newton stop's bound, 2 * reml_tol. Mutation check: on af53c8d4 the
        failed correction returned None and that iteration took a Newton step on
        the uncorrected Jacobian, recorded as "newton"."""
        reference, frame, y, offset = _scop_newton_fixture()
        reference.fit_reml(frame, y, offset=offset)
        reml_tol = reference.reml_diagnostics()["profile"]["reml_tol_resolved"]

        class NotExp:
            """The block's map with its second derivative doubled: not the exp map.
            Seen only by the correction; the inner solves keep the real map."""

            def __init__(self, inner):
                self._inner = inner

            def jacobian_diagonal(self, beta_eff):
                return self._inner.jacobian_diagonal(beta_eff)

            def second_derivative_diagonal(self, beta_eff):
                return 2.0 * self._inner.second_derivative_diagonal(beta_eff)

        real_correction = scop_efs_module._scop_reparam_jacobian_correction
        calls = []

        def not_exp_on_the_third(mode, system):
            calls.append(1)
            if len(calls) == 3:
                states = {
                    index: {**state, "reparam": NotExp(state["reparam"])}
                    for index, state in mode.scop_states.items()
                }
                mode = replace(mode, scop_states=states)
            return real_correction(mode, system)

        monkeypatch.setattr(
            scop_efs_module, "_scop_reparam_jacobian_correction", not_exp_on_the_third
        )
        model, _, _, _ = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.scop_outer_steps[:3] == ["newton", "newton", "efs_uncorrected"]
        assert "newton" in result.scop_outer_steps[3:]
        assert result.scop_newton_fallback is None
        assert result.converged
        _assert_stopped_on_a_full_newton_step(reference._reml_result)
        _assert_stopped_on_a_full_newton_step(result)
        for name, value in reference._reml_result.lambdas.items():
            assert abs(np.log(result.lambdas[name] / value)) <= 2.0 * reml_tol, name

    def test_the_correction_distinguishes_none_from_failure(self, monkeypatch):
        """None means the Jacobian has no reparameterisation terms (no block
        carries a positivity coordinate); terms that exist but cannot be formed
        raise, so the loop can take the EFS step rather than a Newton step on
        the fixed-curvature Jacobian alone. Mutation check: on af53c8d4 a
        non-finite correction returned None, the same value as no terms."""
        from superglm.reml.scop_efs import (
            _scop_newton_system,
            _scop_reparam_jacobian_correction,
            _SCOPCorrectionUnavailableError,
        )

        captured = {}
        real_final = scop_efs_module._finalize_scop_reml_mode

        def capture(context, mode):
            captured["mode"] = mode
            return real_final(context, mode)

        monkeypatch.setattr(scop_efs_module, "_finalize_scop_reml_mode", capture)
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        mode = captured["mode"]
        system = _scop_newton_system(
            mode.penalty_components,
            mode.result.beta,
            mode.hessian_inverse,
            1.0,
            mode.lambdas,
            {pc.name for pc in mode.penalty_components},
            mode.scop_states,
        )
        correction = _scop_reparam_jacobian_correction(mode, system)
        assert correction is not None and np.all(np.isfinite(correction))
        assert _scop_reparam_jacobian_correction(replace(mode, scop_states={}), system) is None
        with np.errstate(invalid="ignore", over="ignore"):
            overflowed = replace(mode, penalty=mode.penalty * np.inf)
            with pytest.raises(_SCOPCorrectionUnavailableError, match="not finite"):
                _scop_reparam_jacobian_correction(overflowed, system)

    def test_a_newton_step_leaves_no_efs_step_size_history(self, monkeypatch):
        """The adaptive EFS step size halves when a step reverses the previous
        EFS step's direction and grows when it agrees. A Newton step is no EFS
        step: at the "efs_fisher" iterate after two Newton steps the EFS step
        reads no previous direction and leaves every step size at its start, 1.
        Forced as in ``test_a_fisher_fallback_iterate_takes_an_efs_step``.
        Mutation check: on af53c8d4 that step read the second Newton step's
        direction and moved the step sizes off 1."""
        real_search = scop_efs_module._backtrack_scop_efs_candidate
        searches = []

        def mark_second(context, current, proposed, **kwargs):
            mode, accepted = real_search(context, current, proposed, **kwargs)
            searches.append(mode)
            if len(searches) == 2 and accepted:
                states = {
                    index: {**state, "last_fisher_fallback": True}
                    for index, state in mode.scop_states.items()
                }
                mode = replace(mode, scop_states=states)
            return mode, accepted

        steps = []
        real_step = scop_efs_module._joint_efs_lambda_step

        def recording_step(*args, **kwargs):
            previous = dict(args[8])
            out = real_step(*args, **kwargs)
            steps.append((previous, dict(args[7])))
            return out

        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", mark_second)
        monkeypatch.setattr(scop_efs_module, "_joint_efs_lambda_step", recording_step)
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.scop_outer_steps[:3] == ["newton", "newton", "efs_fisher"]
        # The bootstrap's step, then one per EFS iterate; the third is the first.
        assert len(steps) == 1 + result.scop_outer_steps.count("efs_fisher")
        previous, step_sizes = steps[1]
        assert previous == {}
        assert set(step_sizes.values()) == {1.0}
        assert result.converged

    def test_the_reparameterisation_terms_cut_the_inner_fits(self, monkeypatch):
        """Counts, not time: on a Gaussian fit Newton takes no more inner IRLS fits
        than the EFS search does (measured 6 against 32), because its Jacobian
        carries the SCOP reparameterisation terms. Mutation check: without them
        (``system.hessian`` alone) the steps overshot, the guard fired at
        iteration 6 and the restarted search took 43 inner fits in all."""
        import functools

        newton_fits = _count_irls_fits(monkeypatch)
        model, frame, y, offset = _scop_newton_fixture("gaussian")
        model.fit_reml(frame, y, offset=offset)
        assert model._reml_result.scop_newton_fallback is None
        monkeypatch.undo()

        efs_fits = _count_irls_fits(monkeypatch)
        real = scop_efs_module.optimize_scop_efs_reml
        monkeypatch.setattr(
            scop_efs_module, "optimize_scop_efs_reml", functools.partial(real, _outer_step="efs")
        )
        efs_model, _, _, _ = _scop_newton_fixture("gaussian")
        efs_model.fit_reml(frame, y, offset=offset)
        assert len(newton_fits) <= len(efs_fits)

    def test_a_newton_fit_cut_short_still_warns(self):
        """Non-convergence is disclosed whatever the step: two Newton iterations
        cannot finish this fit, and the run warns and reports converged=False."""
        from superglm import ConvergenceWarning

        model, frame, y, offset = _scop_newton_fixture()
        with pytest.warns(ConvergenceWarning, match="max_reml_iter"):
            model.fit_reml(frame, y, offset=offset, max_reml_iter=2)
        result = model._reml_result
        assert not result.converged
        assert result.scop_outer_steps == ["newton", "newton"]

    @pytest.mark.parametrize("family", ["gamma", "poisson"])
    def test_a_fisher_fallback_geometry_takes_an_efs_step(self, monkeypatch, family):
        """The joint geometry can fall back to Fisher curvature with no block
        flagged: the observed builder's indefinite branch (Gamma) and the cached
        builder's decomposition failure (Poisson). Its H is J F J + S, without
        the map's -diag(e S beta) term that the reparameterisation correction
        reads, so those iterates take an EFS step ("efs_fisher") and the
        correction is never formed on them. Forced from the first Newton line
        search on by failing each geometry build's first decomposition, which
        sends the builder down its own Fisher branch. Mutation check: on
        941f9ce8 every later iterate took a Newton step with the correction
        formed on the Fisher geometry."""
        import superglm.reml.scop_geometry as scop_geometry

        armed = {"search": False, "build": False}
        for builder_name, decomposer_name in (
            ("build_observed_scop_joint_geometry", "decompose_gram"),
            ("build_cached_scop_joint_geometry", "_decompose_with_factor_certification"),
        ):
            real_builder = getattr(scop_efs_module, builder_name)
            real_decomposer = getattr(scop_geometry, decomposer_name)

            def builder(*args, _real=real_builder, **kwargs):
                armed["build"] = armed["search"]
                try:
                    return _real(*args, **kwargs)
                finally:
                    armed["build"] = False

            def decomposer(*args, _real=real_decomposer, **kwargs):
                if armed["build"]:
                    armed["build"] = False
                    raise np.linalg.LinAlgError("forced indefinite geometry")
                return _real(*args, **kwargs)

            monkeypatch.setattr(scop_efs_module, builder_name, builder)
            monkeypatch.setattr(scop_geometry, decomposer_name, decomposer)

        real_search = scop_efs_module._backtrack_scop_efs_candidate

        def arming_search(*args, **kwargs):
            armed["search"] = True
            return real_search(*args, **kwargs)

        sources = []
        real_correction = scop_efs_module._scop_reparam_jacobian_correction

        def recording_correction(mode, system):
            sources.append(mode.curvature_source)
            return real_correction(mode, system)

        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", arming_search)
        monkeypatch.setattr(
            scop_efs_module, "_scop_reparam_jacobian_correction", recording_correction
        )
        model, frame, y, offset = _scop_newton_fixture(family)
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.curvature_source == "fisher"
        assert result.scop_outer_steps[0] == "newton"
        assert set(result.scop_outer_steps[1:]) == {"efs_fisher"}
        assert sources == ["observed"]
        assert result.converged

    def test_a_rejected_first_newton_step_hands_over_to_efs_in_place(self, monkeypatch):
        """When the first Newton step's forward trials are all rejected, EFS
        takes over from the same mode with the fresh state it starts from, so
        the run is the EFS search's own, step for step. The reason names that
        hand-over, not the restart from the bootstrap ("line_search"), which
        did not happen. Mutation check: on 941f9ce8 the reason read
        "line_search"."""
        import functools

        real_search = scop_efs_module._backtrack_scop_efs_candidate

        def reject_first_newton(context, current, proposed, **kwargs):
            if kwargs.get("reflect", True) is False and kwargs["reml_iteration"] == 1:
                return current, False
            return real_search(context, current, proposed, **kwargs)

        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", reject_first_newton)
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        monkeypatch.undo()

        real = scop_efs_module.optimize_scop_efs_reml
        monkeypatch.setattr(
            scop_efs_module, "optimize_scop_efs_reml", functools.partial(real, _outer_step="efs")
        )
        efs_model, _, _, _ = _scop_newton_fixture()
        efs_model.fit_reml(frame, y, offset=offset)
        efs = efs_model._reml_result
        assert result.scop_newton_fallback == "line_search_first_iteration"
        assert result.scop_newton_fallback_iter == 1
        assert result.scop_outer_steps == efs.scop_outer_steps
        assert result.lambdas == efs.lambdas
        assert (result.n_reml_iter, result.termination_reason) == (
            efs.n_reml_iter,
            efs.termination_reason,
        )

    def test_a_rejected_newton_step_at_the_cap_stops_and_says_so(self, monkeypatch):
        """A guard that fires on the last allowed iteration has no iteration to
        restart in: the run stops at that mode on ``max_reml_iter``, records
        "line_search_at_cap", and the warning says the last Newton step failed
        and nothing was left to restart with. Forced as in
        ``test_a_failed_newton_line_search_restarts_the_search_with_efs``, with
        the cap at the third iteration. Mutation check: on 941f9ce8 the reason
        read "line_search", the label of a restart that never ran, and the
        warning did not mention the failed step."""
        from superglm import ConvergenceWarning

        real_step = scop_efs_module._scop_newton_step
        calls = []

        def reversed_third(system, jacobian=None):
            step = real_step(system, jacobian)
            calls.append(step)
            if len(calls) == 3:
                return -4.0 * np.sign(step) * (np.abs(step) > 0)
            return step

        monkeypatch.setattr(scop_efs_module, "_scop_newton_step", reversed_third)
        model, frame, y, offset = _scop_newton_fixture()
        with pytest.warns(ConvergenceWarning, match="no iteration was left to restart"):
            model.fit_reml(frame, y, offset=offset, max_reml_iter=3)
        result = model._reml_result
        assert not result.converged
        assert result.termination_reason == "max_reml_iter"
        assert result.scop_newton_fallback == "line_search_at_cap"
        assert result.scop_newton_fallback_iter == 3
        assert result.scop_outer_steps == ["newton", "newton", "newton"]
        assert result.lambdas == result.lambda_history[-1] == result.lambda_history[-2]

    def test_aitken_never_extrapolates_from_newton_steps(self, monkeypatch):
        """The Aitken limit reads a history of EFS steps contracting at one
        ratio. At an isolated "efs_fisher" iterate between Newton steps the
        history starts afresh, so no Newton step enters it. Forced by marking
        the modes of the second and fourth line searches as Fisher fallbacks.
        Mutation check: on 941f9ce8 the history held the preceding Newton step
        at the first such iterate and two Newton steps at the second."""
        real_search = scop_efs_module._backtrack_scop_efs_candidate
        searches = []

        def mark(context, current, proposed, **kwargs):
            mode, accepted = real_search(context, current, proposed, **kwargs)
            searches.append(mode)
            if len(searches) in (2, 4) and accepted:
                states = {
                    index: {**state, "last_fisher_fallback": True}
                    for index, state in mode.scop_states.items()
                }
                mode = replace(mode, scop_states=states)
            return mode, accepted

        lengths = []
        real_aitken = scop_efs_module._aitken_step

        def recording_aitken(state, name, prev_dlsp, dlsp, scaled_step):
            out = real_aitken(state, name, prev_dlsp, dlsp, scaled_step)
            lengths.append(len(state[name]))
            return out

        monkeypatch.setattr(scop_efs_module, "_backtrack_scop_efs_candidate", mark)
        monkeypatch.setattr(scop_efs_module, "_aitken_step", recording_aitken)
        model, frame, y, offset = _scop_newton_fixture()
        model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert result.scop_outer_steps[:5] == [
            "newton",
            "newton",
            "efs_fisher",
            "newton",
            "efs_fisher",
        ]
        assert result.scop_outer_steps.count("efs") == 0
        assert lengths and set(lengths) == {0}
        assert result.converged


class TestSCOPObservedGeometryCentring:
    """The observed-curvature geometry takes ``build_centered_system`` as its authority."""

    def test_a_frequent_indicator_forms_no_design_rows(self, monkeypatch):
        """Gamma with a log link takes the observed geometry. Its Diesel
        indicator carries about 60% of the weight, so its weighted mean exceeds
        its centred RMS at every geometry build, and the system there comes
        from the compact anchor-centred supports, which form no design row.
        The answer agrees with one whose geometry centres row chunks
        (``_force_chunked``) to the stopping bound of two converged Newton fits
        of one criterion, 2 * reml_tol in log lambda (the centring routes
        differ at the rounding level). Mutation check: on 941f9ce8 every build
        was redone from 8,192-row chunks after the guard rejected it: 24 row
        materialisations and 8 chunked builds on this fit."""
        import functools

        import superglm.reml.scop_geometry as scop_geometry
        from superglm._group_matrix._group_matrix_centered import _raw_centering_well_scaled
        from superglm.group_matrix import DesignMatrix

        real_build = scop_geometry.build_centered_system
        tripped = []

        def recording_build(**kwargs):
            system = real_build(**kwargs)
            scale = np.sqrt(np.maximum(np.diag(system.data_gram), 0.0) / system.sum_w)
            tripped.append(not _raw_centering_well_scaled(system.mean_x, scale))
            return system

        def no_rows(self, idx):
            raise AssertionError("the observed geometry must not materialise design rows")

        with monkeypatch.context() as patch:
            patch.setattr(scop_geometry, "build_centered_system", recording_build)
            patch.setattr(DesignMatrix, "row_subset", no_rows)
            model, frame, y, offset = _scop_newton_fixture("gamma", diesel_share=0.6)
            model.fit_reml(frame, y, offset=offset)
        result = model._reml_result
        assert tripped and all(tripped)
        assert result.converged and result.curvature_source == "observed"

        monkeypatch.setattr(
            scop_geometry,
            "build_centered_system",
            functools.partial(real_build, _force_chunked=True),
        )
        chunked, _, _, _ = _scop_newton_fixture("gamma", diesel_share=0.6)
        chunked.fit_reml(frame, y, offset=offset)
        reml_tol = model.reml_diagnostics()["profile"]["reml_tol_resolved"]
        _assert_stopped_on_a_full_newton_step(result)
        _assert_stopped_on_a_full_newton_step(chunked._reml_result)
        for name, value in chunked._reml_result.lambdas.items():
            assert abs(np.log(result.lambdas[name] / value)) <= 2.0 * reml_tol, name


class TestCandidateStepBackoff:
    """A candidate certification failure backs the lambda step off (#179).

    The iteration-1 candidate consumes the one EFS proposal that bypasses
    the line search, so it was the one lambda movement with no damping
    behind it: four call sites raised on a rejection the line search
    survives. The backoff applies the line search's own trial formula --
    damped geometric steps in log-lambda -- between the certified mode the
    step was taken from and the proposal that failed. The fixed-lambda
    site, with no certified predecessor, keeps raising; a bootstrap with none
    is published unconverged (``TestColdBootstrapStart``).
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

    def test_a_failed_bootstrap_is_published_unconverged(self, monkeypatch):
        """No certified mode at either bootstrap start: disclosed, not refused.

        Rejecting every certification fails the cold bootstrap and its
        Hessian-scaled retry, and leaves the search no mode to start from or
        damp toward. The fit at the retry's start is published with
        ``converged=False`` and a ConvergenceWarning that names the stage and
        what to change (owner decision 3, 2026-09-30). Mutation check:
        37f73863 raised "SCOP REML bootstrap did not converge to a coefficient
        mode".

        What is published is the last inner fit the retry's certification
        ladder made (rung 3: cold, at the ladder's tightest tolerance), at the
        retry's own start, and nothing is fitted twice: four inner fits per
        start, one per rung. Mutation check: 2a4e28c7 fitted the retry's start
        again from scratch at the loose tolerance, a ninth inner fit that
        repeated the retry's rung 0 bit for bit, and published that.
        """
        from superglm import ConvergenceWarning

        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", lambda mode: 1.0)
        fits: list[tuple[str, float, object]] = []
        real_fit = scop_efs_module.fit_irls_direct

        def counting_fit(**kwargs):
            out = real_fit(**kwargs)
            fits.append((kwargs["debug_context"]["phase"], kwargs["tol"], out[0]))
            return out

        scaled_starts: list[dict[str, float]] = []
        real_scaled = scop_efs_module._hessian_scaled_bootstrap_lambdas

        def recording_scaled(*args, **kwargs):
            scaled_starts.append(real_scaled(*args, **kwargs))
            return scaled_starts[-1]

        monkeypatch.setattr(scop_efs_module, "fit_irls_direct", counting_fit)
        monkeypatch.setattr(scop_efs_module, "_hessian_scaled_bootstrap_lambdas", recording_scaled)
        model, frame, y = self._model()
        with pytest.warns(ConvergenceWarning, match="starting smoothing parameters"):
            model.fit_reml(frame, y, max_reml_iter=5)
        diagnostics = model.reml_diagnostics()
        assert not diagnostics["converged"]
        assert diagnostics["termination_reason"] == "bootstrap_uncertified"
        assert diagnostics["n_reml_iter"] == 0
        assert np.all(np.isfinite(model.predict(frame)))
        # Poisson/log certifies at Fisher curvature, so rung 0 runs at the
        # default pirls_tol and the ladder tightens to 1e-10, then 1e-11 twice.
        assert [(phase, tol) for phase, tol, _ in fits] == [
            ("bootstrap", tol) for tol in (1e-6, 1e-10, 1e-11, 1e-11)
        ] * 2
        assert model._reml_result.pirls_result is fits[-1][2]
        assert len(scaled_starts) == 1
        assert diagnostics["lambdas"] == scaled_starts[0]
        assert diagnostics["lambdas"] != {"x": 1e-4}

    def test_a_failed_warm_bootstrap_restarts_every_component_scaled(self, monkeypatch):
        """A warm bootstrap with no certified mode is retried at the scaled start too.

        A complete ``lambda2_init`` warms every estimated component, so
        2a4e28c7 had no cold component to rescale: a failed warm bootstrap
        went straight to the disclosure, which still claimed a scaled retry,
        and the published fit sat at the warm value that had failed. Cross-
        validation folds after the first, NB2 theta refits and user mappings
        start that way. Every estimated component now restarts at its
        Hessian-scaled value, and the message names only the starts that ran.
        """
        from superglm import ConvergenceWarning

        monkeypatch.setattr(scop_efs_module, "_scop_mode_newton_relative", lambda mode: 1.0)
        starts: list[dict[str, float]] = []
        real = scop_efs_module._fit_scop_reml_mode

        def recording(context, lambdas, **kwargs):
            if kwargs.get("phase") == "bootstrap" and kwargs.get("_certification_retry", 0) == 0:
                starts.append(dict(lambdas))
            return real(context, lambdas, **kwargs)

        monkeypatch.setattr(scop_efs_module, "_fit_scop_reml_mode", recording)
        model, frame, y = self._model()
        with pytest.warns(ConvergenceWarning) as caught:
            model.fit_reml(frame, y, lambda2_init={"x": 3.0}, max_reml_iter=5)
        assert len(starts) == 2
        assert starts[0] == {"x": 3.0}
        assert starts[1]["x"] != 3.0
        diagnostics = model.reml_diagnostics()
        assert diagnostics["termination_reason"] == "bootstrap_uncertified"
        assert diagnostics["lambdas"] == starts[1]
        message = " ".join(
            str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
        )
        assert "scaled to the data's curvature that it retried at" in message
        assert "data-scaled starting ones" in message

    def test_the_bootstrap_message_names_only_the_starts_that_ran(self):
        """With no retry (no scaled start differed), the message claims none.

        Mutation check: 2a4e28c7 claimed a retry at the scaled start whatever
        ran.
        """
        from superglm.diagnostics.convergence import reml_nonconvergence_message

        def reml(*starts):
            return SimpleNamespace(
                converged=False,
                termination_reason="bootstrap_uncertified",
                n_reml_iter=0,
                terminal_refit_termination=None,
                lambda_history=[{"x": value} for value in starts],
                scop_states=None,
            )

        alone = reml_nonconvergence_message(reml(1e-4))
        assert "retried at" not in alone and "data-scaled" not in alone
        assert "no retry ran" in alone
        retried = reml_nonconvergence_message(reml(1e-4, 2.5))
        assert "scaled to the data's curvature that it retried at" in retried
        assert "no retry ran" not in retried

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
        fitted = SuperGLM(family="poisson", features={"x": Numeric()}).fit(
            frame,
            response,
            record_diagnostics=True,
        )
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            [projected_residual],
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

    def test_joint_delta_keeps_projected_residual_for_near_collinear_fit(self):
        """Large fitted components cannot cancel the joint linear delta term."""
        from superglm.solvers.scop import build_scop_solver_reparam
        from superglm.solvers.scop_newton import (
            _build_joint_objective_cache,
            _safe_joint_trial_objective_delta,
        )

        rng = np.random.default_rng(1)
        u = rng.normal(size=100)
        v = rng.normal(size=100)
        v -= u * float(u @ v) / float(u @ u)
        basis = np.column_stack([u, u + 1.0e-9 * v])
        weights = np.ones(100)
        reparam = build_scop_solver_reparam(3, direction="increasing")
        beta = np.log(np.full(2, 1.0e8))
        gamma = reparam.forward(beta)
        response = basis @ gamma - v
        penalty = np.zeros((2, 2))
        state = {
            "B_scop": basis,
            "S_scop": penalty,
            "beta_scop": beta,
            "reparam": reparam,
            "bin_idx": None,
        }
        scop_items = [(0, state)]
        slices = [slice(0, 2)]
        gram = basis.T @ basis
        cache = _build_joint_objective_cache(scop_items, weights, response)
        cache.diag_btwb = [gram]
        cache.cross_btwb = {}

        beta_trial = np.log(gamma + np.array([1.0, -1.0]))
        gamma_delta = reparam.forward(beta_trial) - gamma
        residual = response - basis @ gamma
        projected_residual = basis.T @ (weights * residual)
        expected_delta = -float(gamma_delta @ projected_residual) + 0.5 * float(
            gamma_delta @ gram @ gamma_delta
        )
        reported_delta = _safe_joint_trial_objective_delta(
            scop_items,
            beta_trial,
            slices,
            [0.0],
            [gamma],
            [projected_residual],
            cache,
        )

        assert expected_delta < 0.0
        assert reported_delta == pytest.approx(expected_delta, rel=1.0e-10, abs=1.0e-14)

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
            weight_semantics="frequency",
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

    @pytest.mark.parametrize("discretized", [False, True], ids=["dense", "discrete"])
    def test_joint_curvature_mixed_map_objective_matches_brute_force(self, discretized):
        """Joint line search must use mapped coefficients, not Jacobian diagonals."""
        from superglm.solvers.scop_newton import (
            _safe_joint_objective,
            scop_joint_newton_step,
        )

        rng = np.random.default_rng(216)
        n = 180
        q1, q2 = 5, 4
        reparam1 = _curvature_solver_reparam(q1 + 1, "convex")
        reparam2 = _curvature_solver_reparam(q2 + 1, "concave")
        if discretized:
            bin1 = rng.integers(0, 37, size=n)
            bin2 = rng.integers(0, 29, size=n)
            basis1 = rng.normal(size=(37, q1))
            basis2 = rng.normal(size=(29, q2))
        else:
            bin1 = None
            bin2 = None
            basis1 = rng.normal(size=(n, q1))
            basis2 = rng.normal(size=(n, q2))

        beta1 = rng.normal(scale=0.25, size=q1)
        beta2 = rng.normal(scale=0.25, size=q2)
        beta1[0] = -0.7
        beta2[0] = 1.4
        eta1 = basis1 @ reparam1.forward(beta1)
        eta2 = basis2 @ reparam2.forward(beta2)
        if discretized:
            eta1 = eta1[bin1]
            eta2 = eta2[bin2]
        weights = rng.uniform(0.2, 2.0, size=n)
        response = eta1 + eta2 + rng.normal(scale=0.2, size=n)
        states = {
            0: {
                "B_scop": basis1,
                "S_scop": reparam1.penalty_matrix(),
                "beta_scop": beta1,
                "reparam": reparam1,
                "bin_idx": bin1,
                "group_sl": slice(0, q1),
                "group_name": "x1",
            },
            1: {
                "B_scop": basis2,
                "S_scop": reparam2.penalty_matrix(),
                "beta_scop": beta2,
                "reparam": reparam2,
                "bin_idx": bin2,
                "group_sl": slice(q1, q1 + q2),
                "group_name": "x2",
            },
        }
        groups = self._make_mock_groups(states)
        scop_items = sorted(states.items())
        slices = [slice(0, q1), slice(q1, q1 + q2)]
        lambdas = {"x1": 0.7, "x2": 0.3}
        lambda_list = [lambdas["x1"], lambdas["x2"]]
        beta_before = np.concatenate((beta1, beta2))
        objective_before = _safe_joint_objective(
            scop_items,
            weights,
            response,
            beta_before,
            slices,
            lambda_list,
        )

        results = scop_joint_newton_step(
            states,
            weights,
            response,
            lambdas,
            groups,
        )
        beta_after = np.concatenate((results[0].beta_new, results[1].beta_new))
        objective_after = _safe_joint_objective(
            scop_items,
            weights,
            response,
            beta_after,
            slices,
            lambda_list,
        )

        assert beta1[0] != reparam1.jacobian_diagonal(beta1)[0]
        assert beta2[0] != reparam2.jacobian_diagonal(beta2)[0]
        for result in results.values():
            np.testing.assert_allclose(result.objective_before, objective_before, atol=1e-10)
            np.testing.assert_allclose(result.objective_after, objective_after, atol=1e-10)
        assert objective_after <= objective_before + 1e-12

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

        total_eta = np.zeros_like(z)
        for gamma, (_, state) in zip(gammas, scop_items, strict=True):
            eta = state["B_scop"] @ gamma
            total_eta += eta[state["bin_idx"]]
        residual = z - total_eta
        projected_residuals = []
        for _, state in scop_items:
            weighted_residual = np.bincount(
                state["bin_idx"],
                weights=W * residual,
                minlength=state["B_scop"].shape[0],
            )
            projected_residuals.append(state["B_scop"].T @ weighted_residual)

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
            projected_residuals,
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
