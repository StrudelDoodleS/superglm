"""Private docstrings in ``src/superglm`` stay at 12 lines or fewer, except those allowlisted.

"Private" means the name: a function, method or class whose own name, or an enclosing one's,
starts with an underscore. A public name inside an underscore module is not checked.

The allowlist is a ratchet: an entry records its docstring's exact length, so a trimmed
docstring lowers its count in the same commit and a longer one fails.
"""

import ast
from functools import cache
from pathlib import Path

LIMIT = 12
SRC = Path(__file__).resolve().parents[1] / "src" / "superglm"
PR571 = "file touched by open PR #571"
PR577 = "file touched by open PR #577"

# "path under src/superglm::qualname": (recorded line count, why it stays long)
ALLOWLIST: dict[str, tuple[int, str]] = {
    "diagnostics/discretize.py::_TotalCurveSE": (20, PR571),
    "diagnostics/discretize.py::_axis_column_labels": (13, PR571),
    "diagnostics/discretize.py::_distinct_grid_nodes": (19, PR571),
    "diagnostics/discretize.py::_grid_log_surface": (25, PR571),
    "diagnostics/discretize.py::_grid_lookup_index": (31, PR571),
    "diagnostics/discretize.py::_joint_exact_bands": (41, PR571),
    "diagnostics/discretize.py::_non_grid_builtin_interactions": (21, PR571),
    "distributional/fit_state.py::_curvature_disclosures": (17, PR571),
    "distributional/fit_state.py::_limit_row_disclosure": (26, PR571),
    "distributional/fit_state.py::_wall_disclosure": (23, PR571),
    "distributional/results/solver.py::_score_within_rounding_floor": (30, PR577),
    "distributional/smoothing/objective.py::_discarded_space_error": (16, PR577),
    "distributional/smoothing/objective.py::_restricted_objective": (46, PR577),
    "distributional/smoothing/objective.py::_stable_isolated_gfs_update": (14, PR577),
    "distributional/solver/_formation.py::_AbsTensor": (16, PR577),
    "distributional/solver/solver.py::_half_penalty_quadratic": (14, PR577),
    "editor/free_levels.py::_gap_variances": (14, PR571),
    "editor/free_levels.py::_influence": (15, PR571),
    "editor/free_levels.py::_observed_rows": (14, PR571),
    "editor/free_levels.py::_on_binding_face": (17, PR571),
    "export/rating_tables.py::_base_relativity": (40, PR571),
    "export/rating_tables.py::_continuous_block": (22, PR571),
    "export/rating_tables.py::_continuous_ppform_block": (67, PR571),
    "export/rating_tables.py::_emitted_relativities": (16, PR571),
    "export/rating_tables.py::_export_weights": (21, PR571),
    "export/rating_tables.py::_format_interval": (14, PR571),
    "export/rating_tables.py::_format_number": (34, PR571),
    "export/rating_tables.py::_impact_sweep": (20, PR571),
    "export/rating_tables.py::_offset_per_unit_block": (19, PR571),
    "export/rating_tables.py::_offset_per_unit_block_from_multiplier": (18, PR571),
    "export/rating_tables.py::_piecewise_block": (28, PR571),
    "export/rating_tables.py::_ppform_reachable_relativities": (20, PR571),
    "export/rating_tables.py::_require_log_link_export": (23, PR571),
    "export/rating_tables.py::_require_supported_offset_kind": (13, PR571),
    "export/rating_tables.py::_require_unclamped_response_export": (31, PR571),
    "export/rating_tables.py::_require_unsaturated_predictor_export": (25, PR571),
    "export/rating_tables.py::_require_usable_offset_multipliers": (39, PR571),
    "export/rating_tables.py::_require_usable_relativities_export": (52, PR571),
    "export/rating_tables.py::_significant_digits": (20, PR571),
    "export/rating_tables.py::_spline_categorical_block": (18, PR571),
    "export/rating_tables.py::_total_centering_shift": (27, PR571),
    "features/spline.py::_BSplineBase._assemble_knot_vector": (13, PR571),
    "features/spline.py::_SplineBase": (53, PR571),
    "features/spline.py::_SplineBase.build_knots_and_penalty": (18, PR571),
    "model/fit_state.py::_carry_revised_columns": (20, PR571),
    "model/fit_state.py::_read_solver_state_from_public": (18, PR571),
    "model/report_ops.py::_withhold_repaired_term_inference": (32, PR571),
    "reml/objective.py::_value_roundoff": (16, PR571),
    "reml/penalty_algebra.py::_kronecker_margins": (18, PR571),
    "reml/penalty_algebra.py::_penalty_range_basis": (18, PR571),
    "reml/penalty_algebra.py::_plain_single_raw_route": (26, PR571),
    "reml/penalty_algebra.py::_single_penalty_raw_family": (43, PR571),
    "reml/penalty_algebra.py::_solver_penalty_agreement": (39, PR571),
    "reml/penalty_algebra.py::build_penalty_components._rank_and_logdet": (27, PR571),
    "screening/_pair_factor.py::_pair_scale": (33, PR571),
    "screening/_pair_factor.py::_profiled_factor": (43, PR571),
    "screening/_pair_factor.py::_profiled_rank_scale": (28, PR571),
    "solvers/irls_direct/_fit.py::_IRLSDirectCurvaturePolicy": (22, PR571),
    "solvers/irls_direct/_fit.py::_IRLSDirectFit": (129, PR571),
    "solvers/irls_direct/_fit.py::_authoritative_gram_rank": (17, PR571),
    "solvers/irls_direct/_mean_space.py::_IRLSDirectMeanSpaceOverlay": (28, PR571),
    "solvers/irls_direct/_mean_space.py::_IRLSDirectMeanSpaceOverlay.certify": (55, PR571),
    "solvers/irls_direct/_mean_space.py::_IRLSDirectMeanSpaceOverlay.line_search": (14, PR571),
    "solvers/irls_direct/_search.py::_IRLSDirectTrialSearch": (29, PR571),
    "solvers/irls_state.py::_mean_space_halving_budget": (18, PR571),
    "solvers/irls_state.py::_poisson_sqrt_halving_budget": (13, PR571),
    "solvers/irls_state.py::_stable_penalized_deviance_delta": (23, PR571),
    "solvers/mode_score.py::_form_row_sets": (34, PR571),
    "solvers/mode_score.py::_joint_cell_sets": (32, PR571),
    "solvers/mode_score.py::_scaled_ratio": (13, PR571),
    "solvers/rank.py::_compact_factor_qr": (14, PR577),
    "solvers/rank.py::_conditioned_representatives": (51, PR577),
    "solvers/rank.py::_decompose_gram": (22, PR577),
    "solvers/rank.py::_earliest_representatives": (42, PR577),
    "solvers/rank.py::_eigensolver_relative_bar": (127, PR577),
    "solvers/rank.py::_factor_representative_clears_cutoff": (31, PR577),
    "solvers/rank.py::_formation_slack": (15, PR577),
    "solvers/rank.py::_leverage_pivot_representatives": (59, PR577),
    "solvers/rank.py::_null_basis": (14, PR577),
    "solvers/rank.py::_principal_block_clears_cutoff": (21, PR577),
    "solvers/rank.py::_selection_amplification": (26, PR577),
    "solvers/rank.py::_shifted_cholesky_clears": (15, PR577),
    "solvers/rank.py::_tsqr_weighted_factor": (29, PR577),
}


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _docstring_lengths(node: ast.AST, prefix: str = "", private: bool = False):
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield from _docstring_lengths(child, prefix, private)
            continue
        name, hidden = prefix + child.name, private or _is_private(child.name)
        first = child.body[0]
        if hidden and isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                yield name, first.value.end_lineno - first.value.lineno + 1
        yield from _docstring_lengths(child, name + ".", hidden)


@cache
def _long_docstrings() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, lines in _docstring_lengths(tree):
            key = f"{path.relative_to(SRC).as_posix()}::{name}"
            if lines > LIMIT:
                found[key] = max(lines, found.get(key, 0))
    return found


def test_no_new_long_private_docstring():
    new = {k: n for k, n in _long_docstrings().items() if k not in ALLOWLIST}
    assert not new, f"private docstrings over {LIMIT} lines; shorten them to 6-12: {new}"


def test_allowlist_only_shrinks():
    stale = sorted(set(ALLOWLIST) - set(_long_docstrings()))
    assert not stale, f"gone or now within {LIMIT} lines; delete these allowlist entries: {stale}"


def test_allowlisted_docstrings_keep_their_recorded_length():
    found = _long_docstrings()
    moved = {k: (found[k], n) for k, (n, _) in ALLOWLIST.items() if k in found and found[k] != n}
    assert not moved, (
        "allowlisted docstrings differ from their recorded length (now, recorded); "
        f"lower the count of a shorter one, shorten a longer one: {moved}"
    )
