"""Small read-only check of exceptional PSST pencil reconstruction.

Run from the psst-reference-variance worktree with:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    uv run --no-sync python benchmarks/screening_extreme_penalty.py --output /tmp/closure.json

The normalized residuals are
diagnostics, not certified bounds for arbitrary ill-conditioned inputs.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

import superglm.screening._score_stat as score_stat
from superglm.screening._pair_factor import PairFactor


def make_pair(data_factor):
    width = data_factor.shape[1]
    joint = np.eye(width + 1)
    joint[:width, :width] = data_factor
    joint[:width, -1] = 1.0
    return PairFactor(joint=joint, overlap_width=0, tensor_width=width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(713)
    rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    rotated_data = np.array(
        [[1.0, 0.3, -0.2, 0.1], [0.0, 1.5, 0.4, -0.1], [0.0, 0.0, 0.8, 0.2], [0.0, 0.0, 0.0, 1.2]]
    )
    rotated_penalty = np.diag([0.5, 1.0, 1.4, 2.0]) @ rotation
    tiny = np.finfo(float).smallest_subnormal
    fixtures = [
        ("uniform-huge", np.eye(4), np.ldexp(np.eye(4), 538), tiny),
        ("mixed-null-huge", np.eye(4), np.ldexp(np.diag([0.0, 0.5, 1.0, 2.0]), 538), tiny),
        ("mixed-magnitudes", np.eye(2), np.diag([np.ldexp(1.0, 538), 1.0]), 1.0),
        ("tiny-curvature", np.diag([1.0, np.ldexp(1.0, -540)]), np.ldexp(np.eye(2), -540), 1.0),
        ("rotated-full-rank", rotated_data, np.ldexp(rotated_penalty, 538), tiny),
    ]
    records = []
    for name, data_factor, physical_root, lam in fixtures:
        captured = {}

        def capture_return(frame, event, arg):
            if frame.f_code is score_stat._pair_pencil.__code__ and event == "return":
                captured.update(frame.f_locals)

        previous_profiler = sys.getprofile()
        sys.setprofile(capture_return)
        try:
            pencil = score_stat._pair_pencil(make_pair(data_factor), physical_root)
        finally:
            sys.setprofile(previous_profiler)

        rank = captured["rank"]
        triangular = captured["triangular"][:rank, :rank]
        top, bottom = captured["top"], captured["bottom"]
        recovered_q = np.vstack((top, bottom))
        selected = captured["pivot"][:rank]
        selected_data = captured["R_eff"][:, selected]
        selected_penalty = captured["root"][:, selected]
        dimension = max(captured["stack"].shape)
        eps = np.finfo(float).eps
        condition = float(np.linalg.cond(triangular))
        closure = float(np.linalg.norm(recovered_q.T @ recovered_q - np.eye(rank), 2))
        data_residual = float(
            np.linalg.norm(top @ triangular - selected_data) / np.linalg.norm(selected_data)
        )
        penalty_residual = float(
            np.linalg.norm(bottom @ triangular - selected_penalty)
            / np.linalg.norm(selected_penalty)
        )
        actual = np.array(score_stat._scaled_pencil_moments(pencil, lam))
        if name == "rotated-full-rank":
            v = data_factor.T @ data_factor
            normal_matrix = v + 4.0 * rotated_penalty.T @ rotated_penalty
            left_factor = scipy.linalg.solve_triangular(
                np.linalg.cholesky(normal_matrix), data_factor.T, lower=True
            )
            filter_matrix = left_factor @ left_factor.T
            score = data_factor.T @ np.ones(4)
            expected = np.array(
                [
                    np.trace(filter_matrix),
                    score @ np.linalg.solve(normal_matrix, score),
                    2.0 * np.sum(filter_matrix**2),
                ]
            )
            oracle_description = (
                "Finite SPD oracle: A=R.T@R+4*P.T@P; "
                "H=solve(cholesky(A),R.T); EDF=tr(H@H.T), "
                "T=(R.T@ones).T@solve(A,R.T@ones), variance=2*||H@H.T||_F^2."
            )
        else:
            filters = {
                "uniform-huge": np.full(4, 0.2),
                "mixed-null-huge": 1.0 / (1.0 + 4.0 * np.array([0.0, 0.5, 1.0, 2.0]) ** 2),
                "mixed-magnitudes": np.array([0.0, 0.5]),
                "tiny-curvature": np.array([1.0, 0.5]),
            }[name]
            expected = np.array([np.sum(filters), np.sum(filters), 2.0 * np.sum(filters**2)])
            oracle_description = (
                "Diagonal analytic filters with z=ones; moments=(sum(a),sum(a),2*sum(a^2)). "
                "For mixed-magnitudes the first contribution is below float64's "
                "minimum subnormal and cannot change these rounded totals."
            )
        moment_error = float(np.max(np.abs(actual - expected)))
        moment_scale = dimension * eps * condition**2 * max(1.0, float(np.max(np.abs(expected))))
        record = {
            "fixture": name,
            "R_eff_input": data_factor.tolist(),
            "physical_penalty_root": physical_root.tolist(),
            "lambda": float(lam),
            "stack_shape": list(captured["stack"].shape),
            "rank": rank,
            "dimension": dimension,
            "epsilon": eps,
            "condition_R11_2": condition,
            "orthogonality_defect_2": closure,
            "orthogonality_over_dimension_epsilon_condition": closure
            / (dimension * eps * condition),
            "relative_data_reconstruction_F": data_residual,
            "relative_penalty_reconstruction_F": penalty_residual,
            "data_reconstruction_over_dimension_epsilon": data_residual / (dimension * eps),
            "penalty_reconstruction_over_dimension_epsilon": penalty_residual / (dimension * eps),
            "moments_order": ["EDF", "statistic", "reference_variance"],
            "actual_moments": actual.tolist(),
            "oracle_moments": expected.tolist(),
            "oracle_description": oracle_description,
            "maximum_absolute_moment_error": moment_error,
            "moment_error_over_dimension_epsilon_condition_squared_scale": moment_error
            / moment_scale,
        }
        records.append(record)
        print(json.dumps(record, allow_nan=False))

    output = {
        "source_path": str(Path(score_stat.__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(score_stat.__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scope": "Five tiny fixtures; residual diagnostics do not certify arbitrary ill-conditioned factors.",
        "orthogonality_formula": "||[Q1;Q2].T@[Q1;Q2]-I||_2 / (d*eps*cond_2(R11))",
        "reconstruction_formula": "(||Qi@R11-Ai_selected||_F/||Ai_selected||_F)/(d*eps)",
        "dimension_formula": "d=max(shape(vstack((R_eff,balanced_penalty_root))))",
        "fixtures": records,
    }
    output_path = args.output
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
