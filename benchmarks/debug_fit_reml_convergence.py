from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import BSplineSmooth, Constraint, PSpline, SuperGLM
from superglm.model.reml_debug import (
    list_reml_debug_run_ids,
    load_reml_debug_run,
    plot_reml_debug_trajectory,
    summarize_reml_debug_run,
    write_reml_debug_summary_csv,
)

RESULTS_DIR = Path("benchmarks/results/reml_debug")
SUMMARY_CSV = RESULTS_DIR / "convergence_summary.csv"


@dataclass(frozen=True)
class Scenario:
    name: str
    engine: str
    discrete: bool
    n_constrained: int


def _make_dataset(n: int, n_constrained: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    eta = np.full(n, -0.1)

    for j in range(n_constrained):
        x = np.sort(rng.uniform(0.0, 1.0, size=n))
        data[f"x{j + 1}"] = x
        eta += 0.4 * x + 0.8 * x**2

    y = eta + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame(data), y.astype(np.float64)


def _make_features(engine: str, n_constrained: int) -> dict[str, object]:
    features: dict[str, object] = {}
    for j in range(n_constrained):
        name = f"x{j + 1}"
        constraint = Constraint.fit.convex if j % 2 == 0 else Constraint.fit.concave
        if engine == "scop":
            features[name] = PSpline(n_knots=10, constraint=constraint)
        elif engine == "qp":
            features[name] = BSplineSmooth(n_knots=10, constraint=constraint)
        else:
            raise ValueError(f"Unsupported engine: {engine!r}")
    return features


def _run_scenario(scenario: Scenario, *, n: int, seed: int, max_reml_iter: int) -> tuple[dict, str]:
    X, y = _make_dataset(n=n, n_constrained=scenario.n_constrained, seed=seed)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=scenario.discrete,
        features=_make_features(scenario.engine, scenario.n_constrained),
    )

    before = set(list_reml_debug_run_ids(RESULTS_DIR))
    model.fit_reml(X, y, max_reml_iter=max_reml_iter)
    after = set(list_reml_debug_run_ids(RESULTS_DIR))
    new_ids = sorted(after - before)
    if len(new_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one new debug run for {scenario.name}, found {new_ids!r}"
        )

    run_id = new_ids[0]
    run = load_reml_debug_run(RESULTS_DIR, run_id)
    summary = summarize_reml_debug_run(run)
    summary.update(
        {
            "scenario_name": scenario.name,
            "engine": scenario.engine,
            "mode": "discrete" if scenario.discrete else "exact",
            "n_constrained": scenario.n_constrained,
        }
    )
    return summary, run_id


def _plot_step_norms(run, output_path: Path) -> Path | None:
    if not run.scop_rows:
        return None

    xs = list(range(1, len(run.scop_rows) + 1))
    ys = [float(row.get("step_norm", 0.0)) for row in run.scop_rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, ys, marker="o", color="#2a9d8f")
    ax.set_xlabel("SCOP step row index")
    ax.set_ylabel("step norm")
    ax.set_title(f"SCOP step norms: {run.run_id}")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    os.environ.setdefault("SUPERGLM_DEBUG", "2")
    os.environ.setdefault("SUPERGLM_DEBUG_DIR", str(RESULTS_DIR))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        Scenario("single_scop_discrete", "scop", True, 1),
        Scenario("multi_scop_discrete", "scop", True, 2),
        Scenario("single_scop_exact", "scop", False, 1),
        Scenario("single_qp_discrete", "qp", True, 1),
    ]

    summary_rows = []
    for idx, scenario in enumerate(scenarios, start=1):
        print(
            f"[{idx}/{len(scenarios)}] running {scenario.name} "
            f"(engine={scenario.engine}, mode={'discrete' if scenario.discrete else 'exact'}, "
            f"n_constrained={scenario.n_constrained})"
        )
        summary, run_id = _run_scenario(
            scenario,
            n=20_000 if scenario.n_constrained == 1 else 50_000,
            seed=42 + idx,
            max_reml_iter=20,
        )
        summary_rows.append(summary)
        print(
            f"done {scenario.name}: reml_iters={summary['reml_iterations']} "
            f"strict={summary['strict_converged']} plateau={summary['plateau_converged']}"
        )
        run = load_reml_debug_run(RESULTS_DIR, run_id)
        traj = plot_reml_debug_trajectory(
            run,
            RESULTS_DIR / f"{scenario.name}_trajectory.png",
            title=f"{scenario.name} trajectory",
        )
        print(f"wrote {traj}")
        step_plot = _plot_step_norms(run, RESULTS_DIR / f"{scenario.name}_scop_steps.png")
        if step_plot is not None:
            print(f"wrote {step_plot}")

    summary_path = write_reml_debug_summary_csv(summary_rows, SUMMARY_CSV)
    print(f"wrote {summary_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
