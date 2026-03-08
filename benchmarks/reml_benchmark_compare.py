"""Compare SuperGLM and mgcv REML benchmark results.

Reads JSON outputs from both harnesses and prints a formatted comparison
table with wall time, deviance, EDF, scale, and iteration counts.

Usage:
    uv run python benchmarks/reml_benchmark_compare.py
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_results(tool: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, f"{tool}_results.json")
    if not os.path.exists(path):
        print(f"  {path} not found — run the {tool} harness first")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {path} ({data.get('timestamp', '?')})")
    return data.get("results", [])


def format_table(rows: list[dict], title: str):
    """Print a formatted comparison table."""
    if not rows:
        return

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    header = (
        f"{'Name':<42s} {'Time':>7s} {'Dev':>10s} {'EDF':>7s} "
        f"{'Scale':>8s} {'Iters':>6s} {'Conv':>5s}"
    )
    print(header)
    print("-" * 90)

    for r in rows:
        name = r.get("name", "?")
        time_s = r.get("wall_time_s", 0)
        dev = r.get("deviance", 0)
        edf = r.get("effective_df", 0)
        scale = r.get("scale", r.get("phi", 0))
        iters = r.get("n_reml_iter", r.get("n_outer_iter", "?"))
        conv = r.get("converged", "?")

        print(
            f"  {name:<40s} {time_s:>6.2f}s {dev:>10.1f} {edf:>7.2f} "
            f"{scale:>8.4f} {str(iters):>6s} {str(conv):>5s}"
        )

    print()


def match_and_compare(sg_results: list[dict], mgcv_results: list[dict]):
    """Match comparable runs and print side-by-side comparison."""
    # Group by dataset/family
    comparisons = {
        "Synthetic Poisson (n=800)": {
            "superglm": [r for r in sg_results if "synthetic_poisson" in r["name"]],
            "mgcv": [r for r in mgcv_results if "synthetic_poisson" in r["name"]],
        },
        "Synthetic Gamma (n=800)": {
            "superglm": [r for r in sg_results if "synthetic_gamma" in r["name"]],
            "mgcv": [r for r in mgcv_results if "synthetic_gamma" in r["name"]],
        },
        "MTPL2 Poisson (678k)": {
            "superglm": [r for r in sg_results if "mtpl2" in r["name"]],
            "mgcv": [r for r in mgcv_results if "mtpl2" in r["name"]],
        },
    }

    for title, groups in comparisons.items():
        all_rows = []
        for tool_name, runs in groups.items():
            for r in runs:
                r["name"] = f"[{tool_name}] {r['name']}"
                all_rows.append(r)
        if all_rows:
            format_table(all_rows, title)


def print_lambda_comparison(sg_results: list[dict], mgcv_results: list[dict]):
    """Print smoothing parameter comparison for matched runs.

    NOTE: Raw lambda/sp values are NOT directly comparable between SuperGLM
    and mgcv due to different basis parametrizations (SSP reparametrized
    B-splines vs mgcv's cr basis with sum-to-zero constraint).  We show
    them for reference but compare fit outcomes (deviance, EDF, scale)
    as the primary parity metrics.
    """
    for family in ["poisson", "gamma"]:
        sg_exact = next(
            (r for r in sg_results if r["name"].startswith(f"synthetic_{family}_exact")),
            None,
        )
        mgcv_gam = next(
            (r for r in mgcv_results if f"synthetic_{family}_gam" in r.get("name", "")),
            None,
        )

        if sg_exact and mgcv_gam:
            print(f"  Smoothing parameters — synthetic {family} (not directly comparable):")
            sg_lam = sg_exact.get("lambdas", {})
            mgcv_sp = mgcv_gam.get("smoothing_params", {})

            print(f"    {'Term':<20s} {'SuperGLM λ':>12s} {'mgcv sp':>12s}")
            print(f"    {'-' * 48}")

            mgcv_keys = list(mgcv_sp.keys())
            sg_keys = sorted(sg_lam.keys())

            for i, sg_key in enumerate(sg_keys):
                sg_val = sg_lam[sg_key]
                if i < len(mgcv_keys):
                    mgcv_val = mgcv_sp[mgcv_keys[i]]
                    print(f"    {sg_key:<20s} {sg_val:>12.4f} {mgcv_val:>12.4f}")
                else:
                    print(f"    {sg_key:<20s} {sg_val:>12.4f} {'—':>12s}")
            print()


def main():
    print("=" * 90)
    print("  REML Benchmark Comparison: SuperGLM vs mgcv")
    print("=" * 90)

    print("\nLoading results...")
    sg_results = load_results("superglm")
    mgcv_results = load_results("mgcv")

    if not sg_results and not mgcv_results:
        print("\nNo results found. Run the harness scripts first:")
        print("  uv run python scratch/benchmarks/reml_benchmark_harness.py")
        print("  Rscript scratch/benchmarks/reml_benchmark_harness.R")
        sys.exit(1)

    # Print individual tables
    if sg_results:
        format_table(sg_results, "SuperGLM Results")
    if mgcv_results:
        format_table(mgcv_results, "mgcv Results")

    # Print side-by-side comparison
    if sg_results and mgcv_results:
        # Work on copies so we don't mutate originals for lambda comparison
        sg_copy = [dict(r) for r in sg_results]
        mgcv_copy = [dict(r) for r in mgcv_results]
        match_and_compare(sg_copy, mgcv_copy)

        print("── Smoothing parameter comparison ──")
        print_lambda_comparison(sg_results, mgcv_results)

    # Print summary
    print("── Summary ──")
    if sg_results:
        synth_exact = [r for r in sg_results if "synthetic" in r["name"] and "exact" in r["name"]]
        synth_disc = [r for r in sg_results if "synthetic" in r["name"] and "disc" in r["name"]]
        if synth_exact and synth_disc:
            avg_exact = sum(r["wall_time_s"] for r in synth_exact) / len(synth_exact)
            avg_disc = sum(r["wall_time_s"] for r in synth_disc) / len(synth_disc)
            print(
                f"  Synthetic avg: exact={avg_exact:.2f}s, disc={avg_disc:.2f}s "
                f"(disc/exact={avg_disc / avg_exact:.2f}x)"
            )

        mtpl2 = [r for r in sg_results if "mtpl2" in r["name"]]
        for r in mtpl2:
            print(
                f"  {r['name']}: {r['wall_time_s']:.2f}s, "
                f"dev={r['deviance']:.1f}, edf={r['effective_df']:.2f}"
            )

    if mgcv_results:
        for r in mgcv_results:
            name = r.get("name", "?")
            print(
                f"  {name}: {r.get('wall_time_s', 0):.2f}s, "
                f"dev={r.get('deviance', 0):.1f}, edf={r.get('effective_df', 0):.2f}"
            )

    print()


if __name__ == "__main__":
    main()
