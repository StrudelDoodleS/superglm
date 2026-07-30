"""Compare 30-rep timing results from SuperGLM and reference.

Usage:
    uv run python benchmarks/timing_30rep_compare.py
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}_30rep.json")
    if not os.path.exists(path):
        print(f"  {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    sg = load("superglm")
    reference = load("reference")

    if not sg and not reference:
        print("No results found. Run the timing scripts first.")
        return

    print("=" * 70)
    print("  30-rep Timing Comparison: SuperGLM vs reference (MTPL2 678k)")
    print("=" * 70)

    for label, d in [
        ("SuperGLM (discrete cached-W)", sg),
        ("reference bam (fREML discrete)", reference),
    ]:
        if d is None:
            continue
        print(f"\n  {label}")
        print(f"  {'─' * 50}")
        print(f"    threads:  {d.get('threads', '?')}")
        print(f"    warmup:   {d['warmup_s']:.3f}s")
        print(f"    median:   {d['median_s']:.3f}s")
        print(f"    mean:     {d['mean_s']:.3f}s")
        print(f"    std:      {d['std_s']:.3f}s")
        print(f"    min:      {d['min_s']:.3f}s")
        print(f"    max:      {d['max_s']:.3f}s")
        print(f"    p10:      {d['p10_s']:.3f}s")
        print(f"    p90:      {d['p90_s']:.3f}s")
        print(f"    deviance: {d['deviance']:.1f}")
        print(f"    edf:      {d['effective_df']:.2f}")

    if sg and reference:
        ratio = sg["median_s"] / reference["median_s"]
        print(f"\n  {'=' * 50}")
        print(f"  Ratio (median SuperGLM / median reference): {ratio:.2f}x")
        print(f"  SuperGLM median: {sg['median_s']:.3f}s")
        print(f"  reference median:     {reference['median_s']:.3f}s")
        print(f"  {'=' * 50}")


if __name__ == "__main__":
    main()
