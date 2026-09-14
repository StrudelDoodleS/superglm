"""Audit entry points must fail closed when Python disables assertions."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RESEARCH = Path(__file__).resolve().parents[1] / "docs" / "research"


def load_tool(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, RESEARCH / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script", ["check_broad_interaction_measurements.py", "plot_broad_interaction_surfaces.py"]
)
@pytest.mark.parametrize("optimization", ["flag", "environment"])
def test_audit_entry_points_reject_disabled_assertions(script, optimization):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    flags = ["-O"] if optimization == "flag" else []
    if optimization == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    command = [
        sys.executable,
        *flags,
        "-c",
        "import runpy, sys; runpy.run_path(sys.argv[1])",
        str(RESEARCH / script),
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "requires enabled assertions" in result.stderr


@pytest.mark.parametrize(
    "script", ["check_broad_interaction_measurements.py", "plot_broad_interaction_surfaces.py"]
)
def test_data_identity_allows_relocation_but_preserves_content_and_split_evidence(script):
    tool = load_tool(script)
    original = {
        "source_path": "/old/data/source.csv",
        "source_registry": "/old/checkout/manifest.json",
        "data_sha256": "data bytes",
        "source_registry_sha256": "registry bytes",
        "split_sha256": "split positions",
        "preprocessing_sha256": "preprocessor",
        "adapter_source_sha256": "adapter bytes",
        "source_bytes_verified": True,
    }
    relocated = {
        **original,
        "source_path": "/new/cache/source.csv",
        "source_registry": "/new/checkout/manifest.json",
    }
    assert tool.data_identity(original) == tool.data_identity(relocated)
    for key in original.keys() - {"source_path", "source_registry"}:
        changed = {**relocated, key: False if key == "source_bytes_verified" else "changed"}
        assert tool.data_identity(original) != tool.data_identity(changed), key


def test_audit_rederives_all_100_arms_and_rejects_an_omission():
    audit = load_tool("check_broad_interaction_measurements.py")
    measurement = json.loads(
        (RESEARCH / "2026-09-14-broad-interaction-measurements.json").read_text()
    )
    counts = []
    for case in measurement["datasets"].values():
        admission = {
            "pairs": case["admitted_pairs"],
            "parent_resolutions": sorted({fit["parent_k"] for fit in case["fits"].values()}),
        }
        counts.append(audit.check_menu(measurement["protocol"], admission, case["fits"]))
        incomplete = dict(case["fits"])
        incomplete.pop(next(iter(incomplete)))
        with pytest.raises(ValueError, match="arm menu"):
            audit.check_menu(measurement["protocol"], admission, incomplete)
    assert sum(counts) == 100


@pytest.mark.parametrize("matching", [None, "k6_s0"])
def test_missing_matching_control_remains_an_unavailable_comparison(matching):
    audit = load_tool("check_broad_interaction_measurements.py")
    summaries = {
        "k4_s0": {"pairs": [], "fit_seconds": 1, "evaluation": {"test": {"primary_loss": 4}}},
        "k6_s1": {
            "pairs": [["x", "z"]],
            "fit_seconds": 2,
            "evaluation": {"test": {"primary_loss": 1}},
        },
        "k6_s0": {"status": "not_converged", "fit_seconds": 3},
    }
    choice = {
        "chosen_arm": "k6_s1",
        "additive_arm": "k4_s0",
        "matching_additive_arm": matching,
        "matching_comparison_status": "unavailable",
    }
    comparison = audit.compare_selected_models(choice, summaries)
    assert comparison["test_comparison"]["vs_best_additive_percent"] == 75
    assert comparison["test_comparison"]["vs_matching_k_percent"] is None
    assert comparison["selected_fit_ratio_vs_matching_k"] is None
    assert comparison["outcome"] == "matching_additive_unavailable"


def test_one_pair_plot_writes_artifacts_and_hides_the_unused_axis(tmp_path, monkeypatch):
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    plot = load_tool("plot_broad_interaction_surfaces.py")
    monkeypatch.setattr(plot, "OUTPUT", tmp_path)
    closed = []
    close = plot.plt.close

    def capture(figure):
        if hasattr(figure, "axes"):
            closed.append(figure)
        close(figure)

    monkeypatch.setattr(plot.plt, "close", capture)
    surface = {
        "name": "frequency:attack-angle",
        "parents": ["frequency", "attack-angle"],
        "axes": [np.array([100.0, 1000.0]), np.array([1.0, 2.0])],
        "observed": [np.array([200.0, 500.0]), np.array([1.2, 1.8])],
        "effect": np.array([[-2.0, 1.0], [2.0, -1.0]]),
        "supported": np.ones((2, 2), dtype=bool),
    }
    case = {"choice": {"chosen_arm": "k4_s1"}, "test_comparison": {"vs_best_additive_percent": 1}}
    with PdfPages(tmp_path / "test.pdf") as pdf:
        receipt = plot.plot_case(
            "uci_airfoil", case, {"model_pickle_sha256": "fixture"}, [surface], pdf
        )
    assert len(receipt["surfaces"]) == 1
    assert (tmp_path / "airfoil.png").is_file()
    assert (tmp_path / "airfoil.svg").is_file()
    assert not closed[-1].axes[1].axison
