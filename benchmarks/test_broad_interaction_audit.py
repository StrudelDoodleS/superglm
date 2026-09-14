"""Audit entry points must fail closed when Python disables assertions."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RESEARCH = Path(__file__).resolve().parents[1] / "notes" / "research"


def load_tool(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, RESEARCH / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script",
    [
        "check_broad_interaction_measurements.py",
        "plot_broad_interaction_surfaces.py",
        "check_sympy_quadratic_gap.py",
        "check_sympy_interaction_factors.py",
    ],
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
def test_source_root_selects_the_package_used_for_replay(script, tmp_path):
    source = tmp_path / "source"
    package = source / "src" / "superglm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# Import-location fixture; no fit code is needed.\n")
    (source / "benchmarks").symlink_to(RESEARCH.parents[1] / "benchmarks", target_is_directory=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1]); import superglm; print(superglm.__file__)",
            str(RESEARCH / script),
            "--source-root",
            str(source),
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (package / "__init__.py").resolve()


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


@pytest.mark.parametrize("changed_field", ["pairs", "parent_resolutions"])
def test_audit_rederives_admission_before_trusting_the_menu(tmp_path, monkeypatch, changed_field):
    audit = load_tool("check_broad_interaction_measurements.py")
    state = {"features": {"x": {"kind": "spline"}, "z": {"kind": "spline"}}}
    pairs = [["x", "z"]]
    admission = audit.broad.admit_pairs(state, pairs)
    admission[changed_field] = [] if changed_field == "pairs" else [4]
    metadata = {"fixture": True}
    protocol = {"source": "fixture", "prefix_counts": list(audit.broad.COUNTS)}
    proposal = {
        "status": "proposed",
        "data": metadata,
        "data_identity_sha256": audit.base.digest_json(metadata),
        "proposal": {"pairs": pairs},
        "admission": admission,
    }
    suite = {"protocol": protocol, "datasets": {"fixture": {"proposal": proposal, "arms": {}}}}
    audit.base.write_json(tmp_path / "protocol.json", protocol)
    audit.base.write_json(tmp_path / "suite.json", suite)
    monkeypatch.setattr(audit.broad, "source_identity", lambda: "fixture")
    monkeypatch.setattr(
        audit.data, "load_prepared", lambda *args, **kwargs: {"metadata": metadata, "state": state}
    )
    with pytest.raises(ValueError, match="admission"):
        audit.summarize(tmp_path)


def test_audit_explains_the_scope_boundary_without_an_additive_comparator():
    audit = load_tool("check_broad_interaction_measurements.py")
    with pytest.raises(ValueError, match="scope.*converged additive"):
        audit.compare_selected_models({"chosen_arm": None, "additive_arm": None}, {})


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


@pytest.fixture
def isolated_plot_replay(tmp_path, monkeypatch):
    # Execute the real CLI from a relocated copy, with synthetic inputs and
    # one harmless PDF page. The frozen receipt must survive an ordinary replay.
    research = tmp_path / "checkout" / "notes" / "research"
    research.mkdir(parents=True)
    script = research / "plot_broad_interaction_surfaces.py"
    script.write_bytes((RESEARCH / script.name).read_bytes())
    spec = importlib.util.spec_from_file_location("isolated_plot_replay", script)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    archive = research / "figures" / "2026-09-14-broad-interactions"
    archive.mkdir(parents=True)
    (archive / "receipt.json").write_text("original frozen evidence\n")
    (research / "2026-09-14-broad-interaction-measurements.json").write_text(
        json.dumps({"protocol": {"source": "fixture"}})
    )
    monkeypatch.setattr(plot.broad, "source_identity", lambda: "fixture")
    monkeypatch.setattr(plot, "CASES", {"fixture": None})
    monkeypatch.setattr(plot, "case_surfaces", lambda *args, **kwargs: ({}, {}, []))

    def one_page(dataset, case, fitted, surfaces, pdf):
        figure = plot.plt.figure(figsize=(1, 1))
        pdf.savefig(figure)
        plot.plt.close(figure)
        return {"fixture": True}

    monkeypatch.setattr(plot, "plot_case", one_page)
    monkeypatch.setattr(sys, "argv", [str(script)])
    return plot, archive


def test_default_plot_replay_preserves_the_frozen_archive(isolated_plot_replay):
    plot, archive = isolated_plot_replay
    plot.main()
    assert (archive / "receipt.json").read_text() == "original frozen evidence\n"
    assert [item.name for item in archive.iterdir()] == ["receipt.json"]
    assert (
        plot.REPO / ".benchmark-artifacts/broad-interaction-surfaces-replay/receipt.json"
    ).is_file()


def test_plot_replay_refuses_the_frozen_archive_as_output(isolated_plot_replay, monkeypatch):
    plot, archive = isolated_plot_replay
    monkeypatch.setattr(sys, "argv", [plot.__file__, "--output", str(archive)])
    with pytest.raises(SystemExit) as error:
        plot.main()
    assert error.value.code == 2
    assert (archive / "receipt.json").read_text() == "original frozen evidence\n"


def test_audit_refuses_a_dataset_skipped_before_proposal_without_loading_data(
    tmp_path, monkeypatch
):
    audit = load_tool("check_broad_interaction_measurements.py")
    protocol = {"source": "fixture"}
    case = {
        "status": "budget_exhausted",
        "arms": {},
        "search_worker_process_seconds": 0.0,
        "evaluation_worker_process_seconds": 0.0,
        "all_worker_process_seconds": 0.0,
    }
    audit.base.write_json(tmp_path / "protocol.json", protocol)
    audit.base.write_json(
        tmp_path / "suite.json", {"protocol": protocol, "datasets": {"late": case}}
    )
    monkeypatch.setattr(audit.broad, "source_identity", lambda: "fixture")

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Skipped datasets must be refused before data loading")

    monkeypatch.setattr(audit.data, "load_prepared", unexpected_load)
    with pytest.raises(ValueError, match="scope.*skipped before proposal.*late"):
        audit.summarize(tmp_path)
