"""scripts/update_test_durations.py: junit cases map back to the ids pytest-split looks up."""

from __future__ import annotations

import json

from scripts.update_test_durations import COVERAGE_RUNTIME, main, node_id, read_times

CASES = """<testsuites><testsuite>
<testcase classname="tests.test_ci_contracts" name="test_module_level" time="1.0"/>
<testcase classname="tests.test_api.TestBinomialDistribution" name="test_method" time="2.0"/>
<testcase classname="tests.editor.test_editor_knots_browser" name="test_nested[a.b::c]" time="3.0"/>
<testcase classname="" name="tests.docs.test_notebooks" time="0.0"/>
<testcase classname="tests.test_renamed_away" name="test_gone" time="4.0"/>
</testsuite></testsuites>"""


def test_a_junit_case_maps_to_its_pytest_node_id() -> None:
    assert node_id("tests.test_ci_contracts", "test_x") == "tests/test_ci_contracts.py::test_x"
    # the longest dotted prefix that is a file: a class, then its method
    assert (
        node_id("tests.test_api.TestBinomialDistribution", "test_m")
        == "tests/test_api.py::TestBinomialDistribution::test_m"
    )
    # a test in a subpackage, with a parametrized id holding "." and "::"
    assert (
        node_id("tests.editor.test_editor_knots_browser", "test_n[a.b::c]")
        == "tests/editor/test_editor_knots_browser.py::test_n[a.b::c]"
    )
    assert node_id("", "tests.docs.test_notebooks") is None  # a module skipped at collection
    assert node_id("tests.test_renamed_away", "test_gone") is None  # no such file here


def test_the_refresh_averages_runs_and_keeps_tests_ci_never_ran(tmp_path, capsys) -> None:
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        scale = 1.0 if run == "a" else 3.0
        text = CASES.replace('time="1.0"', f'time="{scale}"')
        (tmp_path / run / "pytest-results.xml").write_text(text, encoding="utf-8")
    times, unmapped = read_times(sorted(tmp_path.rglob("pytest-results.xml")))
    assert times["tests/test_ci_contracts.py::test_module_level"] == [1.0, 3.0]
    assert unmapped == {"::tests.docs.test_notebooks", "tests.test_renamed_away::test_gone"}

    output = tmp_path / "durations.json"
    output.write_text(
        json.dumps({"tests/test_ci_contracts.py::test_module_level": 9.0, "tests/x.py::kept": 5.0})
    )
    assert main([str(tmp_path / "a"), str(tmp_path / "b"), "--output", str(output)]) == 0
    merged = json.loads(output.read_text())
    assert merged["tests/test_ci_contracts.py::test_module_level"] == 2.0  # the mean of 1 and 3
    assert merged["tests/x.py::kept"] == 5.0
    reported = capsys.readouterr().out
    assert "1 module-level skips unmapped" in reported
    assert "1 cases name no file in this checkout" in reported


def test_a_coverage_runtime_given_by_path_is_flagged(tmp_path, capsys) -> None:
    run = tmp_path / f"pytest-Linux-X64{COVERAGE_RUNTIME}1"
    run.mkdir()
    (run / "pytest-results.xml").write_text(CASES, encoding="utf-8")
    main([str(run), "--output", str(tmp_path / "durations.json")])
    assert "ran under coverage" in capsys.readouterr().err
