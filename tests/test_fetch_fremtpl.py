"""The job that gives CI the real book, and the fetch it depends on.

``SUPERGLM_REQUIRE_DATA`` turns a dataset skip into a failure, but on its own it
changes nothing: with no parquet anywhere, CI simply had no data to require.
The half that closes #261 is a job that *fetches* the datasets, so the switch
has something to enforce against.

Two things can silently undo that, and both are covered here:

* the job stops covering a suite -- someone adds a fourth data-guarded module,
  or drops one from the pytest invocation, and it goes back to skipping in CI
  with nothing to say so;
* the fetch stops being the artifact we measured against -- a cache restores a
  truncated file, or upstream re-encodes, and the suites' six-decimal anchors
  are then compared against data nobody checked.

Nothing here touches the network or a dataset.  The suite modules are read as
source, and every download is stubbed.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest
from scripts.fetch_fremtpl import (
    _RETRY_DELAYS,
    ARTIFACTS,
    Artifact,
    FetchError,
    _download,
    cache_key,
    normalise,
    obtain,
)

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "real-data.yml"

FREQ = ARTIFACTS[0]


# ── which suites guard on a dataset, discovered rather than restated ─────────


def _data_guarded_suites() -> dict[str, set[str]]:
    """Map each ``tests/test_*.py`` that guards on a dataset to the files it needs.

    Discovered from the source, deliberately, so the workflow contract below
    cannot pass by agreeing with a stale list.  A module qualifies when it binds
    ``*_SKIP_REASON`` at MODULE scope from ``_datasets.skip_reason(...)``, which
    is the form the enforcement hook keys on.  Calls inside a function body do
    not qualify: ``tests/test_dataset_guard.py`` exercises ``skip_reason`` on a
    fake name and is not itself a real-data suite.
    """
    found: dict[str, set[str]] = {}
    for path in sorted((_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id.endswith("_SKIP_REASON") for t in node.targets
            ):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr == "skip_reason"):
                continue
            names.update(
                arg.value
                for arg in call.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            )
        if names:
            found[path.name] = names
    return found


def _steps(workflow: str) -> dict[str, str]:
    """Split the job into its named step blocks, keyed by step name."""
    starts = [
        (m.start(), m.group(1).strip()) for m in re.finditer(r"(?m)^      - name: (.+)$", workflow)
    ]
    return {
        name: workflow[start : starts[i + 1][0] if i + 1 < len(starts) else len(workflow)]
        for i, (start, name) in enumerate(starts)
    }


def _suite_step(workflow: str) -> str:
    """The step block that runs pytest."""
    matches = [block for block in _steps(workflow).values() if "uv run pytest" in block]
    assert len(matches) == 1, f"expected exactly one pytest step, found {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def workflow() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


# ── the discovery itself must not silently find nothing ─────────────────────


def test_the_discovery_finds_the_suites_that_are_known_to_guard_on_data():
    """A scan that matched nothing would make every contract below vacuous."""
    found = _data_guarded_suites()
    assert set(found) == {
        "test_realdata_parity.py",
        "test_screening_guide_numbers.py",
        "test_mixed_interaction_screening.py",
    }, found
    assert found["test_realdata_parity.py"] == {
        "freMTPL2freq.parquet",
        "freMTPL2sev.parquet",
    }


# ── the job covers every data-guarded suite, and arms the switch ────────────


def _assert_job_runs_every_data_guarded_suite(workflow: str) -> None:
    step = _suite_step(workflow)
    for module in _data_guarded_suites():
        assert f"tests/{module}" in step, (
            f"{module} guards on a dataset but the real-data job never runs it, "
            f"so it keeps skipping in CI with nothing to report that"
        )


def test_the_job_runs_every_data_guarded_suite(workflow):
    _assert_job_runs_every_data_guarded_suite(workflow)


def _drop_a_suite(workflow: str) -> str:
    """Mutation: stop running the parity suite."""
    return workflow.replace("          tests/test_realdata_parity.py\n", "")


def test_the_coverage_contract_rejects_a_dropped_suite(workflow):
    """The check above must actually bite; a passing tautology is worse than none."""
    mutant = _drop_a_suite(workflow)
    assert mutant != workflow, "mutation did not apply"
    with pytest.raises(AssertionError, match="keeps skipping in CI"):
        _assert_job_runs_every_data_guarded_suite(mutant)


def _assert_switch_is_armed(workflow: str) -> None:
    step = _suite_step(workflow)
    assert re.search(r'SUPERGLM_REQUIRE_DATA: *"?1"?', step), (
        "the pytest step does not set SUPERGLM_REQUIRE_DATA, so a dataset that "
        "failed to arrive would skip and this job would pass green"
    )


def test_the_job_arms_the_enforcement_switch(workflow):
    _assert_switch_is_armed(workflow)


def test_the_switch_contract_rejects_an_unarmed_job(workflow):
    mutant = workflow.replace('SUPERGLM_REQUIRE_DATA: "1"', 'SUPERGLM_REQUIRE_DATA: "0"')
    assert mutant != workflow, "mutation did not apply"
    with pytest.raises(AssertionError, match="pass green"):
        _assert_switch_is_armed(mutant)


def test_every_parquet_a_suite_needs_is_pinned_in_the_fetch_script():
    """Fetching only freq would turn the sev-guarded tests red under the switch.

    That is the trap in arming enforcement: an unmet dependency stops being a
    skip and starts being a failure, so the fetch has to cover every file the
    suites ask for, not just the one the issue was written about.
    """
    needed = set().union(*_data_guarded_suites().values())
    pinned = {art.name for art in ARTIFACTS}
    assert needed <= pinned, f"unpinned datasets the suites require: {sorted(needed - pinned)}"


# ── a fetch failure has to be legible as a fetch failure ────────────────────


def test_the_fetch_and_the_suites_report_through_separate_steps(workflow):
    """A red X on a download and a red X on an assertion need different people."""
    blocks = _steps(workflow)
    fetch_report = blocks["Report the fetch failure as infrastructure, not code"]
    suite_report = blocks["Report the suite failure as code"]
    assert "steps.fetch.outcome == 'failure'" in fetch_report
    assert "steps.suites.outcome == 'failure'" in suite_report
    assert "NOT a test failure" in fetch_report
    # The ids the conditions name must be the ids the steps actually carry.
    assert "id: fetch" in workflow
    assert "id: suites" in workflow


def test_the_cache_key_is_derived_from_the_script_not_restated_in_yaml(workflow):
    """A hash written twice is a hash that drifts."""
    assert "--cache-key" in workflow
    assert "key: ${{ steps.pins.outputs.key }}" in workflow
    for art in ARTIFACTS:
        assert art.sha256 not in workflow, (
            f"{art.name}'s pin is restated in the workflow; it belongs only in "
            f"scripts/fetch_fremtpl.py, which the cache key is derived from"
        )


def test_the_cache_key_needs_nothing_but_the_standard_library():
    """The workflow resolves the key BEFORE ``uv sync``, on the runner's python.

    That ordering is what lets the cache restore start without waiting for an
    environment, and it holds only while importing this script costs nothing.  A
    module-scope ``import pandas`` would break that step with a
    ``ModuleNotFoundError`` that reads as a broken script rather than as a step
    in the wrong place, so the pandas import lives inside :func:`normalise`.
    """
    tree = ast.parse((_ROOT / "scripts" / "fetch_fremtpl.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    outside = imported - set(sys.stdlib_module_names)
    assert not outside, f"module-scope imports outside the standard library: {sorted(outside)}"


# ── the cache key tracks every pin ──────────────────────────────────────────


def test_the_cache_key_changes_when_any_pin_changes(monkeypatch):
    """Otherwise a re-pin silently restores the previous bytes from cache."""
    before = cache_key()
    for index in range(len(ARTIFACTS)):
        mutated = list(ARTIFACTS)
        mutated[index] = Artifact(
            **{**vars(ARTIFACTS[index]), "sha256": "0" * 64},
        )
        monkeypatch.setattr("scripts.fetch_fremtpl.ARTIFACTS", tuple(mutated))
        assert cache_key() != before, f"pin {index} does not reach the cache key"
        monkeypatch.undo()


def test_the_cache_key_changes_when_the_normalisation_changes(monkeypatch):
    before = cache_key()
    monkeypatch.setattr("scripts.fetch_fremtpl.SCHEMA_REVISION", 99)
    assert cache_key() != before


# ── verification is not optional, and a cache is not trusted ────────────────


def _stub_download(monkeypatch, payload: bytes, log: list[str] | None = None):
    def fake(art, target):
        if log is not None:
            log.append(art.name)
        target.write_bytes(payload)

    monkeypatch.setattr("scripts.fetch_fremtpl._download", fake)


def test_a_cached_file_that_fails_its_pin_is_re_downloaded(tmp_path, monkeypatch):
    """A cache restore is exactly how a truncated file comes back."""
    raw = tmp_path / f"dataset_{FREQ.openml_id}.pq"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"truncated")

    good = b"the real bytes"
    art = Artifact(**{**vars(FREQ), "sha256": hashlib.sha256(good).hexdigest()})
    log: list[str] = []
    _stub_download(monkeypatch, good, log)

    assert obtain(art, tmp_path).read_bytes() == good
    assert log == [art.name], "the bad cached copy was trusted instead of replaced"


def test_a_cached_file_that_matches_its_pin_is_not_re_downloaded(tmp_path, monkeypatch):
    good = b"the real bytes"
    art = Artifact(**{**vars(FREQ), "sha256": hashlib.sha256(good).hexdigest()})
    (tmp_path / f"dataset_{art.openml_id}.pq").write_bytes(good)
    log: list[str] = []
    _stub_download(monkeypatch, b"should not be used", log)

    assert obtain(art, tmp_path).read_bytes() == good
    assert log == []


def test_a_download_that_misses_its_pin_is_a_fetch_error(tmp_path, monkeypatch):
    _stub_download(monkeypatch, b"not what was pinned")
    with pytest.raises(FetchError, match="does not match the pinned"):
        obtain(FREQ, tmp_path)
    assert not (tmp_path / f"dataset_{FREQ.openml_id}.pq").exists(), (
        "an unverified download was left where the next run would trust it"
    )


# ── retry only what a retry can fix ────────────────────────────────────────


def _stub_urlopen(monkeypatch, exc, attempts: list[int]):
    def fake(url, timeout=None):
        attempts.append(1)
        raise exc

    monkeypatch.setattr("scripts.fetch_fremtpl.urllib.request.urlopen", fake)
    monkeypatch.setattr("scripts.fetch_fremtpl.time.sleep", lambda seconds: None)


def test_a_permanent_status_is_not_retried(tmp_path, monkeypatch):
    """Waiting out a 404 burns the job's clock and reports the same error later."""
    attempts: list[int] = []
    _stub_urlopen(monkeypatch, urllib.error.HTTPError(FREQ.url, 404, "gone", {}, None), attempts)
    with pytest.raises(FetchError, match="HTTP 404"):
        _download(FREQ, tmp_path / "out.pq")
    assert len(attempts) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_a_transient_status_is_retried_before_it_is_reported(tmp_path, monkeypatch, status):
    attempts: list[int] = []
    _stub_urlopen(
        monkeypatch, urllib.error.HTTPError(FREQ.url, status, "later", {}, None), attempts
    )
    with pytest.raises(FetchError, match="could not download"):
        _download(FREQ, tmp_path / "out.pq")
    assert len(attempts) == len(_RETRY_DELAYS) + 1


def test_a_connection_error_is_retried(tmp_path, monkeypatch):
    attempts: list[int] = []
    _stub_urlopen(monkeypatch, urllib.error.URLError("no route to host"), attempts)
    with pytest.raises(FetchError, match="could not download"):
        _download(FREQ, tmp_path / "out.pq")
    assert len(attempts) == len(_RETRY_DELAYS) + 1


def test_an_interrupted_download_leaves_neither_a_target_nor_litter(tmp_path, monkeypatch):
    """The raw directory is what actions/cache stores, so it must stay clean."""

    class _Stream:
        def read(self, *args):
            raise TimeoutError("connection dropped mid-body")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "scripts.fetch_fremtpl.urllib.request.urlopen", lambda url, timeout=None: _Stream()
    )
    monkeypatch.setattr("scripts.fetch_fremtpl.time.sleep", lambda seconds: None)

    target = tmp_path / "out.pq"
    with pytest.raises(FetchError):
        _download(FREQ, target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], f"left behind {list(tmp_path.iterdir())}"


# ── normalisation: the shape is asserted, not assumed ──────────────────────


def _tiny_freq(tmp_path: Path, **overrides) -> Path:
    frame = pd.DataFrame(
        {
            "IDpol": [1.0, 3.0],
            "ClaimNb": pd.array([1, 0], dtype="uint8"),
            "Exposure": [0.1, 0.77],
            "Area": pd.Categorical(["D", "B"]),
            "VehPower": pd.array([5, 6], dtype="uint8"),
            "VehAge": pd.array([0, 2], dtype="uint8"),
            "DrivAge": pd.array([55, 52], dtype="uint8"),
            "BonusMalus": pd.array([50, 50], dtype="uint8"),
            "VehBrand": pd.Categorical(["B12", "B12"]),
            "VehGas": ["'Regular'", "'Diesel'"],
            "Density": [1217.0, 54.0],
            "Region": pd.Categorical(["R82", "R22"]),
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    path = tmp_path / "raw.pq"
    frame.to_parquet(path, index=False)
    return path


def _tiny_artifact(**overrides) -> Artifact:
    return Artifact(**{**vars(FREQ), "rows": 2, **overrides})


def test_normalise_gives_the_dtypes_a_local_copy_carries(tmp_path):
    """The mirror stores counts as uint8 and Density as float; a dtype is what a
    discrete or categorical path dispatches on, so CI must not see a different
    one from the developer whose numbers are pinned."""
    out = normalise(_tiny_artifact(), _tiny_freq(tmp_path), tmp_path / "dest")
    frame = pd.read_parquet(out)
    for column in FREQ.integer_columns:
        assert frame[column].dtype == "int64", f"{column} is {frame[column].dtype}"
    assert frame["IDpol"].dtype == "float64", "IDpol is not a count and must stay float"
    assert sorted(frame["VehGas"].astype(str).unique()) == ["Diesel", "Regular"], (
        "an ARFF round-trip's literal quotes survived into the level labels"
    )


def test_normalise_rejects_a_frame_with_the_wrong_row_count(tmp_path):
    with pytest.raises(FetchError, match="rows, expected"):
        normalise(_tiny_artifact(rows=678013), _tiny_freq(tmp_path), tmp_path / "dest")


def test_normalise_rejects_a_frame_with_different_columns(tmp_path):
    art = _tiny_artifact(columns=FREQ.columns[:-1])
    with pytest.raises(FetchError, match="columns"):
        normalise(art, _tiny_freq(tmp_path), tmp_path / "dest")


def test_normalise_writes_where_the_loader_actually_looks(tmp_path, monkeypatch):
    """The written path is one ``_datasets`` resolves, not merely one that exists.

    Scope, stated exactly, because the obvious reading is wrong: this does NOT
    detect a renamed :data:`ARTIFACTS` entry -- it reads ``art.name`` on both
    sides and a rename moves both together.  A rename is caught by
    ``test_every_parquet_a_suite_needs_is_pinned_in_the_fetch_script``, which
    compares against the names the suites ask for.  What this catches is
    ``normalise`` writing somewhere the loader does not search: measured, a
    ``dest / "dataset.parquet"`` mutation turns it red and leaves the other 23
    green.  The consequence either way is the same and is worth the two tests:
    the fetch succeeds, the suites find nothing, and the switch reports a red
    suite rather than a failed download.
    """
    from . import _datasets

    dest = tmp_path / "dest"
    out = normalise(_tiny_artifact(), _tiny_freq(tmp_path), dest)
    monkeypatch.setattr(_datasets, "_SEARCH_DIRS", [dest])
    assert _datasets.find(FREQ.name) == out
    assert _datasets.usable(FREQ.name) == out
    assert _datasets.skip_reason(FREQ.name) is None
