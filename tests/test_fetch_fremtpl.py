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
import io
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


def data_guarded_suites() -> dict[str, set[str]]:
    """Map each ``tests/test_*.py`` that guards on a dataset to the files it needs.

    Discovered from the source, deliberately, so the workflow contract below
    cannot pass by agreeing with a stale list.  A module qualifies when it binds
    ``*_SKIP_REASON`` at MODULE scope from ``_datasets.skip_reason(...)``, which
    is the form the enforcement hook keys on.  Calls inside a function body do
    not qualify: ``tests/test_dataset_guard.py`` exercises ``skip_reason`` on a
    fake name and is not itself a real-data suite.

    Public because ``tests/test_dataset_guard.py`` parametrizes on it too.  It
    used to hardcode the same three modules, so a fourth suite would have been
    covered by the workflow contract here and by nothing there -- two lists
    that must agree and no check that they do.
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
    found = data_guarded_suites()
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
    for module in data_guarded_suites():
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
    needed = set().union(*data_guarded_suites().values())
    pinned = {art.name for art in ARTIFACTS}
    assert needed <= pinned, f"unpinned datasets the suites require: {sorted(needed - pinned)}"


# ── a fetch failure has to be legible as a fetch failure ────────────────────


def _trigger_block(workflow: str, trigger: str) -> str:
    """The body of one ``on:`` trigger, up to the next key at its indentation."""
    match = re.search(rf"(?m)^  {trigger}:\n((?:(?:    .*)?\n)*)", workflow)
    assert match, f"no `{trigger}:` trigger in {_WORKFLOW.name}"
    return match.group(1)


def _assert_the_job_runs_on_a_stacked_pull_request(workflow: str) -> None:
    step = _trigger_block(workflow, "pull_request")
    assert "branches:" not in step, (
        "the pull_request trigger filters on the base branch, so this job cannot run "
        "on a stacked pull request -- including the one that introduces it. A workflow "
        "that never runs reports nothing, and nothing reads as a pass"
    )


def test_the_job_runs_on_a_pull_request_whatever_its_base(workflow):
    """The job has to be able to run on the pull request that changes it.

    ``ci.yml`` and ``dev-ci.yml`` filter ``pull_request`` to ``branches:
    [master]``, so a stacked pull request gets neither.  Measured on #284 before
    this change: all six checks reporting were ``security.yml``'s supply-chain
    jobs, and "6/6 pass" said nothing whatever about the suites.  This workflow
    declines that filter, which is a choice it can make alone: ``security.yml``
    already takes a bare ``pull_request:``.
    """
    _assert_the_job_runs_on_a_stacked_pull_request(workflow)


def test_the_stacked_pull_request_contract_rejects_a_base_branch_filter(workflow):
    """The check above must bite; it is one line away from being a tautology."""
    mutant = workflow.replace("  pull_request:\n", "  pull_request:\n    branches: [master]\n")
    assert mutant != workflow, "mutation did not apply"
    with pytest.raises(AssertionError, match="cannot run on a stacked pull request"):
        _assert_the_job_runs_on_a_stacked_pull_request(mutant)


def test_the_push_trigger_still_only_fires_on_master(workflow):
    """Widening ``pull_request`` must not silently widen ``push`` as well.

    A branch push would then duplicate the pull-request run on every commit.
    """
    assert "branches: [master]" in _trigger_block(workflow, "push")


def _assert_re_pinning_is_advised_only_for_a_confirmed_re_encode(workflow: str) -> None:
    report = _steps(workflow)["Report the fetch failure as infrastructure, not code"]
    assert "re-pinning" in report, "the summary no longer mentions re-pinning at all"
    assert "two independent downloads agree" in report, (
        "the summary advises re-pinning without naming the one diagnosis that "
        "distinguishes a re-encoded upstream from a damaged transfer, so a truncated "
        "download reads as an instruction to re-pin the anchors against corrupt bytes"
    )
    assert "Do not re-pin" in report


def test_the_fetch_failure_summary_only_advises_re_pinning_for_a_confirmed_re_encode(workflow):
    _assert_re_pinning_is_advised_only_for_a_confirmed_re_encode(workflow)


def test_the_re_pin_advice_contract_rejects_an_unconditional_re_pin(workflow):
    # The advice reverted to the undiscriminating form: any pin miss is called a
    # re-encode, which is what sends a truncated download to the re-pin path.
    mutant = workflow.replace("two independent downloads agree", "the pin does not match")
    assert mutant != workflow, "mutation did not apply"
    with pytest.raises(AssertionError, match="reads as an instruction to re-pin"):
        _assert_re_pinning_is_advised_only_for_a_confirmed_re_encode(mutant)


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
    wrong = b"not what was pinned"
    art = Artifact(**{**vars(FREQ), "size": len(wrong)})
    _stub_download(monkeypatch, wrong)
    with pytest.raises(FetchError, match="does not match the pinned"):
        obtain(art, tmp_path)
    assert not (tmp_path / f"dataset_{FREQ.openml_id}.pq").exists(), (
        "an unverified download was left where the next run would trust it"
    )


# ── a damaged transfer must not be reported as an upstream re-encode ────────
#
# The re-encode branch tells the operator to re-pin, and the workflow repeats
# that as the recommended action.  Reaching it with corrupt bytes is the worst
# outcome this script has: it invites re-pinning the suites' six-decimal anchors
# against data nobody checked.


class _ShortStream:
    """A body that stops early WITHOUT raising, as CPython's own client does.

    ``http.client.HTTPResponse.read(amt)`` returns short and closes the
    connection rather than raising ``IncompleteRead`` -- its source carries the
    comment "Ideally, we would raise IncompleteRead if the content-length wasn't
    satisfied, but it might break compatibility".  ``shutil.copyfileobj`` reads
    with an ``amt``, so that is the path a truncated transfer takes, and
    ``_download``'s retry loop -- which only covers RAISED errors -- never sees
    it.  ``test_an_interrupted_download_leaves_neither_a_target_nor_litter``
    covers the raising path and shares this blind spot.
    """

    def __init__(self, payload: bytes, cut: int):
        self._buf = io.BytesIO(payload[:cut])

    def read(self, amt=-1):
        return self._buf.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_urlopen_bodies(monkeypatch, bodies):
    """Serve *bodies* in order, one per ``urlopen`` call."""
    remaining = list(bodies)

    def fake(url, timeout=None):
        return remaining.pop(0)

    monkeypatch.setattr("scripts.fetch_fremtpl.urllib.request.urlopen", fake)
    monkeypatch.setattr("scripts.fetch_fremtpl.time.sleep", lambda seconds: None)


def test_a_silently_truncated_body_is_reported_as_transport_not_a_re_encode(tmp_path, monkeypatch):
    """Measured before the fix: this reached "The upstream artifact changed"."""
    whole = b"the whole pinned body" * 64
    art = Artifact(
        **{**vars(FREQ), "sha256": hashlib.sha256(whole).hexdigest(), "size": len(whole)}
    )
    _stub_urlopen_bodies(
        monkeypatch,
        [_ShortStream(whole, len(whole) // 2), _ShortStream(whole, len(whole))],
    )

    with pytest.raises(FetchError) as excinfo:
        obtain(art, tmp_path)
    message = str(excinfo.value)
    assert "DAMAGED TRANSFER" in message, message
    assert "Do not re-pin" in message, message
    assert "upstream artifact changed" not in message, message


def test_agreeing_downloads_at_an_unpinned_length_are_reported_as_ambiguous(tmp_path, monkeypatch):
    """The realistic re-encode, and the deterministic truncation, are ONE branch.

    A re-encode almost always changes the byte length, so the length check --
    which used to return before the second download ever ran -- caught every
    realistic re-encode and told the operator, as fact, that it was a damaged
    transfer and not to re-pin.  Every rerun repeated it, and the workflow
    repeated it again, so the documented recovery was reachable only by a
    re-encode that preserved the length exactly.  The suite could not see that:
    its only re-encode case set the PIN's length to the re-encoded length.

    The honest answer is that these bytes are ambiguous.  A proxy truncating at
    a fixed offset survives a repeat download exactly as a re-encode does, and
    nothing available here separates them -- so the message says both and sends
    the reader to the artifact's published length, rather than picking one.

    Serving two bodies is itself the assertion that the second download runs:
    the early return took one, so it never reached ``remaining.pop(0)`` twice.
    """
    reencoded = b"upstream wrote this instead" * 49
    assert len(reencoded) != FREQ.size, "the point of the case is a length that is not the pin's"
    _stub_urlopen_bodies(monkeypatch, [_ShortStream(reencoded, len(reencoded)) for _ in range(2)])

    with pytest.raises(FetchError) as excinfo:
        obtain(FREQ, tmp_path)
    message = str(excinfo.value)
    assert "AMBIGUOUS" in message, message
    assert str(len(reencoded)) in message and str(FREQ.size) in message, message
    assert FREQ.url in message, message
    # Neither of the two routed diagnoses may claim this one.
    assert "re-pin deliberately after re-measuring" not in message, message
    assert "DAMAGED TRANSFER" not in message, message


def test_same_length_corruption_is_reported_as_transport_not_a_re_encode(tmp_path, monkeypatch):
    """A length check cannot see this one, so a second download decides it.

    Two downloads that disagree with each other cannot both be upstream.
    """
    whole = b"the whole pinned body" * 64
    art = Artifact(
        **{**vars(FREQ), "sha256": hashlib.sha256(whole).hexdigest(), "size": len(whole)}
    )
    first = bytearray(whole)
    first[7] ^= 0xFF
    second = bytearray(whole)
    second[9] ^= 0xFF
    _stub_urlopen_bodies(
        monkeypatch,
        [_ShortStream(bytes(first), len(whole)), _ShortStream(bytes(second), len(whole))],
    )

    with pytest.raises(FetchError, match="corrupted in transit"):
        obtain(art, tmp_path)


def test_two_downloads_agreeing_at_the_pinned_length_is_the_re_encode_case(tmp_path, monkeypatch):
    """The one case where "re-pin" is the right advice, and the only one that gives it."""
    reencoded = b"upstream wrote this instead" * 49
    art = Artifact(**{**vars(FREQ), "size": len(reencoded)})
    _stub_urlopen_bodies(monkeypatch, [_ShortStream(reencoded, len(reencoded)) for _ in range(2)])

    with pytest.raises(FetchError, match="re-pin deliberately after re-measuring"):
        obtain(art, tmp_path)


def test_the_second_opinion_download_leaves_no_litter_in_the_cached_directory(
    tmp_path, monkeypatch
):
    """The raw directory is what ``actions/cache`` stores; a stray file rides along."""
    reencoded = b"upstream wrote this instead" * 49
    art = Artifact(**{**vars(FREQ), "size": len(reencoded)})
    _stub_urlopen_bodies(monkeypatch, [_ShortStream(reencoded, len(reencoded)) for _ in range(2)])

    with pytest.raises(FetchError):
        obtain(art, tmp_path)
    assert list(tmp_path.iterdir()) == [], f"left behind {list(tmp_path.iterdir())}"


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


def test_every_artifact_declares_the_dtypes_it_promises(tmp_path):
    """The sev artifact carried a pin and no schema contract, and freq carried both.

    That asymmetry is the finding: ``normalise`` exists because "a dtype is what
    a discrete or categorical path dispatches on", and that argument was applied
    to one of the two files.  ``float_columns`` is the contract, ASSERTED rather
    than cast, and it is non-empty for both.
    """
    for art in ARTIFACTS:
        assert art.float_columns, f"{art.name} promises nothing about its dtypes"
        assert set(art.float_columns) <= set(art.columns)
        assert not set(art.float_columns) & set(art.integer_columns)


def test_the_two_artifacts_agree_on_the_column_the_suites_join_them_on():
    """``_load_gamma_data`` merges freq onto sev on ``IDpol``.

    Measured on the fetched pair: both sides ``float64``, 24,944 joined rows.
    Pinning it on both sides is what stops a future re-pin quietly making that a
    cross-dtype join.
    """
    joined = {art.name: art for art in ARTIFACTS}
    freq, sev = joined["freMTPL2freq.parquet"], joined["freMTPL2sev.parquet"]
    assert "IDpol" in freq.float_columns and "IDpol" in sev.float_columns


def test_normalise_rejects_a_dtype_the_artifact_did_not_promise(tmp_path):
    """A cast would hide this; the point is to report it.

    The raw bytes are pinned, so the only thing that can move a dtype is the
    reader -- and the reader is exactly what differs between a developer's
    machine and the locked CI environment.
    """
    art = _tiny_artifact(float_columns=("IDpol", "ClaimNb"))
    with pytest.raises(FetchError, match="ClaimNb is int64 after normalisation"):
        normalise(art, _tiny_freq(tmp_path), tmp_path / "dest")


def test_normalise_leaves_no_partial_file_where_the_loader_would_trust_it(tmp_path, monkeypatch):
    """``usable()`` accepts a path on existence alone, so a half-written parquet
    is not re-fetched -- it raises from inside a test body on the next run, one
    step removed from the thing that broke."""
    from . import _datasets

    raw = _tiny_freq(tmp_path)  # built BEFORE to_parquet is sabotaged
    dest = tmp_path / "dest"

    def explode(self, path, *args, **kwargs):
        Path(path).write_bytes(b"half a parquet")
        raise OSError("no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(OSError, match="no space left"):
        normalise(_tiny_artifact(), raw, dest)
    monkeypatch.undo()

    monkeypatch.setattr(_datasets, "_SEARCH_DIRS", [dest])
    assert _datasets.find(FREQ.name) is None, f"a partial write is at {dest / FREQ.name}"
    assert list(dest.iterdir()) == [], f"left behind {list(dest.iterdir())}"


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
