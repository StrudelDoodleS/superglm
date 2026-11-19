"""scripts/pre_push.py: the backlog it enforces, the tests it picks, and the hook it installs."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pre_push", ROOT / "scripts" / "pre_push.py")
pre_push = importlib.util.module_from_spec(_spec)
sys.modules["pre_push"] = pre_push
_spec.loader.exec_module(pre_push)


def test_the_ty_backlog_is_the_type_check_jobs():
    workflow = (ROOT / ".github" / "workflows" / "dev-ci.yml").read_text(encoding="utf-8")
    limits = re.findall(r"diagnostics > (\d+)", workflow)
    assert limits and pre_push._ty_backlog() == int(limits[0])


def _repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "master")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_old.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_gone.py").write_text("", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    git("update-ref", "refs/remotes/origin/master", "HEAD")
    (tmp_path / "tests" / "test_old.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_new.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "_helper.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_gone.py").unlink()
    git("add", "-A")
    git("commit", "-q", "-m", "change")
    return tmp_path


def test_changed_tests_are_the_test_files_the_branch_adds_or_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    assert sorted(pre_push._changed_tests()) == ["tests/test_new.py", "tests/test_old.py"]


def test_install_writes_and_upgrades_its_hook_and_leaves_a_foreign_one_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    pre_push._install()
    hook = tmp_path / ".git" / "hooks" / "pre-push"
    assert hook.read_text(encoding="utf-8") == pre_push.HOOK
    pre_push._install()  # the same hook again is fine
    # An earlier version's hook carries the mark and is upgraded in place.
    hook.write_text(
        "#!/bin/sh\n# Installed by scripts/pre_push.py --install.\nexec old\n", encoding="utf-8"
    )
    pre_push._install()
    assert hook.read_text(encoding="utf-8") == pre_push.HOOK
    hook.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not this script's"):
        pre_push._install()
    assert "mine" in hook.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("present", "kernels"),
    [
        ({"AVX2", "FMA3", "AVX512_SKX"}, ["Haswell", "SkylakeX"]),
        ({"AVX2", "FMA3"}, ["Haswell"]),  # Zen 2/3 and Intel without AVX-512
        (set(), [None]),  # ARM64: the tests run once, as dispatched
    ],
)
def test_only_kernels_the_cpu_can_execute_are_forced(monkeypatch, present, kernels):
    from numpy._core._multiarray_umath import __cpu_features__ as features

    for name in ("AVX2", "FMA3", "AVX512_SKX"):
        monkeypatch.setitem(features, name, name in present)
    assert pre_push._runnable_kernels() == kernels


def test_a_run_passes_only_on_the_exit_codes_it_allows():
    command = [sys.executable, "-c", "raise SystemExit(5)"]
    pre_push._run("all skipped", command, passing=frozenset({0, pre_push.NO_TESTS_COLLECTED}))
    with pytest.raises(SystemExit, match="exit 5"):
        pre_push._run("all skipped", command)


@pytest.mark.skipif(shutil.which("sh") is None, reason="git's hooks run under a POSIX sh")
def test_the_hook_lets_a_branch_without_the_gate_push(tmp_path):
    hook = tmp_path / "pre-push"
    hook.write_text(pre_push.HOOK, encoding="utf-8")
    # No scripts/pre_push.py here, and no uv on the empty PATH: the hook exits 0 before uv.
    shell = shutil.which("sh")
    result = subprocess.run([shell, str(hook)], cwd=tmp_path, env={"PATH": ""}, capture_output=True)
    assert result.returncode == 0
    # A script from before the stamps, without --verify-push: still no uv.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "pre_push.py").write_text("# no verifier\n", encoding="utf-8")
    path = {"PATH": os.path.dirname(shutil.which("grep"))}  # grep, but no uv
    result = subprocess.run([shell, str(hook)], cwd=tmp_path, env=path, capture_output=True)
    assert result.returncode == 0
    (tmp_path / "scripts" / "pre_push.py").write_text("# --verify-push\n", encoding="utf-8")
    result = subprocess.run([shell, str(hook)], cwd=tmp_path, env=path, capture_output=True)
    assert result.returncode != 0  # with the verifier, the hook runs it (here: no uv)


def test_the_gate_and_the_suite_drop_the_variables_git_gives_a_hook():
    # The list both carry is git's own, so a git that adds one shows up here.
    local = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True
    ).stdout.split()
    from tests import conftest

    assert set(local) <= set(pre_push.GIT_REPOSITORY_ENV)
    assert set(local) <= set(conftest.GIT_REPOSITORY_ENV)


def test_a_suite_run_from_the_hook_leaves_the_pushed_repository_alone(tmp_path):
    """As a hook, git sets GIT_DIR; a test that builds its own repository must not use it."""
    pushed = tmp_path / "pushed"
    subprocess.run(["git", "init", "-q", str(pushed)], check=True)
    before = (pushed / ".git" / "config").read_text(encoding="utf-8")
    env = {**os.environ, "GIT_DIR": str(pushed / ".git"), "GIT_WORK_TREE": str(pushed)}
    selected = (
        "tests/test_pre_push.py::test_changed_tests_are_the_test_files_the_branch_adds_or_edits"
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            selected,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-2000:]
    assert (pushed / ".git" / "config").read_text(encoding="utf-8") == before


def _quiet_gate(monkeypatch):
    """The gate with every check passing at once, so only its stamping is under test."""
    for name in ("_run", "_ty"):
        monkeypatch.setattr(pre_push, name, lambda *a, **k: None)
    monkeypatch.setattr(pre_push, "_changed_tests", lambda: [])
    monkeypatch.setattr(pre_push, "_quick_suite", lambda full, start: "the whole quick suite")


def test_a_passing_gate_stamps_a_clean_head_and_only_a_clean_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    _quiet_gate(monkeypatch)
    head = pre_push._git("rev-parse", "HEAD")
    (tmp_path / "untracked.txt").write_text("not pushed", encoding="utf-8")
    pre_push.main([])
    assert pre_push._stamp(head).exists()  # untracked files are not pushed
    pre_push._stamp(head).unlink()
    pre_push.main(["--no-suite"])  # an incomplete run passes nothing
    assert not pre_push._stamp(head).exists()
    (tmp_path / "tests" / "test_old.py").write_text("x = 2\n", encoding="utf-8")
    pre_push.main([])  # what was checked is not what HEAD holds
    assert not pre_push._stamp(head).exists()


def test_the_verifier_refuses_exactly_the_commits_without_a_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    head = pre_push._git("rev-parse", "HEAD")
    line = f"refs/heads/master {head} refs/heads/master {pre_push.NO_COMMIT}"
    deletion = f"(delete) {pre_push.NO_COMMIT} refs/heads/old {head}"
    with pytest.raises(SystemExit, match=head[:12]):
        pre_push._verify_push([line, deletion])
    pre_push._stamp(head).parent.mkdir()
    pre_push._stamp(head).touch()
    pre_push._verify_push([line, deletion])


def test_git_push_through_the_hook_waits_for_a_stamp(tmp_path, monkeypatch):
    """The hook as git runs it: stdin from git, an unstamped push refused, a stamped one sent."""
    repo = _repo(tmp_path / "work")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
    (repo / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "pre_push.py", repo / "scripts" / "pre_push.py")
    hook = repo / ".git" / "hooks" / "pre-push"
    # Quoted, with forward slashes: sh (Git Bash on Windows) eats a path's backslashes.
    python = f'"{Path(sys.executable).as_posix()}"'
    hook.write_text(pre_push.HOOK.replace("uv run --no-sync python", python), "utf-8")
    hook.chmod(0o755)
    push = ["git", "push", "-q", "origin", "master"]
    refused = subprocess.run(push, cwd=repo, capture_output=True, text=True)
    assert refused.returncode != 0 and "has not passed the checks" in refused.stderr
    monkeypatch.setattr(pre_push, "ROOT", repo)
    head = pre_push._git("rev-parse", "HEAD")
    pre_push._stamp(head).parent.mkdir()
    pre_push._stamp(head).touch()
    sent = subprocess.run(push, cwd=repo, capture_output=True, text=True)
    assert sent.returncode == 0, sent.stderr
    # An annotated tag reaches the hook as a tag object; its commit's stamp covers it.
    subprocess.run(["git", "tag", "-a", "v9.9.9", "-m", "release"], cwd=repo, check=True)
    tagged = subprocess.run(
        ["git", "push", "-q", "origin", "v9.9.9"], cwd=repo, capture_output=True, text=True
    )
    assert tagged.returncode == 0, tagged.stderr


def test_the_stamp_says_which_suite_passed_and_the_hook_repeats_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    _quiet_gate(monkeypatch)
    monkeypatch.setattr(pre_push, "_quick_suite", lambda full, start: "a selection of 7 tests")
    pre_push.main([])
    head = pre_push._git("rev-parse", "HEAD")
    assert pre_push._stamp_fields(pre_push._stamp(head))["passed"] == "a selection of 7 tests"
    pre_push._verify_push([f"refs/heads/b {head} refs/heads/b {pre_push.NO_COMMIT}"])
    assert f"{head[:12]} passed a selection of 7 tests" in capsys.readouterr().out


def _commit(repo: Path, name: str, text: str, message: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return pre_push._git("rev-parse", "HEAD")


def test_a_clean_rebase_carries_the_stamp_and_a_conflicted_one_needs_the_gate(
    tmp_path, monkeypatch, capsys
):
    repo = _repo(tmp_path)
    monkeypatch.setattr(pre_push, "ROOT", repo)
    git = lambda *args: subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("update-ref", "refs/remotes/origin/master", "HEAD")
    git("checkout", "-q", "-b", "branch")
    stamped = _commit(repo, "notes.txt", "one\ntwo\nthree\n", "branch work")
    pre_push._stamp(stamped).parent.mkdir()
    pre_push._stamp(stamped).write_text(
        pre_push._stamp_text("a selection of 3 tests", pre_push._series(stamped)), "utf-8"
    )
    # master moves on elsewhere; the branch rebases onto it without a conflict.
    git("checkout", "-q", "master")
    moved = _commit(repo, "other.txt", "unrelated\n", "master work")
    git("update-ref", "refs/remotes/origin/master", moved)
    git("checkout", "-q", "branch")
    git("rebase", "-q", "master")
    rebased = pre_push._git("rev-parse", "HEAD")
    assert rebased != stamped and not pre_push._stamp(rebased).exists()
    pre_push._verify_push([f"refs/heads/branch {rebased} refs/heads/branch {pre_push.NO_COMMIT}"])
    out = capsys.readouterr().out
    assert f"carries the stamp of {stamped[:12]}" in out and "a selection of 3 tests" in out
    # master edits the line the branch added: the rebase resolves a conflict, changing the diff.
    git("checkout", "-q", "master")
    clash = _commit(repo, "notes.txt", "one\nTWO\n", "master edits notes")
    git("update-ref", "refs/remotes/origin/master", clash)
    git("checkout", "-q", "-B", "resolved", clash)
    resolved = _commit(repo, "notes.txt", "one\ntwo\nthree\n", "branch work")
    with pytest.raises(SystemExit, match=resolved[:12]):
        pre_push._verify_push([f"refs/heads/r {resolved} refs/heads/r {pre_push.NO_COMMIT}"])


def test_the_quick_suite_runs_the_selection_or_everything_and_records_a_map(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    test_map = pre_push._test_map()
    monkeypatch.setattr(pre_push, "_test_map", lambda: test_map)
    commands = []

    def run(label, command, env=None, passing=None):
        commands.append((command, env))
        listed = [word for word in command if word.endswith("selected.txt")]
        if listed:
            commands.append(Path(listed[0]).read_text(encoding="utf-8").split())

    monkeypatch.setattr(pre_push, "_run", run)
    chosen = test_map.Selection(files={"tests/test_a.py": "changed"}, map_commit="c" * 40)
    monkeypatch.setattr(test_map, "choose", lambda root: chosen)
    passed = pre_push._quick_suite(full=False, start=None)
    (command, env), listed = commands
    assert "--files-from" in command and listed == ["tests/test_a.py"]
    assert "test_map_plugin" not in command and "a selection of 0 tests" in passed
    commands.clear()
    head = pre_push._git("rev-parse", "HEAD")
    assert "whole quick suite" in pre_push._quick_suite(full=True, start=head)
    ((command, env),) = commands
    assert "--files-from" not in command and "test_map_plugin" in command
    assert test_map.OUT_ENV in " ".join(command) or test_map.OUT_ENV in (env or {})
    assert (test_map.map_directory(pre_push.ROOT) / f"{head}.json").exists()


def test_a_map_is_not_kept_when_the_checkout_moved_during_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    test_map = pre_push._test_map()
    monkeypatch.setattr(pre_push, "_test_map", lambda: test_map)

    def run(label, command, env=None, passing=None):
        # Another session edits a tracked file while the suite runs.
        (pre_push.ROOT / "tests" / "test_old.py").write_text("x = 2\n", encoding="utf-8")

    monkeypatch.setattr(pre_push, "_run", run)
    head = pre_push._git("rev-parse", "HEAD")
    pre_push._quick_suite(full=True, start=head)
    assert not (test_map.map_directory(pre_push.ROOT) / f"{head}.json").exists()
