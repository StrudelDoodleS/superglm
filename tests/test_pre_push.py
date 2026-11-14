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
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "pre_push.py").write_text("", encoding="utf-8")
    result = subprocess.run([shell, str(hook)], cwd=tmp_path, env={"PATH": ""}, capture_output=True)
    assert result.returncode != 0  # with the script, the hook runs the gate (here: no uv)


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
