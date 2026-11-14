"""scripts/pre_push.py: the backlog it enforces, the tests it picks, and the hook it installs."""

from __future__ import annotations

import importlib.util
import re
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


def test_install_writes_the_hook_once_and_leaves_a_foreign_one_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_push, "ROOT", _repo(tmp_path))
    pre_push._install()
    hook = tmp_path / ".git" / "hooks" / "pre-push"
    assert hook.read_text(encoding="utf-8") == pre_push.HOOK
    pre_push._install()  # the same hook again is fine
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
