"""A process that can write nowhere but the temporary directory still loads a model and predicts.

Numba needs a directory it can write to even to read a full compile cache, and stops the import of
a module whose kernels are declared ``cache=True`` when it finds none (a read-only file system with
a writable ``/tmp``, as on AWS Lambda). Importing superglm then gives Numba a private temporary
directory, unless the user chose one. The subprocesses force the event instead of relying on file
permissions, so it holds for root and on Windows and macOS: they refuse every write outside the
temporary directory, to Numba's own probe of a cache location and to superglm's.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import _numba_cache
from superglm.distributional import GaussianLS, Predictor
from superglm.features import Categorical, Spline, _basis_kernels
from tests.bound_predictor_fixtures import model_from_templates

PREFIX = _numba_cache._PREFIX
U = np.finfo(np.float64).eps / 2

READ_ONLY = """
import errno, json, os, stat, sys, tempfile

temp = os.path.normcase(os.path.realpath(tempfile.gettempdir()))
def may_write(path):
    path = os.path.normcase(os.path.realpath(path))
    return path == temp or path.startswith(temp + os.sep)
access, temporary_file = os.access, tempfile.TemporaryFile
def read_only_access(path, mode, **kwargs):
    return access(path, mode, **kwargs) and (not mode & os.W_OK or may_write(path))
def read_only_temporary_file(*args, dir=None, **kwargs):
    if dir is not None and not may_write(dir):
        raise OSError(errno.EROFS, "Read-only file system", str(dir))
    return temporary_file(*args, dir=dir, **kwargs)
os.access, tempfile.TemporaryFile = read_only_access, read_only_temporary_file
"""

CHILD = (
    READ_ONLY
    + """
import numpy as np, pandas as pd

model_path, rows, output, first = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
if first == "numba":
    import numba
import superglm
import joblib

cache = numba.config.CACHE_DIR if first == "numba" else os.environ.get("NUMBA_CACHE_DIR")
model = joblib.load(model_path)
quote = model.predict_link(pd.DataFrame({"x": [0.1, 0.2], "g": ["a", "b"]})).to_numpy()
rng = np.random.default_rng(0)
batch = pd.DataFrame({"x": rng.uniform(-1, 1, rows), "g": rng.choice(["a", "b", "c"], rows)})
np.save(output, np.vstack([quote, model.predict_link(batch).to_numpy()]))
report = {"cache": cache}
if cache:
    status = os.stat(cache)
    report["mode"] = stat.S_IMODE(status.st_mode)
    report["mine"] = status.st_uid == os.getuid() if hasattr(os, "getuid") else None
    report["index_files"] = sum(name.endswith(".nbi") for _, _, names in os.walk(cache) for name in names)
print(json.dumps(report))
"""
)

SETTINGS_IN_CODE = (
    READ_ONLY
    + """
import numba

numba.config.THREADING_LAYER = "workqueue"
numba.config.BOUNDSCHECK = True

@numba.njit(parallel=True)
def started(n):
    total = 0
    for i in numba.prange(n):
        total += i
    return total

started(4)  # the worker threads run, so Numba refuses to re-read a different thread count
numba.config.NUMBA_NUM_THREADS = 1

import superglm

@numba.njit
def compiled(x):
    return x + 1

print(json.dumps({
    "value": compiled(1),
    "layer": numba.config.THREADING_LAYER,
    "boundscheck": numba.config.BOUNDSCHECK,
    "threads": numba.config.NUMBA_NUM_THREADS,
    "cache": numba.config.CACHE_DIR,
}))
"""
)


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """A fitted model, its file, and the predictions of a quote and of a batch big enough to reach the kernels."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.uniform(-1, 1, 500), "g": rng.choice(["a", "b", "c"], 500)})
    y = 0.5 * np.sin(2 * frame.x) + 0.2 * (frame.g == "a") + 0.3 * rng.standard_normal(500)
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(
            Predictor("location", {"x": Spline("cr", k=6), "g": Categorical(base="a")}),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = model.fit_reml(frame, y.to_numpy())
    path = tmp_path_factory.mktemp("numba_cache_model") / "model.joblib"
    joblib.dump(model, path)
    rows = _basis_kernels.KERNEL_MIN_ROWS
    rng = np.random.default_rng(0)  # the child draws the same batch
    batch = pd.DataFrame({"x": rng.uniform(-1, 1, rows), "g": rng.choice(["a", "b", "c"], rows)})
    quote = pd.DataFrame({"x": [0.1, 0.2], "g": ["a", "b"]})
    expected = np.vstack(
        [model.predict_link(quote).to_numpy(), model.predict_link(batch).to_numpy()]
    )
    return path, rows, expected


def _run(saved_model, tmp_path, first: str, **setting):
    """Run the child with only ``tmp_path / "tmp"`` writable; return its report and its predictions."""
    path, rows, _ = saved_model
    temp = tmp_path / "tmp"
    temp.mkdir(exist_ok=True)
    cache_settings = ("NUMBA_CACHE_DIR", "NUMBA_CACHE_LOCATOR_CLASSES")
    environment = {k: v for k, v in os.environ.items() if k not in cache_settings}
    environment.update(TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp), **setting)
    output = tmp_path / f"{first}.npy"
    command = [sys.executable, "-c", CHILD, str(path), str(rows), str(output), first]
    done = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1]), np.load(output)


def _assert_predictions(saved_model, predictions) -> None:
    expected = saved_model[2]
    np.testing.assert_allclose(
        predictions, expected, rtol=0.0, atol=64 * U * np.abs(expected).max()
    )


def test_a_process_with_no_writable_cache_location_loads_a_model_and_predicts(
    saved_model, tmp_path
):
    """A quote and a batch that compiles kernels both predict, with Numba's cache in a private directory.

    Mutation checks: without the call in ``superglm/__init__.py`` the child dies with Numba's
    ``no locator available``; a fixed name under the temporary directory fails the uniqueness and the
    mode assertions; a directory left behind fails the last; without ``numba.config.CACHE_DIR =
    directory`` the Numba-first run dies as the unfixed one does; a directory made in the
    temporary directory itself, not in the user's own, fails the parent assertions.
    """
    temp = tmp_path / "tmp"
    caches = []
    for first in ("superglm", "numba"):
        report, predictions = _run(saved_model, tmp_path, first)
        cache = Path(report["cache"])
        caches.append(cache)
        assert cache.name.startswith(PREFIX)
        assert report["index_files"] >= 1
        if sys.platform != "win32":  # the directory is the owner's alone, in the owner's own
            assert report["mode"] == 0o700 and report["mine"]
            assert cache.parent == temp / f"{PREFIX}u{os.getuid()}"
            assert stat.S_IMODE(cache.parent.stat().st_mode) == 0o700
        else:
            assert cache.parent == temp
        _assert_predictions(saved_model, predictions)
        assert not cache.exists()
    assert caches[0] != caches[1]


def test_a_cache_directory_the_user_set_is_used_and_no_private_one_is_made(saved_model, tmp_path):
    """Mutation check: a fix that overwrites ``NUMBA_CACHE_DIR`` fails the first assertion."""
    mine = tmp_path / "tmp" / "my-numba-cache"
    mine.mkdir(parents=True)
    report, predictions = _run(saved_model, tmp_path, "superglm", NUMBA_CACHE_DIR=str(mine))
    assert Path(report["cache"]) == mine
    assert report["index_files"] >= 1
    assert not [path for path in (tmp_path / "tmp").iterdir() if path.name.startswith(PREFIX)]
    _assert_predictions(saved_model, predictions)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No cache setting anywhere, and a fallback that cannot leave anything in the session.

    A setting is recorded before it is deleted, so teardown removes it even when it was absent
    and the code under test sets it; ``numba.config.CACHE_DIR`` is recorded and emptied; the
    private directory is made under ``tmp_path`` and its removal at exit is not registered.
    """
    import numba

    for name in ("NUMBA_CACHE_DIR", "NUMBA_CACHE_LOCATOR_CLASSES"):
        monkeypatch.setenv(name, "")  # recorded, so teardown deletes whatever is there
        monkeypatch.delenv(name)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("atexit.register", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _allow_writes(monkeypatch, allowed) -> None:
    """Answer ``allowed(path)`` both ways a location is asked: the query, and Windows's temporary file."""
    import tempfile

    temporary_file = tempfile.TemporaryFile

    def make_or_refuse(*args, dir=None, **kwargs):
        if not allowed(dir):
            raise PermissionError(13, "Access is denied", str(dir))
        return temporary_file(*args, dir=dir, **kwargs)

    monkeypatch.setattr(os, "access", lambda path, mode, **kwargs: allowed(path))
    monkeypatch.setattr(tempfile, "TemporaryFile", make_or_refuse)


@pytest.fixture
def nothing_is_writable(isolated, monkeypatch):
    """Every location refuses writes."""
    _allow_writes(monkeypatch, lambda path: False)
    return isolated


@pytest.fixture
def numba_not_loaded(nothing_is_writable, monkeypatch):
    """As when superglm is imported first: the setting goes in the environment, and Numba reads it."""
    monkeypatch.delitem(sys.modules, "numba")
    return nothing_is_writable


def test_nothing_changes_while_an_ordinary_location_is_writable(isolated, monkeypatch):
    """Mutation check: a fallback made regardless of the writability test fails this."""
    _allow_writes(monkeypatch, lambda path: True)
    assert _numba_cache.ensure_cache_location() is None
    assert "NUMBA_CACHE_DIR" not in os.environ


def test_locators_the_user_chose_are_left_alone(nothing_is_writable, monkeypatch):
    monkeypatch.setenv("NUMBA_CACHE_LOCATOR_CLASSES", "UserWideCacheLocator")
    assert _numba_cache.ensure_cache_location() is None
    assert "NUMBA_CACHE_DIR" not in os.environ


def test_no_temporary_directory_leaves_numba_to_report_the_problem(
    nothing_is_writable, monkeypatch
):
    def refuse(*args, **kwargs):
        raise OSError("no temporary directory")

    monkeypatch.setattr("tempfile.mkdtemp", refuse)
    assert _numba_cache.ensure_cache_location() is None
    assert "NUMBA_CACHE_DIR" not in os.environ


def test_a_writable_user_cache_directory_is_an_ordinary_location(isolated, monkeypatch):
    """The install directory is read-only and the user's cache directory is not: nothing changes.

    Mutation check: leaving the user-wide location out of the search fails this.
    """
    tmp_path = isolated
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(
        _numba_cache, "_user_cache_directory", lambda: str(tmp_path / "cache" / "numba")
    )
    _allow_writes(monkeypatch, lambda path: os.fspath(path).startswith(str(tmp_path)))
    assert _numba_cache.ensure_cache_location() is None


@pytest.mark.parametrize(
    ("platform", "environment", "expected"),
    [
        ("linux", {"XDG_CACHE_HOME": "xdg"}, ("xdg", "numba")),
        ("linux", {}, ("home", ".cache", "numba")),
        ("darwin", {}, ("home", "Library", "Caches", "numba")),
        ("win32", {"LOCALAPPDATA": "local"}, ("local", "numba")),
        ("win32", {}, ("home", "AppData", "Local", "numba")),
    ],
)
def test_the_user_wide_location_is_the_platforms_cache_directory_plus_numba(
    platform, environment, expected, monkeypatch
):
    """Numba's documented user-wide location, the per-user cache directory, platform by platform."""
    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os.path, "expanduser", lambda path: "home")
    assert _numba_cache._user_cache_directory() == os.path.join(*expected)


def test_a_cache_directory_set_in_code_is_the_users(nothing_is_writable, monkeypatch):
    """``numba.config.CACHE_DIR``, set after Numba read its environment: nothing changes.

    Mutation check: ignoring ``numba.config.CACHE_DIR`` replaces it with a private directory.
    """
    import numba

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(nothing_is_writable / "mine"))
    assert _numba_cache.ensure_cache_location() is None
    assert numba.config.CACHE_DIR == str(nothing_is_writable / "mine")
    assert "NUMBA_CACHE_DIR" not in os.environ


@pytest.mark.parametrize(
    ("text", "names_a_directory"),
    [
        ("opt: 3\ncache_dir: /somewhere\n", True),
        ("CACHE_DIR: /somewhere\n", True),
        ("Cache_Dir: /somewhere\n", True),
        ('"cache_dir": /somewhere\n', True),
        ("{opt: 3, cache_dir: /somewhere}\n", True),
        ("# cache_dir: /somewhere\nopt: 3\n", False),
        ("other:\n  cache_dir: /somewhere\n", False),
        ("cache_dir: [unclosed\n", False),
    ],
)
def test_a_yaml_file_is_the_users_when_numba_would_read_a_cache_directory_from_it(
    numba_not_loaded, monkeypatch, text, names_a_directory
):
    """Numba upper-cases each top-level key, so every spelling of it counts, and only those.

    Mutation checks: ignoring the file, or matching only the exact lower-case line, makes the
    fallback replace a setting of the user's; a line search also takes the nested key for one.
    """
    (numba_not_loaded / ".numba_config.yaml").write_text(text)
    assert (_numba_cache.ensure_cache_location() is None) is names_a_directory
    assert ("NUMBA_CACHE_DIR" not in os.environ) is names_a_directory


def test_a_yaml_file_numba_cannot_read_sets_nothing(numba_not_loaded, monkeypatch):
    """Without PyYAML Numba ignores the file, so the fallback still acts or the import would stop."""
    (numba_not_loaded / ".numba_config.yaml").write_text("cache_dir: /somewhere\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert _numba_cache.ensure_cache_location() is not None


def test_settings_made_in_code_survive_the_next_compile_when_numba_was_imported_first(tmp_path):
    """The fallback acts on a loaded Numba, and the program's settings outlive its next compile.

    Numba's compiler re-reads every setting once a ``NUMBA_*`` variable has changed, so a variable
    set by the fallback reset the threading layer and the bounds check at the next compile and, with
    a changed thread count once threads had started, made that compile raise ``RuntimeError``.
    Mutation checks: setting ``NUMBA_CACHE_DIR`` for a loaded Numba too, or calling
    ``numba.config.reload_config()``, fails the run.
    """
    temp = tmp_path / "tmp"
    temp.mkdir()
    cache_settings = ("NUMBA_CACHE_DIR", "NUMBA_CACHE_LOCATOR_CLASSES")
    environment = {k: v for k, v in os.environ.items() if k not in cache_settings}
    environment.update(TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp), NUMBA_NUM_THREADS="2")
    done = subprocess.run(
        [sys.executable, "-c", SETTINGS_IN_CODE],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert Path(report.pop("cache")).name.startswith(PREFIX)
    assert report == {"value": 2, "layer": "workqueue", "boundscheck": True, "threads": 1}


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o707])
def test_the_inherited_directory_of_a_parent_is_used_only_while_it_is_private_to_the_caller(
    numba_not_loaded, monkeypatch, mode
):
    """A child finds ``NUMBA_CACHE_DIR`` naming the private directory of a parent process.

    Numba loads pickles from it, so it is used only while it is a real directory owned by the caller
    and closed to group and others, and a directory that is gone, open to others, a link or someone
    else's is replaced by a new private one. Mutation check: using any inherited directory fails
    every assertion below it.
    """
    if sys.platform == "win32":
        pytest.skip("Windows has no mode bits to check")
    parent = numba_not_loaded / f"{PREFIX}u{os.getuid()}"
    parent.mkdir(mode=0o700)
    good = parent / f"{PREFIX}good"
    good.mkdir(mode=0o700)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(good))
    assert _numba_cache.ensure_cache_location() is None
    assert os.environ["NUMBA_CACHE_DIR"] == str(good)
    gone = parent / f"{PREFIX}gone"
    loose = parent / f"{PREFIX}loose"
    loose.mkdir()
    loose.chmod(mode)
    link = parent / f"{PREFIX}link"
    link.symlink_to(good)
    for stale in (gone, loose, link):
        monkeypatch.setenv("NUMBA_CACHE_DIR", str(stale))
        replaced = _numba_cache.ensure_cache_location()
        assert replaced is not None and Path(replaced) != stale and not gone.exists()
        assert os.environ["NUMBA_CACHE_DIR"] == replaced
        assert stat.S_IMODE(os.stat(replaced).st_mode) == 0o700
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(good))
    monkeypatch.setattr(os, "getuid", lambda: os.stat(good).st_uid + 1)  # someone else's
    assert _numba_cache.ensure_cache_location() not in (None, str(good))


def test_the_directory_is_made_in_one_of_the_users_own_and_never_in_a_loose_one(
    numba_not_loaded, monkeypatch
):
    """The per-user parent is kept, mode 0700; one that is not private to the caller is not used.

    A child that outlives the process makes Numba create the path again, and only a directory no
    other user can enter makes that harmless. Mutation check: ``mkdtemp`` in the temporary directory
    itself fails the first assertion; trusting an existing parent whatever its mode, the second.
    """
    if sys.platform == "win32":
        pytest.skip("the temporary directory of a Windows user is the user's own")
    temp = numba_not_loaded
    parent = temp / f"{PREFIX}u{os.getuid()}"
    first = Path(_numba_cache.ensure_cache_location())
    assert first.parent == parent and stat.S_IMODE(parent.stat().st_mode) == 0o700
    parent.chmod(0o755)  # as another user, or an older version, might leave it
    monkeypatch.delenv("NUMBA_CACHE_DIR")  # a new process: the first one's setting is not inherited
    second = Path(_numba_cache.ensure_cache_location())
    assert second.parent == temp and second.name.startswith(PREFIX)
    assert stat.S_IMODE(second.stat().st_mode) == 0o700


def test_a_child_that_outlives_the_process_recreates_the_path_only_inside_a_private_directory(
    tmp_path,
):
    """The reproduction: the process ends, its directory is removed, a child it left compiles.

    The child makes the directory again (Numba's ``os.makedirs``) after the parent is gone: the
    path must be in a directory that only its owner can enter, since Numba loads pickles from it.
    Mutation check: ``mkdtemp`` in the shared temporary directory leaves the recreated path
    reachable by group and others.
    """
    if sys.platform == "win32":
        pytest.skip("the temporary directory of a Windows user is the user's own")
    temp = tmp_path / "tmp"
    temp.mkdir()
    go = tmp_path / "go"
    report = tmp_path / "report.json"
    child = (
        "import json, os, stat, sys, time\n"
        f"for _ in range(2400):\n    if os.path.exists({str(go)!r}): break\n"
        "    time.sleep(0.05)\nelse:\n    sys.exit(1)\n"
        "cache = os.environ['NUMBA_CACHE_DIR']\n"
        "os.makedirs(cache, exist_ok=True)  # what Numba does for a cache=True kernel\n"
        "mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)\n"
        f"json.dump({{'cache': cache, 'parent': mode(os.path.dirname(cache)), 'leaf': mode(cache)}},"
        f" open({str(report)!r}, 'w'))\n"
    )
    parent = (
        "import errno, os, subprocess, sys, tempfile\n"
        "access = os.access\n"
        "temp = os.path.realpath(tempfile.gettempdir())\n"
        "os.access = lambda p, m, **k: access(p, m, **k) and (not m & os.W_OK or"
        " os.path.realpath(p).startswith(temp))\n"
        "import superglm\n"
        "assert os.environ.get('NUMBA_CACHE_DIR'), 'no fallback'\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], stdin=subprocess.DEVNULL,"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(os.environ['NUMBA_CACHE_DIR'], flush=True)\n"
    )
    environment = {k: v for k, v in os.environ.items() if not k.startswith("NUMBA_CACHE")}
    environment.update(TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp))
    done = subprocess.run(
        [sys.executable, "-c", parent], env=environment, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    cache = Path(done.stdout.strip().splitlines()[-1])
    assert not cache.exists()  # the process that made it has gone and removed it
    go.write_text("go")
    for _ in range(600):
        if report.exists():
            break
        time.sleep(0.1)
    result = json.loads(report.read_text())
    assert result["cache"] == str(cache)
    assert result["parent"] & 0o077 == 0, result


@pytest.mark.parametrize("probe_allows", [True, False])
def test_windows_is_asked_by_making_a_file_because_os_access_says_yes_to_every_directory(
    probe_allows, monkeypatch, tmp_path
):
    """CPython's Windows ``os.access`` returns true for any directory with ``W_OK``.

    On Windows the writable test makes a file in the directory, as Numba does, so a read-only
    install directory is seen; elsewhere it is the permission query, which creates nothing and
    imports nothing. Mutation check: ``os.access`` alone on Windows says yes in the second case.
    """
    import tempfile

    made = []

    def temporary_file(*args, dir=None, **kwargs):
        made.append(dir)
        if not probe_allows:
            raise PermissionError(13, "Access is denied", str(dir))
        return tempfile.SpooledTemporaryFile()

    monkeypatch.setattr(os, "access", lambda path, mode, **kwargs: True)
    monkeypatch.setattr(tempfile, "TemporaryFile", temporary_file)
    monkeypatch.setattr(sys, "platform", "win32")
    assert _numba_cache._can_write(str(tmp_path)) is probe_allows
    assert made == [str(tmp_path)]
    monkeypatch.setattr(sys, "platform", "linux")
    made.clear()
    assert _numba_cache._can_write(str(tmp_path)) is True and made == []
