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
import subprocess
import sys
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

CHILD = """
import errno, json, os, stat, sys, tempfile
import numpy as np, pandas as pd

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

model_path, rows, output, first = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
if first == "numba":
    import numba
import superglm
import joblib

cache = os.environ.get("NUMBA_CACHE_DIR")
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
    mode assertions; a directory left behind fails the last; without ``reload_config`` the
    Numba-first run dies as the unfixed one does.
    """
    temp = tmp_path / "tmp"
    caches = []
    for first in ("superglm", "numba"):
        report, predictions = _run(saved_model, tmp_path, first)
        cache = Path(report["cache"])
        caches.append(cache)
        assert cache.parent == temp and cache.name.startswith(PREFIX)
        assert report["index_files"] >= 1
        if sys.platform != "win32":  # the directory is the owner's alone
            assert report["mode"] == 0o700 and report["mine"]
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
def nothing_is_writable(monkeypatch):
    """No ``NUMBA_CACHE_DIR`` or locator list, and every location refuses writes."""
    for name in ("NUMBA_CACHE_DIR", "NUMBA_CACHE_LOCATOR_CLASSES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(os, "access", lambda path, mode, **kwargs: False)


def test_nothing_changes_while_an_ordinary_location_is_writable(monkeypatch):
    """Mutation check: a fallback made regardless of the writability test fails this."""
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_LOCATOR_CLASSES", raising=False)
    monkeypatch.setattr(os, "access", lambda path, mode, **kwargs: True)
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


def test_a_writable_user_cache_directory_is_an_ordinary_location(monkeypatch, tmp_path):
    """The install directory is read-only and the user's cache directory is not: nothing changes.

    Mutation check: leaving the user-wide location out of the search fails this.
    """
    for name in ("NUMBA_CACHE_DIR", "NUMBA_CACHE_LOCATOR_CLASSES"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(
        _numba_cache, "_user_cache_directory", lambda: str(tmp_path / "cache" / "numba")
    )
    monkeypatch.setattr(
        os, "access", lambda path, mode, **kwargs: os.fspath(path).startswith(str(tmp_path))
    )
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
