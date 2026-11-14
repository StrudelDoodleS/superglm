"""``superglm.warmup``'s worker processes: the stage table, the cache lock, and when workers start.

The cold census in ``test_import_compat.py`` runs the real workers; these
tests stand in for them where a real compile would take a minute.
"""

from __future__ import annotations

import ast
import importlib
import sys
import threading
import types

import pytest

from superglm import _warmup


def _locked_elsewhere(path: str) -> bool:
    """Whether another open file cannot take the lock on ``path`` now."""
    with open(path, "a+b") as other:
        try:
            if sys.platform == "win32":
                import msvcrt

                other.seek(0)
                msvcrt.locking(other.fileno(), msvcrt.LK_NBLCK, 1)
                other.seek(0)
                msvcrt.locking(other.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other.fileno(), fcntl.LOCK_UN)
        except OSError:
            return True
    return False


def test_every_stage_resolves_and_every_unit_has_a_start_order():
    for _, module, function, _args in _warmup._STAGES:
        assert callable(getattr(importlib.import_module(module), function)), (module, function)
    assert set(_warmup._UNIT_SECONDS) == set(_warmup._units())


def test_a_worker_saves_to_and_loads_from_the_numba_cache_under_the_lock(tmp_path, monkeypatch):
    """Two workers saving different signatures of one kernel must not interleave,
    and a load must not fall between a save's index and its data file."""
    from numba.core import caching

    lock = str(tmp_path / "cache.lock")
    seen = []
    monkeypatch.setattr(
        caching.IndexDataCacheFile,
        "save",
        lambda self, key, data: seen.append(("save", key, data, _locked_elsewhere(lock))),
    )
    monkeypatch.setattr(
        caching.IndexDataCacheFile,
        "load",
        lambda self, key: seen.append(("load", key, _locked_elsewhere(lock))) or "payload",
    )
    _warmup._locked_cache(lock)
    caching.IndexDataCacheFile.save(None, "key", "data")
    assert caching.IndexDataCacheFile.load(None, "key") == "payload"
    assert seen == [("save", "key", "data", True), ("load", "key", True)]
    assert not _locked_elsewhere(lock)


def _fake_stages(monkeypatch):
    calls = []
    module = types.ModuleType("_superglm_fake_stages")
    module.stage = lambda name: calls.append(name)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stages = (
        ("first", module.__name__, "stage", ("a",)),
        ("second", module.__name__, "stage", ("b",)),
        ("first", module.__name__, "stage", ("c",)),
    )
    monkeypatch.setattr(_warmup, "_STAGES", stages)
    monkeypatch.setattr(_warmup, "_UNIT_SECONDS", {"first": 2.0, "second": 1.0})
    return calls


class _Started:
    """Stands in for ``subprocess.Popen``: records each worker and what it was sent."""

    def __init__(self, record, command, **options):
        self.command, self.env, self.code = command, options["env"], None
        record.append(self)

    def communicate(self, code):
        self.code = code

    def kill(self):
        raise AssertionError("no worker to stop")


def _fake_workers(monkeypatch, n_jobs: str, record):
    monkeypatch.setenv("SUPERGLM_N_JOBS", n_jobs)
    monkeypatch.setenv("SUPERGLM_MAX_MEMORY", "64G")
    monkeypatch.setattr(_warmup.subprocess, "Popen", lambda *a, **k: _Started(record, *a, **k))


def test_one_worker_runs_every_stage_here_and_starts_no_process(monkeypatch):
    calls, started = _fake_stages(monkeypatch), []
    _fake_workers(monkeypatch, "1", started)
    _warmup.warmup()
    assert calls == ["a", "b", "c"]
    assert started == []


def test_more_workers_start_each_unit_once_then_run_every_stage_here(monkeypatch, tmp_path):
    calls, started = _fake_stages(monkeypatch), []
    _fake_workers(monkeypatch, "2", started)
    import numba

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path))
    # A path no ASCII locale can encode reaches the worker unchanged, as ASCII.
    monkeypatch.syspath_prepend(str(tmp_path / "caf\u00e9"))
    _warmup.warmup()
    assert sorted(worker.code.rsplit("'", 2)[-2] for worker in started) == ["first", "second"]
    for worker in started:
        assert worker.command == [sys.executable, "-P", "-"]
        assert worker.code.isascii() and "_run_unit(" in worker.code
        assert (
            ast.literal_eval(worker.code.split("sys.path[:] = ", 1)[1].split("\n", 1)[0])
            == sys.path
        )
        assert worker.env["NUMBA_CACHE_DIR"] == str(tmp_path)
    assert calls == ["a", "b", "c"]


def test_a_worker_that_cannot_start_leaves_its_unit_to_this_process(monkeypatch):
    calls = _fake_stages(monkeypatch)
    _fake_workers(monkeypatch, "2", [])

    def cannot_start(*args, **options):
        raise OSError("no interpreter")

    monkeypatch.setattr(_warmup.subprocess, "Popen", cannot_start)
    _warmup.warmup()
    assert calls == ["a", "b", "c"]


def test_no_place_for_the_lock_leaves_every_unit_to_this_process(monkeypatch):
    calls, started = _fake_stages(monkeypatch), []
    _fake_workers(monkeypatch, "2", started)

    def no_scratch(*args, **options):
        raise FileNotFoundError("no temporary directory")

    monkeypatch.setattr(_warmup.tempfile, "TemporaryDirectory", no_scratch)
    _warmup.warmup()
    assert started == []
    assert calls == ["a", "b", "c"]


def test_an_interrupt_stops_the_running_workers_and_starts_no_more(monkeypatch):
    """A Jupyter interrupt reaches this process alone: its workers must not outlive it."""
    _fake_stages(monkeypatch)
    started, killed = [], []
    live, released = threading.Event(), threading.Event()

    class Running(_Started):
        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            live.set()

        def communicate(self, code):
            released.wait(10)

        def kill(self):
            killed.append(self)
            released.set()

    class Pool:
        """Runs the first unit, is interrupted, then drains its queue as a real pool would."""

        def __init__(self, workers):
            self.thread = None

        def map(self, start, units):
            first, *self.queued = units
            self.start = start
            self.thread = threading.Thread(target=start, args=(first,))
            self.thread.start()
            assert live.wait(10)
            raise KeyboardInterrupt

        def shutdown(self, wait):
            self.thread.join()
            for unit in self.queued:
                self.start(unit)

    _fake_workers(monkeypatch, "2", [])
    monkeypatch.setattr(_warmup.subprocess, "Popen", lambda *a, **k: Running(started, *a, **k))
    monkeypatch.setattr(_warmup, "ThreadPoolExecutor", Pool)
    with pytest.raises(KeyboardInterrupt):
        _warmup._compile_in_workers(["first", "second"], 2)
    assert len(started) == 1
    assert killed == started
