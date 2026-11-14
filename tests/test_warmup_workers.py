"""``superglm.warmup``'s worker processes: the stage table, the cache lock, and when workers start.

The cold census in ``test_import_compat.py`` runs the real workers; these
tests stand in for them where a real compile would take a minute.
"""

from __future__ import annotations

import ast
import importlib
import pickle
import sys
import threading
import types
from pathlib import Path

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


def _cache_file(path, stamp):
    from numba.core import caching

    return caching.IndexDataCacheFile(str(path), "kernel-1.py313", stamp)


def test_a_worker_saves_to_and_loads_from_the_numba_cache_under_its_directory_lock(
    tmp_path, monkeypatch
):
    """Two workers saving different signatures of one kernel must not interleave.

    The lock is a file in the cache directory, so the workers of two warmups
    writing one cache (two cluster nodes) take the same one.
    """
    from numba.core import caching

    lock, seen = str(tmp_path / _warmup._LOCK_NAME), []
    cache_file = caching.IndexDataCacheFile
    for name in ("_save_data", "_save_index", "load"):
        method = getattr(cache_file, name)

        def recorded(self, *args, method=method, name=name):
            seen.append((name, _locked_elsewhere(lock)))
            return method(self, *args)

        monkeypatch.setattr(cache_file, name, recorded)
    original = cache_file.save, cache_file.load
    entry = _cache_file(tmp_path, "stamp")
    with _warmup._locked_cache():
        entry.save("key", "kernel")
        assert entry.load("key") == "kernel"
    assert seen == [("_save_data", True), ("_save_index", True), ("load", True)]
    assert (cache_file.save, cache_file.load) == original
    assert not _locked_elsewhere(lock)


def test_a_save_stopped_between_its_writes_leaves_no_entry_naming_an_old_kernel(
    tmp_path, monkeypatch
):
    """A worker killed mid-save must not leave a fresh index naming a stale data file."""
    from numba.core import caching

    _cache_file(tmp_path, "old source").save("old signature", "old kernel")
    writes = []
    for name in ("_save_data", "_save_index"):
        method = getattr(caching.IndexDataCacheFile, name)

        def stopped_on_second(self, *args, method=method):
            writes.append(args)
            if len(writes) == 2:
                raise KeyboardInterrupt  # the worker is stopped after its first write
            return method(self, *args)

        monkeypatch.setattr(caching.IndexDataCacheFile, name, stopped_on_second)
    fresh = _cache_file(tmp_path, "new source")
    with _warmup._locked_cache():
        with pytest.raises(KeyboardInterrupt):
            fresh.save("new signature", "new kernel")
    monkeypatch.undo()
    assert _cache_file(tmp_path, "new source").load("new signature") is None
    # The stale index on disk, which a process on the old source (or a
    # rollback to it) still reads, names only the old kernel.
    assert _cache_file(tmp_path, "old source").load("old signature") == "old kernel"


def test_a_load_from_a_cache_directory_not_made_yet_proceeds_unlocked(tmp_path):
    """No directory, no lock file: the load misses, as Numba's own does, rather than raise."""
    with _warmup._locked_cache():
        assert _cache_file(tmp_path / "not made yet", "stamp").load("key") is None


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


@pytest.mark.parametrize("method", _warmup._CACHE_FILE_METHODS)
def test_a_numba_without_the_cache_methods_the_lock_uses_warms_up_here_alone(
    monkeypatch, method
):
    from numba.core import caching

    calls, started = _fake_stages(monkeypatch), []
    _fake_workers(monkeypatch, "2", started)
    monkeypatch.delattr(caching.IndexDataCacheFile, method)
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


def test_this_process_saves_under_the_workers_lock_while_they_run(monkeypatch, tmp_path):
    """A fit on another thread saves while the workers do: it must take their lock."""
    from numba.core import caching

    _fake_stages(monkeypatch)
    original, held = caching.IndexDataCacheFile.save, []

    class Checking(_Started):
        def communicate(self, code):
            lock = str(tmp_path / _warmup._LOCK_NAME)
            save_data = caching.IndexDataCacheFile._save_data

            def recorded(entry, name, data):
                held.append(_locked_elsewhere(lock))
                return save_data(entry, name, data)

            with pytest.MonkeyPatch.context() as scoped:
                scoped.setattr(caching.IndexDataCacheFile, "_save_data", recorded)
                _cache_file(tmp_path, "stamp").save("key", "kernel")

    _fake_workers(monkeypatch, "2", [])
    monkeypatch.setattr(_warmup.subprocess, "Popen", lambda *a, **k: Checking([], *a, **k))
    _warmup._compile_in_workers(["first"], 2)
    assert held == [True]
    assert caching.IndexDataCacheFile.save is original


def test_one_warmup_runs_at_a_time(monkeypatch):
    calls = _fake_stages(monkeypatch)
    entered, gate = threading.Event(), threading.Event()

    def stage(name):
        calls.append(name)
        if len(calls) == 1:
            entered.set()
            gate.wait(10)

    monkeypatch.setattr(sys.modules["_superglm_fake_stages"], "stage", stage)
    monkeypatch.setenv("SUPERGLM_N_JOBS", "1")
    first, second = (threading.Thread(target=_warmup.warmup) for _ in range(2))
    first.start()
    assert entered.wait(10)
    second.start()
    second.join(0.2)  # the second waits for the first to finish
    assert calls == ["a"]
    gate.set()
    first.join(10)
    second.join(10)
    assert calls == ["a", "b", "c", "a", "b", "c"]


def test_a_worker_that_cannot_start_leaves_its_unit_to_this_process(monkeypatch):
    calls = _fake_stages(monkeypatch)
    _fake_workers(monkeypatch, "2", [])

    def cannot_start(*args, **options):
        raise OSError("no interpreter")

    monkeypatch.setattr(_warmup.subprocess, "Popen", cannot_start)
    _warmup.warmup()
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


def test_an_unreadable_index_leaves_every_existing_data_file_alone(tmp_path):
    """An index another Numba wrote, which this one cannot read, may still name any data file."""
    old = _cache_file(tmp_path, "old source")
    old.save("old signature", "old kernel")
    # Numba reads the version first and skips the rest when it differs.
    Path(old._index_path).write_bytes(pickle.dumps("another numba") + b"not a pickle")
    with _warmup._locked_cache():
        _cache_file(tmp_path, "new source").save("new signature", "new kernel")
    assert pickle.loads(Path(old._data_path(old._data_name(1))).read_bytes()) == "old kernel"
