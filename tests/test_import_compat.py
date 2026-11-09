"""Import surface compatibility tests.

Verifies that the currently supported public import surfaces remain
importable after the src/ cleanup. These tests cover canonical package
entry points and submodule import paths that the codebase still treats as
supported.
"""

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import superglm

# ── Old paths (must keep working after moves) ──────────────────


def test_reml_imports():
    from superglm.reml import REMLResult  # noqa: F401


def test_inference_imports():
    from superglm.inference import (  # noqa: F401
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
    )


def test_diagnostics_imports():
    from superglm.diagnostics import (  # noqa: F401
        SplineRedundancyReport,
        spline_redundancy,
        term_drop_diagnostics,
        term_importance,
    )


def test_validation_imports():
    from superglm.validation import (  # noqa: F401
        DoubleLiftChartResult,
        LiftChartResult,
        LorenzCurveResult,
        LossRatioChartResult,
        double_lift_chart,
        lift_chart,
        lorenz_curve,
        loss_ratio_chart,
    )


# ── Top-level public API ───────────────────────────────────────


def test_toplevel_reexports():
    """Everything in __all__ is importable from the superglm namespace."""
    import superglm

    for name in superglm.__all__:
        assert hasattr(superglm, name), f"superglm.{name} not accessible"


# ── Lazy package roots ─────────────────────────────────────────

_LAZY_PACKAGES = (
    "superglm",
    "superglm.diagnostics",
    "superglm.inference",
    "superglm.features",
    "superglm.model",
    "superglm.reml",
    "superglm.solvers",
    "superglm.solvers.irls_direct",
    "superglm.distributional.solver",
    "superglm._group_matrix",
)
_LIBRARIES = (
    "numpy",
    "pandas",
    "scipy",
    "numba",
    "matplotlib",
    "sklearn",
    "polars",
    "tabmat",
    "pyarrow",
    "plotly",
)


def _loaded_after(statement: str, watched: tuple[str, ...]) -> list[str]:
    """The ``watched`` modules a fresh interpreter has imported once ``statement`` has run."""
    code = f"import sys\n{statement}\nprint([m for m in {watched!r} if m in sys.modules])"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return ast.literal_eval(completed.stdout.strip())


def test_importing_superglm_imports_no_third_party_library():
    """A name costs its import when it is read, not when the package is imported."""
    assert _loaded_after("import superglm", _LIBRARIES) == []


def test_a_population_level_warning_loads_no_fitting_code():
    """A quote at a held ``sz`` level warns on a request thread of a scoring service.

    Mutation check: taking ``warn_caller`` from ``superglm.profiling`` loads ``superglm.model``.
    """
    code = (
        "import warnings; warnings.simplefilter('ignore');"
        "from superglm.distributional.prediction_design import warn_population_levels;"
        "warn_population_levels(['a term'])"
    )
    assert _loaded_after(code, ("superglm.profiling", "superglm.model")) == []


def test_the_module_a_fitted_model_loads_through_needs_no_plotting_or_fitting_code():
    """Unpickling a ``SuperLSS`` imports ``superglm.distributional.api`` and what it imports.

    Plotting, scikit-learn, the normal and chi-squared distributions of ``scipy.stats``, the
    tabmat design backend, and the classic model, summary, diagnostic and profiling code are
    read when a plot, a summary or a fit asks for them. Mutation checks: restoring
    ``from scipy import stats`` at the top of ``distributional/posterior.py``, ``import
    tabmat`` at the top of ``_group_matrix/_group_matrix_tabmat.py``, or the eager imports in
    ``diagnostics/__init__.py`` puts one of them back and fails this.
    """
    libraries = ("matplotlib", "sklearn", "scipy.stats", "tabmat", "polars", "plotly")
    code = tuple(
        f"superglm.{name}"
        for name in (
            "plotting",
            "sklearn",
            "model",
            "inference",
            "profiling",
            "diagnostics.term_diagnostics",
        )
    )
    assert _loaded_after("import superglm.distributional.api", libraries + code) == []


def _type_checking_imports(package: str) -> set[tuple[str, str]]:
    """``(module, name)`` of every import a package keeps under ``typing.TYPE_CHECKING``."""
    path = Path(__file__).resolve().parents[1] / "src" / Path(*package.split(".")) / "__init__.py"
    found: set[tuple[str, str]] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_typing.TYPE_CHECKING":
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom)
                module = statement.module or ""
                if statement.level:
                    module = f"{package}.{module}"
                if module != package:  # ``from superglm import families`` names a submodule
                    found.update((module, alias.name) for alias in statement.names)
    return found


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_a_lazy_package_resolves_every_name_to_the_object_its_module_defines(package):
    """The table the package reads from and the imports a type checker reads are one list.

    Mutation checks: a name dropped from ``_LAZY_EXPORTS``, or added to it without its
    ``TYPE_CHECKING`` import, fails the first assertion; a name filed under the wrong module
    fails the second.
    """
    module = importlib.import_module(package)
    table = {(m, name) for m, names in module._LAZY_EXPORTS.items() for name in names}
    assert table == _type_checking_imports(package)
    for owner, name in sorted(table):
        assert getattr(module, name) is getattr(importlib.import_module(owner), name)
        assert name in dir(module)
    assert not hasattr(module, "no_such_name")


_FIRST_READ_RACE = """
import sys, threading
import superglm
loaded = sorted(m for m in sys.modules if m.startswith("superglm.") and m != "superglm._lazy")
assert loaded == ["superglm._numba_cache"], loaded  # a cold package: nothing it defines is loaded
specs = sys.argv[1:]
start, failures = threading.Barrier(len(specs)), []
def read(spec):
    start.wait()
    try:
        if spec.startswith("import:"):
            __import__(spec[7:])
        else:
            getattr(superglm, spec)
    except BaseException as error:
        failures.append((spec, type(error).__name__, str(error)[:100]))
threads = [threading.Thread(target=read, args=(spec,)) for spec in specs]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(90)
print("hung" if any(t.is_alive() for t in threads) else failures or "ok", flush=True)
sys.stdout.flush()
import os
os._exit(0)
"""


@pytest.mark.parametrize(
    "reads",
    [
        ("SuperGLM", "TweedieProfileResult"),
        ("NBProfileResult", "TweedieProfileResult"),
        ("PriorWeightLatticeWarning", "TweedieProfileResult"),
        ("import:superglm.model", "TweedieProfileResult"),
    ],
    ids=lambda reads: "+".join(reads),
)
def test_threads_that_first_read_names_of_one_import_cycle_at_once_all_succeed(reads):
    """Two threads that first read ``SuperGLM`` and ``TweedieProfileResult`` of a cold package.

    ``superglm.model`` and ``superglm.profiling.tweedie`` import each other, so two loads at
    once make CPython's import locks raise ``_DeadlockError`` in one thread and a half-built
    module in the other (``ImportError: cannot import name ...``): all of the pairs below, in
    every fresh process, on the package without the lock and with the ``superglm.model`` body
    importing its modules. Each pair runs in two fresh interpreters at once, because a load
    that has happened cannot race again, and ``tests/conftest.py`` imports every module in this
    one. The pair of different subpackages ``SuperGLM`` + ``SuperLSS`` never fails and would
    not catch it. Mutation check: ``superglm/model/__init__.py`` importing ``api`` and
    ``fit_ops`` in its body again fails the last pair, a name against an import by name; the
    lock is held to account by the forced overlap below.
    """
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _FIRST_READ_RACE, *reads],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [child.communicate(timeout=180) for child in children]
    assert [out.strip() for out, _ in results] == ["ok", "ok"], results


_DEMO_INIT = (
    "from superglm._lazy import attach\n"
    "__getattr__, __dir__ = attach(__name__, __path__, {exports})\n"
)
_DEMO_DRIVER = """
import os, sys, threading, time
sys.path.insert(0, sys.argv[1])
sys.lazydemo_sync = sync = {{
    "a_in": threading.Event(), "b_holds": threading.Event(), "inside": 0, "peak": 0,
    "mutex": threading.Lock(),
}}
import lazydemo
failures = []
def run(action):
    try:
        action()
    except BaseException as error:
        failures.append(repr(error))
threads = [threading.Thread(target=run, args=(action,)) for action in [{actions}]]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(20)
print("hung" if any(t.is_alive() for t in threads) else (failures or "ok", sync["peak"]), flush=True)
os._exit(0)
"""


def _run_lazydemo(tmp_path, exports, modules, actions):
    """Run ``actions`` on threads against a throwaway package ``lazydemo`` of ``modules``."""
    package = tmp_path / "lazydemo"
    package.mkdir()
    (package / "__init__.py").write_text(_DEMO_INIT.format(exports=exports), encoding="utf-8")
    for name, text in modules.items():
        (package / f"{name}.py").write_text(text, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", _DEMO_DRIVER.format(actions=actions), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip(), completed


_BODY_THAT_STAYS = (
    "import sys, time\nsync = sys.lazydemo_sync\n"
    "with sync['mutex']:\n    sync['inside'] += 1\n    sync['peak'] = max(sync['peak'], sync['inside'])\n"
    "time.sleep(0.6)\n"
    "with sync['mutex']:\n    sync['inside'] -= 1\n{name} = 1\n"
)


def test_first_loads_on_different_threads_run_one_at_a_time(tmp_path):
    """Two names' modules are never being imported at the same moment.

    Each module's body stays for 0.6 s and counts the bodies running at once; two threads that
    first read their names together would overlap. This is the serialisation that keeps
    two loads from meeting inside one import cycle, which no import of superglm's own modules
    can be made to do on demand. Mutation check: ``module_getattr`` calling ``load`` without
    the lock gives a peak of 2.
    """
    result, completed = _run_lazydemo(
        tmp_path,
        "{'lazydemo.p': ('first',), 'lazydemo.q': ('second',)}",
        {"p": _BODY_THAT_STAYS.format(name="first"), "q": _BODY_THAT_STAYS.format(name="second")},
        "lambda: lazydemo.first, lambda: lazydemo.second",
    )
    assert result == "('ok', 1)", completed


def test_a_thread_inside_an_import_never_waits_for_the_first_load_lock(tmp_path):
    """The lock cannot deadlock against an import already running.

    One thread's first read holds the lock while its module's body imports ``a`` by name; the
    thread that is running ``a``'s body, holding ``a``'s import lock, reads a name that needs a
    first load. If it waited for the lock, each thread would wait for the other and neither
    lock manager could see it: the process hangs. It does not wait: it loads the name itself.
    Mutation check: ``_inside_an_import`` returning ``False`` hangs this.
    """
    result, completed = _run_lazydemo(
        tmp_path,
        "{'lazydemo.b': ('held',), 'lazydemo.c': ('other',)}",
        {
            # ``a`` holds its import lock while it reads a name; it does so once ``b``'s body, which
            # holds the first-load lock, is waiting to import ``a``.
            "a": (
                "import sys\nsync = sys.lazydemo_sync\nsync['a_in'].set()\n"
                "assert sync['b_holds'].wait(30)\nimport lazydemo\nlazydemo.other\n"
            ),
            "b": (
                "import sys\nsync = sys.lazydemo_sync\nheld = 1\nsync['b_holds'].set()\n"
                "assert sync['a_in'].wait(30)\nimport lazydemo.a\n"
            ),
            "c": "other = 2\n",
        },
        "lambda: lazydemo.held, lambda: __import__('lazydemo.a')",
    )
    assert result.startswith("('ok'"), completed


_FORK_DRIVER = """
import os, signal, sys, threading, time, warnings
warnings.simplefilter("ignore")
signal.alarm(30)  # shorter than the fork's own wait: a parent stuck in a fork fails, it does not pass late
sys.path.insert(0, sys.argv[1])
sys.lazydemo_sync = sync = {{"body": threading.Event(), "forking": threading.Event()}}
import lazydemo
{driver}
"""


def _fork_in_lazydemo(tmp_path, modules, driver, exports="{'lazydemo.slow': ('value',)}"):
    """Run ``driver`` against a throwaway package; what it prints last, and the process."""
    package = tmp_path / "lazydemo"
    package.mkdir()
    (package / "__init__.py").write_text(_DEMO_INIT.format(exports=exports), encoding="utf-8")
    for name, text in modules.items():
        (package / f"{name}.py").write_text(text, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", _FORK_DRIVER.format(driver=driver), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip().splitlines()[-1:], completed


_REAP_THE_CHILD = """
if pid == 0:
    signal.alarm(15)
    try:
        code = 0 if lazydemo.value == 1 else 3
    finally:
        os._exit(locals().get("code", 4))
_, status = os.waitpid(pid, 0)
print("finished" if status == 0 else f"child status {status}", flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this platform")
def test_a_child_forked_during_a_first_load_reads_that_name_and_the_parent_is_not_stuck(tmp_path):
    """The fork waits for the load, so no module in the child is half run under a dead thread's lock.

    The body being waited for calls ``logging.getLogger`` once the fork has begun: ``logging`` holds
    its lock across a fork, and a fork waiting while it holds that lock would leave the body waiting
    too. A hook of the test's own, registered last and so run first, marks the fork's start. The
    parent has printed nothing on stderr: a hook registered during the wait would run its after-fork
    half unbalanced. Mutation checks: no hold before the fork leaves the child waiting for the
    module's import lock; hooks registered when ``_lazy`` is imported, before ``logging`` is loaded,
    leave the parent stuck until its alarm.
    """
    result, completed = _fork_in_lazydemo(
        tmp_path,
        {
            "slow": (
                "import logging, sys, time\nsync = sys.lazydemo_sync\nsync['body'].set()\n"
                "assert sync['forking'].wait(30)\ntime.sleep(0.2)\n"
                "logging.getLogger('lazydemo.slow')\nvalue = 1\n"
            )
        },
        """
threading.Thread(target=lambda: lazydemo.value, daemon=True).start()
assert sync["body"].wait(30)
os.register_at_fork(before=sync["forking"].set)
pid = os.fork()
"""
        + _REAP_THE_CHILD,
    )
    assert result == ["finished"], completed
    assert "Exception ignored" not in completed.stderr, completed.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this platform")
def test_a_thread_that_forks_inside_an_import_never_waits_for_the_first_load_lock(tmp_path):
    """A module body forks while another thread's first load waits to import that module.

    The forking thread holds the module's import lock and the other thread holds the first-load
    lock and needs it: a fork that waited would wait for a thread that waits for it, and neither lock
    manager could see that. Mutation check: holding the lock whatever the stack hangs the parent
    until its alarm.
    """
    result, completed = _fork_in_lazydemo(
        tmp_path,
        {
            "forks": (
                "import os, sys, threading\nimport lazydemo\nsync = sys.lazydemo_sync\n"
                "threading.Thread(target=lambda: lazydemo.value, daemon=True).start()\n"
                "assert sync['body'].wait(30)\n"
                "sync['child'] = os.fork()\n"
                "if sync['child'] == 0:\n    os._exit(0)\n"
            ),
            "slow": (
                "import sys\nsys.lazydemo_sync['body'].set()\nimport lazydemo.forks\nvalue = 1\n"
            ),
        },
        """
import lazydemo.forks
_, status = os.waitpid(sync["child"], 0)
print("finished" if status == 0 and lazydemo.value == 1 else status, flush=True)
""",
    )
    assert result == ["finished"], completed


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this platform")
def test_a_module_body_that_forks_during_its_first_load_leaves_a_child_that_can_finish_it(tmp_path):
    """The forking thread itself holds the first-load lock, and in the child it still leaves it.

    Mutation check: reinitialising the lock in every child makes the child's ``with`` exit raise
    ``RuntimeError: cannot release un-acquired lock``.
    """
    result, completed = _fork_in_lazydemo(
        tmp_path,
        {"slow": "import os, sys\nsys.lazydemo_sync['child'] = os.fork()\nvalue = 1\n"},
        """
try:
    value = lazydemo.value
except BaseException:
    value = None
if sync["child"] == 0:
    os._exit(0 if value == 1 else 3)
_, status = os.waitpid(sync["child"], 0)
print("finished" if status == 0 and value == 1 else status, flush=True)
""",
    )
    assert result == ["finished"], completed


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this platform")
def test_a_fork_outlasting_its_wait_goes_ahead_and_the_child_is_not_stuck_on_the_lock():
    """A load stuck past the wait does not stop the fork, and the lock is free in the child.

    Mutation checks: an unbounded wait leaves the parent stuck until its alarm; no reinitialisation
    in the child leaves it waiting for a lock whose owner does not exist there.
    """
    code = """
import os, signal, sys, threading, time, warnings
warnings.simplefilter("ignore")
signal.alarm(45)
import superglm
from superglm import _lazy
_lazy._FORK_WAIT_SECONDS = 0.05
_lazy._register_fork_hooks()
held, release = threading.Event(), threading.Event()
def hold():
    with _lazy._FIRST_LOAD:
        held.set()
        release.wait(90)
holder = threading.Thread(target=hold)
holder.start()
held.wait(30)
pid = os.fork()
if pid == 0:
    superglm.warmup  # a name defined in the package itself needs no load
    getattr(superglm, "Numeric")
    os._exit(0)
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    time.sleep(0.05)
else:
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    status = -1
release.set()
print("ok" if status == 0 else "stuck", flush=True)
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert completed.stdout.strip() == "ok", completed


def test_the_root_lists_everything_it_exports_and_every_submodule_resolves():
    names = dir(superglm)
    assert set(superglm.__all__) <= set(names)
    assert {"features", "inference", "families", "solvers"} <= set(names)
    for name in ("features", "inference", "families", "solvers"):
        assert getattr(superglm, name) is importlib.import_module(f"superglm.{name}")


# Every numba kernel in the package compiles in ``superglm.warmup()``, is
# inlined into its callers, or is listed here: a kernel left to the first fit
# that reaches it, with the reason.  tests/_numba_dispatchers.py finds every
# kernel, so one added without that decision fails the test below.
_STRUCTURED = "superglm.solvers._structured."
_GROUP = "superglm._group_matrix._group_matrix_kernels."
_TWEEDIE = "superglm.distributional.kernels._tweedie_numba."
_NB = "superglm.distributional.kernels._negative_binomial_numba."
_DECLARED = "superglm.distributional.kernels._declared."


def test_scalar_observation_warmup_covers_core_dispatchers_and_operands(tmp_path):
    """A bounded first fit must reuse the warm core, including readonly inputs.

    This short subprocess exercises just the new warmup stage. The full census
    below still owns every package dispatcher, import laziness and cold starts.
    """
    code = r"""
import importlib
import numpy as np
from numba.core.dispatcher import Dispatcher
import superglm

superglm._warmup_scalar_observations()
modules = [importlib.import_module("superglm.distributional.kernels." + name)
           for name in ("_incomplete_beta", "_negative_binomial_windows", "_incomplete_gamma",
                        "_gamma_windows", "_tweedie_windows")]
dispatchers = {f"{m.__name__}.{name}": obj for m in modules
               for name, obj in vars(m).items() if isinstance(obj, Dispatcher)}
# The gamma operand module also has uncommon fallbacks owned by the full
# census. Here require every dispatcher defined by the bounded core modules.
for m in (modules[0], modules[1], modules[3], modules[4]):
    for name, obj in vars(m).items():
        if isinstance(obj, Dispatcher) and obj.py_func.__module__ == m.__name__:
            assert obj.signatures, (m.__name__, name)
before = {name: tuple(obj.signatures) for name, obj in dispatchers.items()}
superglm._warmup_scalar_observations()
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import bounded_observation
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
for family, theta in ((NegativeBinomialLS(), [2., .7]), (TweedieLSS(), [2., 1.2, 1.5])):
    y = np.array([1., 2., 3.])
    bounds = ResponseBounds.from_arrays(np.zeros(3), np.full(3, 5.), np.array([-1, 0, 1]))
    for semantics in ("prior", "frequency"):
        w = resolve_likelihood_weights(np.full(3, 2.), n_observations=3, contract=WeightContract(semantics=semantics))
        plan = family.bind_likelihood(y, w, bounded_observation(bounds))
        matrix = np.tile(theta, (3, 1))
        matrix.setflags(write=False)
        for order in (0, 1, 2):
            result = family.evaluate_natural(y, matrix, plan, derivative_order=order)
            assert result.valid.all()
after = {name: tuple(obj.signatures) for name, obj in dispatchers.items()}
assert before == after, {name: after[name] for name in before if before[name] != after[name]}
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=210,
        env={**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "scalar-core-cache")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"

    # The first process filled this private cache. A fresh process must load
    # the same core code without recompiling an imported-callee composition.
    reload_code = r"""
import json
from numba.core import event
import superglm

compiles = []
class Listener(event.Listener):
    def on_start(self, ev):
        function = getattr(ev.data.get("dispatcher"), "py_func", None)
        module = getattr(function, "__module__", "") or ""
        if module.startswith("superglm."):
            compiles.append(module + "." + function.__name__)
    def on_end(self, ev):
        pass

with event.install_listener("numba:compile", Listener()):
    superglm._warmup_scalar_observations()
print(json.dumps(compiles))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", reload_code],
        capture_output=True,
        text=True,
        timeout=210,
        env={**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "scalar-core-cache")},
    )
    assert loaded.returncode == 0, loaded.stderr
    assert json.loads(loaded.stdout) == [], loaded.stdout


def test_dirichlet_warmup_covers_compiled_rows_and_directions(tmp_path):
    code = r"""
import numpy as np
import superglm
from superglm.distributional.kernels import _dirichlet_numba as compiled
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.links import LogLink

superglm._warmup_dirichlet()
names = ("_fsum", "_evaluate_batch", "_curvature_batch")
before = {name: tuple(getattr(compiled, name).nopython_signatures) for name in names}
assert all(before.values()), before
superglm._warmup_dirichlet()
for k in (2, 5):
    family = superglm.DirichletLSS(tuple(f"c{j}" for j in range(k)))
    response = np.full((2, k), 1. / k)
    weights = resolve_likelihood_weights(np.ones(2), n_observations=2,
                                        contract=WeightContract(semantics="frequency"))
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    for readonly in (False, True):
        theta = np.full((2, k), 2.)
        theta.setflags(write=not readonly)
        if not readonly:
            family.expected_information_natural(theta, plan)
        for order in (0, 1, 2):
            assert family.evaluate_natural(response, theta, plan, derivative_order=order).valid.all()
    eta, direction = np.log(theta), np.full_like(theta, .2)
    links = [LogLink()] * k
    family.predictor_curvature_directional_derivative(response, eta, direction, links, plan)
    family.predictor_curvature_second_directional_derivative(response, eta, direction, direction, links, plan)
after = {name: tuple(getattr(compiled, name).nopython_signatures) for name in names}
assert before == after, {name: after[name] for name in names if before[name] != after[name]}
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=210,
        env={**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "dirichlet-core-cache")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


_FIRST_USE: dict[str, str] = {
    # Measured over cold first fits on the freMTPL2 book: 71 on 2026-10-20
    # (every family and link, SuperGLM and SuperLSS, exact and binned,
    # RandomEffect and FactorSmooth terms) and 17 more on 2026-10-21 for the
    # routes those missed (continuous covariates, tensors over disjoint and
    # P-spline margins, a decomposed and a lone tensor, factor smooths on the
    # Gram backend and beside numerics only): which fits reach each kernel,
    # its compile time on top of warmup(), and its load from a filled cache.
    **dict.fromkeys(
        [
            *(
                _TWEEDIE + name
                for name in (
                    "_polygamma_012",
                    "_third_term_channels",
                    "_series_third_cumulants",
                    "_tweedie_third_row",
                    "_tweedie_hessian_direction_rows",
                    "_fourth_term_channels",
                    "_series_fourth_cumulants",
                    "_tweedie_fourth_row",
                    "_tweedie_fourth_rows",
                    "_positive_closed",
                    "_zero_closed",
                    "_zeta_derivatives",
                    "_term_derivatives",
                    "_series_cumulants",
                )
            ),
            *(
                _NB + name
                for name in (
                    "_log_third_rows",
                    "_log_fourth_rows",
                    "_log_third_row",
                    "_log_fourth_row",
                    "_tail_moment_one",
                    "_tail_moment_two",
                    "_tail_moment_three",
                )
            ),
            "superglm.distributional.kernels.gamma._small_shape_jdd_series",
        ],
        "a SuperLSS endpoint step's third and fourth derivatives: none of 26 SuperLSS fits "
        "reached them (3 with a pure-noise smooth); 7 s to compile after warmup(), 0.06 s "
        "from cache",
    ),
    **dict.fromkeys(
        (
            _STRUCTURED + "balance_tree._leaf_path_row_forms",
            _STRUCTURED + "block_leaves._level_row_forms",
        ),
        "a structured FactorSmooth fit's hat diagonal (Cook's distance, standardized "
        "residuals): diagnostics, not the fit",
    ),
    **dict.fromkeys(
        (
            *(
                _DECLARED + f"{module}.{law}_uniform"
                for module in ("gamma", "gamma_log")
                for law in ("logsf", "logcdf")
            ),
        ),
        "GammaLS reads a bounded row's tail by the uniform expansion only above shape 1e4 "
        "(cv below 0.01 at unit weight), a response nearly constant in its cell: no census fit "
        "reaches it",
    ),
    _STRUCTURED + "block_leaves._moment_segments": (
        "the signed moment segments of a binned FactorSmooth term: no census fit reaches "
        "them (fs or sz, Fisher or signed, exact or binned)"
    ),
    _STRUCTURED + "leaf_kernels._sparse_rows": (
        "a RandomEffect leaf pass whose border has a row its tables cannot code: none in the census"
    ),
    _GROUP + "_operand_exponent_bounds": (
        "the non-float64 scan: fits pass float64 operands to _float64_operand_exponent_bounds"
    ),
    **dict.fromkeys(
        (
            _GROUP + "_dense_small_weighted_moments",
            _GROUP + "_factor_smooth_support_dense_cell_aggregates",
            _GROUP + "_factor_smooth_csr_dense_cross",
            _GROUP + "_factor_smooth_support_dense_cross",
        ),
        "a FactorSmooth moment system built without its leaf layout, or crossed with a dense "
        "term of two or more columns: no census fit reaches either (fs or sz, exact or binned, "
        "beside numerics, splines or a polynomial)",
    ),
    _GROUP + "_fused_2d_bincount_2": (
        "a discrete tensor's two weight grids on the chunked moment route, which the raw-moment "
        "and pattern routes now serve: no census fit reaches it"
    ),
    "superglm.solvers._exact_sums.scaled_exact_sum": (
        "the overflow fallback: only a sum whose products or partials leave float64's range "
        "reaches it (0.6 s)"
    ),
    "superglm.plotting.diagnostics._lowess_core": "a diagnostic plot's smoother, not a fit (2.2 s)",
    **dict.fromkeys(
        (
            *(
                "superglm.distributional.kernels._gg_uniform_tail." + name
                for name in ("gg_uniform_log_tail", "_log_tail_rows", "_domain_rows")
            ),
        ),
        "the generalized gamma tails' own scalar and array entry points, for tests: bounded "
        "fits reach both engines through gg_log_tail's row kernel",
    ),
    **dict.fromkeys(
        (
            _DECLARED + f"{module}.carrier"
            for module in (
                "generalized_pareto",
                "log_normal_mean",
                "two_piece_location",
                "two_piece_mean",
            )
        ),
        "generated with every declaration, but these densities have no parameter-free part, "
        "so their rows and windows never read the carrier",
    ),
}

_WARMUP_CENSUS = r"""
import json

import numpy as np
import superglm
from superglm._group_matrix import _group_matrix_algebra as group_algebra
from superglm._group_matrix import _group_matrix_kernels as group_kernels
from superglm._group_matrix import _group_matrix_support as support_scans
from superglm.features import interaction
from superglm.reml import multi_penalty
from tests._numba_dispatchers import is_inlined, package_dispatchers, signatures

dispatchers, problems = package_dispatchers()
at_import = signatures(dispatchers)
superglm.warmup()
warmed = signatures(dispatchers)

# These core observation routes have Python callers with scalar/readonly
# operands distinct from the complete-row kernels. Census them after warmup.
superglm._warmup_scalar_observations()
superglm._warmup_dirichlet()


def readonly(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays


# The row Kronecker build and the support scans, through their production entry
# points: a first interaction fit must reuse the signatures warmup compiled.
import scipy.sparse as sp

basis = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]]))
interaction._row_kron(basis, basis)
row_hashes = support_scans._row_hashes(basis, 2)
_, first_rows, row_groups = np.unique(row_hashes, return_index=True, return_inverse=True)
assert support_scans._verified_representatives(basis, first_rows, row_groups, 2) is not None

# The SuperLSS penalty value passes one operand form, whatever its coefficients' flags.
from superglm.distributional.solver.solver import _half_penalty_quadratic, _penalty_entries

# np.nonzero's index layout differs with the nonzero count: none, one and several.
for diagonal, expected in (([0.0, 0.0, 0.0], 0.0), ([0.0, 2.0, 0.0], 4.0), ([0.0, 2.0, 3.0], 17.5)):
    penalty = np.diag(diagonal)
    entries = _penalty_entries(penalty)
    for coefficients in (np.array([1.0, 2.0, 3.0]), readonly(np.array([1.0, 2.0, 3.0]))[0]):
        assert _half_penalty_quadratic(penalty, coefficients, entries) == expected

# Every layout a caller holds reaches the range scans through ``_native``, as
# their call sites pass it: C, Fortran and strided, writable or read-only.
for values in (
    np.ones((3, 4)),
    np.asfortranarray(np.ones((3, 4))),
    np.ones((6, 4))[::2, ::2],
    np.ones((3, 4))[::-1],
    np.ones(9)[::2, None],
):
    for operand in (values, *readonly(values.copy() if values.flags.owndata else values.view())):
        assert group_kernels._tensor_operand_in_reassociation_range(
            group_kernels._scan_form(operand)
        )
        assert group_algebra._operand_exponent_bounds(operand) == (0, 1)

# A weight kernel's call site hands it a read-only C operand (``_native``),
# whatever the caller holds: SuperGLM's weights, SuperLSS's read-only channel
# columns (binned) and strided ones (exact).
from superglm._group_matrix._group_matrix_core import _bincount_levels
from superglm.group_matrix import _disc_disc_2d_hist

four = np.array([0, 1, 1, 0], dtype=np.intp)
for weights in (np.ones(4), np.ones(8)[::2], *readonly(np.ones(4), np.ones(8)[::2])):
    np.testing.assert_array_equal(_bincount_levels(four, weights, 2), [2.0, 2.0])
    joint = _disc_disc_2d_hist(four, four, weights, 2, 2)
    np.testing.assert_array_equal(joint, np.diag([2.0, 2.0]))

frozen_response, frozen_mean = readonly(np.array([0.0, 1.0, 2.5]), np.ones(3))
assert np.all(np.isfinite(superglm.tweedie_logpdf(frozen_response, frozen_mean, 1.0, 1.5)))

values = np.array([1.0, 2.0], dtype=np.float64)
codes = np.array([0, 1], dtype=np.intp)
csr_indices = np.array([0, 1], dtype=np.int32)
csr_indptr = np.array([0, 1, 2], dtype=np.int32)
cell_ptr, cell_order = group_kernels._cell_csr(codes, codes, 2, 2)
assert group_kernels._cell_csr_matches(cell_ptr, cell_order, codes, codes, 2, 2)
group_kernels._csr_weighted_gram(
    values, csr_indices, csr_indptr, group_kernels._native(values), 2, absolute_weights=True
)

row_patterns, unique_codes, offsets, pair_left, pair_right, pair_offsets, right_sizes = readonly(
    np.array([0, 1], dtype=np.int32),
    np.array([[0, 0], [1, 1]], dtype=np.int32),
    np.array([0, 2, 4], dtype=np.intp),
    codes[:1],
    codes[1:],
    np.array([0, 4], dtype=np.intp),
    np.array([2], dtype=np.intp),
)
group_kernels._pattern_support_summaries(
    row_patterns,
    unique_codes,
    values,
    values,
    offsets,
    pair_left,
    pair_right,
    pair_offsets,
    right_sizes,
)

# Dot2 refinement passes one operand form, whatever its callers' layouts and flags.
rng = np.random.default_rng(0)
left, right = rng.normal(size=(3, 300)), rng.normal(size=(300, 2))
for first in (left, readonly(left.copy())[0], np.asfortranarray(left)):
    for second in (right, readonly(right.copy())[0]):
        multi_penalty._refine_product(first, second, *multi_penalty._matmul_enclosed(first, second))
        multi_penalty._compensated_dot(first[0], second[:, 0])
        multi_penalty._compensated_dot(first[0], np.ascontiguousarray(second[:, 1]))

entry_points = signatures(dispatchers)
superglm.warmup()
rewarmed = signatures(dispatchers)

# First fits: a binned SuperGLM fit and binned Tweedie and negative binomial
# SuperLSS fits, each with a spline, a categorical and a RandomEffect of 300
# levels (the structured engine's leaf pass).
import warnings

import pandas as pd
from superglm import cat, re, s

warnings.simplefilter("ignore")
rng = np.random.default_rng(7)
n = 4000
frame = pd.DataFrame({
    "x": rng.uniform(0, 1, n),
    "g": rng.choice(list("pqrs"), n),
    "u": rng.integers(0, 300, n).astype(str),
})
mu = np.exp(np.sin(3 * frame.x) + rng.normal(0, 0.3, 300)[frame.u.astype(int)])
counts = rng.poisson(mu).astype(np.float64)
amounts = rng.gamma(1.5, mu / 1.5) * (rng.uniform(size=n) < 0.6)
terms = (s("x", kind="cr", k=8), cat("g"), re("u"))
superglm.SuperGLM(family="poisson", discrete=True, terms=list(terms)).fit_reml(frame, counts)
tweedie = superglm.TweedieLSS()
model = superglm.SuperLSS(tweedie, tweedie.mu(*terms), tweedie.phi(), tweedie.p(), discrete=True)
model.fit_reml(frame, amounts)
negative_binomial = superglm.NegativeBinomialLS()
model = superglm.SuperLSS(
    negative_binomial, negative_binomial.mean(*terms), negative_binomial.theta(), discrete=True
)
model.fit_reml(frame, counts)

# Complete compositions use the warmed row and both directional kernels.
dirichlet = superglm.DirichletLSS(("a", "b", "c"))
composition = np.array([rng.dirichlet([2., 3., 4.]) for _ in range(120)])
for discrete in (False, True):
    model = superglm.SuperLSS(
        dirichlet,
        *(dirichlet.alpha(c, s("x", kind="cr", k=5)) for c in dirichlet.components),
        discrete=discrete,
    )
    model.fit_reml(frame.iloc[:120], composition)

# Both declared pilots reuse the warmed generated rows, including their
# carriers on immutable prepared responses and their bounded windows.
for family in (superglm.LomaxLSS(), superglm.LogLogisticLSS()):
    pilot_frame=frame.iloc[:200]
    severity=family.quantile((np.arange(200)+.5)/200, np.tile([2.,2.5],(200,1)))
    for bounded in (False,True):
        options={"truncation":(0.,np.inf),"censored":severity>3.} if bounded else {}
        y=np.minimum(severity,3.) if bounded else severity
        superglm.SuperLSS(family,family.scale(),family.shape()).fit(pilot_frame,y,**options)

# An exact ordinary spline: the sparse-SSP Gram's direct kernel call.
superglm.SuperGLM(
    family="poisson", discrete=False, terms=[s("x", kind="ps", k=12)]
).fit_reml(frame.iloc[:2000], counts[:2000])

# Binned factor smooths of both bases on a 50-level group: the structured
# engine's FactorSmooth leaf pass, which warmup compiles by hand.
from superglm import FactorSmooth
from superglm import interaction as term_interaction

frame["h"] = rng.integers(0, 50, n).astype(str)
for smooths in (
    [term_interaction(FactorSmooth("x", group="h", basis="fs", k=5))],
    [s("x", kind="ps", k=7, m=2), term_interaction(FactorSmooth("x", group="h", basis="sz", kind="ps", k=5, m=2))],
):
    superglm.SuperGLM(family="poisson", discrete=True, terms=smooths).fit_reml(frame, counts)

# An exact gamma (log link) fs fit: observed curvature takes the signed half of
# the leaf pass (the signed leaf segments and trials), and an exact term its
# sorted CSR basis rows, raw moments and compensated level sums.
exact = frame.iloc[:2000]
superglm.SuperGLM(
    family="gamma", terms=[term_interaction(FactorSmooth("x", group="h", basis="fs", k=5))]
).fit_reml(exact, rng.gamma(2.0, mu.to_numpy()[:2000] / 2.0))
print(json.dumps({
    "problems": problems,
    "inlined": sorted(name for name, value in dispatchers.items() if is_inlined(value)),
    "at_import": {name: found for name, found in at_import.items() if found},
    "warmed": warmed,
    "entry_points": entry_points,
    "rewarmed": rewarmed,
    "fitted": signatures(dispatchers),
}))
"""


def _changed(before: dict, after: dict) -> dict:
    return {name: after[name] for name in after if after[name] != before.get(name)}


@pytest.mark.slow
def test_public_warmup_compiles_every_kernel_but_the_first_use_allowlist(tmp_path) -> None:
    """After ``superglm.warmup()`` a first fit compiles only the kernels in ``_FIRST_USE``.

    On an empty numba cache a kernel compiles the kernels it calls, so after
    warmup each of the package's dispatchers has a signature, is inlined
    (``inline="always"``: it never compiles on its own), or is listed in
    ``_FIRST_USE``.  A listed kernel that warmup compiles, or that no longer
    exists, fails as well, so the list cannot rot.  The production entry points
    then run with the layouts and flags their callers pass and must reuse
    warmup's signatures, a second warmup compiles nothing, and first fits
    (SuperGLM, and Tweedie and negative binomial SuperLSS, binned, with a
    spline, a categorical and a structured RandomEffect; an exact ordinary
    spline; binned ``fs`` and ``sz`` factor smooths; and an exact gamma ``fs``
    fit, which takes the signed rows) compile nothing outside ``_FIRST_USE``.
    ``warmup()`` stays a bare call: import compiles nothing.
    """
    assert inspect.signature(superglm.warmup).parameters == {}
    completed = subprocess.run(
        [sys.executable, "-c", _WARMUP_CENSUS],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba-cache")),
        timeout=1800,  # a cold warmup takes about a minute; this bounds a hang
    )
    assert completed.returncode == 0, completed.stderr
    census = json.loads(completed.stdout)
    assert not census["problems"], census["problems"]
    assert not census["at_import"], f"importing superglm compiled {sorted(census['at_import'])}"
    warmed, inlined = census["warmed"], set(census["inlined"])
    unclassified = sorted(
        name
        for name, found in warmed.items()
        if not found and not {name} & (inlined | _FIRST_USE.keys())
    )
    assert not unclassified, (
        f"warmup() compiles none of {unclassified}: warm it, or list it in _FIRST_USE with the reason"
    )
    stale = sorted(
        name for name in _FIRST_USE if name not in warmed or warmed[name] or name in inlined
    )
    assert not stale, (
        f"_FIRST_USE lists kernels warmup() compiles or the package no longer has: {stale}"
    )
    compiled_inline = sorted(name for name in inlined if warmed[name])
    assert not compiled_inline, (
        f"warmup() compiled the inlined {compiled_inline} on its own: numba caches no inlined "
        "kernel, so it would compile again in every process"
    )
    assert not _changed(warmed, census["entry_points"]), "a first fit's operands compiled anew"
    assert not _changed(warmed, census["rewarmed"]), "a second warmup() compiled anew"
    first_fits = sorted(set(_changed(census["rewarmed"], census["fitted"])) - _FIRST_USE.keys())
    assert not first_fits, f"a first fit after warmup() compiled {first_fits}"


def test_an_exception_raised_by_a_root_export_is_catchable_from_the_root():
    """A caller must be able to catch what the top-level API can raise.

    ``export_rating_tables`` is exported from the package root, so a user
    following the documented surface writes ``from superglm import
    export_rating_tables`` and never imports ``superglm.export`` at all.  If the
    exception it raises lives only on the submodule, that user cannot name the
    failure mode in an ``except`` clause without reaching past the API they were
    given.  Asserted as a rule over the raising functions rather than as one
    hard-coded pair, so the next exception added to this module is covered by
    the same check instead of needing a new test.
    """
    import superglm
    from superglm import export as export_module

    assert "export_rating_tables" in superglm.__all__
    raisable = [
        name
        for name in export_module.__all__
        if isinstance(getattr(export_module, name), type)
        and issubclass(getattr(export_module, name), Exception)
    ]
    assert raisable, "superglm.export exports no exception, so this rule pins nothing"

    for name in raisable:
        assert hasattr(superglm, name), (
            f"{name} can be raised through the root-exported export_rating_tables "
            f"but is not importable from superglm"
        )
        assert name in superglm.__all__, f"{name} is reachable but undocumented in __all__"
        assert getattr(superglm, name) is getattr(export_module, name), (
            f"superglm.{name} is a different object from superglm.export.{name}, so "
            "an except clause written against one would not catch the other"
        )


def test_the_structured_solver_error_is_catchable_from_the_root():
    """A structured fit that cannot proceed raises ``StructuredSolverError``
    (no other solver is tried); a caller catches it as ``superglm.StructuredSolverError``
    or, as before it existed, as ``numpy.linalg.LinAlgError``."""
    import numpy as np

    import superglm
    from superglm.solvers import irls_direct

    assert "StructuredSolverError" in superglm.__all__
    assert superglm.StructuredSolverError is irls_direct.StructuredSolverError
    assert issubclass(superglm.StructuredSolverError, np.linalg.LinAlgError)


def test_the_observed_mode_errors_are_catchable_from_the_root():
    """A REML fit through observed geometry can raise ``ObservedModeNotConvergedError``
    (0.36.0 notes), a subclass of ``ObservedModeNotCertifiedError``; both are named
    from ``superglm`` and are the classes ``observed_geometry`` raises (#449)."""
    import superglm
    from superglm import ObservedModeNotCertifiedError, ObservedModeNotConvergedError
    from superglm.reml import observed_geometry

    for name in ("ObservedModeNotCertifiedError", "ObservedModeNotConvergedError"):
        assert name in superglm.__all__
    assert ObservedModeNotCertifiedError is observed_geometry.ObservedModeNotCertifiedError
    assert ObservedModeNotConvergedError is observed_geometry.ObservedModeNotConvergedError
    assert issubclass(ObservedModeNotConvergedError, ObservedModeNotCertifiedError)
    assert issubclass(ObservedModeNotCertifiedError, RuntimeError)


def test_public_model_signatures_do_not_expose_private_frame_adapter():
    from superglm import SuperGLM

    for name, method in inspect.getmembers(SuperGLM, inspect.isfunction):
        if not name.startswith("_"):
            assert "EagerFrame" not in str(inspect.signature(method)), name


def test_public_plotting_signatures_do_not_expose_private_frame_adapter():
    import superglm.plotting as plotting

    for name in plotting.__all__:
        assert "EagerFrame" not in str(inspect.signature(getattr(plotting, name))), name


def test_parallel_conftest_hook_loads_every_warning_class_module() -> None:
    """Under ``-n``, the conftest leaves xdist no warning module to import.

    pytest-xdist rebuilds a worker's warning by importing the warning class's
    module in a receiver thread. ``pytest_configure`` in tests/conftest.py
    imports superglm on the main thread first when workers will run. That
    protects the receiver threads only while the import loads every module
    defining a warning class. Serial runs must not pay for the import.
    """
    root = Path(__file__).parents[1]
    sources = {
        path: path.read_text(encoding="utf-8") for path in (root / "src" / "superglm").rglob("*.py")
    }
    modules = sorted(
        ".".join(path.relative_to(root / "src").with_suffix("").parts).removesuffix(".__init__")
        for path, text in sources.items()
        # Parse only files that can define one: parsing all of them takes 1.5 s.
        if "Warning" in text
        and any(
            isinstance(node, ast.ClassDef)
            and any(
                getattr(base, "id", getattr(base, "attr", "")).endswith("Warning")
                for base in node.bases
            )
            for node in ast.walk(ast.parse(text))
        )
    )
    assert modules
    script = f"""
import sys, types
import tests.conftest as conftest

def configure(**option):
    conftest.pytest_configure(types.SimpleNamespace(option=types.SimpleNamespace(**option)))

configure()  # CI: pytest-xdist is not installed, so there is no numprocesses option
configure(numprocesses=None)  # xdist installed, no -n
assert "superglm" not in sys.modules, "a serial run imported superglm"
configure(numprocesses=2)
print([module for module in {modules!r} if module not in sys.modules])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_pandas_fit_does_not_import_optional_polars_backend():
    script = r"""
import importlib.abc
import importlib.util
import sys

class RejectPolars(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "polars" or fullname.startswith("polars."):
            raise AssertionError(f"unexpected optional import: {fullname}")
        return None

sys.meta_path.insert(0, RejectPolars())

# Tabmat probes optional dataframe packages with find_spec before importing
# them. Simulate the answer from an environment where Polars is not installed;
# the rejecting finder above still fails any actual import attempt.
real_find_spec = importlib.util.find_spec
def optional_polars_is_absent(name, package=None):
    if name == "polars" or name.startswith("polars."):
        return None
    return real_find_spec(name, package)
importlib.util.find_spec = optional_polars_is_absent

import numpy as np
import pandas as pd
from superglm import Numeric, SuperGLM

X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
y = np.array([0.1, 1.1, 2.1, 3.1])
model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={"x": Numeric()},
).fit(X, y)
prediction = model.predict(X)
assert prediction.shape == (4,)
assert not any(name == "polars" or name.startswith("polars.") for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


# ── Supported canonical paths ───────────────────────────────────


def test_reml_result_canonical():
    from superglm.reml.result import PenaltyCache, REMLResult, _map_beta_between_bases  # noqa: F401


def test_reml_penalty_algebra_canonical():
    from superglm.reml.penalty_algebra import (  # noqa: F401
        build_penalty_caches,
        build_penalty_components,
        cached_logdet_s_plus,
        compute_logdet_s_derivatives,
        compute_logdet_s_plus,
        compute_total_penalty_rank,
    )


def test_reml_optimizer_canonical():
    from superglm.reml.direct import optimize_direct_reml  # noqa: F401
    from superglm.reml.discrete import optimize_discrete_reml_cached_w  # noqa: F401
    from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian  # noqa: F401
    from superglm.reml.objective import reml_laml_objective  # noqa: F401

    # ``superglm.reml.runner.run_reml_once`` was here until the dead covariance
    # chain was deleted (PR "Delete the covariance chain no production fit
    # reaches"). Its whole module is gone: the only inbound chain ended at
    # ``SuperGLM._run_reml_once``, which nothing called, and a full-suite
    # runtime trace recorded zero production frames on it. This entry is
    # dropped deliberately -- it is a supported-surface removal, not an
    # oversight, and it belongs in 0.30.0's changelog.
    from superglm.reml.w_derivatives import (  # noqa: F401
        compute_d2W_deta2,
        compute_dW_deta,
        reml_w_correction,
    )


def test_reml_multi_penalty_canonical():
    from superglm.reml.multi_penalty import (  # noqa: F401
        SimilarityTransformResult,
        logdet_s_gradient,
        logdet_s_hessian,
        similarity_transform_logdet,
    )


def test_inference_term_old_path_still_resolves_and_warns():
    """``superglm.inference.term`` was a re-export shell; each name resolves to its owner."""
    import superglm.inference.term as old
    from superglm.inference import _term_ops, _term_types

    with pytest.warns(FutureWarning, match=r"use superglm\.inference\.TermInference\.") as caught:
        assert old.TermInference is _term_types.TermInference
    assert caught[0].filename == __file__  # the caller's line, as from ``superglm.inference``
    with pytest.warns(FutureWarning, match=r"use SuperGLM\.term_inference\."):
        assert old.term_inference is _term_ops.term_inference
    with pytest.raises(AttributeError):
        old._safe_exp  # noqa: B018


@pytest.mark.parametrize("name", sorted(superglm.inference._MOVED))
def test_inference_moved_name_resolves_and_warns_with_its_replacement(name):
    """Each helper ``superglm.inference`` used to re-export resolves, naming what replaces it."""
    import importlib
    import re

    import superglm.inference as package

    owner, replacement = package._MOVED[name]
    match = f"use {replacement}." if replacement is not None else f"its code is in {owner}."
    with pytest.warns(FutureWarning, match=re.escape(match)) as caught:
        resolved = getattr(package, name)
    assert resolved is getattr(importlib.import_module(owner), name)
    assert caught[0].filename == __file__  # the caller's line, not a line of superglm's


# The public names each module's ``__all__`` listed before its helpers moved out.
_LEGACY_STAR_NAMES = {
    "superglm.inference": (
        "FactorSmoothResult",
        "InteractionInference",
        "ModelMetrics",
        "ModelSummary",
        "RandomEffectResult",
        "SmoothCurve",
        "SplineMetadata",
        "TermInference",
        "build_basis_detail",
        "build_coef_rows",
        "drop1",
        "feature_se_from_cov",
        "refit_unpenalised",
        "relativities",
        "simultaneous_bands",
        "spline_group_enrichment",
        "term_inference",
    ),
    "superglm.inference.term": (
        "InteractionInference",
        "SmoothCurve",
        "SplineMetadata",
        "TermInference",
        "drop1",
        "feature_se_from_cov",
        "refit_unpenalised",
        "relativities",
        "simultaneous_bands",
        "spline_group_enrichment",
        "term_inference",
    ),
}


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_a_star_import_in_a_cold_process_binds_the_packages_public_names(package):
    """A star import reads only what a lazy package has loaded unless ``__all__`` names the rest,
    so each has one, and the internal packages' hold every public name of their tables.

    Mutation check: no ``__all__`` on ``superglm.solvers`` or ``superglm.reml`` binds only ``attach``.
    """
    import importlib

    module = importlib.import_module(package)
    if package in ("superglm.reml", "superglm.solvers"):
        table = {name for names in module._LAZY_EXPORTS.values() for name in names}
        assert set(module.__all__) == {name for name in table if not name.startswith("_")}
    code = f"namespace = {{}}; exec('from {package} import *', namespace); print(*namespace)"
    completed = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert set(module.__all__) <= set(completed.stdout.split())


@pytest.mark.parametrize("module_name", sorted(_LEGACY_STAR_NAMES))
def test_inference_star_import_still_binds_every_legacy_name(module_name):
    """``from <module> import *`` binds each legacy name, warning for the moved ones."""
    import importlib
    import warnings

    module = importlib.import_module(module_name)
    namespace: dict[str, object] = {}
    with pytest.warns(FutureWarning, match="is deprecated and will be removed"):
        exec(f"from {module_name} import *", namespace)  # noqa: S102
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = {name: getattr(module, name) for name in _LEGACY_STAR_NAMES[module_name]}
    for name, value in expected.items():
        assert namespace[name] is value, name


def test_inference_unknown_name_raises_attribute_error():
    import superglm.inference as package

    with pytest.raises(AttributeError, match="no attribute 'not_a_moved_name'"):
        package.not_a_moved_name  # noqa: B018


def test_simultaneous_bands_resolves_from_both_legacy_paths_and_returns_the_bands():
    import numpy as np
    import pandas as pd

    from superglm import Spline, SuperGLM
    from superglm.inference._term_covariance import simultaneous_band_and_critical_value

    with pytest.warns(FutureWarning, match=r"use SuperGLM\.simultaneous_bands\."):
        from superglm.inference import simultaneous_bands as package_bands
    with pytest.warns(FutureWarning, match=r"use SuperGLM\.simultaneous_bands\."):
        from superglm.inference.term import simultaneous_bands as term_bands
    assert package_bands is term_bands

    rng = np.random.default_rng(575)
    X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 200)})
    y = np.sin(3.0 * X["x"].to_numpy()) + rng.normal(0.0, 0.3, 200)
    model = SuperGLM(
        family="gaussian", features={"x": Spline(n_knots=6)}, selection_penalty=0.0
    ).fit(X, y)
    inputs = {
        "result": model.result,
        "groups": model._groups,
        "specs": model._specs,
        "covariance_fn": lambda: model._coef_covariance,
        "n_sim": 500,
        "n_points": 25,
    }
    expected, _ = simultaneous_band_and_critical_value("x", **inputs)
    pd.testing.assert_frame_equal(package_bands("x", **inputs), expected)


def test_inference_metrics_canonical():
    from superglm.inference.metrics import ModelMetrics  # noqa: F401


def test_inference_coef_tables_canonical():
    from superglm.inference.coef_tables import (  # noqa: F401
        build_basis_detail,
        build_coef_rows,
    )


def test_inference_summary_canonical():
    from superglm.inference.summary import (  # noqa: F401
        ModelSummary,
        _BasisDetailRow,
        _CoefRow,
        _compute_coef_stats,
    )


def test_inference_covariance_canonical():
    # ``_penalised_xtwx_inv`` and ``_penalised_xtwx_inv_gram`` were listed here
    # until the dead covariance chain was deleted (PR "Delete the covariance
    # chain no production fit reaches").  Neither had a production caller: the
    # gram form was reached only from ``compute_coef_covariance`` and
    # ``reml/runner.py``, both themselves dead, and the dense form was a
    # test-only oracle.  Their supported-surface removal lands in 0.30.0.
    # ``_active_penalty_matrix`` is the live half of that assembly and is added
    # here in their place, since ``inference/metrics.py`` and
    # ``model/state_ops.py`` both import it across package boundaries.
    from superglm.inference.covariance import (  # noqa: F401
        _active_penalty_matrix,
        _second_diff_penalty,
    )


def test_profiling_tweedie_canonical():
    from superglm.profiling.tweedie import TweedieProfileResult  # noqa: F401


def test_profiling_nb_canonical():
    from superglm.profiling.nb import NBProfileResult, NBThetaBoundWarning  # noqa: F401


def test_stats_model_tests_canonical():
    from superglm.stats.model_tests import (  # noqa: F401
        DispersionTestResult,
        ScoreTestZIResult,
        VuongTestResult,
        ZeroInflationResult,
        dispersion_test,
        score_test_zi,
        vuong_test,
        zero_inflation_index,
    )


def test_stats_davies_canonical():
    from superglm.stats.davies import psum_chisq, satterthwaite  # noqa: F401


def test_stats_wood_pvalue_canonical():
    from superglm.stats.wood_pvalue import wood_test_smooth  # noqa: F401


def test_diagnostics_spline_checks_canonical():
    from superglm.diagnostics.spline_checks import (  # noqa: F401
        SplineRedundancyReport,
        spline_redundancy,
    )


def test_diagnostics_term_diagnostics_canonical():
    from superglm.diagnostics.term_diagnostics import (  # noqa: F401
        term_drop_diagnostics,
        term_importance,
    )


def test_diagnostics_discretize_canonical():
    from superglm.diagnostics.discretize import (  # noqa: F401
        DiscretizationResult,
        discretization_impact,
    )
