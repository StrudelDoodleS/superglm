# Cached Tweedie Trace Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Matplotlib-only `TweedieProfileResult.trace_plot()` that visualizes cached search evaluations without performing any new profile evaluations.

**Architecture:** The result object will prepare finite `p`/mean-NLL pairs directly from its immutable `search_trace`, sort them by `p`, and render their scaled objective differences using the existing SuperGLM Matplotlib style constants. The existing dense `profile_plot()` remains unchanged; tests enforce that the new method never accesses the lazy objective or mutates evaluation/CI state.

**Tech Stack:** Python 3.10+, pandas, NumPy, Matplotlib, pytest, Ruff.

---

### Task 1: Specify the cached-only plotting contract

**Files:**
- Modify: `tests/test_profile_ci.py`

- [ ] **Step 1: Add a deterministic trace-result fixture and failing rendering tests**

Add focused tests to `TestTweedieProfileCI` using a deliberately unsorted trace:

```python
@staticmethod
def _cached_trace_result(phi_method="mle"):
    trace = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "p": [1.8, 1.2, 1.5],
            "phi": [0.9, 1.2, 1.0],
            "nll": [2.1, 2.2, 2.0],
            "n_iter": [1, 5, 2],
            "fit_converged": [True, True, True],
            "source": ["brent", "brent", "brent"],
        }
    )

    def unexpected_objective(_p):
        raise AssertionError("trace_plot must not evaluate the profile objective")

    return tweedie_module.TweedieProfileResult(
        p_hat=1.5,
        phi_hat=1.0,
        nll=2.0,
        n_evaluations=3,
        converged=True,
        method="brent",
        phi_method=phi_method,
        search_trace=trace,
        _objective=unexpected_objective,
        _ll_scale=10.0,
        _evaluation_count=lambda: 3,
    )

def test_trace_plot_uses_only_sorted_cached_evaluations(self):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = self._cached_trace_result()
    before_count = result.n_total_evaluations
    before_trace = result.search_trace.copy(deep=True)

    fig = result.trace_plot()
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.2, 1.5, 1.8])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [4.0, 0.0, 2.0])
    assert ax.get_xlabel() == "p"
    assert ax.get_ylabel() == "Profile deviance"
    assert result.n_total_evaluations == before_count
    pd.testing.assert_frame_equal(result.search_trace, before_trace)
    assert result._ci_cache == {}
    assert result._ci_details_cache == {}
    plt.close(fig)

def test_trace_plot_uses_supplied_axis_and_neutral_pearson_label(self):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outer_fig, supplied_ax = plt.subplots()
    result = self._cached_trace_result(phi_method="pearson")

    returned_fig = result.trace_plot(ax=supplied_ax)

    assert returned_fig is outer_fig
    assert supplied_ax.get_ylabel() == "Profile objective difference"
    assert "likelihood" not in supplied_ax.get_title().lower()
    plt.close(outer_fig)
```

- [ ] **Step 2: Add failing finite-row and empty-trace tests**

```python
def test_trace_plot_ignores_nonfinite_rows(self):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = self._cached_trace_result()
    result.search_trace.loc[0, "nll"] = np.nan

    fig = result.trace_plot()

    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1.2, 1.5])
    plt.close(fig)

@pytest.mark.parametrize(
    "trace",
    [
        pd.DataFrame(columns=["p", "nll"]),
        pd.DataFrame({"p": [np.nan], "nll": [np.inf]}),
    ],
)
def test_trace_plot_rejects_trace_without_finite_points(self, trace):
    result = self._cached_trace_result()
    result.search_trace = trace

    with pytest.raises(RuntimeError, match="finite p/nll"):
        result.trace_plot()
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
rtk proxy uv run pytest tests/test_profile_ci.py -q -k "trace_plot"
```

Expected: failures report that `TweedieProfileResult` has no `trace_plot` attribute.

- [ ] **Step 4: Commit the failing tests**

```bash
rtk git add tests/test_profile_ci.py
rtk git commit -m "Test cached Tweedie trace plotting"
```

### Task 2: Implement the Matplotlib trace renderer

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_profile_ci.py`

- [ ] **Step 1: Implement the minimal cached-only method**

Add `trace_plot()` immediately before `profile_plot()` on `TweedieProfileResult`:

```python
def trace_plot(self, *, ax=None):
    """Plot cached Tweedie p-search evaluations without fitting new models."""
    import matplotlib.pyplot as plt

    from superglm.plotting.common import (
        _LINE_COLOR,
        _LINE_WIDTH,
        _PW_FILL,
        _REF_COLOR,
        _REF_LW,
    )

    try:
        trace_p = np.asarray(self.search_trace["p"], dtype=np.float64)
        trace_nll = np.asarray(self.search_trace["nll"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Tweedie search trace must contain numeric p/nll values") from exc

    finite = np.isfinite(trace_p) & np.isfinite(trace_nll)
    if not np.any(finite):
        raise RuntimeError("Tweedie search trace contains no finite p/nll evaluations")

    order = np.argsort(trace_p[finite], kind="stable")
    plotted_p = trace_p[finite][order]
    plotted_difference = 2.0 * self._ll_scale * (trace_nll[finite][order] - self.nll)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.get_figure()

    ax.plot(
        plotted_p,
        plotted_difference,
        color=_LINE_COLOR,
        linewidth=_LINE_WIDTH,
        marker="o",
        markersize=5.5,
        markerfacecolor=_PW_FILL,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=f"Search evaluations ({len(plotted_p)})",
        zorder=4,
    )
    ax.axvline(
        self.p_hat,
        linestyle=":",
        color=_REF_COLOR,
        linewidth=_REF_LW,
        label=rf"$\hat{{p}}$ = {self.p_hat:.3f}",
    )
    ax.set_xlabel("p")
    ax.set_ylabel(
        "Profile deviance" if self.phi_method == "mle" else "Profile objective difference"
    )
    ax.set_title("Tweedie p profile search trace")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper right")
    return fig
```

- [ ] **Step 2: Run the trace tests and verify GREEN**

Run:

```bash
rtk proxy uv run pytest tests/test_profile_ci.py -q -k "trace_plot"
```

Expected: all selected trace-plot tests pass.

- [ ] **Step 3: Run the complete profile-CI test module**

Run:

```bash
rtk proxy uv run pytest tests/test_profile_ci.py -q
```

Expected: 73 existing tests plus the new trace-plot tests pass.

- [ ] **Step 4: Commit the implementation**

```bash
rtk git add src/superglm/profiling/tweedie.py
rtk git commit -m "Add cached Tweedie trace plot"
```

### Task 3: Document cost semantics and verify the branch

**Files:**
- Modify: `docs/guide/families.md`

- [ ] **Step 1: Document the cheap trace plot and expensive dense profile plot**

Replace the Tweedie profile-plot example with:

````markdown
### Search trace and profile plots

```python
result.trace_plot()    # cached search evaluations; performs no new fits
result.profile_plot()  # dense profile curve; performs additional fixed-p fits
```

`trace_plot()` is the quick diagnostic for the evaluations already performed by
`estimate_p()`. It sorts and connects only those cached points. `profile_plot()` evaluates a
dense grid of additional *p* values and can therefore be substantially more expensive when
`phi_method="mle"`.
````

- [ ] **Step 2: Run formatting and static checks**

Run:

```bash
rtk proxy uv run ruff format --check src/superglm/profiling/tweedie.py tests/test_profile_ci.py
rtk proxy uv run ruff check src/superglm/profiling/tweedie.py tests/test_profile_ci.py
rtk git diff --check
```

Expected: every command exits successfully with no formatting, lint, or whitespace errors.

- [ ] **Step 3: Run focused regression suites**

Run:

```bash
rtk proxy uv run pytest tests/test_profile_ci.py tests/test_tweedie_profile.py -q
```

Expected: all profile-CI and Tweedie-profile tests pass.

- [ ] **Step 4: Commit documentation**

```bash
rtk git add docs/guide/families.md
rtk git commit -m "Document Tweedie trace plot costs"
```

- [ ] **Step 5: Request independent code review and publish a draft PR**

Run a specification review and a code-quality review against `origin/master`, address any
blocking findings, rerun focused verification, then push `feat/tweedie-trace-plot` and open a
draft PR describing the zero-evaluation guarantee and validation commands.
