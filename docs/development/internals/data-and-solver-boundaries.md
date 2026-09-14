# Data and Solver Boundaries

SuperGLM has two useful levels of explanation. The user layer is intentionally
small; the developer layer shows where named data becomes numerical solver
state. Keeping these layers separate makes it possible to add dataframe support
without putting dataframe dispatch inside performance-sensitive algorithms.

## User layer

```text
pandas.DataFrame | eager polars.DataFrame
                   -> fit / REML / profile / predict / metrics / CV
```

The model accepts either native eager frame. A Polars frame is not converted as
a whole to pandas. A caller must collect a `polars.LazyFrame` before using it.
Existing table-oriented results—summaries, inference tables, diagnostics, plot
data, and rating-table payloads—remain pandas objects.

## Developer layer

```text
native frame
    -> EagerFrame boundary
    -> feature compiler
    -> DesignMatrix + GroupMatrix blocks
    -> construction-time execution plans
    -> IRLS / PIRLS / REML
```

`superglm._frame.EagerFrame` is operation-local. It supplies column names,
schema classification, cached one-column arrays, native row selection, and fit
fingerprints. It is not stored in a `DesignMatrix` and never enters IRLS,
PIRLS, REML, line search, Gram, matvec, or centering kernels.

The feature compiler turns equivalent pandas and Polars columns into the same
feature specs, penalties, solver slices, and `GroupMatrix` classes. From that
point onward, backend choice is over. Dense arrays, sparse bases, categorical
codes, compressed spline support, and tensor grids are the numerical contract.
Construction-time execution plans then select specialised kernels, eligible
Tabmat partitions, or stable fallbacks.

## Where to make a change

| Change | Owner |
| --- | --- |
| Accept/normalize a dataframe dtype | `superglm._frame` |
| Change a basis or categorical encoding | feature spec and `dm_builder` |
| Change block storage | `GroupMatrix` construction |
| Change Gram/matvec/centering | group-matrix algebra/execution plan |
| Change working responses or weights | working-row geometry |
| Change coefficient iteration or line search | IRLS/PIRLS solver |
| Change smoothing selection | REML objective/update/finalization |
| Add a future AFT objective | response/objective/working-geometry contract, then only required kernels |

This is the shortest route to the owning abstraction. For example, changing
how a weighted Gram is assembled normally starts in the group-matrix execution
plan and algebra—not in pandas handling, feature syntax, or REML. Changing the
working response for a family starts in working-row geometry, before deciding
whether any specialised matrix kernel is actually required.

## Inspecting a fitted design

`design_summary()` creates a new pandas table from fitted, immutable metadata.
It reports each solver span, physical representation, support-row count,
compressed status, ordinary Tabmat partition membership, specialised discrete
route, and the construction-time eligibility reason.

With pandas input:

```python
import pandas as pd

from superglm import Categorical, Numeric, SuperGLM

X = pd.DataFrame({"age": [20.0, 30.0, 40.0], "region": ["N", "S", "N"]})
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features={"age": Numeric(), "region": Categorical(base="first")},
).fit(X, [0.0, 1.0, 2.0])

print(model.design_summary())
```

With Polars input, only frame construction changes:

```python
import polars as pl

X = pl.DataFrame({"age": [20.0, 30.0, 40.0], "region": ["N", "S", "N"]})
model.fit(X, [0.0, 1.0, 2.0])
print(model.design_summary())  # still a pandas DataFrame
```

The summary reports immutable storage and construction-time eligibility. It
does not build a SplitMatrix, run a kernel, or prove that an eligible route was
used. Fit, REML, and profile traces—and benchmark kernel call counters—remain
authoritative for actual execution, fallbacks, certification, convergence, and
line-search decisions. With `retain_fit_state=False`, the fitted design is
deliberately discarded, so no design summary is available.

## Conceptual path for a future AFT objective

An accelerated-failure-time objective would cross the developer layer in a
deliberate sequence:

1. Validate duration, event, and censoring inputs and define their ownership.
2. Define the objective value and its gradient/curvature, or an equivalent
   working-response and working-weight geometry.
3. Connect that geometry to a compatible coefficient solver and its
   convergence, line-search, and rollback contracts.
4. Add only the specialised matrix kernels demonstrated to be necessary; reuse
   existing `DesignMatrix` operations where they already match the algebra.
5. Recompute and publish the terminal objective and fitted state atomically.

This branch adds no AFT API and no generic objective framework. The sequence is
a map for future work, not an implementation commitment; the likelihood,
censoring semantics, and numerical certificates would need their own design and
theory review first.
