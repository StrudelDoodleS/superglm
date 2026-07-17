# Transactional Tweedie Model State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete model configuration during profiling and make successful Tweedie result installation atomic while every failure leaves the caller unchanged.

**Architecture:** Replace parent-pair interaction reconstruction with one faithful cloning primitive shared by profiling and the editor. Run final fitting and synchronization entirely on a staged model, validate it, prepare a complete installable state dictionary, and swap that dictionary into the original object under rollback protection. The original object identity remains stable for callers.

**Tech Stack:** Python `copy`, dataclasses, NumPy, pandas, pytest monkeypatch/state snapshots.

---

## File map

- Modify `src/superglm/model/base.py`: faithful configuration cloning and interaction filtering.
- Modify `src/superglm/profiling/tweedie.py`: use the shared clone without profiling-only repairs.
- Modify `src/superglm/model/profile_ops.py`: staged final fit, validation, transactional state commit.
- Modify `src/superglm/model/api.py`: public atomicity/convergence documentation and eager-CI forwarding added by the consumer plan.
- Modify `src/superglm/editor/session.py`: rely on faithful clone and replace only after success.
- Create `tests/test_model_cloning.py`: custom resolved/pending interaction and option preservation.
- Create `tests/test_tweedie_profile_state.py`: callback/refit/synchronization failure atomicity and successful-state coherence.
- Modify `tests/test_editor.py`: real custom-interaction reprofiling regression.
- Modify `tests/test_tweedie_convergence.py`: final solver/REML convergence gate.
- Modify `tests/test_weighted_forwarding.py`: staging forwards all inputs and solver controls.

### Task 1: Make configuration cloning faithful

**Files:**
- Modify: `src/superglm/model/base.py:535-607`
- Create: `tests/test_model_cloning.py`

- [ ] **Step 1: Add the failing custom tensor clone regression**

```python
def test_clone_preserves_resolved_custom_tensor_configuration(tensor_model, tensor_frame):
    model, X, y = tensor_model, tensor_frame[0], tensor_frame[1]
    model.fit(X, y)
    source = model._interaction_specs["custom_surface"]

    clone = model._clone_without_features(set())
    copied = clone._interaction_specs["custom_surface"]

    assert copied is not source
    assert copied.parent_names == source.parent_names
    assert copied._n_knots == (3, 4)
    assert copied._decompose is True
    assert clone._interaction_order == model._interaction_order
    assert clone._pending_interactions == model._pending_interactions
    assert clone._retain_fit_state is model._retain_fit_state

    clone.fit(X, y)
    assert [(g.feature_name, g.n_cols) for g in clone._groups] == [
        (g.feature_name, g.n_cols) for g in model._groups
    ]
```

Construct `tensor_model` with two spline parents and:

```python
model.add_interaction(
    "x1",
    "x2",
    name="custom_surface",
    kind="tensor",
    n_knots=(3, 4),
    decompose=True,
)
```

- [ ] **Step 2: Verify the current clone loses the interaction**

Run: `rtk pytest tests/test_model_cloning.py -k custom_tensor`

Expected: clone contains default `x1:x2` with a different width instead of `custom_surface`.

- [ ] **Step 3: Deep-copy complete surviving interaction state**

Construct the new model without the `interactions=` parent-pair argument. After creating it with
deep-copied surviving feature specs and all existing solver options, assign:

```python
surviving = {
    name: copy.deepcopy(spec)
    for name, spec in model._interaction_specs.items()
    if not set(spec.parent_names) & drop
}
new_model._interaction_specs = surviving
new_model._interaction_order = [
    name for name in model._interaction_order if name in surviving
]
new_model._pending_interactions = [
    copy.deepcopy(interaction)
    for interaction in model._pending_interactions
    if not set(interaction[:2]) & drop
]
```

Pass `retain_fit_state=model._retain_fit_state` to the constructor. Also preserve `direct_solve`,
`discrete`, bin settings, convergence mode, tolerances, active-set settings, spline shorthand,
categorical defaults, and deep-copied penalties/lambda configuration. Do not copy fitted results,
design matrices, caches, or editor history into a configuration clone.

- [ ] **Step 4: Cover pending interactions and feature drops**

Add tests proving pending custom interactions remain pending with their complete tuple/dict payload,
and interactions disappear only when one parent appears in `drop`. Verify clone mutation does not
mutate source specs.

- [ ] **Step 5: Run and commit**

Run: `rtk pytest tests/test_model_cloning.py tests/test_interactions.py tests/test_drop1_weights.py`

Expected: clone, interaction, and drop-one suites pass.

```bash
rtk git add src/superglm/model/base.py tests/test_model_cloning.py
rtk git commit -m "Preserve model interaction configuration"
```

### Task 2: Remove profiling-only clone repairs and prove structural parity

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:3401-3417`
- Modify: `tests/test_model_cloning.py`
- Modify: `tests/test_tweedie_profile.py`

- [ ] **Step 1: Add a failing profile clone parity test**

```python
def test_profile_clone_matches_source_design_groups(tensor_model, tensor_frame):
    model, X, y = tensor_model, tensor_frame[0], tensor_frame[1]
    model.fit(X, y)
    clone = _clone_profile_model(model, X, np.ones(len(X)))
    clone.fit(X, y)
    assert clone._interaction_order == ["custom_surface"]
    assert [(g.feature_name, g.n_cols) for g in clone._groups] == [
        (g.feature_name, g.n_cols) for g in model._groups
    ]
```

- [ ] **Step 2: Simplify `_clone_profile_model`**

Delete the manual assignments to `_interaction_specs`, `_interaction_order`, and
`_pending_interactions`. Retain only the shared configuration clone and unresolved shorthand
resolution on the scratch clone. Assert in development that source and clone do not share mutable
spec/penalty objects.

- [ ] **Step 3: Run and commit**

Run: `rtk pytest tests/test_model_cloning.py tests/test_tweedie_profile.py -k 'clone or caller or interaction'`

Expected: profiling and generic clone parity tests pass.

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_model_cloning.py tests/test_tweedie_profile.py
rtk git commit -m "Share faithful Tweedie profile cloning"
```

### Task 3: Introduce final-state certification on a staged model

**Files:**
- Modify: `src/superglm/model/profile_ops.py:16-154`
- Create: `tests/test_tweedie_profile_state.py`
- Modify: `tests/test_tweedie_convergence.py`

- [ ] **Step 1: Add failing final-fit certification tests**

```python
@pytest.mark.parametrize("attribute", ["converged", "solver_converged", "phi_converged"])
def test_uncertified_profile_is_rejected_before_staging(
    fitted_tweedie_model, deterministic_profile, monkeypatch, attribute
):
    model, X, y = fitted_tweedie_model
    setattr(deterministic_profile, attribute, False)
    before = pickle.dumps(model.__dict__, protocol=5)
    monkeypatch.setattr(profile_ops, "estimate_tweedie_p", lambda *args, **kwargs: deterministic_profile)
    with pytest.raises(TweedieProfileError, match="not installable"):
        model.estimate_p(X, y)
    assert pickle.dumps(model.__dict__, protocol=5) == before


def test_nonconverged_staged_final_fit_is_not_installed(
    fitted_tweedie_model, deterministic_profile, monkeypatch
):
    model, X, y = fitted_tweedie_model
    before = _observable_model_snapshot(model, X)
    monkeypatch.setattr(profile_ops, "estimate_tweedie_p", lambda *args, **kwargs: deterministic_profile)
    monkeypatch.setattr(profile_ops, "_staged_fit_converged", lambda staged: False)
    with pytest.raises(TweedieProfileError, match="final fit did not converge"):
        model.estimate_p(X, y)
    _assert_observable_model_snapshot(model, before, X)
```

- [ ] **Step 2: Confirm the current wrapper installs invalid results**

Run: `rtk pytest tests/test_tweedie_profile_state.py -k 'uncertified or nonconverged'`

Expected: family power and/or final fitted state changes despite the failure flags.

- [ ] **Step 3: Split synchronization into mutation and validation phases**

Keep `_synchronize_tweedie_profile_refit(staged, y, result)` operating only on the staged model.
Add `_validate_staged_tweedie_profile(staged, X, offset, result)` that raises
`TweedieProfileError` unless all of these hold:

```python
checks = {
    "profile aggregate convergence": result.converged,
    "profile exact density": result.density_exact,
    "profile density certification": result.density_certified,
    "profile objective finite": result.objective_finite,
    "profile dispersion convergence": result.phi_converged,
    "final public solver convergence": staged.result.converged,
    "family/distribution power agreement": staged.family.p == staged._distribution.p == result.p_hat,
    "public dispersion agreement": staged.result.phi == result.phi_hat,
    "solver dispersion agreement": staged._solver_pirls_result().phi == result.phi_hat,
}
```

For REML, also require `_reml_result.converged`. Recompute predictions for `X` and require they are
finite. Require synchronized fit statistics and covariance/inference caches to exist before release.

- [ ] **Step 4: Add staged-fit helper**

```python
def _fit_staged_tweedie_model(model, X, y, sample_weight, offset, result, resolved_mode):
    staged = model._clone_without_features(set(), lambda2=copy.deepcopy(model.lambda2))
    staged.family = Tweedie(result.p_hat)
    retain = staged._retain_fit_state
    staged._retain_fit_state = True
    if resolved_mode == "fit_reml":
        staged.fit_reml(X, y, sample_weight=sample_weight, offset=offset)
    else:
        staged.fit(X, y, sample_weight=sample_weight, offset=offset)
    _synchronize_tweedie_profile_refit(staged, y, result)
    _validate_staged_tweedie_profile(staged, X, offset, result)
    staged._retain_fit_state = retain
    if not retain:
        fit_ops._maybe_release_fit_state(staged)
    return staged
```

No caller reference is passed into any mutating helper.

- [ ] **Step 5: Run and commit**

Run: `rtk pytest tests/test_tweedie_profile_state.py tests/test_tweedie_convergence.py -k 'staged or convergence or installable'`

Expected: staged certification tests pass.

```bash
rtk git add src/superglm/model/profile_ops.py tests/test_tweedie_profile_state.py tests/test_tweedie_convergence.py
rtk git commit -m "Certify staged Tweedie refits"
```

### Task 4: Make public installation atomic under every failure

**Files:**
- Modify: `src/superglm/model/profile_ops.py:16-72`
- Modify: `tests/test_tweedie_profile_state.py`
- Modify: `tests/test_tweedie_profile.py:2355-2599`

- [ ] **Step 1: Add callback/refit/synchronization failure snapshots**

Create `_observable_model_snapshot` that captures `pickle.dumps(model.__dict__)`, top-level field
identities, predictions, summary text, family power, distribution power, and the prior profile
result. Parameterize failures at `best_found`, `final_refit`, staged `fit`, synchronization, and
validation. Each test asserts byte/identity/prediction equality after the exception.

```python
@pytest.mark.parametrize("event", ["best_found", "final_refit"])
def test_progress_callback_failure_is_atomic(fitted_tweedie_model, event):
    model, X, y = fitted_tweedie_model
    before = _observable_model_snapshot(model, X)

    def callback(name, payload):
        payload["profile_estimate"].clear()
        if name == event:
            raise RuntimeError(f"failed at {event}")

    with pytest.raises(RuntimeError, match=event):
        model.estimate_p(X, y, progress_callback=callback)
    _assert_observable_model_snapshot(model, before, X)
```

- [ ] **Step 2: Confirm current callback failure mutates family state**

Run: `rtk pytest tests/test_tweedie_profile_state.py -k progress_callback_failure`

Expected: the `final_refit` case leaves `family.p` changed.

- [ ] **Step 3: Implement a rollback-protected state swap**

```python
def _commit_tweedie_profile_state(model, staged, result):
    original_state = model.__dict__
    installed_state = staged.__dict__.copy()
    installed_state["_tweedie_profile_result"] = None
    try:
        model.__dict__ = installed_state
        model._tweedie_profile_result = result
    except BaseException:
        model.__dict__ = original_state
        raise
```

Rewrite `estimate_p` in this order: low-level profile; validate installability; copied `best_found`
callback; copied `final_refit` callback; stage/final-fit/synchronize/validate; detach result state as
specified by the consumer plan; atomic commit. Never set `model.family` before the final commit.

- [ ] **Step 4: Test success coherence**

On a successful fit assert family/distribution/result powers, public/solver dispersions, deviance,
log likelihood, predictions, covariance state, metrics, summary, and profile result all originate
from the staged final fit. Assert the original Python model object identity remains unchanged.

- [ ] **Step 5: Run and commit**

Run: `rtk pytest tests/test_tweedie_profile_state.py tests/test_tweedie_profile.py -k 'final_refit or progress or synchronize or atomic or install'`

Expected: all failure paths preserve state and success paths are coherent.

```bash
rtk git add src/superglm/model/profile_ops.py tests/test_tweedie_profile_state.py tests/test_tweedie_profile.py
rtk git commit -m "Install Tweedie fits atomically"
```

### Task 5: Prove complete input and solver-control forwarding

**Files:**
- Modify: `tests/test_weighted_forwarding.py`
- Modify: `tests/test_tweedie_profile_state.py`

- [ ] **Step 1: Extend spies to the staged final fit**

Assert copied input values and object ownership for `X`, response, prior weights, offset, lambda
configuration, direct solver, active-set, convergence mode, tolerances, maximum iterations,
discrete settings, and REML mode. Verify offset appears exactly once in reconstructed means and
predictions.

- [ ] **Step 2: Run forwarding tests**

Run: `rtk pytest tests/test_weighted_forwarding.py tests/test_tweedie_profile_state.py -k 'forward or offset or solver or reml'`

Expected: all profile and staged-fit controls match the public request.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/test_weighted_forwarding.py tests/test_tweedie_profile_state.py
rtk git commit -m "Cover staged Tweedie fit forwarding"
```

### Task 6: Repair editor reprofiling with custom interactions

**Files:**
- Modify: `src/superglm/editor/session.py:735-798`
- Modify: `tests/test_editor.py:2644-2799`

- [ ] **Step 1: Add the real custom-interaction editor regression**

```python
def test_editor_reprofile_preserves_custom_tensor_model(tweedie_tensor_editor_fixture):
    session, X, y = tweedie_tensor_editor_fixture
    source = session.model
    source_groups = [(group.feature_name, group.n_cols) for group in source._groups]

    result = session.reprofile_distribution(
        "tweedie_p", X=X, y=y, method="grid", grid=np.array([1.4, 1.5, 1.6])
    )

    assert result.converged
    assert session.model._interaction_order == ["custom_surface"]
    assert session.model._interaction_specs["custom_surface"]._n_knots == (3, 4)
    assert session.model._interaction_specs["custom_surface"]._decompose is True
    assert [(group.feature_name, group.n_cols) for group in session.model._groups] == source_groups
```

- [ ] **Step 2: Verify current editor clone degrades the interaction**

Run: `rtk pytest tests/test_editor.py -k reprofile_preserves_custom_tensor_model`

Expected: custom name and decomposed group widths are replaced by a default interaction.

- [ ] **Step 3: Replace only after a certified public success**

Keep editor reprofiling on a clone, but rely exclusively on the fixed shared clone. Call
`replace_in_force_model` only after `estimate_p` returns an installable result and the clone's final
state passes the same coherence checks. On any exception preserve the source model and collapse
history.

- [ ] **Step 4: Run editor and model-state gates**

Run: `rtk pytest tests/test_editor.py -k 'reprofile_distribution or custom_tensor'`

Run: `rtk pytest tests/test_model_cloning.py tests/test_tweedie_profile_state.py tests/test_tweedie_convergence.py tests/test_weighted_forwarding.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/editor/session.py tests/test_editor.py
rtk git commit -m "Preserve editor Tweedie interactions"
```
