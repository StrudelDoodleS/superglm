# Editor Refit Timing And Debugging

## Goal

Make structural editor refits feel controlled on slower machines by showing that the editor is
busy, timing the backend refit work, and timing the browser recovery work after the fit returns.

## Scope

This applies to structural level operations:

- collapse selected categorical or ordered-categorical levels
- ungroup selected levels
- uncollapse the last collapsed-level refit

Fixed-offset refits and distribution profiling already have separate visible busy states. They can
reuse the same timing shape later, but they are not changed by this feature.

## Timing Payload

Structural refit responses include a `timing` object:

```json
{
  "operation": "collapse_levels",
  "fit_ms": 1234.5,
  "summary_ms": 12.3,
  "server_total_ms": 1280.0
}
```

`fit_ms` measures the synchronous model replacement/refit call in Python. For uncollapse it measures
the model restore path, which is intentionally reported in the same field so the browser can display
one simple timing shape.

`summary_ms` measures compact summary generation after the model state has changed.

`server_total_ms` measures the whole Python route-side operation from entry to summary payload.

## Browser Timing

The browser wraps each structural refit and adds client-side timings:

- `client_request_ms`: time spent waiting for the refit request to return
- `client_recovery_ms`: time spent fetching state, redrawing, refreshing metrics, and refreshing the
  active report after the refit response
- `client_total_ms`: total visible busy time for the structural operation

The app displays these values in the model-summary note after the operation finishes.

## Busy State

During a structural refit the editor shows an app-level overlay with a spinner and elapsed timer.
The overlay blocks pointer interactions with the chart and toolbar until the post-refit state,
metrics, and active report are refreshed. This avoids the impression that the editor has frozen or
that a second edit can be safely queued while the model is being rebuilt.

## Performance Policy

This remains a synchronous request first. That keeps the code small and avoids building a generic
job system before we know it is needed. If real usage shows structural refits repeatedly exceeding
about five to eight seconds, the next step is to move these operations to the existing profile-job
pattern: start endpoint, poll endpoint, live phase updates, and final payload.
