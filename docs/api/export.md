# Export

Rating-table export for a fitted model.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.export_rating_tables
```

When a base level cannot be represented in the requested block shape, export
raises {py:exc}`superglm.RatingTableBaseNotRepresentableError`, documented with
the rest of the package's errors on
[Warnings and exceptions](warnings-and-exceptions.md).

## Structure files

A structure file holds a fitted model's structural decisions without its
coefficients: groupings, references, where new levels go, and polynomial
ranges. `Structure.apply` builds them into another model, ready to fit.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.Structure
   superglm.read_structure
```
