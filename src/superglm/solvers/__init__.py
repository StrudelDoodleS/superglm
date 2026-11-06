import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.solvers.constrained_qp import QPResult as QPResult
    from superglm.solvers.constrained_qp import solve_constrained_qp as solve_constrained_qp
    from superglm.solvers.irls_direct import fit_irls_direct as fit_irls_direct
    from superglm.solvers.pirls import IterationDiagnostics as IterationDiagnostics
    from superglm.solvers.pirls import PIRLSResult as PIRLSResult
    from superglm.solvers.pirls import fit_pirls as fit_pirls

# Each name is read from its module on first access (``superglm._lazy``), so importing the
# package, or one of its modules, loads only what that module imports.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.solvers.constrained_qp": ("QPResult", "solve_constrained_qp"),
    "superglm.solvers.irls_direct": ("fit_irls_direct",),
    "superglm.solvers.pirls": ("IterationDiagnostics", "PIRLSResult", "fit_pirls"),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)
