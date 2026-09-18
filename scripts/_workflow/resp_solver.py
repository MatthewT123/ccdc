"""Accurate small-system RESP linear solves, retaining PsiRESP's fit objective."""
from contextlib import contextmanager
import warnings
import numpy as np


@contextmanager
def precise_resp_solver():
    # PsiRESP's singular-matrix fallback uses default LSMR tolerances. Redundant
    # equivalence constraints can then leave material charge-constraint residuals.
    from psiresp.constraint import SparseGlobalConstraintMatrix
    original = SparseGlobalConstraintMatrix._solve
    def solve(self):
        self._previous_charges = None if self._charges is None else self._charges.copy()
        if self.coefficient_matrix.shape[0] > 1024:
            raise ValueError('Precise RESP solver limited to small constraint systems (<=1024)')
        matrix = self.coefficient_matrix.toarray()
        q, _, _, _ = np.linalg.lstsq(matrix, self.constant_vector, rcond=1e-14)
        residual = np.max(np.abs(matrix @ q - self.constant_vector))
        if not np.isfinite(q).all() or residual > 1e-8:
            raise ValueError(f'RESP linear-system residual too large: {residual}')
        self._charges = q
    SparseGlobalConstraintMatrix._solve = solve
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Charge fitting did not converge.*')
            yield
    finally:
        SparseGlobalConstraintMatrix._solve = original
