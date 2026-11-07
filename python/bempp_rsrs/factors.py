"""SciPy view of the native RSRS factor object."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator


class RsrsFactors:
    """Python owner of completed native RSRS factors.

    ``transpose`` means the algebraic transpose. Combining it with
    ``conjugate`` gives the Hermitian adjoint expected by SciPy.
    """

    def __init__(self, native: Any, *, report: dict[str, Any] | None = None):
        self._native = native
        self._report = dict(report or {})

    @property
    def report(self) -> dict[str, Any]:
        return dict(self._report)

    @property
    def dimension(self) -> int:
        return int(self._native.dimension)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._native.dtype)

    def apply(
        self,
        values,
        *,
        transpose: bool = False,
        conjugate: bool = False,
        inverse: bool = False,
    ):
        array = np.ascontiguousarray(values, dtype=self.dtype)
        if conjugate:
            array = np.ascontiguousarray(np.conjugate(array))
        if array.ndim == 1:
            result = np.asarray(
                self._native.apply_vector(
                    array, transpose=transpose, inverse=inverse
                )
            )
        elif array.ndim == 2:
            result = np.asarray(
                self._native.apply_matrix(
                    array, transpose=transpose, inverse=inverse
                )
            )
        else:
            raise ValueError("factor input must be a vector or matrix")
        return np.conjugate(result) if conjugate else result

    def matvec(self, vector):
        return self.apply(vector)

    def matmat(self, matrix):
        return self.apply(matrix)

    def transpose_matvec(self, vector):
        return self.apply(vector, transpose=True)

    def transpose_matmat(self, matrix):
        return self.apply(matrix, transpose=True)

    def inverse_matvec(self, vector):
        return self.apply(vector, inverse=True)

    def inverse_matmat(self, matrix):
        return self.apply(matrix, inverse=True)

    def inverse_transpose_matvec(self, vector):
        return self.apply(vector, transpose=True, inverse=True)

    def inverse_transpose_matmat(self, matrix):
        return self.apply(matrix, transpose=True, inverse=True)

    def aslinearoperator(self, *, inverse: bool = False) -> "RsrsLinearOperator":
        return RsrsLinearOperator(self, inverse=inverse)


class RsrsLinearOperator(LinearOperator):
    def __init__(self, factors: Any, *, inverse: bool = False):
        self.factors = factors
        self.inverse = inverse
        dimension = int(factors.dimension)
        super().__init__(dtype=np.dtype(factors.dtype), shape=(dimension, dimension))

    def _call(self, values, *, transpose=False, conjugate=False):
        return np.asarray(
            self.factors.apply(
                np.ascontiguousarray(values, dtype=self.dtype),
                transpose=transpose,
                conjugate=conjugate,
                inverse=self.inverse,
            )
        )

    def _matvec(self, vector):
        return self._call(vector)

    def _matmat(self, matrix):
        return self._call(matrix)

    def _rmatvec(self, vector):
        return self._call(vector, transpose=True, conjugate=True)
