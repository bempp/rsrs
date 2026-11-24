"""Sample-fed RSRS factors with independently scoped BLAS threads."""

from __future__ import annotations

import json
from typing import Any

from threadpoolctl import threadpool_limits

from ._bempp_rsrs import (
    RsrsFactorsC64 as _NativeRsrsFactorsC64,
    RsrsFactorsF64 as _NativeRsrsFactorsF64,
    RsrsSessionC64 as _NativeRsrsSessionC64,
    RsrsSessionF64 as _NativeRsrsSessionF64,
)


class _Factors:
    def __init__(self, native: Any, blas_threads: int):
        self._native = native
        self._blas_threads = max(1, int(blas_threads))

    @property
    def report_json(self) -> str:
        return self._native.report_json

    def matmat(self, values: Any, **kwargs: Any) -> Any:
        # This limit applies only while native RSRS/RLST kernels run. Python
        # target operators and FMMs used before or after this call retain their
        # own OpenMP/BLAS settings.
        with threadpool_limits(limits=self._blas_threads, user_api="blas"):
            return self._native.matmat(values, **kwargs)


class RsrsFactorsF64(_Factors):
    pass


class RsrsFactorsC64(_Factors):
    pass


class _Session:
    _native_type: type
    _factors_type: type[_Factors]

    def __init__(self, points: Any, options_json: str):
        options = json.loads(options_json)
        self._blas_threads = max(
            1, int(options.get("threads", {}).get("blas_threads", 1))
        )
        self._native = self._native_type(points, options_json)

    @property
    def dimension(self) -> int:
        return self._native.dimension

    @property
    def required_sample_rows(self) -> int:
        return self._native.required_sample_rows

    def factorize(self, *args: Any, **kwargs: Any) -> _Factors:
        with threadpool_limits(limits=self._blas_threads, user_api="blas"):
            native = self._native.factorize(*args, **kwargs)
        return self._factors_type(native, self._blas_threads)

    def factorize_sketches(self, *args: Any, **kwargs: Any) -> _Factors:
        with threadpool_limits(limits=self._blas_threads, user_api="blas"):
            native = self._native.factorize_sketches(*args, **kwargs)
        return self._factors_type(native, self._blas_threads)

    def factorize_dense(self, *args: Any, **kwargs: Any) -> _Factors:
        with threadpool_limits(limits=self._blas_threads, user_api="blas"):
            native = self._native.factorize_dense(*args, **kwargs)
        return self._factors_type(native, self._blas_threads)

    def factorize_hdf5(self, *args: Any, **kwargs: Any) -> _Factors:
        with threadpool_limits(limits=self._blas_threads, user_api="blas"):
            native = self._native.factorize_hdf5(*args, **kwargs)
        return self._factors_type(native, self._blas_threads)


class RsrsSessionF64(_Session):
    _native_type = _NativeRsrsSessionF64
    _factors_type = RsrsFactorsF64


class RsrsSessionC64(_Session):
    _native_type = _NativeRsrsSessionC64
    _factors_type = RsrsFactorsC64


__all__ = ["RsrsSessionF64", "RsrsSessionC64", "RsrsFactorsF64", "RsrsFactorsC64"]
