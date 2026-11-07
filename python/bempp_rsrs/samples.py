"""Validated sample arrays at the Python/native boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SampleConvention(str, Enum):
    TRANSPOSE = "transpose"


def _matrix(name: str, value: ArrayLike) -> NDArray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have numeric dtype")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class SampleBundle:
    y_test: ArrayLike
    y_sketch: ArrayLike
    z_test: ArrayLike | None = None
    z_sketch: ArrayLike | None = None
    z_convention: SampleConvention = SampleConvention.TRANSPOSE

    def __post_init__(self) -> None:
        object.__setattr__(self, "y_test", _matrix("y_test", self.y_test))
        object.__setattr__(self, "y_sketch", _matrix("y_sketch", self.y_sketch))
        if self.z_test is not None:
            object.__setattr__(self, "z_test", _matrix("z_test", self.z_test))
        if self.z_sketch is not None:
            object.__setattr__(self, "z_sketch", _matrix("z_sketch", self.z_sketch))
        self.validate()

    @property
    def sample_count(self) -> int:
        return int(self.y_test.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.y_test.shape[1])

    def validate(self, *, require_z: bool = False) -> None:
        if self.y_test.shape != self.y_sketch.shape:
            raise ValueError("y_test and y_sketch shapes differ")
        if self.y_test.dtype != self.y_sketch.dtype:
            raise TypeError("y_test and y_sketch dtypes differ")
        if (self.z_test is None) != (self.z_sketch is None):
            raise ValueError("z_test and z_sketch must both be present or absent")
        if require_z and self.z_test is None:
            raise ValueError("nonsymmetric factorisation requires z samples")
        if self.z_test is not None:
            if self.z_test.shape != self.y_test.shape or self.z_sketch.shape != self.y_test.shape:
                raise ValueError("y and z families must have identical shapes")
            if self.z_test.dtype != self.y_test.dtype or self.z_sketch.dtype != self.y_test.dtype:
                raise TypeError("all sample arrays must have identical dtypes")

