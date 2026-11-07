"""Restart-based Python boundary for sample-fed RSRS factorisation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Protocol

import numpy as np

from .config import RsrsOptions
from .factors import RsrsFactors
from .samples import SampleBundle


@dataclass(frozen=True)
class NeedsSamples:
    y_rows: int
    z_rows: int
    total_required: int
    level: int
    limiting_box: int | None = None


class FactorizationStatus(str, Enum):
    NEEDS_SAMPLES = "needs_samples"
    PROGRESSED = "progressed"
    COMPLETE = "complete"


class InsufficientSamplesError(RuntimeError):
    """Rust determined that the supplied cache cannot finish factorisation.

    The current native factorisation must be discarded. Extend the cache to
    at least ``total_required`` rows and construct a fresh :class:`RsrsSession`.
    """

    def __init__(self, request: NeedsSamples):
        self.request = request
        super().__init__(
            "RSRS cannot continue with the supplied cache: "
            f"need {request.y_rows} additional Y rows and "
            f"{request.z_rows} additional Z rows "
            f"({request.total_required} total rows required)"
        )


class NativeSession(Protocol):
    """Contract implemented by the forthcoming PyO3 session object."""

    def advance(self) -> tuple[FactorizationStatus, NeedsSamples | None]: ...

    def add_samples(self, samples: SampleBundle) -> None: ...


def _native_options(options: RsrsOptions) -> dict[str, Any]:
    """Translate the versioned Python configuration to the Rust wire schema."""

    values = options.to_mapping()
    values.pop("schema_version")
    tree = values.pop("tree")
    threads = values.pop("threads")
    values["num_threads"] = threads["factorization_threads"]
    values["apply_threads"] = threads["apply_threads"]
    # Tree geometry belongs to session construction, not RsrsArgs.
    values.pop("run_seed", None)
    return values | {"_tree": tree}


class RsrsSession:
    """One disposable attempt at a sample-fed RSRS factorisation.

    Supply the cache once and call :meth:`factorize`. If Rust reports a sample
    deficit, this object is no longer reusable: Python should extend the cache
    and restart with a new session. Python never plans tree levels, ranks, or
    sample counts.
    """

    def __init__(self, points, options: RsrsOptions, *, dtype=np.float64):
        """Create a real or complex sample-fed factorisation session."""
        try:
            from ._bempp_rsrs import RsrsSessionC64, RsrsSessionF64
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise ImportError(
                "The native RSRS extension is not installed. Install "
                "the rsrs/python-native package in this Python environment."
            ) from exc
        coordinates = np.ascontiguousarray(points, dtype=np.float64)
        scalar_dtype = np.dtype(dtype)
        if scalar_dtype == np.dtype(np.float64):
            native_type = RsrsSessionF64
        elif scalar_dtype == np.dtype(np.complex128):
            native_type = RsrsSessionC64
        else:
            raise TypeError(
                "RSRS Python sessions support float64 and complex128, "
                f"not {scalar_dtype}"
            )
        wire = _native_options(options)
        tree = wire.pop("_tree")
        self._native = native_type(
            coordinates,
            json.dumps(wire),
            max_tree_depth=tree["max_tree_depth"],
            leaf_size=tree["leaf_size"],
        )
        self._started = False

    @property
    def dimension(self) -> int:
        return int(self._native.dimension)

    @property
    def requires_z(self) -> bool:
        return bool(self._native.requires_z)

    def add_samples(self, samples: SampleBundle) -> None:
        if self._started:
            raise RuntimeError(
                "this RSRS attempt has already started; create a new session "
                "to use an enlarged sample cache"
            )
        samples.validate(require_z=self.requires_z)
        if samples.dimension != self.dimension:
            raise ValueError(
                f"sample dimension {samples.dimension} does not match {self.dimension}"
            )
        self._native.add_samples(
            samples.y_test,
            samples.y_sketch,
            samples.z_test,
            samples.z_sketch,
        )

    def factorize(self, samples: SampleBundle) -> RsrsFactors:
        """Run until completion or report the exact Rust-computed deficit.

        This method deliberately does not accept additional samples after a
        pause. A deficit invalidates the attempt and callers must restart RSRS
        from the beginning with an enlarged cache.
        """

        if self._started:
            raise RuntimeError("this RSRS factorisation attempt was already used")
        self.add_samples(samples)
        self._started = True
        while True:
            status, request = self.advance()
            if status is FactorizationStatus.COMPLETE:
                report = json.loads(self._native.report_json())
                native_factors = self._native.take_factors()
                return RsrsFactors(native_factors, report=report)
            if status is FactorizationStatus.NEEDS_SAMPLES:
                assert request is not None
                raise InsufficientSamplesError(request)

    def advance(self) -> tuple[FactorizationStatus, NeedsSamples | None]:
        """Low-level Rust step used internally by :meth:`factorize`."""
        result = dict(self._native.advance())
        status = FactorizationStatus(result["status"])
        if status is not FactorizationStatus.NEEDS_SAMPLES:
            return status, None
        return status, NeedsSamples(
            y_rows=int(result["y_rows"]),
            z_rows=int(result["z_rows"]),
            total_required=int(result["total_required"]),
            level=-1 if result["level"] is None else int(result["level"]),
            limiting_box=result.get("limiting_box"),
        )
