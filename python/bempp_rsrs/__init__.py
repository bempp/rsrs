"""Python interface to sample-fed RSRS factorisation."""

from .config import RsrsOptions, ThreadOptions, TreeOptions
from .cache import load_sample_bundle
from .factors import RsrsFactors, RsrsLinearOperator
from .samples import SampleBundle, SampleConvention
from .session import (
    FactorizationStatus,
    InsufficientSamplesError,
    NeedsSamples,
    NativeSession,
    RsrsSession,
)

try:
    from ._bempp_rsrs import RsrsSessionC64, RsrsSessionF64
except ImportError:  # Native extension is optional while developing pure Python tools.
    RsrsSessionC64 = None
    RsrsSessionF64 = None

__all__ = [
    "FactorizationStatus",
    "InsufficientSamplesError",
    "NativeSession",
    "NeedsSamples",
    "RsrsLinearOperator",
    "RsrsFactors",
    "RsrsOptions",
    "RsrsSessionF64",
    "RsrsSessionC64",
    "RsrsSession",
    "SampleBundle",
    "SampleConvention",
    "ThreadOptions",
    "TreeOptions",
    "load_sample_bundle",
]
