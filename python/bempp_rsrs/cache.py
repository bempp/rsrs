"""Load Python-generated or legacy multipart RSRS sample caches."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .samples import SampleBundle


_PART_PATTERN = re.compile(r"^(?P<base>.+)\.(?P<part>[0-9]{5})\.h5$")


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("loading RSRS HDF5 caches requires h5py") from exc
    return h5py


def _parts(directory: Path, base: str) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in directory.glob(f"{base}.*.h5"):
        match = _PART_PATTERN.match(path.name)
        if match is not None and match.group("base") == base:
            indexed.append((int(match.group("part")), path))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"{base} cache has missing or non-contiguous part files")
    return [path for _, path in indexed]


def _stored_shape(file) -> tuple[int, int]:
    shape = tuple(int(value) for value in np.asarray(file.attrs["shape"]).reshape(-1))
    if len(shape) != 2:
        raise ValueError(f"invalid RSRS shape attribute {shape}")
    return shape


def _read_values(file) -> np.ndarray:
    real = np.asarray(file["real"])
    if "imag" not in file:
        return real
    imag = np.asarray(file["imag"])
    if real.shape != imag.shape:
        raise ValueError("cache real and imaginary datasets have different shapes")
    return real + 1j * imag


def load_family(directory: str | Path, base: str) -> np.ndarray:
    """Load one row-oriented multipart sample family.

    RSRS part files split the matrix by coordinate columns. Values within each
    file are stored column-major, so each part is reshaped as
    ``(part_columns, rows).T`` before horizontal concatenation.
    """

    directory = Path(directory)
    h5py = _h5py()
    parts = _parts(directory, base)
    if not parts:
        legacy = directory / base
        if not legacy.exists():
            legacy = directory / f"{base}.h5"
        if not legacy.exists():
            raise FileNotFoundError(f"no cache files found for {base!r} in {directory}")
        with h5py.File(legacy, "r") as file:
            values = _read_values(file).reshape(-1)
            if "shape" in file.attrs:
                rows, columns = _stored_shape(file)
                return np.ascontiguousarray(values.reshape(columns, rows).T)
            raise ValueError(f"legacy cache {legacy} does not record its matrix shape")

    matrices = []
    expected_shape: tuple[int, int] | None = None
    for path in parts:
        with h5py.File(path, "r") as file:
            shape = _stored_shape(file)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"cache part {path} has shape {shape}, expected {expected_shape}")
            rows, _ = shape
            values = _read_values(file).reshape(-1)
            if values.size % rows:
                raise ValueError(f"cache part {path} length is not divisible by row count")
            matrices.append(values.reshape(values.size // rows, rows).T)

    result = np.ascontiguousarray(np.hstack(matrices))
    if result.shape != expected_shape:
        raise ValueError(f"assembled {base} shape {result.shape}, expected {expected_shape}")
    return result


def load_sample_bundle(directory: str | Path, *, require_z: bool = False) -> SampleBundle:
    directory = Path(directory)
    y_test = load_family(directory, "y_test_file")
    y_sketch = load_family(directory, "y_sketch_file")

    z_test_parts = _parts(directory, "z_test_file")
    z_sketch_parts = _parts(directory, "z_sketch_file")
    has_z = bool(z_test_parts or z_sketch_parts)
    if bool(z_test_parts) != bool(z_sketch_parts):
        raise ValueError("cache contains only one member of the z sample family")

    bundle = SampleBundle(
        y_test=y_test,
        y_sketch=y_sketch,
        z_test=load_family(directory, "z_test_file") if has_z else None,
        z_sketch=load_family(directory, "z_sketch_file") if has_z else None,
    )
    bundle.validate(require_z=require_z)
    return bundle

