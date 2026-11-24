# RSRS

This repository implements dense strong recursive skeletonisation (SRS) and
sample-driven randomised strong recursive skeletonisation (RSRS). The same
factor object supports forward, transpose, adjoint, conjugate, and inverse
applications.

## Workspace

The project is a virtual Cargo workspace:

```text
crates/rsrs/          public Rust facade
crates/rsrs-core/     numerical factors, extraction, scheduling, and reports
crates/rsrs-octree/   Bempp octree planning and sample budgets
crates/rsrs-samples/  portable seeded/HDF5 sample-cache format
bindings/python/      PyO3 extension
python/bempp_rsrs/    Python configuration and session interface
```

The former monolithic implementation is retained in Git history and is not
compiled as a second backend. `rsrs-core` is the single production
implementation.

## Numerical choices

The configuration exposes:

- tolerance-selected or fixed-rank compression;
- RRQR and complex-capable strong RRQR;
- projection, QR, or SVD nullification;
- LU, Cholesky, or SVD least-squares solves, with optional zero,
  scaled, or historical regularisation;
- direct-inverse, LU, and LU-hybrid pivots;
- real symmetric, complex symmetric, Hermitian, and general matrices;
- per-level and constant sample budgets;
- independent Rayon and BLAS thread counts.

Projection intentionally keeps every projected sample row. Historical RSRS
discarded all but `sample_rows - near_columns`; that truncation is not part of
the production algorithm.

## Samples

Rust and Python use the same counter-based Gaussian generator. Real operators
use real standard normals; complex operators use circular complex standard
normals. A cache may contain explicit legacy tests or store only sketches plus
the seed metadata required to reproduce tests exactly.

The HDF5 provider reads only the prefix requested by the Rust factorisation.
It also validates metadata and can perform a quick first/last-block sketch
check. The `hdf5-store` feature requires HDF5 1.x; it is optional so the core
workspace remains buildable on systems that only provide HDF5 2.x.

## Local development

The current development checkout uses the sibling RLST compatibility branch:

```text
WIP/
  rlst-comparison/current-latest/rlst
  rsrs
```

Build and test with:

```bash
cargo check --workspace
OPENBLAS_NUM_THREADS=1 cargo test --workspace
```

To install the Python interface in an activated environment:

```bash
cd bindings/python
maturin develop --release
```

Enable native HDF5 cache loading on an HDF5 1.x system with:

```bash
maturin develop --release --features hdf5-store
```

The Python factorisation boundary is restart-based. If Rust reports that the
cache is too short, the attempt stops with the exact Y/Z deficit. The user may
then extend the cache and explicitly start a new factorisation; RSRS never
enters an implicit sampling loop.

## Reports and tests

Factorisation reports include total and per-level time, ranks, active degrees
of freedom, maximum sample rows used, sample-buffer bytes, and retained ID,
elimination, and diagonal-factor storage. Regression coverage includes all
four real/complex symmetry classes, transpose and adjoint caches, extraction
and pivot choices, sample modes, adaptive octrees, dense SRS, and sampled RSRS.
