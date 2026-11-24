# RSRS sample cache format

The canonical schema is versioned by `rsrs_samples::FORMAT_VERSION`. New
sketch-only caches use one `samples.h5` file. The legacy multipart cache remains
readable through the Python and native legacy adapters.

## Root attributes

- `cache_kind = "rsrs_sketch_cache"`
- `format_version` (version 2 for new caches; version 1 remains readable)
- `digest_algorithm = "sha256"`
- `committed_rows`
- `metadata_json`

`metadata_json` records the scalar kind, operator dimension and fingerprint,
Y/Z generator versions and seeds, distributions, row capacity, and whether Z
means transpose or adjoint sampling.

## Datasets

Sketches are appendable, row-oriented datasets:

- `/sketches/y/real`, optionally `/sketches/y/imag`
- `/sketches/z/real`, optionally `/sketches/z/imag`; the entire Z family is
  absent for symmetric or Hermitian Y-only caches

Dense test matrices are deliberately absent. They are regenerated in bounded
row blocks by the canonical Rust generator, which is also exposed to Python.
Real operators use independent real standard normals. Complex operators use
circular complex standard normals `(x + i y) / sqrt(2)`.

Append manifests live at:

- `/manifests/first_row`
- `/manifests/rows`
- `/manifests/sha256` with shape `(batches, 4, 32)`

The four digest slots are Y recipe, Y sketch bytes, Z recipe, and Z sketch
bytes. Y-only caches retain the fixed manifest shape and store zero digests in
the two Z slots.
Recipe digests are independent of floating-point `libm` details. Sketch digests
cover canonical little-endian row-major bytes.

## Transaction and validation rules

An append writes and flushes sketch data, appends and flushes its manifest, and
only then advances `committed_rows`. Readers ignore any tail beyond that value.
Quick validation checks metadata and the first and last committed manifests;
full validation checks every manifest. Numerical operator consistency remains
a separate tolerance-based check of `A Omega_y = Y` and the configured
transpose/adjoint relation for Z.
