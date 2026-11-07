# Multi-repository development architecture

RSRS is developed alongside, but independently from, the kernel and experiment
repositories:

```text
WIP/
  rsrs/             # factorisation core and bempp_rsrs Python bindings
  bempp-rs/         # Bempp backend
  kifmm-for-rsrs/   # KiFMM backend
  so-kernels/    # installable Python operator/sampling package
  rsrs-exps/        # experiment definitions, analysis, GMRES, and plots
```

The dependency direction is:

```text
bempp-rs ───────┐
                ├─> so-kernels ──> rsrs-exps
kifmm-for-rsrs ─┘                       ↑
                                        │
rsrs ───────────────────────────────────┘
```

RSRS does not depend on the Python kernel package. Its native Rust API can
sample a Rust operator. Its Python API accepts precomputed arrays or cache
paths and never calls a Python operator from Rust.

The initial Python integration uses restart-on-shortage semantics. Rust owns
all tree traversal, rank selection, sample-budget decisions, and
factorisation. Python supplies the current cache once. If Rust reports an
exact Y/Z deficit, that factorisation attempt ends; Python extends the cache
and starts a new RSRS attempt from the beginning. A live Rust factorisation is
not resumed with newly generated Python samples in this stage.

During local development, consumers use editable Python installations and a
Cargo path dependency on `../rsrs`. Reproducible cluster/release manifests pin
the corresponding Git commits instead of embedding source snapshots.
