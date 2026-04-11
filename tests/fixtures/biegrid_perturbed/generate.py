#!/usr/bin/env python3
"""Generate perturbed BIEGrid fixtures for tests/rsrs_operator_mv.rs."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"RSRS_BIEGRID_FIXTURE_V1"


def rel_fro_defect(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1.0e-14))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-per-axis", type=int, default=22)
    parser.add_argument("--scale", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rsrs-exps", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    rsrs_exps = args.rsrs_exps or (script_dir.parents[3] / "rsrs-exps")
    out = args.out or (script_dir / "cells22_scale1e-2_seed12345.bin")

    sys.path.insert(0, str(rsrs_exps.resolve()))
    from python.bie_grid import (  # pylint: disable=import-outside-toplevel
        BIEGrid,
        build_rank_one_box_perturbations,
        build_real_rank_one_box_perturbations,
        make_complex_wrapped_operator,
        make_real_wrapped_operator,
    )

    ndim = 2
    experiment = BIEGrid(args.cells_per_axis**ndim, ndim)
    n = experiment.N
    raw_points = np.asarray(experiment.XX, dtype=np.float64)
    if raw_points.shape[1] == 2:
        points = np.column_stack([raw_points, np.zeros(raw_points.shape[0], dtype=np.float64)])
    else:
        points = raw_points

    real_perturbation = build_real_rank_one_box_perturbations(
        n, ndim, scale=args.scale, seed=args.seed
    )
    complex_perturbation = build_rank_one_box_perturbations(
        n, ndim, scale=args.scale, seed=args.seed
    )

    base_op = experiment.fast_apply_op
    identity_real = np.eye(n, dtype=np.float64)
    identity_complex = np.eye(n, dtype=np.complex128)

    real_symmetric_op = make_real_wrapped_operator(
        base_op, n, perturbation=real_perturbation, symmetry_mode="real_symmetric"
    )
    real_nonsymmetric_op = make_real_wrapped_operator(
        base_op, n, perturbation=real_perturbation, symmetry_mode="none"
    )
    complex_symmetric_op, _ = make_complex_wrapped_operator(
        base_op, n, perturbation=complex_perturbation, symmetry_mode="complex_symmetric"
    )
    complex_nonsymmetric_op, _ = make_complex_wrapped_operator(
        base_op, n, perturbation=complex_perturbation, symmetry_mode="none"
    )

    real_symmetric = np.ascontiguousarray(
        real_symmetric_op.matmat(identity_real), dtype=np.float64
    )
    real_nonsymmetric = np.ascontiguousarray(
        real_nonsymmetric_op.matmat(identity_real), dtype=np.float64
    )
    complex_symmetric = np.ascontiguousarray(
        complex_symmetric_op.matmat(identity_complex), dtype=np.complex128
    )
    complex_nonsymmetric = np.ascontiguousarray(
        complex_nonsymmetric_op.matmat(identity_complex), dtype=np.complex128
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<QQdQ", n, args.cells_per_axis, args.scale, args.seed))
        handle.write(np.ascontiguousarray(points, dtype=np.float64).tobytes(order="C"))
        handle.write(real_symmetric.tobytes(order="C"))
        handle.write(real_nonsymmetric.tobytes(order="C"))
        handle.write(complex_symmetric.tobytes(order="C"))
        handle.write(complex_nonsymmetric.tobytes(order="C"))

    print(f"wrote {out} ({out.stat().st_size} bytes, n={n})")
    print(
        "transpose defects: "
        f"real_sym={rel_fro_defect(real_symmetric, real_symmetric.T):.3e}, "
        f"real_nosymm={rel_fro_defect(real_nonsymmetric, real_nonsymmetric.T):.3e}, "
        f"complex_sym={rel_fro_defect(complex_symmetric, complex_symmetric.T):.3e}, "
        f"complex_nosymm={rel_fro_defect(complex_nonsymmetric, complex_nonsymmetric.T):.3e}"
    )


if __name__ == "__main__":
    main()
