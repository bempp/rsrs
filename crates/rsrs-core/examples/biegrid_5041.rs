extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

use rlst::dense::array::DynArray;
use rsrs_core::{
    dense_multilevel::{DenseCyclePlan, DenseMultilevelFactorization},
    elementary_factor::FactorMode,
    factorized_operator::DenseFactorizedOperator,
    interpolative_decomposition::IdMethod,
};
use std::{collections::HashSet, time::Instant};

const CELLS: usize = 70;
const SIDE: usize = CELLS + 1;
const DIM: usize = SIDE * SIDE;

fn matrix() -> DynArray<f64, 2> {
    let h = 1.0 / CELLS as f64;
    let q = 256usize;
    let mut self_integral = 0.0;
    for jy in 0..q {
        for ix in 0..q {
            let x = ((ix as f64 + 0.5) / q as f64) * h * 0.5;
            let y = ((jy as f64 + 0.5) / q as f64) * h * 0.5;
            self_integral += -(x.hypot(y)).ln() / (2.0 * std::f64::consts::PI);
        }
    }
    let diagonal = 4.0 * self_integral * (h * 0.5 / q as f64).powi(2);
    let mut result = DynArray::<f64, 2>::from_shape([DIM, DIM]);
    for row in 0..DIM {
        let (ry, rx) = (row / SIDE, row % SIDE);
        for column in 0..DIM {
            if row == column {
                result[[row, column]] = diagonal;
            } else {
                let (cy, cx) = (column / SIDE, column % SIDE);
                let dx = (rx as f64 - cx as f64) * h;
                let dy = (ry as f64 - cy as f64) * h;
                result[[row, column]] =
                    -(dx.hypot(dy)).ln() / (2.0 * std::f64::consts::PI * DIM as f64);
            }
        }
    }
    result
}

fn level(boxes_per_axis: usize, rank: usize) -> Vec<DenseCyclePlan> {
    let mut boxes = vec![Vec::new(); boxes_per_axis * boxes_per_axis];
    for iy in 0..SIDE {
        for ix in 0..SIDE {
            let bx = (ix * boxes_per_axis / SIDE).min(boxes_per_axis - 1);
            let by = (iy * boxes_per_axis / SIDE).min(boxes_per_axis - 1);
            boxes[by * boxes_per_axis + bx].push(iy * SIDE + ix);
        }
    }
    let all = (0..DIM).collect::<Vec<_>>();
    let mut plans = Vec::new();
    for by in 0..boxes_per_axis {
        for bx in 0..boxes_per_axis {
            let target = boxes[by * boxes_per_axis + bx].clone();
            if target.len() <= rank {
                continue;
            }
            let mut near = Vec::new();
            for ny in by.saturating_sub(1)..=(by + 1).min(boxes_per_axis - 1) {
                for nx in bx.saturating_sub(1)..=(bx + 1).min(boxes_per_axis - 1) {
                    near.extend_from_slice(&boxes[ny * boxes_per_axis + nx]);
                }
            }
            near.sort_unstable();
            near.dedup();
            let near_set = near.iter().copied().collect::<HashSet<_>>();
            let far = all
                .iter()
                .copied()
                .filter(|index| !near_set.contains(index))
                .collect();
            plans.push(DenseCyclePlan {
                target_indices: target,
                near_indices: near,
                far_indices: far,
                rank,
            });
        }
    }
    plans
}

fn dense_apply(matrix: &DynArray<f64, 2>, input: &DynArray<f64, 2>) -> DynArray<f64, 2> {
    let mut result = DynArray::<f64, 2>::from_shape([DIM, input.shape()[1]]);
    for row in 0..DIM {
        for column in 0..input.shape()[1] {
            let mut value = 0.0;
            for inner in 0..DIM {
                value += matrix[[row, inner]] * input[[inner, column]];
            }
            result[[row, column]] = value;
        }
    }
    result
}

fn relative_l2(actual: &DynArray<f64, 2>, expected: &DynArray<f64, 2>) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in 0..actual.shape()[0] {
        for column in 0..actual.shape()[1] {
            numerator += (actual[[row, column]] - expected[[row, column]]).powi(2);
            denominator += expected[[row, column]].powi(2);
        }
    }
    (numerator / denominator).sqrt()
}

fn main() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    println!("baseline=refactored_dense_biegrid dofs={DIM} cells_per_axis={CELLS}");
    let start = Instant::now();
    let matrix = matrix();
    println!(
        "dense_matrix_build_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let schedule = vec![level(16, 16), level(8, 16), level(4, 16)];
    println!(
        "planned_cycles_per_level={:?}",
        schedule.iter().map(Vec::len).collect::<Vec<_>>()
    );

    let start = Instant::now();
    let (factors, transformed) =
        DenseMultilevelFactorization::factorize(&matrix, &schedule, IdMethod::Rrqr).unwrap();
    let cycles = factors.cycles().len();
    let active = factors.active_indices().len();
    let operator = DenseFactorizedOperator::new(factors, &transformed).unwrap();
    println!("factorization_seconds={:.6}", start.elapsed().as_secs_f64());
    println!("executed_cycles={cycles} active_dofs={active}");

    let mut rhs = DynArray::<f64, 2>::from_shape([DIM, 1]);
    for row in 0..DIM {
        let t = (row + 1) as f64;
        rhs[[row, 0]] = t.sin() + 0.25 * t.cos();
    }
    let dense = dense_apply(&matrix, &rhs);
    let approximate = operator.apply(&rhs, FactorMode::NoTrans, false).unwrap();
    let forward_error = relative_l2(&approximate, &dense);
    let solution = operator.apply(&rhs, FactorMode::NoTrans, true).unwrap();
    let recovered = dense_apply(&matrix, &solution);
    let inverse_residual = relative_l2(&recovered, &rhs);
    println!("forward_relative_l2_error={forward_error:.12e}");
    println!("direct_inverse_true_relative_residual={inverse_residual:.12e}");
}
