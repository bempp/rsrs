//! Matched current RLST sampled-RSRS benchmark for the historical 5,041-DOF case.

extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rlst::{dense::array::DynArray, dot_blas, EvaluateArray};
use rsrs_core::{
    configuration::MatrixSymmetry,
    elementary_factor::FactorMode,
    interpolative_decomposition::IdMethod,
    regularized_least_squares::{Regularization, RegularizedLeastSquaresMethod},
    sampled_id::{InMemorySampleProvider, SampleBlock, ZMode},
    sampled_multilevel::{SampledCyclePlan, SampledMultilevelFactorization},
    scheduling::FactorizationPlan,
};
use std::time::Instant;

const CELLS: usize = 70;
const SIDE: usize = CELLS + 1;
const DIM: usize = SIDE * SIDE;
const SAMPLE_ROWS: usize = 602;
const RANK: usize = 16;

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

fn level(boxes_per_axis: usize) -> Vec<SampledCyclePlan> {
    let mut boxes = vec![Vec::new(); boxes_per_axis * boxes_per_axis];
    for iy in 0..SIDE {
        for ix in 0..SIDE {
            let bx = (ix * boxes_per_axis / SIDE).min(boxes_per_axis - 1);
            let by = (iy * boxes_per_axis / SIDE).min(boxes_per_axis - 1);
            boxes[by * boxes_per_axis + bx].push(iy * SIDE + ix);
        }
    }
    let mut plans = Vec::new();
    for by in 0..boxes_per_axis {
        for bx in 0..boxes_per_axis {
            let target = boxes[by * boxes_per_axis + bx].clone();
            if target.len() <= RANK {
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
            plans.push(SampledCyclePlan {
                box_id: by * boxes_per_axis + bx,
                parent_id: None,
                target_indices: target,
                near_indices: near,
                rank: RANK,
                sample_rows: None,
            });
        }
    }
    plans
}

fn samples(matrix: &DynArray<f64, 2>) -> InMemorySampleProvider<f64> {
    let mut y_test = DynArray::<f64, 2>::from_shape([SAMPLE_ROWS, DIM]);
    let mut y_rng = ChaCha8Rng::seed_from_u64(0x5253_5253_5041_0001);
    for row in 0..SAMPLE_ROWS {
        for column in 0..DIM {
            y_test[[row, column]] = y_rng.sample(StandardNormal);
        }
    }
    let transpose = matrix.r().transpose().eval();
    let y_sketch = dot_blas!(y_test.r(), transpose.r());
    let fingerprint = |array: &DynArray<f64, 2>| {
        let mut sum = 0.0;
        let mut weighted = 0.0;
        for row in 0..array.shape()[0] {
            for column in 0..array.shape()[1] {
                let value = array[[row, column]];
                sum += value;
                weighted += value * ((row * 17 + column * 31 + 1) % 1009) as f64;
            }
        }
        (sum, weighted)
    };
    let test_fingerprint = fingerprint(&y_test);
    let sketch_fingerprint = fingerprint(&y_sketch);
    println!(
        "y_test_fingerprint={:.16e},{:.16e}",
        test_fingerprint.0, test_fingerprint.1
    );
    println!(
        "y_sketch_fingerprint={:.16e},{:.16e}",
        sketch_fingerprint.0, sketch_fingerprint.1
    );
    InMemorySampleProvider::new(
        SampleBlock {
            y_test,
            y_sketch,
            z_test: None,
            z_sketch: None,
        },
        ZMode::Transpose,
    )
    .unwrap()
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
    println!("baseline=refactored_sampled_biegrid dofs={DIM} cells_per_axis={CELLS}");
    let start = Instant::now();
    let matrix = matrix();
    println!(
        "dense_matrix_build_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let schedule = vec![level(16), level(8), level(4)];
    println!(
        "planned_cycles_per_level={:?}",
        schedule.iter().map(Vec::len).collect::<Vec<_>>()
    );

    let start = Instant::now();
    let provider = samples(&matrix);
    println!("sampling_seconds={:.6}", start.elapsed().as_secs_f64());
    let rayon_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let plan = FactorizationPlan::from_cycle_levels(DIM, schedule, SAMPLE_ROWS);
    println!(
        "planned_batches_per_level={:?} rayon_threads={rayon_threads}",
        plan.levels
            .iter()
            .map(|level| level.batches.len())
            .collect::<Vec<_>>()
    );
    let start = Instant::now();
    let (factors, report) =
        SampledMultilevelFactorization::factorize_planned_with_report_and_symmetry(
            &provider,
            SAMPLE_ROWS,
            &plan,
            IdMethod::Rrqr,
            Regularization {
                mode: rsrs_core::regularized_least_squares::RegularizationMode::Scaled,
                relative: 1.0e-16,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
            rayon_threads,
            MatrixSymmetry::Symmetric,
        )
        .unwrap();
    println!("factorization_seconds={:.6}", start.elapsed().as_secs_f64());
    println!(
        "used_samples={} executed_cycles={} active_dofs={}",
        report.used_sample_rows,
        factors.cycles().len(),
        factors.active_indices().len()
    );

    let mut rhs = DynArray::<f64, 2>::from_shape([DIM, 1]);
    for row in 0..DIM {
        let t = (row + 1) as f64;
        rhs[[row, 0]] = t.sin() + 0.25 * t.cos();
    }
    let reference = dot_blas!(matrix.r(), rhs.r());
    let approximation = factors.apply(&rhs, FactorMode::NoTrans, false).unwrap();
    let forward_error = relative_l2(&approximation, &reference);
    let solution = factors.apply(&rhs, FactorMode::NoTrans, true).unwrap();
    let recovered = dot_blas!(matrix.r(), solution.r());
    let inverse_residual = relative_l2(&recovered, &rhs);
    println!("forward_relative_l2_error={forward_error:.12e}");
    println!("direct_inverse_true_relative_residual={inverse_residual:.12e}");
}
