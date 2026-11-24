//! Matched historical/RSRS comparison using the exact bempp-octree
//! hierarchy, deterministic Y samples, and historical numerical choices.

extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

use bempp_octree::{Octree, Point};
use mpi::topology::SimpleCommunicator;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rlst::{dense::array::DynArray, dot_blas, EvaluateArray};
use rsrs_core::{
    configuration::MatrixSymmetry,
    elementary_factor::FactorMode,
    interpolative_decomposition::IdMethod,
    regularized_least_squares::{
        Regularization, RegularizationMode, RegularizedLeastSquaresMethod,
    },
    sampled_id::{InMemorySampleProvider, SampleBlock, ZMode},
    sampled_multilevel::SampledMultilevelFactorization,
    sampled_options::{
        ExtractionOptions, NullificationMethod, PivotOptions, SampledFactorizationOptions,
    },
};
use rsrs_octree::fixed_rank_plan_from_octree;
use std::time::Instant;

const CELLS: usize = 70;
const SIDE: usize = CELLS + 1;
const DIM: usize = SIDE * SIDE;
const RANK: usize = 16;

fn index_fingerprint(values: &[usize]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        (hash ^ (*value as u64)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn matrix_fingerprint(values: &DynArray<f64, 2>) -> (f64, f64) {
    let [rows, columns] = values.shape();
    let mut sum = 0.0;
    let mut weighted = 0.0;
    for row in 0..rows {
        for column in 0..columns {
            let value = values[[row, column]];
            sum += value;
            weighted += value * (1 + row * columns + column) as f64;
        }
    }
    (sum, weighted)
}

fn points_and_matrix() -> (Vec<Point>, DynArray<f64, 2>) {
    let h = 1.0 / CELLS as f64;
    let points = (0..SIDE)
        .flat_map(|iy| (0..SIDE).map(move |ix| Point::new([ix as f64 * h, iy as f64 * h, 0.0], 0)))
        .collect::<Vec<_>>();
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
    let mut matrix = DynArray::<f64, 2>::from_shape([DIM, DIM]);
    for row in 0..DIM {
        let (ry, rx) = (row / SIDE, row % SIDE);
        for column in 0..DIM {
            matrix[[row, column]] = if row == column {
                diagonal
            } else {
                let (cy, cx) = (column / SIDE, column % SIDE);
                let dx = (rx as f64 - cx as f64) * h;
                let dy = (ry as f64 - cy as f64) * h;
                -(dx.hypot(dy)).ln() / (2.0 * std::f64::consts::PI * DIM as f64)
            };
        }
    }
    (points, matrix)
}

fn samples(matrix: &DynArray<f64, 2>, rows: usize) -> InMemorySampleProvider<f64> {
    let mut y_test = DynArray::<f64, 2>::from_shape([rows, DIM]);
    let mut rng = ChaCha8Rng::seed_from_u64(0x5253_5253_5041_0001);
    for row in 0..rows {
        for column in 0..DIM {
            y_test[[row, column]] = rng.sample(StandardNormal);
        }
    }
    let transpose = matrix.r().transpose().eval();
    let y_sketch = dot_blas!(y_test.r(), transpose.r());
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
    let universe = mpi::initialize().unwrap();
    let communicator: SimpleCommunicator = universe.world();
    let build_started = Instant::now();
    let (points, matrix) = points_and_matrix();
    println!(
        "dense_matrix_build_seconds={:.6}",
        build_started.elapsed().as_secs_f64()
    );

    let tree = Octree::new(&points, 6, 50, &communicator);
    let plan = fixed_rank_plan_from_octree(&tree, 1, RANK, 8);
    println!(
        "levels={:?} cycles={:?} batches={:?} required_sample_rows={}",
        plan.levels
            .iter()
            .map(|level| level.level)
            .collect::<Vec<_>>(),
        plan.levels
            .iter()
            .map(|level| level
                .batches
                .iter()
                .map(|batch| batch.cycles.len())
                .sum::<usize>())
            .collect::<Vec<_>>(),
        plan.levels
            .iter()
            .map(|level| level.batches.len())
            .collect::<Vec<_>>(),
        plan.required_sample_rows,
    );
    assert_eq!(plan.required_sample_rows, 602);

    let sampling_started = Instant::now();
    let provider = samples(&matrix, plan.required_sample_rows);
    println!(
        "sampling_seconds={:.6}",
        sampling_started.elapsed().as_secs_f64()
    );
    let historical_regularization = Regularization {
        mode: RegularizationMode::Historical,
        relative: 1.0e-10,
        absolute_floor: 0.0,
    };
    let extraction = ExtractionOptions {
        solver: RegularizedLeastSquaresMethod::Lu,
        regularization: historical_regularization,
    };
    let nullification_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "projection".to_string());
    let nullification = match nullification_name.as_str() {
        "projection" => NullificationMethod::Projection {
            solver: RegularizedLeastSquaresMethod::Lu,
            regularization: historical_regularization,
        },
        "qr" => NullificationMethod::Qr { tolerance: 1.0e-10 },
        "svd" => NullificationMethod::Svd { tolerance: 1.0e-10 },
        other => panic!("unknown nullification method {other:?}; use projection, qr, or svd"),
    };
    println!("nullification={nullification_name}");
    let options = SampledFactorizationOptions {
        nullification,
        near_extraction: extraction,
        diagonal_extraction: extraction,
        near_pivot: PivotOptions {
            method: rsrs_core::configuration::PivotMethod::Lu,
            shift: 1.0e-10,
        },
        diagonal_pivot: PivotOptions {
            method: rsrs_core::configuration::PivotMethod::Lu,
            shift: 0.0,
        },
    };
    let rayon_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let factor_started = Instant::now();
    let (factors, report) =
        SampledMultilevelFactorization::factorize_planned_with_report_and_options(
            &provider,
            plan.required_sample_rows,
            &plan,
            IdMethod::Rrqr,
            options,
            rayon_threads,
            MatrixSymmetry::Symmetric,
        )
        .unwrap();
    println!(
        "factorization_seconds={:.6}",
        factor_started.elapsed().as_secs_f64()
    );
    println!(
        "used_samples={} cycles={} active_dofs={}",
        report.used_sample_rows,
        factors.cycles().len(),
        factors.active_indices().len()
    );
    for (cycle_index, executed) in factors.cycles().iter().enumerate() {
        let interpolation = matrix_fingerprint(executed.cycle.id_update());
        println!(
            "trace_id cycle={} level={} batch={} skeleton_len={} skeleton_hash={:016x} redundant_len={} redundant_hash={:016x} interpolation_sum={:.16e} interpolation_weighted={:.16e}",
            cycle_index,
            executed.level,
            executed.batch,
            executed.cycle.skeleton_indices().len(),
            index_fingerprint(executed.cycle.skeleton_indices()),
            executed.cycle.redundant_indices().len(),
            index_fingerprint(executed.cycle.redundant_indices()),
            interpolation.0,
            interpolation.1,
        );
    }

    let mut rhs = DynArray::<f64, 2>::from_shape([DIM, 1]);
    for row in 0..DIM {
        let t = (row + 1) as f64;
        rhs[[row, 0]] = t.sin() + 0.25 * t.cos();
    }
    let reference = dot_blas!(matrix.r(), rhs.r());
    let approximation = factors.apply(&rhs, FactorMode::NoTrans, false).unwrap();
    let solution = factors.apply(&rhs, FactorMode::NoTrans, true).unwrap();
    let recovered = dot_blas!(matrix.r(), solution.r());
    println!(
        "forward_relative_l2_error={:.12e}",
        relative_l2(&approximation, &reference)
    );
    println!(
        "direct_inverse_true_relative_residual={:.12e}",
        relative_l2(&recovered, &rhs)
    );
}
