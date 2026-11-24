extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rlst::{
    base_types::c64,
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    dot_blas, EvaluateArray, Gemm, Lapack, RlstScalar,
};
use rsrs_core::{
    elementary_factor::FactorMode,
    interpolative_decomposition::IdMethod,
    regularized_least_squares::{Regularization, RegularizedLeastSquaresMethod},
    sampled_id::{InMemorySampleProvider, SampleBlock, ZMode},
    sampled_multilevel::{SampledCyclePlan, SampledMultilevelFactorization},
};
use std::{fs, path::PathBuf, time::Instant};

const MAGIC: &[u8] = b"RSRS_BIEGRID_FIXTURE_V1";
const SAMPLE_ROWS: usize = 384;

struct Reader {
    bytes: Vec<u8>,
    offset: usize,
}
impl Reader {
    fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, count: usize) -> &[u8] {
        let start = self.offset;
        self.offset += count;
        &self.bytes[start..self.offset]
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }
}

struct Fixture {
    points: Vec<[f64; 3]>,
    real_symmetric: DynArray<f64, 2>,
    real_nonsymmetric: DynArray<f64, 2>,
    complex_symmetric: DynArray<c64, 2>,
    complex_nonsymmetric: DynArray<c64, 2>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/biegrid_perturbed/cells22_scale1e-2_seed12345.bin");
    let mut reader = Reader::new(fs::read(path).unwrap());
    assert_eq!(reader.take(MAGIC.len()), MAGIC);
    let n = reader.u64() as usize;
    let _cells = reader.u64();
    let _scale = reader.f64();
    let _seed = reader.u64();
    let points = (0..n)
        .map(|_| [reader.f64(), reader.f64(), reader.f64()])
        .collect();
    let mut real = || {
        let mut matrix = DynArray::<f64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = reader.f64();
            }
        }
        matrix
    };
    let real_symmetric = real();
    let real_nonsymmetric = real();
    let mut complex = || {
        let mut matrix = DynArray::<c64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = c64::new(reader.f64(), reader.f64());
            }
        }
        matrix
    };
    let complex_symmetric = complex();
    let complex_nonsymmetric = complex();
    Fixture {
        points,
        real_symmetric,
        real_nonsymmetric,
        complex_symmetric,
        complex_nonsymmetric,
    }
}

fn plans(points: &[[f64; 3]], boxes_per_axis: usize, rank: usize) -> Vec<Vec<SampledCyclePlan>> {
    let min_x = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let max_x = points
        .iter()
        .map(|p| p[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut boxes = vec![Vec::new(); boxes_per_axis * boxes_per_axis];
    for (index, point) in points.iter().enumerate() {
        let x = (((point[0] - min_x) / (max_x - min_x) * boxes_per_axis as f64).floor() as usize)
            .min(boxes_per_axis - 1);
        let y = (((point[1] - min_y) / (max_y - min_y) * boxes_per_axis as f64).floor() as usize)
            .min(boxes_per_axis - 1);
        boxes[y * boxes_per_axis + x].push(index);
    }
    let mut level = Vec::new();
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
            level.push(SampledCyclePlan {
                box_id: by * boxes_per_axis + bx,
                parent_id: None,
                target_indices: target,
                near_indices: near,
                rank,
                sample_rows: None,
            });
        }
    }
    vec![level]
}

trait GaussianScalar: MatrixId + Lapack + Gemm {
    fn gaussian(rng: &mut ChaCha8Rng) -> Self;
}
impl GaussianScalar for f64 {
    fn gaussian(rng: &mut ChaCha8Rng) -> Self {
        rng.sample(StandardNormal)
    }
}
impl GaussianScalar for c64 {
    fn gaussian(rng: &mut ChaCha8Rng) -> Self {
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        c64::new(rng.sample(StandardNormal), rng.sample(StandardNormal)) * scale
    }
}

fn samples<Item: GaussianScalar>(
    matrix: &DynArray<Item, 2>,
    seed: u64,
) -> InMemorySampleProvider<Item> {
    let n = matrix.shape()[0];
    let mut y_test = DynArray::<Item, 2>::from_shape([SAMPLE_ROWS, n]);
    let mut z_test = DynArray::<Item, 2>::from_shape([SAMPLE_ROWS, n]);
    let mut y_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut z_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e3779b97f4a7c15);
    for row in 0..SAMPLE_ROWS {
        for column in 0..n {
            y_test[[row, column]] = Item::gaussian(&mut y_rng);
            z_test[[row, column]] = Item::gaussian(&mut z_rng);
        }
    }
    let transpose = matrix.r().transpose().eval();
    let y_sketch = dot_blas!(y_test.r(), transpose.r());
    let z_sketch = dot_blas!(z_test.r(), matrix.r());
    InMemorySampleProvider::new(
        SampleBlock {
            y_test,
            y_sketch,
            z_test: Some(z_test),
            z_sketch: Some(z_sketch),
        },
        ZMode::Transpose,
    )
    .unwrap()
}

fn run<Item>(matrix: &DynArray<Item, 2>, points: &[[f64; 3]], label: &str, seed: u64)
where
    Item: GaussianScalar,
    Item::Real: Into<Item> + Into<f64>,
{
    let schedule = plans(points, 4, 16);
    let max_near = schedule[0]
        .iter()
        .map(|plan| plan.near_indices.len())
        .max()
        .unwrap();
    println!(
        "sampled_biegrid label={label} n={} samples={SAMPLE_ROWS} cycles={} max_near={max_near}",
        matrix.shape()[0],
        schedule[0].len()
    );
    let start = Instant::now();
    let provider = samples(matrix, seed);
    let sampling_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let factors = SampledMultilevelFactorization::factorize(
        &provider,
        SAMPLE_ROWS,
        &schedule,
        IdMethod::Rrqr,
        Regularization {
            mode: rsrs_core::regularized_least_squares::RegularizationMode::Scaled,
            relative: Item::real(1.0e-14),
            absolute_floor: Item::real(0.0),
        },
        RegularizedLeastSquaresMethod::Cholesky,
    )
    .unwrap();
    let factorization_seconds = start.elapsed().as_secs_f64();
    let cycles = factors.cycles().len();
    let active = factors.active_indices().len();
    let n = matrix.shape()[0];
    let mut identity = DynArray::<Item, 2>::from_shape([n, n]);
    for index in 0..n {
        identity[[index, index]] = Item::one();
    }
    let approximation = factors
        .apply(&identity, FactorMode::NoTrans, false)
        .unwrap();
    let approximation_error = relative_frobenius(&approximation, matrix);
    let mut rhs = DynArray::<Item, 2>::from_shape([n, 1]);
    for row in 0..n {
        rhs[[row, 0]] = Item::from_real(Item::real(
            ((row + 1) as f64).sin() + 0.25 * ((row + 1) as f64).cos(),
        ));
    }
    let exact_product = dense_apply(matrix, &rhs);
    let approximate_product = factors.apply(&rhs, FactorMode::NoTrans, false).unwrap();
    let forward_error = relative_frobenius(&approximate_product, &exact_product);
    let solution = factors.apply(&rhs, FactorMode::NoTrans, true).unwrap();
    let recovered = dense_apply(matrix, &solution);
    let inverse_residual = relative_frobenius(&recovered, &rhs);
    println!("sampled_result label={label} sampling_seconds={sampling_seconds:.6} factorization_seconds={factorization_seconds:.6} executed_cycles={cycles} active_dofs={active} approximation_error={approximation_error:.12e} forward_error={forward_error:.12e} inverse_residual={inverse_residual:.12e}");
}

fn dense_apply<Item: RlstScalar>(
    matrix: &DynArray<Item, 2>,
    input: &DynArray<Item, 2>,
) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([matrix.shape()[0], input.shape()[1]]);
    for row in 0..matrix.shape()[0] {
        for column in 0..input.shape()[1] {
            let mut value = Item::zero();
            for inner in 0..matrix.shape()[1] {
                value += matrix[[row, inner]] * input[[inner, column]];
            }
            result[[row, column]] = value;
        }
    }
    result
}

fn relative_frobenius<Item: RlstScalar>(
    actual: &DynArray<Item, 2>,
    expected: &DynArray<Item, 2>,
) -> f64
where
    Item::Real: Into<f64>,
{
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in 0..actual.shape()[0] {
        for column in 0..actual.shape()[1] {
            let error: f64 =
                RlstScalar::abs(actual[[row, column]] - expected[[row, column]]).into();
            let reference: f64 = RlstScalar::abs(expected[[row, column]]).into();
            numerator += error * error;
            denominator += reference * reference;
        }
    }
    (numerator / denominator).sqrt()
}

fn main() {
    let fixture = fixture();
    run(
        &fixture.real_symmetric,
        &fixture.points,
        "real-symmetric",
        12345,
    );
    run(
        &fixture.real_nonsymmetric,
        &fixture.points,
        "real-nonsymmetric",
        22345,
    );
    run(
        &fixture.complex_symmetric,
        &fixture.points,
        "complex-symmetric",
        32345,
    );
    run(
        &fixture.complex_nonsymmetric,
        &fixture.points,
        "complex-nonsymmetric",
        42345,
    );
}
